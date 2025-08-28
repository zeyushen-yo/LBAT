"""
FSDP PPO Trainer with Ray-based single controller.
Adapted from the excellently written verl implementation.
"""

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from ragen.trainer import core_algos
from ragen.trainer.core_algos import agg_loss
from ragen.env.lbat.accel import AccelBuffer, BanditParams, PEAParams, OPSParams, LBATItem, perturb_params
from ragen.env import REGISTERED_ENV_CONFIGS
from ragen.env.lbat.config import LBATEnvConfig
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

WorkerType = Type[Worker]


from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager, compute_response_mask, _timer, apply_kl_penalty, AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as VerlRayPPOTrainer

import torch
from verl.utils.torch_functional import masked_mean

from ragen.llm_agent.agent_proxy import LLMAgentProxy
from ragen.utils import GenerationsLogger


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, bi_level_gae=False, high_level_gamma=1.0):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        if bi_level_gae:
            advantages, returns = core_algos.compute_bi_level_gae_advantage_return(
                token_level_rewards=data.batch["token_level_rewards"],
                values=data.batch["values"],
                loss_mask=data.batch["response_mask"],
                gamma=gamma,
                lam=lam,
                high_level_gamma=high_level_gamma,
            )
        else:
            advantages, returns = core_algos.compute_gae_advantage_return(
                token_level_rewards=data.batch["token_level_rewards"],
                values=data.batch["values"],
                response_mask=data.batch["response_mask"],
                gamma=gamma,
                lam=lam,
            )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


class RayAgentTrainer(VerlRayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn)
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0
        # do not use the original val logger, but use this here
        self.generations_logger = GenerationsLogger()
        accel_cfg = getattr(self.config, "accel", None)
        self.accel_enabled = bool(accel_cfg and accel_cfg.get("enabled", True))
        if self.accel_enabled:
            cap = int(accel_cfg.get("buffer_capacity", 64))
            temp = float(accel_cfg.get("temperature", 0.5))
            seed = int(accel_cfg.get("seed", 0))
            # Single unified buffer storing LBATItem (family + params)
            self.accel_buf = AccelBuffer(capacity=cap, temperature=temp, seed=seed)
            self.accel_rng = np.random.default_rng(seed)
            self.accel_warmup_n = int(accel_cfg.get("warmup_random_envs", 4))
            self.accel_generator_per_iter = int(accel_cfg.get("generator_per_iter", 1))
            self.accel_n_eval_repeats = int(accel_cfg.get("n_eval_repeats", 2))
            self.accel_perturb_sigma_small = float(accel_cfg.get("perturb_sigma_small", 0.05))
            self.accel_perturb_sigma_big = float(accel_cfg.get("perturb_sigma_big", 0.15))
            # choose which env from a batch to perturb ("max_gap" or "last")
            self.accel_pick_policy = str(accel_cfg.get("pick_policy", "max_gap"))

            self._accel_bandit_cfg = REGISTERED_ENV_CONFIGS["lbat"]()

            print("_accel_bandit_cfg: ", self._accel_bandit_cfg) # for testing

    # ===== ACCEL helpers =====

    def _accel__extract_infos(self, batch):
        print("batch: ", batch) # for testing
        mi = getattr(batch, "meta_info", {}) or {}
        print("mi: ", mi) # for testing
        return mi.get("accel_env_info", [])

    def _accel__maybe_add_many(self, infos):
        """Group duplicates by canonical params, take median gap, add once per env."""
        from collections import defaultdict
        import numpy as np

        grouped = {}  # key -> {"item": LBATItem, "gaps": [float,...]}
        for it in infos:
            params_dict = it.get("params")
            family = it.get("family")
            # New LBAT payload packs family inside params
            if isinstance(params_dict, dict) and (family is None) and ("family" in params_dict):
                family = params_dict.get("family", "mab")
                params_dict = params_dict.get("params", {})
            gap = it.get("gap")
            if params_dict is None or gap is None:
                continue
            if family == "mab":
                p = BanditParams(**params_dict)
            elif family == "pea":
                p = PEAParams(**params_dict)
            else:
                p = OPSParams(**params_dict)
            item = LBATItem(family=family, params=p)
            key = item.canonical_key()

            entry = grouped.get(key)
            if entry is None:
                grouped[key] = {"item": item, "gaps": [float(gap)]}
            else:
                entry["gaps"].append(float(gap))

        added = 0
        for entry in grouped.values():
            med_gap = float(np.median(entry["gaps"]))
            if self.accel_buf.add(score=med_gap, params=entry["item"]):
                added += 1
        return added

    def _accel__pick_one(self, infos):
        """Choose an env from a list to perturb."""
        if not infos:
            return None
        if self.accel_pick_policy == "max_gap":
            return max(infos, key=lambda d: d["gap"])
        return infos[-1]

    def _accel__eval_override_once(self, params_dict, repeats=None):
        """Run an eval-only episode with an explicit env override and add to buffer."""
        reps = repeats or self.accel_n_eval_repeats
        gaps = []
        last_infos = None
        for _ in range(reps):
            meta = {
                "validate": True,
                "accel": {"params_override": params_dict},
            }
            eval_batch = self.agent_proxy.rollout(DataProto(meta_info=meta), val=True)
            infos = self._accel__extract_infos(eval_batch)
            if not infos:
                continue
            last_infos = infos
            gaps.append(infos[0]["gap"])
        if gaps and last_infos:
            # Use median gap to reduce noise
            med_gap = float(np.median(gaps))
            print("med_gap: ", med_gap) # for testing
            # unpack family from nested params if needed
            fam = last_infos[0].get("family")
            p_dict = last_infos[0].get("params")
            if isinstance(p_dict, dict) and fam is None and ("family" in p_dict):
                fam = p_dict.get("family", "mab")
                p_dict = p_dict.get("params", {})
            if fam == "mab":
                p = BanditParams(**p_dict)
            elif fam == "pea":
                p = PEAParams(**p_dict)
            else:
                p = OPSParams(**p_dict)
            self.accel_buf.add(score=med_gap, params=LBATItem(family=fam, params=p))
            return med_gap
        return None

    def _accel__generator_discovery_once(self):
        """Discover a fresh random env (no training), maybe add to buffer."""
        meta = {
            "validate": True,          # eval-only
            "accel": {"params_override": None},  # force fresh random
        }
        eval_batch = self.agent_proxy.rollout(DataProto(meta_info=meta), val=True)
        infos = self._accel__extract_infos(eval_batch)
        self._accel__maybe_add_many(infos)
        return infos

    def _accel__warmup(self):
        for _ in range(self.accel_warmup_n):
            print("warmup iteration: ", _)
            self._accel__generator_discovery_once()

    def _accel__build_params_override(self, *, val: bool) -> Optional[list]:
        if not self.accel_enabled or len(self.accel_buf) == 0:
            return None
        if val:
            env_groups = int(self.config.es_manager.val.env_groups)
            group_size = int(self.config.es_manager.val.group_size)
        else:
            env_groups = int(self.config.es_manager.train.env_groups)
            group_size = int(self.config.es_manager.train.group_size)

        per_env_overrides = []
        for _ in range(env_groups):
            item = self.accel_buf.sample(1)[0]
            per_env_overrides.extend([item.to_dict()] * group_size)
        return per_env_overrides

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        assert self.config.trainer.total_training_steps is not None, "must determine total training steps"
        total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")
        # val_start = 100000
        # self.train_seeds = [seed for seed in range(0, self.config.trainer.total_training_steps * 1000, 1000)]
        # self.val_seeds = [seed for seed in range(val_start, val_start + self.config.trainer.validation_steps)]

    def init_agent_proxy(self):
        self.agent_proxy = LLMAgentProxy(
            config=self.config,
            actor_rollout_wg=self.actor_rollout_wg,
            tokenizer=self.tokenizer
        )
    def _maybe_log_generations(self, inputs, outputs, scores, _type="val"):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.generations_to_log_to_wandb[_type]

        if generations_to_log == 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.generations_logger.log(self.config.trainer.logger, samples, self.global_steps, _type)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        env_metric_dict = {}
        for step in range(self.config.trainer.validation_steps):
            # Store original inputs
            input_texts = ["" for _ in range(self.config.es_manager.val.env_groups * self.config.es_manager.val.group_size)]
            sample_inputs.extend(input_texts)
            
            meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            test_gen_batch = DataProto(batch=None, non_tensor_batch=None, meta_info=meta_info)
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            import time
            start_time = time.time()
            test_batch = self.agent_proxy.rollout(test_gen_batch, val=True)
            end_time = time.time()
            print(f"validation generation time: {end_time - start_time} seconds")
            for key, value in test_batch.meta_info["metrics"].items():
                if "val-env/" + key not in env_metric_dict:
                    env_metric_dict["val-env/" + key] = []
                env_metric_dict["val-env/" + key].append(value)

            # Store generated outputs
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, _type="val")

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = reduce_metrics(env_metric_dict)

        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
 
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and not self.ref_in_actor:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )


    def _save_checkpoint(self):
        """ 
        Different from VerlRayPPOTrainer, we have no dataloader so we won"t save it. Other logic is the same.
        """
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
         to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            logger.maybe_sync()
            if self.config.trainer.get("val_only", False):
                return

        if self.accel_enabled and self.accel_warmup_n > 0:
            self._accel__warmup()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        def _process_batch_for_logging(batch):
            inputs = batch.batch["input_ids"]
            inputs = [self.tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in inputs]
            outputs = [""] * len(inputs)
            scores = batch.batch["rm_scores"].sum(-1).cpu().tolist()
            return inputs, outputs, scores

        def _filter_rollout(batch):
            """filter rollout based on in-group max - in-group mean. We want those groups to have high-quality rollouts that deviates significantly from the mean"""
            rollout_filter_ratio = self.config.actor_rollout_ref.rollout.rollout_filter_ratio
            num_groups, group_size = self.config.es_manager.train.env_groups, self.config.es_manager.train.group_size

            rm_scores = batch.batch["original_rm_scores"].sum(dim=-1).view(num_groups, group_size)
            in_group_std = rm_scores.std(dim=-1)
            in_group_max = rm_scores.max(dim=-1).values
            in_group_mean = rm_scores.mean(dim=-1)
            if rollout_filter_ratio == 1:
                return batch, {"rollout/in_group_std": in_group_std.mean(), "rollout/in_group_max": in_group_max.mean(), "rollout/in_group_mean": in_group_mean.mean(), "rollout/chosen_in_group_std": in_group_std.mean(), "rollout/chosen_in_group_max": in_group_max.mean(), "rollout/chosen_in_group_mean": in_group_mean.mean()}

            if self.config.actor_rollout_ref.rollout.rollout_filter_type == "std_rev":
                top_groups = (-in_group_std).topk(int(rollout_filter_ratio * num_groups)).indices
            elif self.config.actor_rollout_ref.rollout.rollout_filter_type == "std":
                top_groups = in_group_std.topk(int(rollout_filter_ratio * num_groups)).indices
            else:
                raise ValueError(f"Invalid rollout filter type: {self.config.actor_rollout_ref.rollout.rollout_filter_type}")

            mask = torch.zeros(num_groups, dtype=torch.bool)
            mask[top_groups] = True
            mask = mask.unsqueeze(1).expand(-1, group_size).flatten()

            batch.batch = batch.batch[mask]

            for key, value in batch.non_tensor_batch.items():
                if isinstance(value, np.ndarray):
                    batch.non_tensor_batch[key] = value[mask]
                else:
                    batch.non_tensor_batch[key] = [v for v, m in zip(value, mask) if m]

            metrics = {
                "rollout/in_group_std": in_group_std.mean(),
                "rollout/in_group_max": in_group_max.mean(),
                "rollout/in_group_mean": in_group_mean.mean(),
                "rollout/chosen_in_group_std": in_group_std[top_groups].mean(),
                "rollout/chosen_in_group_max": in_group_max[top_groups].mean(),
                "rollout/chosen_in_group_mean": in_group_mean[top_groups].mean()
            }
            return batch, metrics

        import time
        self.start_time = time.time()
        for step in range(self.total_training_steps):
            # metrics = {}
            timing_raw = {}
            
            if self.accel_enabled and self.accel_generator_per_iter > 0:
                for _ in range(self.accel_generator_per_iter):
                    gen_infos = self._accel__generator_discovery_once()

            batch: DataProto = DataProto()
            is_last_step = self.global_steps >= self.total_training_steps

            with _timer("step", timing_raw):
                # generate a batch
                with _timer("gen", timing_raw):
                    if self.accel_enabled:
                        ovr = self._accel__build_params_override(val=False)
                        meta = {"accel": {"params_override": ovr}}
                        batch = self.agent_proxy.rollout(DataProto(meta_info=meta), val=False)
                    else:
                        batch = self.agent_proxy.rollout(DataProto(), val=False)

                    batch, metrics = _filter_rollout(batch)
                    metrics.update({"train/" + key: value for key, value in batch.meta_info["metrics"].items()})

                    inputs, outputs, scores = _process_batch_for_logging(batch)
                    # self._maybe_log_generations(inputs=inputs, outputs=outputs, scores=scores, _type="train")

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    # TODO: check if this is correct. Not tested yer
                    logger.log("[NotImplemented] REMAX implementation is not tested yet in RAGEN. Exiting.")
                    exit()
                    with _timer("gen_max", timing_raw):
                        gen_baseline_batch = deepcopy(batch)
                        gen_baseline_batch.meta_info["do_sample"] = False
                        gen_baseline_output = self.agent_proxy.rollout(gen_baseline_batch, val=False)

                        batch = batch.union(gen_baseline_output)
                        reward_baseline_tensor = self.reward_fn(batch)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_batch, gen_baseline_output

                # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                            # dtype=object)
                # repeat to align with repeated responses in rollout
                # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                # batch = batch.union(gen_batch_output)

                # NOTE reward normalization already done in ctx_manager, so set group size = 1 here
                # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                #                                             dtype=object)
                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["group_ids"]

                # batch.batch["response_mask"] = compute_response_mask(batch)
                batch.batch["response_mask"] = batch.batch["loss_mask"]
                # balance the number of valid tokens on each dp rank.
                # Note that this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                if self.use_rm:
                    with _timer("reward", timing_raw):
                    # compute reward model score
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                # recompute old_log_probs
                with _timer("old_log_prob", timing_raw):
                    # Pad to be divisible by DP world size to avoid empty shards on some ranks
                    padded_batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(padded_batch)
                    old_log_prob = unpad_dataproto(old_log_prob, pad_size)
                    batch = batch.union(old_log_prob)
                    avg_old_log_prob = masked_mean(old_log_prob.batch["old_log_probs"], batch.batch["response_mask"])
                    metrics.update({"rollout/old_log_prob": avg_old_log_prob})

                if self.use_reference_policy:
                    # compute reference log_prob
                    with _timer("ref", timing_raw):
                        if not self.ref_in_actor:
                            padded_batch, pad_size = pad_dataproto_to_divisor(batch, self.ref_policy_wg.world_size)
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(padded_batch)
                            ref_log_prob = unpad_dataproto(ref_log_prob, pad_size)
                        else:
                            padded_batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(padded_batch)
                            ref_log_prob = unpad_dataproto(ref_log_prob, pad_size)
                        batch = batch.union(ref_log_prob)
                        avg_ref_log_prob = masked_mean(ref_log_prob.batch["ref_log_prob"], batch.batch["response_mask"])
                        metrics.update({"rollout/ref_log_prob": avg_ref_log_prob})

                # compute values
                if self.use_critic:
                    with _timer("values", timing_raw):
                        padded_batch, pad_size = pad_dataproto_to_divisor(batch, self.critic_wg.world_size)
                        values = self.critic_wg.compute_values(padded_batch)
                        values = unpad_dataproto(values, pad_size)
                        batch = batch.union(values)

                with _timer("adv", timing_raw):
                    # we combine with rule-based rm
                    reward_extra_infos_dict: dict[str, list]
                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    batch.batch["token_level_scores"] = reward_tensor

                    print(f"{list(reward_extra_infos_dict.keys())=}")
                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty, multi_turn=True)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process

                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        multi_turn=True,
                        high_level_gamma=self.config.algorithm.high_level_gamma,
                        bi_level_gae=self.config.algorithm.bi_level_gae,
                    )

                ##### A very different setting, just here for testing: Can I normalize the advantages to have a mean of 0?
                if self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO and self.config.grpo_advantage_length_weight:
                    response_mask = batch.batch["response_mask"]
                    advantages = batch.batch["advantages"]
                    response_relative_lengths = (torch.sum(response_mask, dim=-1) + 1e-6) / torch.sum(response_mask, dim=-1).float().mean()
                    advantages = advantages / response_relative_lengths.unsqueeze(-1) 
                    batch.batch["advantages"] = advantages

                # update critic
                if self.use_critic:
                    with _timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with _timer("update_actor", timing_raw):
                        batch.meta_info["multi_turn"] = True
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    with _timer("dump_rollout_generations", timing_raw):
                        print(batch.batch.keys())
                        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                        self._dump_generations(
                            inputs=inputs,
                            outputs=outputs,
                            scores=scores,
                            reward_extra_infos_dict=reward_extra_infos_dict,
                            dump_path=rollout_data_dir,
                        )

                # validate
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with _timer("testing", timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                        logger.maybe_sync()
                    metrics.update(val_metrics)

                if self.accel_enabled:
                    # 1) add all envs from this training batch to buffer (their current gaps)
                    train_infos = self._accel__extract_infos(batch)
                    self._accel__maybe_add_many(train_infos)

                    # 2) pick one env from the batch, edit it, eval (stop-grad), maybe add
                    pick = self._accel__pick_one(train_infos)
                    if pick is not None:
                        fam = pick.get("family")
                        p_dict = pick.get("params")
                        if isinstance(p_dict, dict) and fam is None and ("family" in p_dict):
                            fam = p_dict.get("family", "mab")
                            p_dict = p_dict.get("params", {})
                        if fam == "mab":
                            base = BanditParams(**p_dict)
                        elif fam == "pea":
                            base = PEAParams(**p_dict)
                        else:
                            base = OPSParams(**p_dict)
                        edited = perturb_params(base, self.accel_rng, self.accel_perturb_sigma_small, self.accel_perturb_sigma_big, self._accel_bandit_cfg)
                        edited_item = LBATItem(family=fam, params=edited)
                        print("pick: ", pick) # for testing
                        print("edited: ", edited_item) # for testing
                        print("buffer_size: ", len(self.accel_buf.items)) # for testing
                        self._accel__eval_override_once(edited_item.to_dict(), repeats=self.accel_n_eval_repeats)

                if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()
                        logger.maybe_sync()

            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # add another timing metric: total time
            metrics.update({"timing_s/total": time.time() - self.start_time})
            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)
            logger.maybe_sync()

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1
