"""
This is the environment state manager for the LLM agent.
author: Pingyue Zhang
date: 2025-03-30
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import PIL.Image
import hydra
import random
import numpy as np

from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from ragen.utils import register_resolvers
register_resolvers()

import multiprocessing
from functools import partial

use_parallel = True

@dataclass
class EnvStatus:
    """Status of an environment"""
    truncated: bool = False # done but not success
    terminated: bool = False # done and success
    num_actions: int = 0 # current action step (single action)
    rewards: List[float] = field(default_factory=list) # rewards for each turn
    seed: Optional[int] = None # what seed is used to reset this environment



class EnvStateManager:
    """Manager for the environment state
    The class is responsible for managing multiple (kinds of) environments
    
    """
    def __init__(self, config, mode: str = "train"):
        self.sys_config = config
        self.mode = mode
        self.config = getattr(self.sys_config.es_manager, mode)
        self.env_groups = int(self.config.env_groups)
        self.group_size = self.config.group_size
        seed_cfg = getattr(self.sys_config, "seed", None)
        if seed_cfg is not None:
            self.base_seed = seed_cfg.get(mode, None)
        else:
            self.base_seed = None
        self.seed_counter = 0
        self.n_cpu = multiprocessing.cpu_count()
        self._init_envs()
        self.rollout_cache = None
        self._accel_infos = []

    def _init_envs(self):
        """Initialize the environments. train_envs and val_envs are lists of envs:
        Input: tags: ["SimpleSokoban", "HarderSokoban"]; n_groups: [1, 1]; group_size: 16
        Output: envs: List[Dict], each **entry** is a dict with keys: tag, group_id, env_id, env, env_config, status
        Example: [{"tag": "SimpleSokoban", "group_id": 0, "env_id": 0, "env": env, "config": env_config, "status": EnvStatus()},
            ...
            {"tag": "SimpleSokoban", "group_id": 0, "env_id": 15 (group_size - 1), ...},
            {"tag": "HarderSokoban", "group_id": 1, "env_id": 16, ...}
            ...]
        """
        assert sum(self.config.env_configs.n_groups) == self.env_groups, f"Sum of n_groups must equal env_groups. Got sum({self.config.env_configs.n_groups}) != {self.env_groups}"
        assert len(self.config.env_configs.tags) == len(self.config.env_configs.n_groups), f"Number of tags must equal number of n_groups. Got {len(self.config.env_configs.tags)} != {len(self.config.env_configs.n_groups)}"
        
        global use_parallel
        if use_parallel:
            env_configs = self.config.env_configs
            done_groups = 0
            self.envs = []  # start with an empty list
            for tag, n_group in zip(env_configs.tags, env_configs.n_groups):
                func = partial(self._init_one_env, sys_config=self.sys_config, 
                            group_size=self.group_size, tag=tag)
                start = done_groups * self.group_size
                end   = (done_groups + n_group) * self.group_size
                with multiprocessing.Pool(max(1, self.n_cpu // 4)) as pool:
                    self.envs.extend(pool.map(func, range(start, end)))
                done_groups += n_group
            return

        self.envs = self._init_env_instances(self.config)
    
    @staticmethod
    def _init_one_env(env_id, sys_config, group_size, tag):
        cfg_template = sys_config.custom_envs[tag]
        env_class = cfg_template.env_type
        max_actions_per_traj = cfg_template.max_actions_per_traj
        if cfg_template.env_config is None:
            env_config = REGISTERED_ENV_CONFIGS[env_class]()
        else:
            env_config = REGISTERED_ENV_CONFIGS[env_class](**cfg_template.env_config)
        env_obj = REGISTERED_ENVS[env_class](env_config)
        entry = {'tag': tag, 'group_id': env_id // group_size, 'env_id': env_id, 
                'env': env_obj, 'config': env_config, 'status': EnvStatus(), 'max_actions_per_traj': max_actions_per_traj}
        return entry

    def _init_env_instances(self, config):
        env_list = []
        done_groups = 0
        for tag, n_group in zip(config.env_configs.tags, config.env_configs.n_groups):
            for env_id in range(done_groups * self.group_size, (done_groups + n_group) * self.group_size):
                cfg_template = self.sys_config.custom_envs[tag]
                env_class = cfg_template.env_type
                max_actions_per_traj = cfg_template.max_actions_per_traj
                if cfg_template.env_config is None:
                    env_config = REGISTERED_ENV_CONFIGS[env_class]()
                else:
                    env_config = REGISTERED_ENV_CONFIGS[env_class](**cfg_template.env_config)
                env_obj = REGISTERED_ENVS[env_class](env_config)
                entry = {'tag': tag, 'group_id': env_id // self.group_size, 'env_id': env_id, 
                        'env': env_obj, 'config': env_config, 'status': EnvStatus(), 'max_actions_per_traj': max_actions_per_traj}
                env_list.append(entry)
            done_groups += n_group
        return env_list

    def collect_accel_infos(self):
        infos = self._accel_infos
        self._accel_infos = []
        return infos

    def reset(self, seed: Optional[int] = None, meta_info: Optional[dict] = None):
        """
        Reset the environments and get initial observation
        build up rollout cache like [{"env_id": int, "history": List[Dict], "group_id": int}, ...]
        """
        def _expand_seed(seed: int):
            seeds = [[seed + i] * self.group_size for i in range(self.env_groups)] # [[seed, ..., seed], [seed+1, ..., seed+1], ...]
            return sum(seeds, [])

        def _normalize_overrides(override, n_envs: int):
            if override is None:
                return [None] * n_envs
            if isinstance(override, dict):
                return [override] * n_envs
            if isinstance(override, list):
                if len(override) == n_envs:
                    return override
                if len(override) == 1:
                    return override * n_envs
                raise ValueError(
                    f"params_override list length {len(override)} != num_envs {n_envs}"
                )
            raise TypeError("params_override must be None, dict, or list[dict]")

        meta = meta_info or {}
        self._accel_infos = []
        accel = meta.get("accel", {}) or {}
        params_override = accel.get("params_override", None)

        envs = self.envs
        n_envs = len(envs)
        rollout_cache = [{"env_id": entry['env_id'], "history": [], "group_id": entry['group_id'], "tag": entry['tag'], "penalty": 0} for entry in envs]

        # reset all environments
        if seed is None:
            if self.mode == "train":
                if self.base_seed is not None:
                    seed = self.base_seed + self.seed_counter
                    self.seed_counter += self.env_groups
                else:
                    seed = random.randint(0, 1000000)
            else:
                seed = 123 if self.base_seed is None else self.base_seed
        else:
            if self.mode == "train" and self.base_seed is not None:
                self.seed_counter = seed - self.base_seed + 1
        seeds = _expand_seed(seed)
        overrides = _normalize_overrides(params_override, n_envs)

        for seed_i, entry, ovr in zip(seeds, envs, overrides):
            if ovr is not None and hasattr(entry["env"], "set_params_for_next_reset"):
                entry["env"].set_params_for_next_reset(ovr)

            entry['env'].reset(seed=seed_i, mode=self.mode)
            entry['status'] = EnvStatus(seed=seed_i)

        # update rollout cache
        for cache, env in zip(rollout_cache, envs):
            next_state = self._handle_mm_state(env['env'].render())
            cache['history'] = self._update_cache_history(cache['history'], next_state=next_state, actions_left=env['max_actions_per_traj'], num_actions_info=None)
            
        self.rollout_cache = rollout_cache
        return rollout_cache

    def step(self, all_env_inputs: List[Dict]):
        """Step the environments.
        1. extract valid actions from the action lookup table (if exists) and execute the actions, and update rollout cache
        2. Since rollout does not need to act over done envs, whenever the environment is done, we only update rollout cache, but not output env_outputs.
        Input:
        all_env_inputs: List[Dict]
            {env_id: int, llm_response: str, actions: List[str]}
            NOTE: should use env_id as index for existing some already done envs
        env_outputs: List[Dict]
            {env_id: int, history: List[Dict][{state: str, actions: List[str], reward: float, info: Dict, llm_response: str, llm_raw_response: str, (Optional)images: List[PIL.Image.Image]}]}
        """
        def _execute_actions(env, actions):
            acc_reward, turn_info, turn_done = 0, {}, False
            executed_actions = []
            for action in actions:
                _, reward, done, info = env.step(action)
                acc_reward += reward
                turn_info.update(info) # NOTE: currently use last info for multi-action
                executed_actions.append(action)
                if done:
                    turn_done = True
                    break
            
            return acc_reward, turn_info, turn_done, executed_actions

        def _log_env_state(status, history, cur_obs, max_actions_per_traj, executed_actions, all_actions, acc_reward, turn_done, turn_info, env_input):
            obs = self._handle_mm_state(cur_obs)
            status.num_actions += len(executed_actions)
            status.rewards.append(acc_reward) # NOTE use turn-wise acc_reward
            actions_left = max_actions_per_traj - status.num_actions
            if turn_done:
                status.terminated = True # TODO check terminated definition in gymnasium
                status.truncated = not turn_info.get('success', False)
            history = self._update_cache_history(history, next_state=obs, actions_left=actions_left, num_actions_info={
                'actions': executed_actions, 'reward': acc_reward, 'info': turn_info,
                'llm_response': env_input['llm_response'], 'llm_raw_response': env_input['llm_raw_response']
            })
            # filter out invalid actions
            # history = [content for content in history[:-1] if content['actions']] + [history[-1]]
            return status, history

        global use_parallel
        if use_parallel and len(all_env_inputs) > 120:
            parallel_env_inputs = [{
                **env_input,
                'env': self.envs[env_input['env_id']],
                'rollout_cache_single': self.rollout_cache[env_input['env_id']]
            } for env_input in all_env_inputs]

            func = partial(self._step_env, sys_config=self.sys_config)
            with multiprocessing.Pool(max(1, self.n_cpu // 8)) as pool:
                results = pool.map(func, parallel_env_inputs)

            env_outputs = []
            for turn_done, env, rollout_cache, accel_info in results:
                self.envs[env['env_id']] = env
                self.rollout_cache[env['env_id']] = rollout_cache
                if accel_info is not None:
                    if not hasattr(self, "_accel_infos"):
                        self._accel_infos = []
                    self._accel_infos.append(accel_info)
                if not turn_done:
                    env_outputs.append(rollout_cache)
            return env_outputs

        envs = self.envs
        env_outputs = []

        for env_input in all_env_inputs:
            acc_reward, turn_info, turn_done = 0, {}, False
            entry = envs[env_input['env_id']]
            env_id, env = entry['env_id'], entry['env']
            actions_left_before = entry['max_actions_per_traj'] - entry['status'].num_actions

            # execute actions in envs
            valid_actions = self._extract_map_valid_actions(entry, env_input['actions'])
            acc_reward, turn_info, turn_done, executed_actions = _execute_actions(env, valid_actions[:actions_left_before])
            if len(valid_actions) != len(env_input['actions']) or not valid_actions:
                self.rollout_cache[env_id]["penalty"] += self.sys_config.es_manager.format_penalty
                
            status, history = _log_env_state(entry['status'], self.rollout_cache[env_id]['history'], entry['env'].render(), entry['max_actions_per_traj'], executed_actions, valid_actions, acc_reward, turn_done, turn_info, env_input)
            entry['status'] = status
            if entry['status'].num_actions >= entry['max_actions_per_traj'] and not turn_done:
                entry['status'].truncated = True
                entry['status'].terminated = True
                turn_done = True

            if turn_done:
                gap = turn_info.get("protagonist_antagonist_gap")
                if gap is not None:
                    params = entry['env'].export_params() if hasattr(entry['env'], "export_params") else None
                    if params is not None:
                        if not hasattr(self, "_accel_infos"):
                            self._accel_infos = []
                        self._accel_infos.append({"gap": float(gap), "params": params})

            self.rollout_cache[env_id]['history'] = history
            if not turn_done: # NOTE done environments are not sent for further llm generation (for efficiency)
                env_outputs.append(self.rollout_cache[env_id])

        return env_outputs

    @staticmethod
    def _step_env(env_input, sys_config):
        def _extract_map_valid_actions(entry: Dict, actions: List[str]):
            """extract valid actions from the action lookup table (if exists)"""
            mapped_actions = []
            action_lookup = getattr(entry['env'].config, 'action_lookup', None)
            if action_lookup is None:
                mapped_actions = actions
            else: # the envs have pre-defined action lookup
                rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
                actions = [action.lower() for action in actions]
                mapped_actions = [rev_action_lookup[action] for action in actions if action in rev_action_lookup]
            return mapped_actions
        
        def _execute_actions(env, actions):
            acc_reward, turn_info, turn_done = 0, {}, False
            executed_actions = []
            for action in actions:
                _, reward, done, info = env.step(action)
                acc_reward += reward
                turn_info.update(info) # NOTE: currently use last info for multi-action
                executed_actions.append(action)
                if done:
                    turn_done = True
                    break
            
            return acc_reward, turn_info, turn_done, executed_actions

        def _log_env_state(status, history, cur_obs, max_actions_per_traj, executed_actions, all_actions, acc_reward, turn_done, turn_info, env_input):
            def _handle_mm_state(state: Union[str, np.ndarray, list[np.ndarray]]):
                """Handle the state from the environment
                """
                if isinstance(state, str): # text state
                    return state
                elif isinstance(state, np.ndarray): # when env state is a single image, convert it to a list to unify output format
                    state = [state]
                results = [PIL.Image.fromarray(_state, mode='RGB') for _state in state]
                return results

            def _update_cache_history(history: List[Dict], next_state, actions_left, num_actions_info: Optional[Dict] = None):
                """
                Update last step info and append state to history
                """
                if num_actions_info is not None: # update last step info
                    assert len(history), "History should not be empty"
                    history[-1].update(num_actions_info)
                
                entry = {} # append state to history
                if isinstance(next_state, str): # text state
                    entry['state'] = next_state
                else: # multimodal state
                    entry['state'] = "<images>" * len(next_state)
                    entry['images'] = next_state
                entry['actions_left'] = actions_left
                history.append(entry)
                return history
            
            obs = _handle_mm_state(cur_obs)
            status.num_actions += len(executed_actions)
            status.rewards.append(acc_reward) # NOTE use turn-wise acc_reward
            actions_left = max_actions_per_traj - status.num_actions
            if turn_done:
                status.terminated = True # TODO check terminated definition in gymnasium
                status.truncated = not turn_info.get('success', False)
            history = _update_cache_history(history, next_state=obs, actions_left=actions_left, num_actions_info={
                'actions': executed_actions, 'reward': acc_reward, 'info': turn_info,
                'llm_response': env_input['llm_response'], 'llm_raw_response': env_input['llm_raw_response']
            })
            # filter out invalid actions
            # history = [content for content in history[:-1] if content['actions']] + [history[-1]]
            return status, history

        acc_reward, turn_info, turn_done = 0, {}, False
        entry = env_input['env']
        env_id, env = entry['env_id'], entry['env']
        actions_left_before = entry['max_actions_per_traj'] - entry['status'].num_actions

        # execute actions in envs
        valid_actions = _extract_map_valid_actions(entry, env_input['actions'])
        rollout_cache_single = env_input['rollout_cache_single']
        acc_reward, turn_info, turn_done, executed_actions = _execute_actions(env, valid_actions[:actions_left_before])
        if len(valid_actions) != len(env_input['actions']) or not valid_actions:
            rollout_cache_single["penalty"] += sys_config.es_manager.format_penalty
            
        status, history = _log_env_state(entry['status'], rollout_cache_single['history'], entry['env'].render(), entry['max_actions_per_traj'], executed_actions, valid_actions, acc_reward, turn_done, turn_info, env_input)
        entry['status'] = status
        if entry['status'].num_actions >= entry['max_actions_per_traj'] and not turn_done:
            entry['status'].truncated = True
            entry['status'].terminated = True
            turn_done = True
        rollout_cache_single['history'] = history

        accel_info = None
        if turn_done:
            gap = turn_info.get("protagonist_antagonist_gap")
            if gap is not None:
                params = entry['env'].export_params() if hasattr(entry['env'], "export_params") else None
                if params is not None:
                    accel_info = {"gap": float(gap), "params": params}
        
        return turn_done, entry, rollout_cache_single, accel_info

    def get_rollout_states(self):
        """Get the final output for all environment"""
        envs = self.envs
        rollout_cache = self.rollout_cache
        TURN_LVL_METRICS = ['action_is_effective', 'action_is_valid', 'end_of_page']

        # add metrics to rollout cache
        for entry, cache in zip(envs, rollout_cache):
            status = entry['status']
            env_metric = {
                'success': float(status.terminated and (not status.truncated)),
                'num_actions': status.num_actions,
            }
            custom_metric = {}
            for turn in cache['history']:
                for k, v in turn.get('info', {}).items():
                    if k == 'success':
                        continue
                    if k not in custom_metric:
                        custom_metric[k] = []
                    custom_metric[k].append(float(v))
            for k, v in custom_metric.items():
                # TODO: Move TURN_LVL_METRICS into the environment
                if "Webshop" not in k or ("Webshop" in k and k in TURN_LVL_METRICS):
                    env_metric[k] = np.sum(v) / (len(cache['history']) - 1) # NOTE: exclude the last observation
                else:
                    env_metric[k] = np.sum(v)


            cache['history'][-1]['metrics'] = custom_metric
            env_metric = {f"{entry['tag']}/{k}": v for k, v in env_metric.items()}
            cache['metrics'] = env_metric
            if entry['tag'] == "MetamathQA":
                cache['correct_answer'] = entry['env'].correct_answer

        # calculate pass@k where k is the group size
        group_success = {}
        for entry, cache in zip(envs, rollout_cache):
            key = (entry['tag'], entry['group_id'])
            success_val = cache['metrics'].get(f"{entry['tag']}/success", 0.0)
            group_success.setdefault(key, []).append(success_val)

        for (tag, gid), succ_list in group_success.items():
            pass_success = float(any(succ_list))
            for entry, cache in zip(envs, rollout_cache):
                if entry['tag'] == tag and entry['group_id'] == gid:
                    cache['metrics'][f"{tag}/pass@{self.group_size}"] = pass_success
        return rollout_cache




    def _update_cache_history(self, history: List[Dict], next_state, actions_left, num_actions_info: Optional[Dict] = None):
        """
        Update last step info and append state to history
        """
        if num_actions_info is not None: # update last step info
            assert len(history), "History should not be empty"
            history[-1].update(num_actions_info)
        
        entry = {} # append state to history
        if isinstance(next_state, str): # text state
            entry['state'] = next_state
        else: # multimodal state
            entry['state'] = "<images>" * len(next_state)
            entry['images'] = next_state
        entry['actions_left'] = actions_left
        history.append(entry)
        return history

    def _extract_map_valid_actions(self, entry: Dict, actions: List[str]):
        """extract valid actions from the action lookup table (if exists)"""
        mapped_actions = []
        action_lookup = getattr(entry['env'].config, 'action_lookup', None)
        if action_lookup is None:
            mapped_actions = actions
        else: # the envs have pre-defined action lookup
            rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
            actions = [action.lower() for action in actions]
            mapped_actions = [rev_action_lookup[action] for action in actions if action in rev_action_lookup]
        return mapped_actions
    
    def _handle_mm_state(self, state: Union[str, np.ndarray, list[np.ndarray]]):
        """Handle the state from the environment
        """
        if isinstance(state, str): # text state
            return state
        elif isinstance(state, np.ndarray): # when env state is a single image, convert it to a list to unify output format
            state = [state]
        results = [PIL.Image.fromarray(_state, mode='RGB') for _state in state]
        return results
        
    def render(self):
        rendered_list = [entry['env'].render() for entry in self.envs]
        return rendered_list

    def close(self):
        for entry in self.envs:
            entry['env'].close()




@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
    """
    Unit test for EnvStateManager
    """
    es_manager = EnvStateManager(config, mode="train")
    print("Initializing environments...")
    es_manager.reset(seed=123)

    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRunning step for training environments...")
    all_env_inputs = [
        {
            "env_id": i,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        } for i in range(128)
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")

    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go left, go up",
            "llm_response": "Go left, go up",
            "actions": ["left", "up"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go up, go up",
            "llm_response": "Go up, go up",
            "actions": ["up", "up", "up", "up", "up"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRendering final output...")
    final_outputs = es_manager.get_rollout_states()
    print(f"final outputs[:4]: {final_outputs[:4]}")
    
    print("\nClosing environments...")
    es_manager.close()
    print("Test completed successfully!")


if __name__ == "__main__":
	main()
