from itertools import zip_longest

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import AutoTokenizer
import hydra
from ragen.utils import register_resolvers
from ragen.env import REGISTERED_ENV_CONFIGS
from tensordict import TensorDict

from dataclasses import asdict
register_resolvers()

def get_special_tokens(tokenizer: AutoTokenizer):
    name = tokenizer.name_or_path.lower()
    if "qwen" in name:
        special_token = tokenizer.convert_tokens_to_ids("<|im_start|>")
        reward_token  = tokenizer.convert_tokens_to_ids("<|im_end|>")
    elif "llama" in name:
        # Not used for Llama masking anymore, but keep for completeness
        special_token = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
        reward_token  = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    else:
        raise ValueError(f"Unsupported model: {tokenizer.name_or_path}")
    if special_token in (None, tokenizer.unk_token_id) or reward_token in (None, tokenizer.unk_token_id):
        raise ValueError("Special tokens not found in tokenizer.")
    return special_token, reward_token

def _last_true_index(mask: torch.Tensor) -> torch.Tensor:
    """
    Return index of the last True in each row of a boolean mask.
    If a row has no True, returns 0 for that row.
    Works on CPU/GPU and avoids argmax on bool dtype.
    mask: (B, L) bool
    returns: (B,) int64
    """
    mask_i64 = mask.to(dtype=torch.int64)
    cm = mask_i64.cumsum(dim=-1)
    totals = mask_i64.sum(dim=-1, keepdim=True)
    is_last = (cm == totals).to(dtype=torch.int64)  # 1 where last True sits, else 0
    return is_last.argmax(dim=-1)

def _llama_special_ids(tokenizer):
    sid = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    eid = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if any(x is None or x == tokenizer.unk_token_id for x in (sid, eid, eot)):
        raise ValueError("Llama special tokens not found in tokenizer.")
    return sid, eid, eot

@torch.no_grad()
def _build_llama_role_masks(input_ids: torch.Tensor, tokenizer) -> Dict[str, torch.Tensor]:
    """
    Marks assistant content spans strictly from <|end_header_id|>+1 to <|eot_id|> (inclusive)
    for headers whose role text is 'assistant'. Also returns header_mask and eot_mask.
    """
    B, L = input_ids.shape
    device = input_ids.device
    sid, eid, eot = _llama_special_ids(tokenizer)

    is_sid = (input_ids == sid)
    is_eid = (input_ids == eid)
    is_eot = (input_ids == eot)

    header_mask   = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
    assistant_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)

    for b in range(B):
        starts = is_sid[b].nonzero(as_tuple=False).squeeze(-1).tolist()
        ends   = is_eid[b].nonzero(as_tuple=False).squeeze(-1).tolist()
        eots   = is_eot[b].nonzero(as_tuple=False).squeeze(-1).tolist()
        end_ptr = 0
        eot_ptr = 0
        for s in starts:
            while end_ptr < len(ends) and ends[end_ptr] <= s:
                end_ptr += 1
            if end_ptr >= len(ends): break
            e_hdr = ends[end_ptr]
            header_mask[b, s:e_hdr+1] = True

            # decode role string between start/end header ids
            role_txt = tokenizer.decode(input_ids[b, s+1:e_hdr].tolist()).strip().lower()

            # find the next eot after this header
            while eot_ptr < len(eots) and eots[eot_ptr] <= e_hdr:
                eot_ptr += 1
            if eot_ptr >= len(eots):
                span_end = L - 1
            else:
                span_end = eots[eot_ptr]  # INCLUDE EOT

            span_start = min(e_hdr + 1, L - 1)

            print("role_txt: ", role_txt)
            if role_txt.startswith("assistant"):
                assistant_mask[b, span_start:span_end+1] = True

    return {
        "header_mask": header_mask,        # only header tokens
        "assistant_mask": assistant_mask,  # includes <|eot_id|>
        "eot_mask": (input_ids == eot),
    }

def place_per_turn_rewards(input_ids, turn_indicators, all_scores, tokenizer,
                           special_token, reward_token):
    """
    Returns score_tensor aligned to input_ids (no slicing yet).
    For each assistant turn k, places that turn's scalar reward on the last
    non-special token of the k-th assistant span; if empty, uses the span's last token.
    """
    device = input_ids.device
    bsz, L = input_ids.shape
    score_tensor = torch.zeros((bsz, L), dtype=torch.float32, device=device)

    # Assistant spans are odd indices >= 3 (system=1, user=2, assistant=3, user=4, assistant=5, ...)
    is_assistant_span = (turn_indicators % 2 == 1) & (turn_indicators > 1)
    header_mask = (input_ids == special_token) | (input_ids == reward_token)

    not_special_body = (~header_mask)
    max_turn_id = int(turn_indicators.max().item())

    # all_scores is list over batch of per-turn scalars; zip_longest groups by turn index
    for k, scores_k in enumerate(zip_longest(*all_scores, fillvalue=0.0)):
        turn_id = 3 + 2 * k
        if turn_id > max_turn_id:
            break

        span_k = is_assistant_span & (turn_indicators == turn_id)
        body_k = span_k & not_special_body

        def last_true(mask):
            cnt = mask.long().sum(-1, keepdim=True)
            csm = mask.long().cumsum(-1)
            last = (csm == cnt).long().argmax(-1)
            return last

        has_body = body_k.any(-1)
        last_body = last_true(body_k)
        last_span = last_true(span_k)
        last_idx  = torch.where(has_body, last_body, last_span)

        scores_k = torch.tensor(scores_k, dtype=torch.float32, device=device)
        score_tensor.scatter_(1, last_idx.unsqueeze(1), scores_k.unsqueeze(1))

    return score_tensor

def get_masks_and_scores_llama(
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    all_scores: List[List[float]],
    use_turn_scores: bool = False,
    debug: bool = False,
):
    B, L = input_ids.shape
    device = input_ids.device

    masks = _build_llama_role_masks(input_ids, tokenizer)
    response_mask = masks["assistant_mask"].clone()  # learn only assistant content
    loss_mask     = response_mask.clone()
    eot_mask      = masks["eot_mask"]

    # scores in sequence space, then shift to label space at the end
    score_tensor = torch.zeros((B, L), dtype=torch.float32, device=device)

    if use_turn_scores:
        # place each turn's scalar on that turn's EOT (inside assistant mask)
        for b in range(B):
            # EOTs inside assistant spans = one per assistant message
            turn_eots = (eot_mask[b] & response_mask[b]).nonzero(as_tuple=False).squeeze(-1).tolist()
            for k, s in enumerate(all_scores[b]):
                if k < len(turn_eots):
                    t = turn_eots[k]
                else:
                    # fallback: last assistant token if the EOT is missing
                    asst_pos = (response_mask[b].nonzero(as_tuple=False).squeeze(-1).tolist() or [L-1])
                    t = asst_pos[-1]
                score_tensor[b, t] = float(s)
    else:
        # single scalar per sequence -> place on *last* assistant token (prefer EOT)
        seq_scores = torch.tensor([sum(x) for x in all_scores], dtype=torch.float32, device=device)
        for b in range(B):
            asst_pos = (response_mask[b].nonzero(as_tuple=False).squeeze(-1)).tolist()
            if not asst_pos: 
                continue
            eots = (eot_mask[b] & response_mask[b]).nonzero(as_tuple=False).squeeze(-1).tolist()
            t = (eots or asst_pos)[-1]
            score_tensor[b, t] = seq_scores[b]

    # align to label space
    score_tensor  = score_tensor[:, 1:]
    loss_mask     = loss_mask[:, 1:]
    response_mask = response_mask[:, 1:]

    if debug:
        bad = (score_tensor.abs() > 0) & (~response_mask.bool())
        if bool(bad.any()):
            rows = torch.nonzero(bad.any(-1)).squeeze(-1).tolist()
            print(f"[SANITY] rewards outside assistant mask at rows: {rows}")

    return score_tensor, loss_mask, response_mask


def get_masks_and_scores(input_ids: torch.Tensor, tokenizer: AutoTokenizer, all_scores: List[List[float]] = None, use_turn_scores: bool = False, enable_response_mask: bool = False, debug: bool = False):
    """
    input_ids: shape (bsz, seq_len)
    Get loss mask that only learns between <|im_start|>assistant and <|im_end|>. Currently only supports qwen.
    Update: should also work for Llama-3? We'll see.
    NOTE: important! This assumes that the input_ids starts with system and then user & assistant in alternative ways
    """
    special_token, reward_token = get_special_tokens(tokenizer)
    
    turn_starts = torch.where(input_ids == special_token, 1, 0)
    turn_indicators = torch.cumsum(turn_starts, dim=-1)
    if enable_response_mask:
        loss_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1) # only learns all assistant turns
    else:
        loss_mask = (turn_indicators > 1) # learns everything after system prompt
    response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)

    header_mask = (input_ids == special_token) | (input_ids == reward_token)

    loss_mask = loss_mask & (~header_mask)
    response_mask = response_mask & (~header_mask)
    
    score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)

    if use_turn_scores:
        score_tensor = place_per_turn_rewards(input_ids, turn_indicators, all_scores,
                                            tokenizer, special_token, reward_token)
    else:
        # single scalar per sequence – place it on the last *masked* position later
        score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
        seq_scores = torch.tensor([sum(x) for x in all_scores], dtype=torch.float32, device=input_ids.device)
        # we'll place it after slicing, see below

    # Align everything to label positions (responses = input_ids[:, 1:])
    score_tensor = score_tensor[:, 1:]
    loss_mask = loss_mask[:, 1:]
    response_mask = response_mask[:, 1:]

    if not use_turn_scores:
        # Put the scalar on the last masked position so it participates in the loss
        last_masked = _last_true_index(response_mask)
        score_tensor.zero_()
        score_tensor.scatter_(1, last_masked.unsqueeze(1), seq_scores.unsqueeze(1))

    # Safety check during debug
    if debug:
        nz = (score_tensor.abs() > 0)
        bad = (nz & (~response_mask.bool())).any(dim=-1)  # any nonzero reward outside mask?
        if bool(bad.any()):
            idxs = torch.nonzero(bad).flatten().tolist()
            print(f"[SANITY] nonzero rewards outside mask for rows: {idxs[:8]}{'...' if len(idxs)>8 else ''}")

    return score_tensor, loss_mask, response_mask



class ContextManager:
    """
    Manages the context for LLM interactions with environments.
    Translates between environment outputs and LLM inputs, and vice versa.
    """

    def __init__(self, 
                 config,
                 tokenizer,
                 processor = None,
                 mode: str = "train",
                 ):
        """
        Initialize the ContextManager.
        Processor is used to process the image data.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.action_sep = self.config.agent_proxy.action_sep
        self.special_token_list = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]
        self.debug_mask_sanity = bool(getattr(self.config, "debug_mask_sanity", False))

        self.es_cfg = self.config.es_manager[mode]
        self.env_nums = {
                env_tag: n_group * self.es_cfg.group_size
                for n_group, env_tag in zip(self.es_cfg.env_configs.n_groups, self.es_cfg.env_configs.tags)
        }
        self._init_prefix_lookup()
        self.tokenizer.padding_side = "left"
    
    def _check_env_installed(self, env_type: str):
        if env_type not in REGISTERED_ENV_CONFIGS:
            raise ValueError(f"Environment {env_type} is not installed. Please install it using the scripts/setup_{env_type}.sh script.")

    def _init_prefix_lookup(self):
        prefix_lookup = {}
        prefixes = {}
        env_config_lookup = {}
        env_config = {}
        for env_tag, env_config in self.config.custom_envs.items():
            if env_tag not in self.es_cfg.env_configs.tags:
                continue

            self._check_env_installed(env_config.env_type)
            env_config_new = asdict(REGISTERED_ENV_CONFIGS[env_config.env_type]())
            for k,v in env_config.items():
                env_config_new[k] = v
            env_instruction = env_config_new.get("env_instruction", "")
            if env_config_new.get("grid_vocab", False):
                grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join([f"{k}: {v}" for k, v in env_config_new["grid_vocab"].items()])
                env_instruction += grid_vocab_str
            if env_config_new.get("action_lookup", False):
                action_lookup_str = "\nYour available actions are:\n" + ", ".join([f"{v}" for k, v in env_config_new["action_lookup"].items()])
                action_lookup_str += f"\nYou can make up to {env_config_new['max_actions_per_traj']} actions, separated by the action separator \" " + self.action_sep + " \"\n"
                env_instruction += action_lookup_str
            prefixes[env_tag] = env_instruction
            env_config_lookup[env_tag] = {'max_tokens': env_config.get("max_tokens", self.config.actor_rollout_ref.rollout.response_length)}

        tags = self.es_cfg.env_configs.tags
        n_groups = self.es_cfg.env_configs.n_groups
        group_size = self.es_cfg.group_size

        cur_group = 0
        for env_tag, n_group in zip(tags, n_groups):
            env_instruction = prefixes[env_tag]
            start_idx = cur_group * group_size
            end_idx = (cur_group + n_group) * group_size
            for i in range(start_idx, end_idx):
                prefix_lookup[i] = env_instruction
                env_config_lookup[i] = env_config_lookup[env_tag]
            cur_group += n_group
            
        self.prefix_lookup = prefix_lookup
        self.env_config_lookup = env_config_lookup

    def _parse_response(self, response: str) -> List:
        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if self.config.agent_proxy.enable_think else r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            # think_content, action_content, actions = "", "", [] # do not remove this kind of invalid string
            llm_response, actions = response, []
            print("experiment name: ", self.config.trainer.experiment_name) # for testing
            if "lbat" in self.config.trainer.experiment_name:
                actions = [""] # force an invalid action step to trigger penalty and state advance
                # print("did I reach here?") # for testing
        else:
            if self.config.agent_proxy.enable_think:
                think_content, action_content = match.group(1), match.group(2)
            else:
                think_content, action_content = "", match.group(1)

                
            for special_token in self.special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
            
            actions = [action.strip() for action in action_content.split(self.action_sep) if action.strip()]
            if not actions and "lbat" in self.config.trainer.experiment_name:
                # If the format is correct but answer is empty, still step an invalid action
                actions = [""]
            max_actions = self.config.agent_proxy.max_actions_per_turn

            if len(actions) > max_actions:
                actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
                action_content = (" " + self.action_sep + " ").join(actions)

            llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>" if self.config.agent_proxy.enable_think else f"<answer>{action_content}</answer>"
        return llm_response, actions
        
    def _normalize_score_tensor(self,
                                score_tensor: torch.Tensor,
                                env_outputs: List[Dict],
                                response_mask: torch.Tensor) -> torch.Tensor:
        """
        Normalize per-sequence scalar scores when use_turn_scores == False.
        Works no matter which position the scalar currently occupies.
        """
        device = score_tensor.device
        bsz, L = score_tensor.shape

        # 1) Read the scalar per row (exactly one nonzero per row by construction)
        acc_scores = score_tensor.sum(dim=-1)  # (bsz,)

        # 2) Apply penalties and groupwise normalization (unchanged logic)
        rn_cfg = self.config.agent_proxy.reward_normalization
        grouping, method = rn_cfg.grouping, rn_cfg.method
        if grouping == "state":
            group_tags = [env_output["group_id"] for env_output in env_outputs]
        elif grouping == "inductive":
            group_tags = [env_output["tag"] for env_output in env_outputs]
        elif grouping == "batch":
            group_tags = [1] * len(env_outputs)
        else:
            raise ValueError(f"Invalid grouping: {grouping}")

        if method == "mean_std":
            def norm_func(x):
                m = x.mean()
                s = x.std()
                return (x - m) / (s + 1e-6) if float(s) > 1e-6 else torch.zeros_like(x)
        elif method == "mean":
            norm_func = lambda x: x - x.mean()
        elif method == "asym_clip":
            def norm_func(x):
                m = x.mean()
                s = x.std()
                y = (x - m) / (s + 1e-6) if float(s) > 1e-6 else torch.zeros_like(x)
                return y.clamp(min=-1, max=3)
        elif method == "identity":
            norm_func = lambda x: x
        else:
            raise ValueError(f"Invalid normalization method: {method}")

        penalty = torch.tensor([eo.get("penalty", 0.0) for eo in env_outputs],
                            dtype=torch.float32, device=device)
        normalized = acc_scores + penalty

        # groupwise normalize if group_size > 1
        from collections import defaultdict
        idxs = defaultdict(list)
        for i, g in enumerate(group_tags):
            idxs[g].append(i)
        normalized = normalized.clone()
        for g, ii in idxs.items():
            ii = torch.tensor(ii, device=device, dtype=torch.long)
            normalized[ii] = norm_func(normalized[ii])

        # 3) Re-scatter normalized scalars to the last masked position
        out = torch.zeros_like(score_tensor)
        last_masked = _last_true_index(response_mask)
        out.scatter_(1, last_masked.unsqueeze(1), normalized.unsqueeze(1))
        return out

    
    def get_lm_inputs(self, env_outputs: List[Dict], prepare_for_update: bool) -> DataProto:
        """
        env_outputs - please see below example
        [
            {"env_id": 1, "history": [{"state": "###\n#x_#", "llm_response": "Response 1", "reward": 0.5}, {"state": "###\n#x_#"}]},
            {"env_id": 2, "history": [{"state": "###\n#x_#"}]},
            ...
        ]
        prefix_lookup - from env_id to initial prompt
        """
        llm_input_texts = []
        messages_list = [] # for api calling

        # ---------- THINK STEP BONUS (scale-invariant) ----------
        step_cfg = getattr(self.config.agent_proxy, "think_step_bonus", None)

        def _count_tokens(txt: str) -> int:
            if not txt:
                return 0
            return len(self.tokenizer.encode(txt, add_special_tokens=False))

        def _think_pass(entry: Dict[str, Any]) -> bool:
            if not (step_cfg and getattr(step_cfg, "enabled", True) and
                    self.config.agent_proxy.enable_think and "llm_response" in entry):
                return False
            m = re.search(r"<think>(.*?)</think>", entry["llm_response"], flags=re.DOTALL)
            if not m:
                return False
            think_txt = (m.group(1) or "").strip()
            n = _count_tokens(think_txt)
            min_tok = int(getattr(step_cfg, "min_tokens", 20))
            if n < min_tok:
                return False
            # diversity gate
            toks = self.tokenizer.encode(think_txt, add_special_tokens=False)
            uniq_ratio = (len(set(toks)) / max(1.0, float(len(toks))))
            if uniq_ratio < float(getattr(step_cfg, "min_unique_ratio", 0.25)):
                return False
            return True

        # Collect all non-zero base rewards across the batch and compute a single scale.
        batch_vals = []
        for eo in env_outputs:
            for entry in eo["history"]:
                r = float(entry.get("reward", 0.0))
                if r != 0.0:
                    batch_vals.append(r)

        def _robust_scale(vals: list[float], how: str) -> float:
            if not vals:
                return 1.0
            t = torch.tensor(vals, dtype=torch.float32)
            how = str(how or "std").lower()
            if how == "std":
                # Use population std for stability on small samples
                s = float(t.std(unbiased=False).item())
                return s if s > 1e-6 else max(1.0, float(t.abs().mean().item()))
            if how == "mad":  # median absolute deviation
                med = t.median().item()
                mad = (t - med).abs().median().item()
                # 1.4826 * MAD ≈ std if normal
                s = 1.4826 * mad
                return s if s > 1e-6 else max(1.0, float(t.abs().mean().item()))
            if how == "mean_abs":
                m = float(t.abs().mean().item())
                return m if m > 1e-6 else 1.0
            return 1.0

        # One common scale for the whole batch
        batch_scale = _robust_scale(batch_vals, getattr(step_cfg, "scale_by", "std"))
        if not np.isfinite(batch_scale) or batch_scale <= 0:
            batch_scale = 1.0

        for env_output in env_outputs:
            if 'state' in env_output['history'][-1] and prepare_for_update:
                env_output['history'] = env_output['history'][:-1] # when prepare for update, we do not add the state from the n+1 turn to the trajectory
            
            max_k = getattr(self.config.agent_proxy, "max_context_window", None)
            if max_k is not None and isinstance(max_k, int) and max_k > 0:
                env_output['history'] = env_output['history'][-max_k:]
            
            messages = [
                {"role": "system", "content": f"You're a helpful assistant. "}, 
                {"role": "user", "content": self.prefix_lookup[env_output["env_id"]]}
            ]

            for idx, content in enumerate(env_output["history"]):
                messages[-1]["content"] += f"\nTurn {idx + 1}:\n"
                if "state" in content:
                    FORMAT_PROMPT = "<think> [Your thoughts] </think> <answer> [your answer] </answer>" if self.config.agent_proxy.enable_think else "<answer> [your answer] </answer>"
                    LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_output['env_id']]['max_tokens']} tokens."
                    # messages[-1]["content"] += f"State:\n{content['state']}\nYou have {content['actions_left']} actions left. Always output: {FORMAT_PROMPT} with no extra text. Strictly follow this format. {LENGTH_PROMPT}\n"
                    messages[-1]["content"] += f"State:\n{content['state']}\nAlways output: {FORMAT_PROMPT} with no extra text. Strictly follow this format. Do not use specific algorithms. {LENGTH_PROMPT}\n"
                if "llm_response" in content:
                    messages.append({"role": "assistant", "content": content["llm_response"]})
                if "reward" in content and not (prepare_for_update and idx == len(env_output["history"]) - 1):
                    # when prepare for update, we do not add the reward from the n+1 turn to the trajectory
                    try:
                        reward_str = f"{float(content['reward']):.4f}"
                    except Exception as e:
                        print(f"[ContextManager] reward formatting failed: {e}")
                        reward_str = str(content.get('reward'))
                    messages.append({"role": "user", "content": f"Reward:\n{reward_str}\n"})
                    

            # NOTE: this assertion is important for loss mask computation        
            assert all(msg["role"] == "assistant" for msg in messages[2::2])

            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=(not prepare_for_update), tokenize=False)
            llm_input_texts.append(text)
            messages_list.append(messages)

        inputs = self.tokenizer(llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False) # do not truncate here. Process later at TODO
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        # position ids should start at 0 for first valid token and stay at 0 over padding
        position_ids = torch.clamp(attention_mask.cumsum(dim=-1) - 1, min=0)
        if prepare_for_update:
            scores = []
            for eo in env_outputs:
                # Always use the same scale for the whole batch
                scale = float(batch_scale)
                beta = float(getattr(step_cfg, "beta", 0.2)) if step_cfg else 0.0
                only_pos = bool(getattr(step_cfg, "only_if_base_positive", True)) if step_cfg else True

                per_turn = []
                for entry in eo["history"]:
                    base_r = float(entry.get("reward", 0.0))
                    step = 0.0
                    if step_cfg and getattr(step_cfg, "enabled", True):
                        only_pos = bool(getattr(step_cfg, "only_if_base_positive", True))
                        if (not only_pos) or (base_r >= 0.0):
                            if _think_pass(entry):
                                step = float(getattr(step_cfg, "beta", 0.2)) * float(batch_scale)
                    per_turn.append(base_r + step)
                scores.append(per_turn)

            # score_tensor, loss_mask, response_mask = get_masks_and_scores(input_ids, self.tokenizer, scores, use_turn_scores=self.config.agent_proxy.use_turn_scores, enable_response_mask=self.config.enable_response_mask, debug=self.debug_mask_sanity)
            if "llama" in self.tokenizer.name_or_path.lower():
                score_tensor, loss_mask, response_mask = get_masks_and_scores_llama(
                    input_ids, self.tokenizer, scores,
                    use_turn_scores=self.config.agent_proxy.use_turn_scores,
                    debug=self.debug_mask_sanity
                )
            else:
                score_tensor, loss_mask, response_mask = get_masks_and_scores(
                    input_ids, self.tokenizer, scores,
                    use_turn_scores=self.config.agent_proxy.use_turn_scores,
                    enable_response_mask=self.config.enable_response_mask,
                    debug=self.debug_mask_sanity
                )

            normalized_score_tensor = score_tensor
            if not self.config.agent_proxy.use_turn_scores:
                normalized_score_tensor = self._normalize_score_tensor(score_tensor, env_outputs, response_mask)
            response_length = response_mask.sum(dim=-1).float().mean().item()

        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, 1:], # remove the first token
        }, batch_size=input_ids.shape[0])

        if prepare_for_update:
            llm_inputs.batch["loss_mask"] = loss_mask # remove the first token
            llm_inputs.batch["rm_scores"] = normalized_score_tensor # remove the first token
            llm_inputs.batch["original_rm_scores"] = score_tensor # remove the first token
        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"] for env_output in env_outputs], dtype=object),
            "group_ids": np.array([env_output["group_id"] for env_output in env_outputs], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
        }

        if prepare_for_update:
            metrics = {}
            for env_output in env_outputs:
                for key, value in env_output["metrics"].items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            mean_metrics = {
                key: np.sum(value) / self.env_nums[key.split("/")[0]]
                for key, value in metrics.items()
            }
            for key, values in metrics.items():
                if not isinstance(values, list):
                    continue
                prefix, suffix = key.split("/", 1)
                non_zero_values = [v for v in values if v != 0]
                if non_zero_values:  # Avoid division by zero
                    non_zero_key = f"{prefix}/non-zero/{suffix}"
                    mean_metrics[non_zero_key] = np.mean(non_zero_values)
            metrics = mean_metrics
            metrics["response_length"] = response_length
            llm_inputs.meta_info = {"metrics": metrics}
        return llm_inputs

    def get_env_inputs(self, lm_outputs: DataProto) -> List[Dict]:
        if lm_outputs.batch is not None and 'responses' in lm_outputs.batch.keys():
            responses = self.tokenizer.batch_decode(
                lm_outputs.batch['responses'], 
                skip_special_tokens=True
            )
        else: # dataproto has textual responses
            responses = lm_outputs.non_tensor_batch['response_texts']
            
        env_ids = lm_outputs.non_tensor_batch['env_ids']
        env_inputs = []
        for env_id, response in zip(env_ids, responses):
            llm_response, actions = self._parse_response(response)
            env_inputs.append({
                "env_id": env_id,
                "llm_raw_response": response,
                "llm_response": llm_response,
                "actions": actions,
            })
        return env_inputs

    def formulate_rollouts(self, env_outputs: List[Dict]) -> DataProto:
        llm_inputs = self.get_lm_inputs(env_outputs, prepare_for_update=True)
        return llm_inputs

    



@hydra.main(version_base = None, config_path = "../../config", config_name = "base")
def main(config):
    import json
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    ctx_manager = ContextManager(config=config, tokenizer=tokenizer)
    print("ctx_manager prefix", ctx_manager.prefix_lookup)
    # batch_list = [
    #     {
    #         "env_ids": 0,
    #         "chat_response": "<think><think></answer> 123. </think><answer> <answer> say | hi </answer></answer>",
    #     },
    #     {
    #         "env_ids": 1,
    #         "chat_response": "<think> 456. </think><answer> 789 </answer><think> 10123 </think><answer> 11111 </answer>",
    #     }
    # ]
    # ctx_manager.action_sep_lookup = {
    #     0: "|",
    #     1: ";"
    # }
    # for item in batch_list:
    #     item["responses"] = tokenizer.encode(item["chat_response"], return_tensors="pt",max_length=512, truncation=True,padding="max_length")[0]
    # batch_dict = collate_fn(batch_list)
    # batch = DataProto.from_single_dict(batch_dict)
    # env_inputs = ctx_manager.get_env_inputs(batch)
    # print(env_inputs)
    


    env_outputs = [
        {
            "env_id": 1,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 1", "reward": 0.5, "actions_left": 2},
                {"state": "###\n#x_#<image>", "llm_response": "Response 2", "reward": 0.8, "actions_left": 1},
                {"state": "###\n#x_#<image>", "actions_left": 0}
            ],
            "group_id": 0,
            "metrics": {}
        },
        {
            "env_id": 2,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 3", "reward": 0.3, "actions_left": 1},
                {"state": "###\n#x_#<image>", "actions_left": 0}
            ],
            "group_id": 1,
            "metrics": {}
        }
    ]
    
    prefix_lookup = {1: "Initial prompt", 2: "Initial prompt 2"}
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    env_prompt = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
    print(env_prompt)
    formulate_rollouts_rst= ctx_manager.formulate_rollouts(env_outputs)
    print(formulate_rollouts_rst)

if __name__ == "__main__":
    main()
    
