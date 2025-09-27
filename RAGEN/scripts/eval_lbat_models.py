import argparse
import json
import os
import random
import re
import sys
import time
import traceback
from dataclasses import asdict
from typing import Any, Dict, List, Optional
from datetime import datetime
from tqdm.auto import tqdm

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure spawn method and vLLM worker method are set before importing vLLM
try:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
except Exception as e:
    print(f"[WARN] Failed to set multiprocessing start method to 'spawn' early: {e}")
    traceback.print_exc()
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams

from ragen.env.lbat.env import LBATEnv
from ragen.env.lbat.config import LBATEnvConfig


def init_messages(max_new_tokens: int) -> List[Dict[str, str]]:
    """Initialize messages like ctx_manager: system + empty user container."""
    system_note = "You're a helpful assistant. "
    return [
        {"role": "system", "content": system_note},
        {"role": "user", "content": ""},
    ]


def append_turn_state(messages: List[Dict[str, str]], turn_index: int, observation_text: str, max_new_tokens: int) -> None:
    """Append a Turn k state block to the trailing user message, mirroring ctx_manager."""
    format_prompt = "<think> [Your thoughts] </think> <answer> [your answer] </answer>"
    length_prompt = f"Max response length: {int(max_new_tokens)} tokens."
    messages[-1]["content"] += (
        f"\nTurn {turn_index}:\n"
        f"State:\n{observation_text}\n"
        f"Always output: {format_prompt} with no extra text. Strictly follow this format. Do not use specific algorithms. {length_prompt}\n"
    )


def append_reward(messages: List[Dict[str, str]], reward: float) -> None:
    try:
        reward_str = f"{float(reward):.4f}"
    except Exception:
        reward_str = str(reward)
    messages.append({"role": "user", "content": f"Reward:\n{reward_str}\n"})


def extract_answer_span(text: str) -> Optional[str]:
    """Extract the content inside <answer>...</answer>. Returns None if not found."""
    try:
        match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            return None
        return match.group(1).strip()
    except Exception as e:
        print(f"[ERROR] extract_answer_span failed: {e}")
        traceback.print_exc()
        return None


def parse_action_from_answer(answer_text: str, env: LBATEnv) -> Any:
    """Parse an action appropriate for the current family from the answer text.

    - MAB: integer index (0-based env expects, but we accept any int; env handles validity)
    - PEA: integer in [0, 10]
    - OPS: a weight vector string or list; we return the raw text and let env parse/validate
    """
    try:
        family = getattr(env, "family", "mab")
        if family in ("mab", "pea"):
            # Prefer integers first
            int_match = re.search(r"[-+]?\d+", answer_text)
            if int_match:
                val = int(int_match.group(0))
                if family == "pea":
                    val = max(0, min(10, val))
                return val

            # Fall back to first float, then cast
            float_match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+\.\d*", answer_text)
            if float_match:
                fval = float(float_match.group(0))
                if family == "pea":
                    return int(max(0, min(10, round(fval * 10))))
                else:
                    return int(round(fval))

            # Could not parse -> deliberately invalid to trigger env penalty/advance
            return ""

        if family == "ops":
            # Return the raw content; env will parse, normalize, and validate
            return answer_text

        # Unknown family: return the raw content
        return answer_text
    except Exception as e:
        print(f"[ERROR] parse_action_from_answer failed: {e}")
        traceback.print_exc()
        return ""


@torch.inference_mode()
def generate_action_text_vllm(
    llm: LLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
) -> str:
    """Generate one turn using vLLM, matching training-time templating."""
    try:
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if os.environ.get("LBAT_DEBUG", "").lower() in ("1", "true", "yes"):
            turns = re.findall(r"\bTurn\s+(\d+):", prompt_text)
            print(f"[DEBUG] Prompt turns seen: {turns[-8:]} (total {len(turns)})")
            print("[DEBUG] Prompt head (1000):\n" + prompt_text[:1000])
            print("[DEBUG] Prompt tail (1000):\n" + prompt_text[-1000:])

        eos_token_id = tokenizer.eos_token_id
        sp = SamplingParams(
            max_tokens=int(max_new_tokens),
            temperature=float(temperature) if bool(do_sample) else 0.0,
            top_p=float(top_p) if bool(do_sample) else 1.0,
            top_k=int(top_k) if bool(do_sample) else -1,
            n=1,
            stop=["</answer>"],
            stop_token_ids=[eos_token_id] if eos_token_id is not None else None,
        )

        outs = llm.generate([prompt_text], sp)
        # vLLM returns text for each request; take the first
        text = outs[0].outputs[0].text if outs and outs[0].outputs else ""
        if os.environ.get("LBAT_DEBUG", "").lower() in ("1", "true", "yes"):
            # vLLM doesn't directly report token count here; show length as proxy
            print(f"[DEBUG] Generated text length: {len(text)} (cap {max_new_tokens})")
            print("[DEBUG] Model output (truncated 1000 chars):\n" + text[:1000])
        return text
    except Exception as e:
        print(f"[ERROR] generate_action_text_vllm failed: {e}")
        traceback.print_exc()
        return ""


def run_single_episode(
    llm: LLM,
    tokenizer: AutoTokenizer,
    seed: int,
    device: torch.device,
    gen_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one full LBAT episode and return episode metrics and metadata."""
    try:
        # Disable prompt rewriting to avoid external API use
        env_cfg = LBATEnvConfig(rewrite_probability=0.0)
        # Allow evaluator to toggle OOD via environment variables for quick testing
        try:
            if os.environ.get("LBAT_OOD", "").lower() in ("1", "true", "yes"):
                env_cfg.ood_enabled = True
                prob = os.environ.get("LBAT_OOD_PROB", None)
                if prob is not None:
                    env_cfg.ood_probability = float(prob)
        except Exception as e:
            print(f"[WARN] could not parse LBAT_OOD env vars: {e}")
        env = LBATEnv(config=env_cfg)
        observation = env.reset(seed=seed)

        done = False
        step_idx = 1
        final_info: Dict[str, Any] = {}

        # Build messages exactly like ctx_manager: start with Turn 1 state
        messages: List[Dict[str, str]] = init_messages(int(gen_cfg.get("max_new_tokens", 128)))
        append_turn_state(messages, step_idx, observation, int(gen_cfg.get("max_new_tokens", 128)))

        while not done:
            response_text = generate_action_text_vllm(
                llm=llm,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=int(gen_cfg.get("max_new_tokens", 128)),
                temperature=float(gen_cfg.get("temperature", 0.5)),
                top_p=float(gen_cfg.get("top_p", 0.9)),
                top_k=int(gen_cfg.get("top_k", 50)),
                do_sample=bool(gen_cfg.get("do_sample", False)),
            )

            answer = extract_answer_span(response_text) or ""
            action = parse_action_from_answer(answer, env)
            # Append assistant response as-is (ctx_manager uses llm_response)
            messages.append({"role": "assistant", "content": response_text})
            try:
                observation, reward, done, info = env.step(action)
            except Exception as step_err:
                print(f"[ERROR] env.step failed at step {step_idx}: {step_err}")
                traceback.print_exc()
                # Attempt an invalid action to advance/reset safely
                observation, reward, done, info = env.step("")

            final_info = info
            # Append Reward after the assistant turn
            append_reward(messages, reward)

            if not done:
                step_idx += 1
                append_turn_state(messages, step_idx, observation, int(gen_cfg.get("max_new_tokens", 128)))
                if os.environ.get("LBAT_DEBUG", "").lower() in ("1", "true", "yes"):
                    print(f"[DEBUG] Advanced to Turn {step_idx}; last reward {float(reward):.4f}; family={getattr(env, 'family', None)}; horizon={getattr(env, 'horizon', None)}")

            if step_idx > int(getattr(env, "horizon", 128)) + 5:
                print("[WARN] Exceeded reasonable horizon steps; forcing termination.")
                break

        # Aggregate episode results
        episode_metrics: Dict[str, Any] = {
            "cumulative_reward": float(getattr(env, "cumulative_reward", 0.0)),
            "family": getattr(env, "family", None),
            "type": getattr(env, "type", None),
            "horizon": int(getattr(env, "horizon", 0)),
        }
        # Include optional final metrics if present
        for k in (
            "protagonist_regret",
            "antagonist_regret",
            "random_regret",
            "protagonist_antagonist_gap",
        ):
            if isinstance(final_info, dict) and k in final_info:
                episode_metrics[k] = float(final_info[k])

        try:
            env.close()
        except Exception as e_close:
            print(f"[WARN] env.close() raised: {e_close}")
            traceback.print_exc()

        return episode_metrics
    except Exception as e:
        print(f"[ERROR] run_single_episode failed: {e}")
        traceback.print_exc()
        return {"cumulative_reward": 0.0}


def load_vllm_and_tokenizer(model_path: str, device: torch.device):
    """Load vLLM engine and tokenizer for generation."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        # Let vLLM select GPU automatically; dtype auto
        llm = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=1)
        return llm, tokenizer
    except Exception as e:
        print(f"[ERROR] Failed to load vLLM/ tokenizer from {model_path}: {e}")
        traceback.print_exc()
        raise


def evaluate_model(
    model_path: str,
    episodes: int,
    base_seed: int,
    device: torch.device,
    gen_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate a single model over multiple episodes; return aggregate and per-episode stats."""
    llm, tokenizer = load_vllm_and_tokenizer(model_path, device)

    episode_results: List[Dict[str, Any]] = []
    for i in tqdm(range(episodes), desc=os.path.basename(model_path), leave=False):
        seed = base_seed + i
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        result = run_single_episode(llm, tokenizer, seed=seed, device=device, gen_cfg=gen_cfg)
        episode_results.append(result)
        print(
            f"[INFO] {os.path.basename(model_path)} episode {i+1}/{episodes}: reward={result.get('cumulative_reward', 0.0):.4f}, family={result.get('family')}"
        )

    # Aggregate metrics
    def _mean(key: str) -> Optional[float]:
        vals = [r[key] for r in episode_results if key in r]
        return float(sum(vals) / len(vals)) if vals else None

    summary: Dict[str, Any] = {
        "model_path": model_path,
        "episodes": episodes,
        "avg_cumulative_reward": _mean("cumulative_reward"),
        "avg_protagonist_regret": _mean("protagonist_regret"),
        "avg_antagonist_regret": _mean("antagonist_regret"),
        "avg_random_regret": _mean("random_regret"),
        "avg_gap": _mean("protagonist_antagonist_gap"),
        "per_episode": episode_results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate one or more HF models on LBATEnv.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more model paths (Hugging Face model IDs or local paths)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes (environment reinitialized each time)",
    )
    parser.add_argument("--seed", type=int, default=123, help="Base seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run generation on",
    )
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--ood", action="store_true", help="Enable OOD sampling in LBATEnv")
    parser.add_argument("--ood_prob", type=float, default=1.0, help="Probability an episode is OOD (when --ood)")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to save JSON results (defaults to outputs/<date>/<time>/eval_results.json)",
    )

    args = parser.parse_args()

    try:
        # Ensure CUDA is initialized in spawned workers, not forked children (required by vLLM)
        try:
            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)
        except Exception as e:
            print(f"[WARN] Failed to set multiprocessing start method to 'spawn': {e}")

        device = torch.device(args.device)
        gen_cfg = {
            "do_sample": bool(args.do_sample),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "max_new_tokens": int(args.max_new_tokens),
        }

        # Set OOD flags via env for the LBATEnv constructor path used in the evaluator
        if args.ood:
            os.environ["LBAT_OOD"] = "1"
            os.environ["LBAT_OOD_PROB"] = str(args.ood_prob)

        all_summaries: List[Dict[str, Any]] = []
        t0 = time.time()
        for model_path in tqdm(args.models, desc="Models"):
            print(f"[INFO] Evaluating model: {model_path}")
            summary = evaluate_model(
                model_path=model_path,
                episodes=int(args.episodes),
                base_seed=int(args.seed),
                device=device,
                gen_cfg=gen_cfg,
            )
            all_summaries.append(summary)

        t1 = time.time()
        print("\n=== Aggregate Results ===")
        for s in all_summaries:
            print(
                f"Model: {s['model_path']}\n"
                f"  episodes: {s['episodes']}\n"
                f"  avg_reward: {s['avg_cumulative_reward']:.4f}"
                + (
                    f", avg_gap: {s['avg_gap']:.4f}"
                    if s.get("avg_gap") is not None
                    else ""
                )
            )

        # Determine default output path if not provided
        if not args.out:
            now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
            args.out = os.path.join("outputs", now, "eval_results.json")

        # Ensure parent directory exists and write results
        try:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(all_summaries, f, indent=2)
            print(f"[INFO] Results written to {args.out}")
        except Exception as e:
            print(f"[ERROR] Failed to write results to {args.out}: {e}")
            traceback.print_exc()

        print(f"[INFO] Total wall time: {t1 - t0:.2f}s")
    except Exception as e:
        print(f"[FATAL] Evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()