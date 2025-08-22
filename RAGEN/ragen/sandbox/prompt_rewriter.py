from __future__ import annotations

import os
from typing import Dict, List
from openai import AzureOpenAI


def _get_sandbox_client() -> AzureOpenAI | None:
    try:
        api_key = os.environ.get("AI_SANDBOX_KEY")
        if not api_key:
            print("[prompt_rewriter] AI_SANDBOX_KEY not set; skipping rewrite.")
            return None
        endpoint = "https://api-ai-sandbox.princeton.edu/"
        api_version = "2025-03-01-preview"
        client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
        return client
    except Exception as e:
        print(f"[prompt_rewriter] Error initializing sandbox client: {e}")
        return None


def _system_prompt_for(mode: str) -> str:
    mode = (mode or "").strip().lower()
    if mode == "reset":
        return (
            "You are a faithful paraphraser and style enhancer for initial task briefings. Diversify surface form while preserving exact semantics. "
            "Follow these rules strictly: "
            "1) Preserve all numbers, units, inequalities, mathematical expressions, identifiers, variable names, code, quoted strings, and control/special tokens exactly as given (e.g., <think>, </think>, <answer>, </answer>, <|im_start|>, <|im_end|>). "
            "2) Do not alter, remove, or add any requirements, options, steps, constraints, or instructions. Do not change list lengths or the ordering of items. Do not modify tables. "
            "3) You may add brief, clearly cosmetic flourishes that do not affect meaning: "
            "   - At the very beginning, add one short, scene-setting sentence (<= 20 words) that does not instruct the model to do anything. "
            "   - When domain entities are present, you may append parenthetical nicknames without replacing original labels: "
            "     • MAB (multi-armed bandit): add a nickname for each arm. "
            "     • OPS (online portfolio selection): add a nickname for each asset. "
            "     • PEA (prediction with expert advice): add a nickname for each expert. "
            "   - Vary sentence structure and connectors; use synonyms; switch between active/passive voice where safe. "
            "4) Never change technical content: keep all facts, counts, values, and logical relationships identical. Do not introduce new facts. "
            "5) Keep content inside control tokens or code blocks untouched; do not insert narrative inside them. "
            "6) Output only the rewritten text, with no explanations. "
        )
    # step mode (default): diversify wording only, no background or nicknames
    return (
        "You are a faithful paraphraser for mid-episode updates. Preserve exact semantics while varying phrasing to reduce homogeneity. "
        "Follow these rules strictly: "
        "1) Preserve all numbers, units, inequalities, mathematical expressions, identifiers, variable names, code, quoted strings, and control/special tokens exactly as given (e.g., <think>, </think>, <answer>, </answer>, <|im_start|>, <|im_end|>). "
        "2) Do not alter, remove, or add any requirements, options, steps, constraints, or instructions. Do not change list lengths or the ordering of items. Do not modify tables. "
        "3) Do not add any background stories, nicknames, personas, roleplay, or extra commentary. Do not prepend or append new sentences. "
        "4) Diversify only via wording changes: use synonyms, vary sentence structure and connectors, switch between active/passive voice where safe. "
        "5) Keep content inside control tokens or code blocks untouched. "
        "6) Output only the rewritten text, with no explanations. "
    )


def rewrite_text_via_sandbox(text: str, model: str = "gpt-4o", temperature: float = 0.9, max_tokens: int = 1000, mode: str = "step") -> str:
    """
    Paraphrase the given text with GPT-4o via Princeton AI Sandbox while preserving meaning and control tokens.
    If the sandbox is unavailable or any error occurs, returns the original text.
    """
    client = _get_sandbox_client()
    if client is None:
        return text

    system_prompt = _system_prompt_for(mode)

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )
        rewritten = response.choices[0].message.content
        if isinstance(rewritten, str) and len(rewritten.strip()) > 0:
            return rewritten
        return text
    except Exception as e:
        print(f"[prompt_rewriter] Rewrite failed: {e}")
        return text