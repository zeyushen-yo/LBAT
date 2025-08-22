import os
import time
from typing import List, Dict, Optional
from openai import AzureOpenAI, OpenAIError
from llm_exploration.inference.inference_engine import LLMInferenceEngine

class OpenAIInferenceEngine(LLMInferenceEngine):
    """
    Inference Engine for GPT-4-class models hosted on Princeton AI-Sandbox.

    Usage (identical to before):
        llm = OpenAIInferenceEngine(model_name="gpt-4o-mini")

        convs = [[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who is the first president of the US?"},
        ]]

        outputs = llm.batched_generate(
            convs=convs,
            max_n_tokens=128,
            temperature=1.0,
            top_p=1.0,
        )
        print(outputs)       # -> ["George Washington."]
    """

    API_RETRY_SLEEP   = 10       # seconds between retries after an error
    API_ERROR_OUTPUT  = "$ERROR$"
    API_QUERY_SLEEP   = 0.5      # gentle throttle between calls
    API_MAX_RETRY     = 5
    API_TIMEOUT       = 20       # seconds (currently unused but kept for parity)

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        *,
        endpoint: str = "https://api-ai-sandbox.princeton.edu/",
        api_version: str = "2025-03-01-preview",
        api_key_env_var: str = "AI_SANDBOX_KEY",
    ):
        """
        Parameters
        ----------
        model_name : str
            Deployed model in the sandbox (e.g. "gpt-4o-mini").
        api_key : str, optional
            If None, value is read from the `AI_SANDBOX_KEY` environment variable.
        endpoint / api_version : str
            Keep defaults unless Princeton changes their deployment.
        """
        self.model_name = model_name
        key = api_key or os.getenv(api_key_env_var)
        if key is None:
            raise ValueError(
                "No sandbox API key provided. "
                f"Set {api_key_env_var}=<your key> or pass api_key=<key>."
            )

        # Azure-style OpenAI client pointing to the sandbox endpoint
        self.client = AzureOpenAI(
            api_key       = key,
            azure_endpoint= endpoint,
            api_version   = api_version,
        )

    # ------------------------------------------------------------------
    # Required public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        conv: List[Dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,   # accepted for signature parity; ignored
    ) -> str:
        """
        Single prompt-response. Retries on transient API errors.
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                chat = self.client.chat.completions.create(
                    model       = self.model_name,
                    messages    = conv,
                    max_tokens  = max_n_tokens,
                    temperature = temperature,
                    top_p       = top_p,
                    n           = 1,
                )
                # Ensure we always return a string, even if the API returns None
                content = chat.choices[0].message.content
                output = content if content is not None else ""
                break
            except OpenAIError as e:
                print(f"[Sandbox API error] {type(e).__name__}: {e}")
                time.sleep(self.API_RETRY_SLEEP)
            except Exception as e:
                # Catch-all to avoid propagating None/attribute errors from SDK
                print(f"[Sandbox client error] {type(e).__name__}: {e}")
                time.sleep(self.API_RETRY_SLEEP)

            # polite spacing between calls even on success paths
            time.sleep(self.API_QUERY_SLEEP)

        # Always return a string
        return output if isinstance(output, str) else str(output)

    def batched_generate(
        self,
        convs: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> List[str]:
        """
        AI-Sandbox does **not** support native batching, so we iterate.
        """
        return [
            self.generate(
                conv,
                max_n_tokens=max_n_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
            )
            for conv in convs
        ]