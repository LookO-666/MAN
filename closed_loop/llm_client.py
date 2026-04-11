"""DeepSeek API client with retry logic and token tracking."""

import time
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

_total_prompt_tokens = 0
_total_completion_tokens = 0


def get_token_usage():
    """Return cumulative token usage."""
    return {"prompt_tokens": _total_prompt_tokens, "completion_tokens": _total_completion_tokens,
            "total_tokens": _total_prompt_tokens + _total_completion_tokens}


def _reset_token_usage():
    """Reset cumulative token counters (used between independent runs)."""
    global _total_prompt_tokens, _total_completion_tokens
    _total_prompt_tokens = 0
    _total_completion_tokens = 0


def call_llm(system_prompt: str, user_prompt: str, *,
             api_key: str, base_url: str = "https://api.deepseek.com",
             model: str = "deepseek-chat", temperature: float = 0.7,
             max_retries: int = 3) -> str:
    """Call DeepSeek API (OpenAI-compatible) with retry and token logging."""
    global _total_prompt_tokens, _total_completion_tokens

    client = OpenAI(api_key=api_key, base_url=base_url)

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            elapsed = time.time() - t0

            usage = response.usage
            p_tok = usage.prompt_tokens if usage else 0
            c_tok = usage.completion_tokens if usage else 0
            _total_prompt_tokens += p_tok
            _total_completion_tokens += c_tok

            logger.info(f"LLM call OK  | {elapsed:.1f}s | prompt_tok={p_tok} compl_tok={c_tok}")
            print(f"  [LLM] {elapsed:.1f}s | prompt={p_tok} completion={c_tok}")

            return response.choices[0].message.content

        except Exception as e:
            logger.warning(f"LLM call attempt {attempt}/{max_retries} failed: {e}")
            print(f"  [LLM] attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"LLM call failed after {max_retries} retries: {e}") from e
