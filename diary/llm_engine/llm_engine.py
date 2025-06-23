import asyncio
import os
import logging
from typing import Dict, List

from omegaconf import DictConfig
from openai import AuthenticationError, BadRequestError, OpenAI, AsyncOpenAI

from .llm_table import LLMS
from .backoff import retry_with_exponential_backoff

logging.getLogger("httpx").setLevel(logging.WARNING)


class LLMEngine:
    def __init__(self, llm_config: DictConfig) -> None:
        self.config = llm_config
        self.client, self.is_instruct, self.is_reasoning = None, False, False
        self.prepare_llm(
            model_name=llm_config.model_name,
            api_provider=llm_config.api_provider,
        )

    def prepare_llm(self, model_name: str, api_provider: str):
        if api_provider == "localhost":
            self.client = OpenAI(
                api_key="EMPTY",
                base_url=self.config.api_base.format(port=self.config.port),
            )
            models = self.client.models.list()
            model = next(iter(models)).id
            assert model == model_name, (
                f"Intended model was {model_name}, but got {model} from localhost."
            )
        else:
            if api_provider == "google":
                self.client = AsyncOpenAI(
                    api_key=os.environ.get("GOOGLE_API_KEY"),
                    base_url=self.config.api_base,
                )
            else:
                raise ValueError("Cannot use other providers.")

        llm = next((llm for llm in LLMS if llm["model_name"] == model_name), None)
        assert llm is not None, "Cannot find model in LLM table."
        self.is_instruct = llm["is_instruct"]
        self.is_reasoning = llm["is_reasoning"]
        return

    @retry_with_exponential_backoff(
        max_retries=20,
        no_retry_on=(AuthenticationError, BadRequestError),
    )
    def prompt_llm(self, prompt: str):
        assert not self.is_instruct, (
            "Called completion query on instruct model. "
            f"Model called: {self.config.model_name}"
        )
        return self.client.completions.create(
            model=self.config.model_name,
            prompt=prompt,
            max_tokens=self.config.get("max_tokens", 1024),
            temperature=self.config.get("temperature", 0.0),
            stop=list(self.config.get("stop", [])),
            top_p=self.config.get("top_p", 1.0),
            n=self.config.get("n", 1),
            extra_headers={"min_p": f"{self.config.min_p:.3f}"},
        )
    
    @retry_with_exponential_backoff(
        max_retries=20,
        no_retry_on=(AuthenticationError, BadRequestError),
    )
    def prompt_llm_chat(self, messages: List[Dict[str, str]]):
        assert self.is_instruct, (
            "Called chat query on non-instruct model. "
            f"Model called: {self.config.model_name}"
        )
        return self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=self.config.get("max_tokens", 512),
            temperature=self.config.get("temperature", 0.0),
            top_p=self.config.get("top_p", 1.0),
            n=self.config.get("n", 1),
        )
    
    def prompt_llm_chat_batch(self, messages_list: List):
        assert self.is_instruct, (
            "Called chat query on non-instruct model. "
            f"Model called: {self.config.model_name}"
        )
        return asyncio.run(self._prompt_llm_chat_batch_async(messages_list))
    
    async def _prompt_llm_chat_batch_async(self, messages_list: List):
        
        @retry_with_exponential_backoff(
            max_retries=20,
            no_retry_on=(AuthenticationError, BadRequestError),
        )
        async def _single(messages):
            return await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.get("max_tokens", 512),
                temperature=self.config.get("temperature", 0.0),
                top_p=self.config.get("top_p", 1.0),
                n=self.config.get("n", 1),
            )

        tasks = [
            asyncio.create_task(_single(messages))
            for messages in messages_list
        ]
        return await asyncio.gather(*tasks)