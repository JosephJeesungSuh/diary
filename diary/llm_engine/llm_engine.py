import os

from omegaconf import DictConfig, OmegaConf
from openai import AuthenticationError, BadRequestError, OpenAI

from .llm_table import LLMS
from .backoff import retry_with_exponential_backoff


class LLMEngine:
    def __init__(self, llm_config: DictConfig) -> None:
        self.config = llm_config
        self.client, self.is_instruct, self.is_reasoning = None, False, False
        self.prepare_llm(
            model_name=llm_config.model_name,
            api_provider=llm_config.api_provider,
        )
        print(f"LLMEngine::Model Name: {self.config.model_name}")
        print(f"LLMEngine::API Provider: {self.config.api_provider}")
        print(f"LLMEngine::is_instruct: {self.is_instruct}")
        print(f"LLMEngine::is_reasoning: {self.is_reasoning}")

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
                self.client = OpenAI(
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
        return self.client.completions.create(
            model=self.config.model_name,
            prompt=prompt,
            max_tokens=self.config.get("max_tokens", 1024),
            temperature=self.config.get("temperature", 0.0),
            stop=list(self.config.get("stop", [])),
            top_p=self.config.get("top_p", 1.0),
            extra_headers={"min_p": f"{self.config.min_p:.3f}"},
        )