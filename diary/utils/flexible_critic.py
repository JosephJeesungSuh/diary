# an LLM critic operation
# usage cases:
# 1. Evaluate the agent's response in the narrative generation process in terms of coherence.
# 2. Parse the agent's response during identity survey.
# 3. Check how many number of cigarettes the agent has smoked on the day.

from typing import Optional

from openai import OpenAI

class FlexibleCritic:

    def __init__(self,
                 model_name: str,
                 provider: Optional[str] = "google",
                 temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature

    def run_model(self, prompt: str) -> str:
        return
    
    def evaluate_narrative(self, response: str, context: str):
        return

    def parse_identity_survey(self, response: str, context: str):
        return
    
    def check_n_smoked(self, response: str, context: str) -> int:
        return