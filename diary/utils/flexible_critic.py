# an LLM critic operation
# usage cases:
# 1. Evaluate the agent's response in the narrative generation process in terms of coherence.
# 2. Parse the agent's response during identity survey.
# 3. Check how many number of cigarettes the agent has smoked on the day.

from typing import Optional, List

from openai import OpenAI

from diary.llm_engine.llm_engine import LLMEngine

    
def evaluate_narrative(engine: LLMEngine,
                       context: str,
                       rollout: str,
                       review_criterion: List[str]) -> bool:
    return True
    

def parse_identity_survey(self, response: str, context: str):
    return

def check_n_smoked(self, response: str, context: str) -> int:
    return