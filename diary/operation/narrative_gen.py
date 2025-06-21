from typing import Dict, Any, Tuple

from openai import OpenAI
from omegaconf import DictConfig

from diary.entity.history import Event, History
from diary.llm_engine.llm_engine import LLMEngine
from diary.utils.flexible_critic import evaluate_narrative

def generate_narrative(
    history: History,
    continuation_prompt: str,
    response_engine: LLMEngine,
    critic_engine: LLMEngine,
    **kwargs) -> Tuple[Event, Event]:

    agent_params: Dict[str, Any] = kwargs.get("agent_params", {})
    interview_params: Dict[str, Any] = kwargs.get("interview_params", {})
    input_prompt: str = (
        history.format_string() + "\n\n"
        + interview_params.entity + continuation_prompt + "\n\n"
        + agent_params.entity
    ).strip()
    query_event = Event(
        entity=interview_params.entity,
        action=continuation_prompt,
        metadata=None,
    )
    
    metadata: Dict = {
        "cost": [],
        "retry": 0
    }
    n_trial: int = 0
    while n_trial < interview_params.max_trial:
        result = response_engine.prompt_llm(prompt=input_prompt)
        response = result.choices[0].text.strip()
        metadata["cost"].append(tuple([
            result.usage.prompt_tokens,
            result.usage.completion_tokens,
        ]))
        pass_critic = evaluate_narrative(
            engine=critic_engine,
            context=input_prompt,
            rollout=response,
            review_criterion=["all"],
        )
        if pass_critic:
            break
        metadata["retry"] += 1
        n_trial += 1

    response_event = Event(
        entity=agent_params.entity,
        action=response.strip(),
        metadata=metadata,
    )
    return query_event, response_event