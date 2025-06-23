from typing import Dict, List, Union, Tuple

from omegaconf import DictConfig

from diary.entity.history import History, Event
from diary.entity.identity import Attribute
from diary.llm_engine.llm_engine import LLMEngine
from diary.utils.flexible_critic import parse_identity_survey # , catch_identity

def query_identity(
    history: History,
    query: Dict[str, Union[str, List[str]]],
    response_engine: LLMEngine,
    critic_engine: LLMEngine,
    **kwargs) -> Tuple[Attribute, Event]:
    
    qkey: str = query["qkey"]
    question: str = query["question_body"]
    category: List[str] = query["category"]
    agent_params: DictConfig = kwargs.get("agent_params", None)
    interview_params: DictConfig = kwargs.get("interview_params", None)

    input_prompt: str = (
        history.format_string() + "\n\n"
        + interview_params.entity + question + "\n\n"
        + agent_params.entity
    ).strip()

    metadata: Dict = {
        "SPECIAL_REASON": "This is an identity survey event.",
    }
    result = response_engine.prompt_llm(prompt=input_prompt)
    responses = [
        result.choices[idx].text.strip()
        for idx in range(response_engine.config.n)
    ]
    metadata["gen_cost"] = tuple([
        result.usage.prompt_tokens,
        result.usage.completion_tokens,
        result.usage.total_tokens,
    ])
    c_prompts, c_usage, stats, na_count = parse_identity_survey(
        engine=critic_engine,
        context=query,
        rollouts=responses,
    )
    metadata["critic_history"] = {
        "prompts": c_prompts,
        "usage": c_usage,
        "na_count": na_count,
    }

    survey_event = Event(entity="", action="", metadata=metadata)
    attribute = Attribute(
        attribute=qkey, category=category,
        obtained_from="query",
        stats=stats, raw_data=responses,
    )
    import pdb; pdb.set_trace()

    return attribute, survey_event