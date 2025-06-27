from typing import Dict, Tuple, List

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

    agent_params: DictConfig = kwargs.get("agent_params", None)
    interview_params: DictConfig = kwargs.get("interview_params", None)
    query_event = Event(
        entity=interview_params.entity,
        action=continuation_prompt,
        metadata=None,
    )
    metadata: Dict = {
        "gen_cost": [],
        "retry": 0,
        "critic_history": [],
    }
    n_trial: int = 0

    input_prompt: str = (
        history.format_string() + "\n\n"
        + interview_params.entity + continuation_prompt + "\n\n"
        + agent_params.entity
    ).strip()
    if response_engine.is_instruct:
        model_input: List[Dict[str, str]] = (
            history.format_chat()
            + [{"role": "user", "content": continuation_prompt.strip()}]
        )
    else:
        model_input: str = input_prompt
    
    while n_trial < interview_params.max_trial:
        result = response_engine.prompt_llm_dispatch(prompt=model_input)
        response = (
            result.choices[0].message.content.strip()
            if response_engine.is_instruct
            else result.choices[0].text.strip()
        )
        metadata["gen_cost"].append(tuple([
            result.usage.prompt_tokens,
            result.usage.completion_tokens,
            result.usage.total_tokens,
        ]))
        if kwargs.get("interview_params", {}).get("bypass_critic", False):
            break
        c_prompts, c_pass, c_usage, good = evaluate_narrative(
            engine=critic_engine,
            context=input_prompt,
            rollout=response,
            review_criterion=["all"],
            entity=[interview_params.entity, agent_params.entity],
        )
        print(f"Evaluation result: {'Success' if good else 'Failure'}")
        metadata["critic_history"].append({
            "prompts": c_prompts,
            "pass": c_pass,
            "usage": c_usage,
        })
        if good:
            break
        metadata["retry"] += 1
        n_trial += 1

    response_event = Event(
        entity=agent_params.entity,
        action=response.strip(),
        metadata=metadata,
    )
    return query_event, response_event