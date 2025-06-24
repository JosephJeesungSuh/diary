# an LLM critic operation
# usage cases:
# 1. Evaluate the agent's response in the narrative generation process in terms of coherence.
# 2. Parse the agent's response during identity survey.
# 3. Check how many number of cigarettes the agent has smoked on the day.

from typing import Optional, List, Tuple, Literal, Dict, Union, Any

from diary.llm_engine.llm_engine import LLMEngine
from .critic_prompt import CRITIC_PROMPT

def generate_critic_prompt(context: Any,
                           rollout: Union[str, List[str]],
                           review_criterion: List[str],
                           purpose: Literal[
                               "evaluate_narrative",
                               "parse_identity_survey",
                               "check_n_smoked"],
                           entity: List[str]) -> List[List[Dict[str, str]]]:
    assert purpose in CRITIC_PROMPT, (
        "invalid purpose for calling critic model."
    )
    assert isinstance(review_criterion, list)

    if purpose == "evaluate_narrative":
        assert isinstance(rollout, str)
        if review_criterion == ["all"]: # special case
            review_criterion = list(CRITIC_PROMPT[purpose]['format_for_critic'].keys())
        
        context = context.strip().replace(
            entity[0].strip(), "Interviewer:"
        ).replace(entity[1].strip(), "Participant:")
        assert context.endswith("Participant:")
        context = context[:-len("Participant:")].strip()
        
        sys_prompt = CRITIC_PROMPT[purpose]['context'].format(conversation=context)
        critic_prompts = [
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": (
                    CRITIC_PROMPT[purpose]['format_for_critic'][
                        criterion].format(response=rollout)
                )},
            ]
            for criterion in review_criterion
        ]
        return critic_prompts
    
    elif purpose == "parse_identity_survey":
        critic_prompts = [
            [
                {"role": "user", "content": (
                    CRITIC_PROMPT[purpose]['context'].format(
                        question_body=context['question_body'],
                        options="\n".join(context['category']),
                        response=r
                    )
                 )}
            ] for r in rollout
        ]
        return critic_prompts


def evaluate_narrative(engine: LLMEngine,
                       context: str,
                       rollout: str,
                       review_criterion: List[str],
                       entity: List[str]) -> Tuple:
    critic_prompts: List[List] = generate_critic_prompt(
        context=context,
        rollout=rollout,
        review_criterion=review_criterion,
        purpose="evaluate_narrative",
        entity=entity,
    )
    critic_runs = [engine.prompt_llm_chat(p) for p in critic_prompts]
    outputs = [run.choices[0].message.content for run in critic_runs]
    critic_pass = [
        (text is not None and text.strip().lower().startswith('no.'))
        for text in outputs
    ]
    critic_usage = [
        tuple([
            run.usage.prompt_tokens,
            run.usage.completion_tokens,
            run.usage.total_tokens,
        ]) for run in critic_runs
    ]
    return (
        critic_prompts, critic_pass, critic_usage,
        True if all(critic_pass) else False
    )
    

def parse_identity_survey(engine: LLMEngine,
                          context: Dict[str, Union[str, List[str]]],
                          rollouts: List[str]) -> Tuple:
    critic_prompts: List[List] = generate_critic_prompt(
        context=context,
        rollout=rollouts,
        review_criterion=[],
        purpose="parse_identity_survey",
        entity=[],
    )
    critic_runs = engine.prompt_llm_chat_batch(critic_prompts)
    outputs = [run.choices[0].message.content for run in critic_runs]
    critic_usage = tuple([
        sum(run.usage.prompt_tokens for run in critic_runs),
        sum(run.usage.completion_tokens for run in critic_runs),
        sum(run.usage.total_tokens for run in critic_runs),
    ])
    category = context['category']
    stats: List[float] = [0.0] * len(category)
    na_count: int = 0
    for o in outputs:
        if o.strip() in category:
            stats[category.index(o.strip())] += 1.0
        else:
            na_count += 1
    stats = [s/sum(stats) if sum(stats) > 0 else 0.0 for s in stats]
    return (critic_prompts, critic_usage, stats, na_count)


def check_n_smoked(self, response: str, context: str) -> int:
    raise NotImplementedError