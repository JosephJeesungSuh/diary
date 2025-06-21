# an LLM critic operation
# usage cases:
# 1. Evaluate the agent's response in the narrative generation process in terms of coherence.
# 2. Parse the agent's response during identity survey.
# 3. Check how many number of cigarettes the agent has smoked on the day.

import re
from typing import Optional, List, Tuple, Literal, Dict

from openai import OpenAI

from diary.llm_engine.llm_engine import LLMEngine


CRITIC_PROMPT = {
    "evaluate_narrative": {
        "context": "You are given the following conversation between an interviewer and a participant:\n\n",
        "format_for_critic": {
            "consistency": """Here is the participant's response to the interviewer's last question: {response}

Question: Does the response contain any comments made by a third person, outside the context of the participant's response?
For example, if the response contains a sentence like:

"Comment: I had a great time at the beach"
"Barbara: That's a great story!"
"Continue writing"

you should mark this as 'Yes.'.

However, for cases where the participant themselves are referring to other's comments, like:

"My friend said that she had a great time at the beach."
"Some people say that it is a great story."

you should mark this as 'No.'.
Answer strictly as 'Yes.' or 'No.' Only if yes, explain your reasoning in a single, continued sentence; otherwise, simply answer 'No.'.""",
            "contains_code": """Here is the participant's response to the interviewer's last question: {response}

Question: Does the response contain any code snippets?
For example, if the response contains a sentence like:

"<div>Thank you for sharing your story</div>"
"return
100"

you should mark this as 'Yes.'.
Answer strictly as 'Yes.' or 'No.' Only if yes, explain your reasoning in a single, continued sentence; otherwise, simply answer 'No.'.""",
            "contains_metadata": """Here is the participant's response to the interviewer's last question: {response}

Question: Does the response contain any metadata or editor notes, i.e. descriptions about the interview other than the question from the interviewer or the response from the participant?
For example, if the response contains a sentence like:

"Recording: Thanks for sharing."
"2025-12-10-13-34"
"00:01:00:00"
"Transcript text: I am not sure what you mean by that."
"stopped recording."
"Editors Note: Each interview is edited from a transcript "
"This is a typical question asked by interviewers."
you should mark this as 'Yes.'.

However, for cases where the participant themselves are indicating references to facial expressions, tone of voice, or other non-verbal cues, like:
For example, if the response only contains snippets like:
(Voice gets lowered.)
(laughs)
you should mark this as 'No.'.

Answer strictly as 'Yes.' or 'No.' Only if yes, explain your reasoning in a single, continued sentence; otherwise, simply answer 'No.'.""",
            "contains_question": """Here is the participant's response to the interviewer's last question: {response}

Question: Does the response for review contain any explicit new questions that is outside the scope of the participant's response?
For example, if the response contains a sentence like:

"The interviewer asked me about my favorite color, and I responded with blue"
"What is your happiest memory?
"Response: I see. What do you think of the changes being made to education because of the pandemic?"

you should mark this as 'Yes.'.
Answer strictly as 'Yes.' or 'No.' Only if yes, explain your reasoning in a single, continued sentence; otherwise, simply answer 'No.'.""",
            "is_irrelevant": """Here is the participant's response to the interviewer's last question: {response}

Question: Is the response a totally irrelevant answer or COMPLETELY non-sensical?
Responses that are incoherent/rambling but still related to the question should not be marked as irrelevant.
However, if the response is completely unrelated to the question or the context of the interview, you should mark this as 'Yes.'.

Answer strictly as 'Yes.' or 'No.' Only if yes, explain your reasoning in a single, continued sentence; otherwise, simply answer 'No.'.""",
            "is_not_interview": """Here is the participant's response to the interviewer's last question: {response}

Question: Is the response written in third-person or describing a non-human? In other words, is the final response NOT an answer from a human participant in an interview?
For example, if the response is like:

"Nora M. is a great storyteller."
"Once upon a time, there was a robot named Robo."
"As an AI model, I was born in 2021."

You should mark this as 'Yes.'.
Answer strictly as 'Yes.' or 'No.' Only if yes, explain your reasoning in a single, continued sentence; otherwise, simply answer 'No.'.""",
        }
    }
}


def generate_critic_prompt(context: str,
                           rollout: str,
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
    if review_criterion == ["all"]: # special case
        review_criterion = list(CRITIC_PROMPT[purpose]['format_for_critic'].keys())
    
    context = context.strip().replace(
        entity[0].strip(), "Interviewer:"
    ).replace(entity[1].strip(), "Participant:")
    assert context.endswith("Participant:")
    context = context[:-len("Participant:")].strip()
    
    sys_prompt = CRITIC_PROMPT[purpose]['context'] + context
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
    print("outputs:\n\n")
    print(outputs)
    critic_pass = [
        (text is not None and text.strip().lower().startswith('no'))
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
    

def parse_identity_survey(self, response: str, context: str):
    raise NotImplementedError

def check_n_smoked(self, response: str, context: str) -> int:
    raise NotImplementedError