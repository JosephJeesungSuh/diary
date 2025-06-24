CRITIC_PROMPT = {
    "evaluate_narrative": {
        "context": """You are given the following conversation (possibly multi-turn) between an interviewer and a participant:"
========== Beginning of conversation ==========
{conversation}
========== End of conversation ==========""",
        "format_for_critic": {
            "consistency": """Here is the participant's response to the interviewer's last question:
========== Beginning of response ==========
{response}
========== End of response ==========

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
            "contains_code": """Here is the participant's response to the interviewer's last question:
========== Beginning of response ==========
{response}
========== End of response ==========

Question: Does the response contain any code snippets?
For example, if the response contains a sentence like:

"<div>Thank you for sharing your story</div>"
"return
100"

you should mark this as 'Yes.'.
Answer strictly as 'Yes.' or 'No.' Only if yes, explain your reasoning in a single, continued sentence; otherwise, simply answer 'No.'.""",
            "contains_metadata": """Here is the participant's response to the interviewer's last question:
========== Beginning of response ==========
{response}
========== End of response ==========

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
            "contains_question": """Here is the participant's response to the interviewer's last question:
========== Beginning of response ==========
{response}
========== End of response ==========

Question: Does the response for review contain any explicit new questions that is outside the scope of the participant's response?
For example, if the response contains a sentence like:

"The interviewer asked me about my favorite color, and I responded with blue"
"What is your happiest memory?
"Response: I see. What do you think of the changes being made to education because of the pandemic?"

you should mark this as 'Yes.'.
Answer strictly as 'Yes.' or 'No.' Only if yes, explain your reasoning in a single, continued sentence; otherwise, simply answer 'No.'.""",
            "is_irrelevant": """Here is the participant's response to the interviewer's last question:
========== Beginning of response ==========
{response}
========== End of response ==========

Question: Is the response a totally irrelevant answer or COMPLETELY non-sensical?
Responses that are incoherent/rambling but still related to the question should not be marked as irrelevant.
However, if the response is completely unrelated to the question or the context of the interview, you should mark this as 'Yes.'.

Answer strictly as 'Yes.' or 'No.' Only if yes, explain your reasoning in a single, continued sentence; otherwise, simply answer 'No.'.""",
            "is_not_interview": """Here is the participant's response to the interviewer's last question:
========== Beginning of response ==========
{response}
========== End of response ==========

Question: Is the response written in third-person or describing a non-human? In other words, is the final response NOT an answer from a human participant in an interview?
For example, if the response is like:

"Nora M. is a great storyteller."
"Once upon a time, there was a robot named Robo."
"As an AI model, I was born in 2021."

You should mark this as 'Yes.'.
Answer strictly as 'Yes.' or 'No.' Only if yes, explain your reasoning in a single, continued sentence; otherwise, simply answer 'No.'.""",
        }
    },
    "parse_identity_survey": {
        "context": """You are given the following question and a person's response to the question:
========== Beginning of question ==========
Question: {question_body}
Available options, each separated by a newline:
{options}
========== End of question ==========
========== Beginning of response ==========
{response}
========== End of response ==========
Instruction: Print strictly which option the person's response corresponds to. Copy the option exactly as it is, without any additional text or explanation or modification.
If the response does not match any of the options, strictly print '[N/A]'.""",
    }
}