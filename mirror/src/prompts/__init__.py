from llama import QUERY as LLAMA_QUERY
from qwen import QUERY as QWEN_QUERY
from qwen import SYSTEM_MESSAGE as QWEN_SYSTEM_MESSAGE

SAMPLE_DIALOG = """Therapist: [Welcoming tone] Hi, Paul. I appreciate you being here today. I understand you've been feeling like a terrible pet owner due to your kittens' behavior. Can you tell me more about that?
Client: [Crosses arms, looking defensive] Yeah, they were loud all night, and I just can't handle it. It makes me feel like I'm failing them somehow.
Therapist: [Nods empathetically] It sounds really frustrating to deal with their noise. When you say you’re failing them, what thoughts come to mind?
Client: [Frowning] I don't know. Just that if I can’t keep them quiet, I must be a bad owner. Simple as that.
Therapist: [Gently probing] I hear you saying it's simple, but I wonder if we could explore some other aspects of being a pet owner together—like times when things have gone well with your kittens?
Client: [Scoffs lightly] Well, they’re cute and all, but what does that matter if they can’t behave?
Therapist: [Calmly] It's understandable to focus on their behavior. But it might help to recognize the positives too, even small ones—what do you think?"""

SAMPLE_UTTER = """Client: [Looks skeptical] Positives? Like what? They just wake me up at night."""


def build_message_for_qwen(dialogue, utter):
    fewshot = [{"role": "user", "content": QWEN_QUERY.format(dialogue=SAMPLE_DIALOG, utterance=SAMPLE_UTTER)},
           {"role": "assistance", "content": """- Facial Expression Description: skeptical expression with raised eyebrows and pursed lips
- Contrasting Facial Expression Description: positive or bright smiling expression"""}]

    messages = [{"role": "system", "content": QWEN_SYSTEM_MESSAGE}]
    messages += fewshot
    messages += [{
            'role': "user",
            'content': QWEN_QUERY.format(
                dialogue=dialogue,
                utterance=utter,
            )   
        }]
    return messages

def build_message_for_llama(dialogue, utter):
    messages = [{"role": "user", "content": LLAMA_QUERY.format(dialogue=SAMPLE_DIALOG, utterance=SAMPLE_UTTER)},
           {"role": "assistance", "content": """Facial Expression Description: skeptical expression with raised eyebrows and pursed lips
Contrasting Facial Expression Description: positive or bright smiling expression"""}]
    messages += [{
            'role': "user",
            'content': LLAMA_QUERY.format(
                dialogue=dialogue,
                utterance=utter,
            )   
        }]
    return messages