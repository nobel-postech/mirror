SAMPLE_DIALOG = """Therapist: [Welcoming tone] Hi, Paul. I appreciate you being here today. I understand you've been feeling like a terrible pet owner due to your kittens' behavior. Can you tell me more about that?
Client: [Crosses arms, looking defensive] Yeah, they were loud all night, and I just can't handle it. It makes me feel like I'm failing them somehow.
Therapist: [Nods empathetically] It sounds really frustrating to deal with their noise. When you say you’re failing them, what thoughts come to mind?
Client: [Frowning] I don't know. Just that if I can’t keep them quiet, I must be a bad owner. Simple as that.
Therapist: [Gently probing] I hear you saying it's simple, but I wonder if we could explore some other aspects of being a pet owner together—like times when things have gone well with your kittens?
Client: [Scoffs lightly] Well, they’re cute and all, but what does that matter if they can’t behave?
Therapist: [Calmly] It's understandable to focus on their behavior. But it might help to recognize the positives too, even small ones—what do you think?"""

SAMPLE_UTTER = """Client: [Looks skeptical] Positives? Like what? They just wake me up at night."""
SAMPLE_RESPONSE = """Facial Expression Description: skeptical expression with raised eyebrows and pursed lips
Contrasting Facial Expression Description: positive or bright smiling expression"""


QUERY = """You are given a transcript of the counseling conversation and the client's utterance. Focus on capturing any visual details, particularly the facial expressions, that would match the client's last utterance. Generate facial expressions that might not align with what is being said.
### Output Format ###
Facial Expression Description: [Facial expression that aligns with the client's statement]
Contrasting Facial Expression Description: [Facial expression that contrasts with the client's statement]

### Dialogue History ###
{dialogue}

### Client's Utterance ###
{utterance}"""

def build_message_for_llama(dialogue, utter):
    messages = [
        {"role": "user", "content": QUERY.format(dialogue=SAMPLE_DIALOG, utterance=SAMPLE_UTTER)},
        {"role": "assistance", "content": SAMPLE_RESPONSE}
    ]
    messages += [{
            'role': "user",
            'content': QUERY.format(
                dialogue=dialogue,
                utterance=utter,
            )   
        }]
    return messages