
QUERY = """You are given a transcript of the counseling conversation and the client's utterance. Focus on capturing any visual details, particularly the facial expressions, that would match the client's last utterance. Generate facial expressions that might not align with what is being said.
### Output Format ###
Facial Expression Description: [Facial expression that aligns with the client's statement]
Contrasting Facial Expression Description: [Facial expression that contrasts with the client's statement]

### Dialogue History ###
{dialogue}

### Client's Utterance ###
{utterance}"""
