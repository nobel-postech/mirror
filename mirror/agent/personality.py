PERSONALITY_TRAITS = [
    {
        "trait_name": "Emotional Resistance",
        "desc": "Tends to avoid expressing emotions. Finds it difficult to show anxiety. Struggles to talk about past wounds. Acts indifferent towards emotions."
    },
    {
        "trait_name": "Cognitive Resistance",
        "desc": "Resists acknowledging their problems. Has a strong negative self-image. Easily rejects the therapist's advice. Maintains distorted thought patterns during therapy."
    },
    {
        "trait_name": "Behavioral Resistance",
        "desc": "Frequently misses therapy sessions. Does not follow through on assigned tasks. Approaches therapy with skepticism. Fears change and avoids new experiences."
    },
    {
        "trait_name": "No Resistance",
        "desc": "Open to discussing feelings. Eager to explore new ideas. Willing to accept feedback. Shows commitment to the therapy process."
    },
]

def generate_personality_description(trait):
    description = (
        f"The client exhibits the personality trait of '{trait['trait_name']}'. "
        f"{trait['desc']}"
    )
    return description