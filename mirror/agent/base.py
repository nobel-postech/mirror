import re
from mirror.utils.openai_utils import chatgpt_request

class Agent:
    def __init__(self, model):
        self.model = model

    def load_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def get_str_context(self, history):
        context = ""
        for utt in history:
            role = "Client" if utt['role'] == 'client' else "Therapist"
            str_utt = utt['content'].strip()
            context += f"{role}: {str_utt}\n"
        return context.strip()

    def post_processing(self, utterance):
        if re.sub(r"\s+", "" ,utterance).startswith(("Client:", "Therapist:")):
            utterance = utterance.split(":", 1)[-1]
        
        if "\n\n" in utterance:
            return utterance.split("\n\n")[0].strip()
        return utterance.strip()
    
    def request(self, messages):
        return chatgpt_request(messages=messages, model=self.model)