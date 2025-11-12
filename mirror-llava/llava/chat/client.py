import os
from glob import iglob
from pathlib import Path
from os.path import join as pjoin
from typing import List

from llava.chat.openai_utils import OpenaiGpt
from llava.chat.data_utils import load_prompt
current_dir  = Path(__file__).parent

QUERY = """### Personal Information ###: 
{client_information}

### Personality Traits ###: {personality_trait}
### Distorted Thoughts ###: {distorted_thoughts}
### Reason for Seeking Counseling ###: {reason_counseling}

### Counseling Dialogue History ###:
{history}"""

INITIAL_QUERY = """### Personal Information ###: 
{client_information}

### Personality Traits ###: {personality_trait}
### Distorted Thoughts ###: {distorted_thoughts}
### Reason for Seeking Counseling ###: {reason_counseling}

You are the client in this scenario. Your initial response should only describe your emotional state non-verbally (e.g., [Shrugs, looking down]) without any verbal statement or explanation."""

class GPTClient:
    def __init__(self, model_name, prompt_name="agent_client.txt") -> None:
        self.model_name = model_name
        self.system_message = load_prompt(pjoin(current_dir, "prompts", prompt_name))
        self.gpt = OpenaiGpt(model_name=self.model_name)
    
    def set_personality(self, client_information, personality_trait, distorted_thoughts, reason_counseling):
        self.client_information = client_information
        self.personality_trait = personality_trait
        self.distorted_thoughts = distorted_thoughts
        self.reason_counseling = reason_counseling
        self.history = []
        
    def _add_to_history(self, role, message):
        self.history.append({"role": role, "content": message})

    def _get_history_text(self, history: List[str]):
        history_text = '\n'.join(
            [
                f"{message['role'].capitalize()}: {message['content']}"
                for message in history
            ]
        )
        return history_text

    def _build_message(self):
        history = self._get_history_text(self.history)
        prompt = QUERY.format(
            client_information=self.client_information,
            personality_trait=self.personality_trait,
            distorted_thoughts=self.distorted_thoughts,
            reason_counseling=self.reason_counseling,
            history=history
        )
        messages = [{"role": "system", "content": self.system_message}]
        messages += [{
                'role': "user",
                'content': prompt 
            }]
        return messages
    
    def _build_initial_message(self):
        prompt = INITIAL_QUERY.format(
            client_information=self.client_information,
            personality_trait=self.personality_trait,
            distorted_thoughts=self.distorted_thoughts,
            reason_counseling=self.reason_counseling,
        )
        messages = [{
                'role': "user",
                'content': prompt 
            }]
        return messages
    
    
    def get_response(self, counselor_statement, **kwargs):
        default_params = {
            "temperature": 0.7,
            "max_tokens": 128,
            "top_p": 0.9,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.2,
            "stop": None
        }
        params = {**default_params, **kwargs}
        self._add_to_history("Counselor", counselor_statement)
        messages = self._build_message()
        response = self.gpt.request(messages=messages, **params)
        self._add_to_history("Client", response['content'])
        return response
    
    def initial_response(self, **kwargs):
        default_params = {
            "temperature": 0.7,
            "max_tokens": 128,
            "top_p": 0.9,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.2,
            "stop": None
        }
        params = {**default_params, **kwargs}
        messages = self._build_initial_message()
        response = self.gpt.request(messages=messages, **params)
        return response