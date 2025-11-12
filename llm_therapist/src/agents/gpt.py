import os
import re
import pandas as pd

from typing import List
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from pathlib import Path
from os.path import join as pjoin

from src.utils.openai_utils import OpenaiGpt
from src.utils.data_utils import load_prompt

current_dir = Path(__file__).parent

QUERY = """### Client Information ###:
{client_information}

### Reason for seeking counseling ###:
{reason_counseling}

### Counseling Dialogue History ###:
{history}"""


class GPTCounselor:
    def __init__(self, model_name, prompt_name="agent_gpt.txt") -> None:
        self.model_name = model_name
        self.system_message = load_prompt(
            pjoin(current_dir.parent, "prompts", prompt_name)
        )
        self.prompt_template = QUERY

        self.gpt = OpenaiGpt(model_name=self.model_name)

    def set_client(self, client_information, reason_counseling):
        self.client_information = client_information
        self.reason_counseling = reason_counseling
        self.history = []

    def _add_to_history(self, role, message):
        self.history.append({"role": role, "content": message})

    def _get_history_text(self, history: List[str]):
        history_text = "\n".join(
            [
                f"{message['role'].capitalize()}: {message['content']}"
                for message in history
            ]
        )
        return history_text

    def build_message(self, prompt):
        messages = [{"role": "system", "content": self.system_message}]
        messages += [{"role": "user", "content": prompt}]
        return messages

    def clean_message(self, response):
        response = response.split("'message':")[1]
        response = response.split(", {")[0]
        response = response.replace('"', "")
        response = response.replace("]", "")
        response = response.replace("}", "")
        return response

    def post_process_response(self, response):
        if "'message':" in response:
            response = self.clean_message(response)
        response = self.extract_counselor_message(response)
        return response.strip()

    def extract_after_title(self, text):
        result = re.split(r"(Counselor|Psychotherapist|Therapist)\s*:", text)
        return result[-1] if result else text

    def extract_counselor_message(self, response):
        response = self.extract_after_title(response)
        for delimiter in ["Client :", "Client:"]:
            if delimiter in response:
                response = response.split(delimiter)[0]

        response = response.replace("\n", "")
        response = response.replace("\\", "")
        response = response.replace('"', "")
        return response

    def respond(self, client_statement, **kwargs):
        self._add_to_history("Client", client_statement)
        default_params = {
            "temperature": 0.7,
            "max_tokens": 128,
            "top_p": 0.9,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.2,
            "stop": None,
        }
        params = {**default_params, **kwargs}

        history = self._get_history_text(self.history)
        prompt = self.prompt_template.format(
            client_information=self.client_information,
            reason_counseling=self.reason_counseling,
            history=history,
        )

        messages = self.build_message(prompt=prompt)
        response = self.gpt.request(messages=messages, **params)
        proc_response = self.post_process_response(response["content"])

        self._add_to_history("Counselor", proc_response)
        return {
            "content": proc_response,
            "usage": response["usage"],
            "model": response["model"],
        }
