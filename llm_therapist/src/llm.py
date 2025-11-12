import os
import re
import pandas as pd

from typing import List
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from pathlib import Path
from os.path import join as pjoin
from src.utils.data_utils import load_prompt

current_dir = Path(__file__).parent

class Llama3:
    def __init__(self, model_name_or_path, device="cuda") -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device

        self._load_model()

    def _load_model(self):
        self.llm = LLM(
            model=self.model_name_or_path,
            tensor_parallel_size=1,
            device=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def batch_generate(self, batch):
        prompts_chat_applied = [
            self.tokenizer.apply_chat_template(
                prompt["messages"] if isinstance(prompt, dict) else prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in batch
        ]
        outputs = self.llm.generate(prompts_chat_applied, self.sampling_params)
        generated_texts = [o.outputs[0].text for o in outputs]
        return generated_texts

    def clean_message(self, response):
        response = response.split("'message':")[1]
        response = response.split(", {")[0]
        response = response.replace('"', "")
        response = response.replace("]", "")
        response = response.replace("}", "")
        return response

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

    def post_process_response(self, response):
        if "'message':" in response:
            response = self.clean_message(response)
        response = self.extract_counselor_message(response)
        return response.strip()

    def generate(self, prompt, **kwargs):
        if isinstance(prompt, dict):
            prompt = prompt["messages"]

        default_params = {
            "temperature": 0.7,
            "max_tokens": 128,
            "top_p": 0.9,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.2,
            "stop": None,
        }
        params = {**default_params, **kwargs}
        sampling_params = SamplingParams(**params)
        prompts_chat_applied = [
            self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        ]
        outputs = self.llm.generate(prompts_chat_applied, sampling_params)
        generated_texts = [o.outputs[0].text for o in outputs]
        return self.post_process_response(generated_texts[0])

class CactusLlama3Counselor(Llama3):
    def __init__(
        self,
        model_name_or_path,
        prompt_name="agent_cactus_llama3.txt",
        cbt_prompt_name="agent_cbt_llama3.txt",
        device="cuda",
    ):
        super().__init__(model_name_or_path=model_name_or_path, device=device)

        self.prompt_template = load_prompt(
            pjoin(current_dir.parent, "prompts", prompt_name)
        )
        self.cbt_prompt_template = load_prompt(
            pjoin(current_dir.parent, "prompts", cbt_prompt_name)
        )
        self.pattern = r"CBT technique:\s*(.*?)\s*Counseling plan:\s*(.*)"

    def set_client(self, client_information, reason_counseling):
        self.client_information = client_information
        self.reason_counseling = reason_counseling
        self.history = []

    def set_plan(self):
        history = self._get_history_text(self.history)
        self.cbt_technique, self.cbt_plan = self.get_cbt_plan(history)

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
        messages = [{"role": "user", "content": prompt}]
        return messages

    def respond(self, client_statement, max_tokens=128):
        self._add_to_history("Client", client_statement)

        # Build Prompt
        history = self._get_history_text(self.history)
        prompt = self.prompt_template.format(
            client_information=self.client_information,
            reason_counseling=self.reason_counseling,
            cbt_plan=self.cbt_plan,
            history=history,
        )
        messages = self.build_message(prompt=prompt)
        response = self.generate(messages)
        proc_response = self.post_process_response(response)
        self._add_to_history("Counselor", proc_response)

        return proc_response

    def get_cbt_plan(self, history, max_tokens=512):
        history = self._get_history_text(self.history)
        prompt = self.cbt_prompt_template.format(
            client_information=self.client_information,
            reason_counseling=self.reason_counseling,
            history=history,
        )
        messages = self.build_message(prompt=prompt)
        response = self.generate(messages, max_tokens=max_tokens)

        try:
            cbt_technique = response.split("Counseling")[0].replace("\n", "")
        except Exception as e:
            cbt_technique = None
            print(e)

        try:
            cbt_plan = response.split("Counseling")[1].split(":\n")[1]
        except Exception as e:
            cbt_plan = None
            print(e)

        if cbt_plan:
            return cbt_technique, cbt_plan
        else:
            error_file_path = Path(f"./invalid_response.txt")
            with open(error_file_path, "a+", encoding="utf-8") as f:
                f.write(response)
            raise ValueError("Invalid response format from LLM")

    def extract_cbt_details(self, response):
        match = re.search(self.pattern, response, re.DOTALL | re.IGNORECASE)
        if not match:
            return None, None

        cbt_technique = match.group(1).strip()
        cbt_plan = match.group(2).strip()
        return cbt_technique, cbt_plan
