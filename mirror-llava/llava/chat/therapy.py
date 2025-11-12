import os
import re
import json
import argparse
import pandas as pd

from tqdm import tqdm
from os.path import join as pjoin

from llava.chat.vlm import (
    LlavaCounselor, LlavaMirrorCounselor, LlavaMirrorEcCounselor, LlavaMirrorPlanningCounselor, LlavaMirrorEcPlanningCounselor
)
from llava.chat.client import GPTClient
from llava.chat.image_gen import ImageGenerator

def get_resistance_desc(personality):
    if "Emotional Resistance" in personality:
        return """This client tends to avoid expressing their emotions, finding it particularly challenging to display vulnerability or anxiety. They struggle to discuss past emotional wounds and often present an indifferent or detached attitude toward their own emotional experiences."""
    elif "Cognitive Resistance" in personality:
        return """The client demonstrates a reluctance to acknowledge their own problems, often maintaining a strong negative self-image. They are quick to reject the therapist’s advice and persist in holding onto distorted or unhelpful thought patterns throughout the therapeutic process."""
    elif "Behavioral Resistance" in personality:
        return """The client frequently skips therapy sessions and does not complete assigned tasks. They approach therapy with skepticism, expressing doubt about its effectiveness. A deep-seated fear of change leads them to avoid new experiences and resist stepping out of their comfort zone."""

def remove_text_in_parentheses(text):
    text = re.sub(r"\[[^)]*\]", "", text)
    return re.sub("[\s]+", " ", text).strip()

class TherapySession:
    def __init__(self, args):
        self.client_agent = GPTClient(args.client_model_name)
        self.max_turns = args.max_turns
        self.ctx_len = args.ctx_len
        
        self.img_gen = ImageGenerator(save_dir=args.image_save_dir)
        self.load_counselor(args)

    def load_counselor(self, args):
        if "ec" in args.model_path and 'planning' in args.model_path:
            self.counselor_agent = LlavaMirrorEcPlanningCounselor(
                args=args
            )
        elif "ec" in args.model_path:
            self.counselor_agent = LlavaMirrorEcCounselor(
                args=args
            )
        elif "planning" in args.model_path:
            self.counselor_agent = LlavaMirrorPlanningCounselor(
                args=args
            )
        elif "base" in args.model_path.lower():
            self.counselor_agent = LlavaMirrorCounselor(
                args=args
            )
        elif args.model_path.split("/")[-1] in ["llava-v1.5-7b", "llava-v1.5-13b"]:
            self.counselor_agent = LlavaCounselor(
                args=args
            )
        else:
            raise NotImplementedError(args.model_path)

    def initialize_session(self, sample):
        self.example = sample
        self.client_agent.set_personality(
            client_information=sample["personal_info"],
            personality_trait=get_resistance_desc(sample["personality"]),
            distorted_thoughts=sample["distorted_thought"],
            reason_counseling=sample["reason_for_seeking_counseling"],
        )
        self.counselor_agent.set_client(
            client_information=sample["personal_info"],
            reason_counseling=sample["reason_for_seeking_counseling"],
        )

    def _get_history_text(self, history):
        history_text = '\n'.join(
            [
                f"{message['role'].capitalize()}: {message['content']}"
                for message in history[-self.ctx_len:]
            ]
        )
        return history_text
        
    def _exchange_statements(self, dialog_idx):
        counselor_statement = None
        
        history = []
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0}
        for turn in range(self.max_turns):
            if turn == 0:
                client_output = self.client_agent.initial_response()
            else:
                client_output = self.client_agent.get_response(counselor_statement)
            total_tokens["prompt_tokens"] += client_output["usage"]["prompt_tokens"]
            total_tokens["completion_tokens"] += client_output["usage"][
                "completion_tokens"
            ]
            client_statement = client_output["content"].split(":", 1)[-1].strip()
            p, np, image_file_path = self.img_gen.run(
                origin=self.example['img_path'], 
                gender=self.example['dominant_gender'], 
                dialogue=self._get_history_text(history), 
                statement=f"Client: {client_statement}", 
                dialog_idx=dialog_idx, 
                turn_idx=turn
            )
            if "[/END]" in client_statement or "END" in client_statement:
                history += [
                    {
                        "role": "Client",
                        "content": client_statement.replace("[/END]", ""),
                        'image_path': image_file_path,
                        'prompt': p,
                        'negative_prompt': np
                    }
                ]
                if len(history) / 2 >= 2: break
            else:
                history += [{
                    "role": "Client", 
                    "content": client_statement.replace("[/END]", ""),
                    'image_path': image_file_path,
                    'prompt': p,
                    'negative_prompt': np
                }]
            client_statement_verbal = remove_text_in_parentheses(client_statement).strip()
            counselor_statement = self.counselor_agent.respond(
                client_statement=client_statement_verbal,
                image_file=image_file_path
            )
            
            if isinstance(counselor_statement, dict):
                total_tokens["prompt_tokens"] += counselor_statement["usage"][
                    "prompt_tokens"
                ]
                total_tokens["completion_tokens"] += counselor_statement["usage"][
                    "completion_tokens"
                ]
                counselor_statement = counselor_statement["content"]
            history += [{"role": "Counselor", "content": counselor_statement}]
        return history, total_tokens

    def run_session(self, dialog_idx):
        history, total_tokens = self._exchange_statements(dialog_idx)
        return {
            "idx": self.example["idx"],
            "history": history,
            "total_tokens": total_tokens,
        }
