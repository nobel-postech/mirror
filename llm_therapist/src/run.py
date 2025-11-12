import os
import re
import json
import argparse
import pandas as pd

from tqdm import tqdm
from os.path import join as pjoin

from src.agents import GPTClient, GPTCounselor, CactusLlama3Counselor, Llama3Counselor


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
    def __init__(self, client_model_name, counselor_model_name, max_turns):
        self.client_agent = GPTClient(client_model_name)
        self.max_turns = max_turns
        self.counselor_model_name = counselor_model_name
        self.load_counselor(self.counselor_model_name)

    def load_counselor(self, counselor_model_name):
        if "camel" in counselor_model_name:
            self.counselor_agent = CactusLlama3Counselor(
                model_name_or_path=counselor_model_name
            )
        elif "llama" in counselor_model_name.lower():
            self.counselor_agent = Llama3Counselor(
                model_name_or_path=counselor_model_name
            )
        elif "gpt" in counselor_model_name:
            self.counselor_agent = GPTCounselor(model_name=counselor_model_name)
        else:
            raise NotImplementedError(counselor_model_name)

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
        if "camel" in self.counselor_model_name:
            self.counselor_agent.set_plan()

    def _exchange_statements(self):
        counselor_statement = "Hello, {name}. Thank you for joining me today. How are you feeling these days?".format(
            name=self.example["name"].split()[0]
        )
        self.counselor_agent._add_to_history("Counselor", counselor_statement)
        history = [{"role": "Counselor", "content": counselor_statement}]
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0}
        for turn in range(self.max_turns):
            client_output = self.client_agent.get_response(counselor_statement)
            total_tokens["prompt_tokens"] += client_output["usage"]["prompt_tokens"]
            total_tokens["completion_tokens"] += client_output["usage"][
                "completion_tokens"
            ]
            try:
                client_statement = client_output["content"].split(":", 1)[1].strip()
            except Exception as e:
                client_statement = client_output["content"].strip()

            if "[/END]" in client_statement or "END" in client_statement:
                history += [
                    {
                        "role": "Client",
                        "content": client_statement.replace("[/END]", ""),
                    }
                ]
                if len(history) / 2 >= 2: break
            else:
                history += [{
                    "role": "Client", 
                    "content": client_statement.replace("[/END]", ""),
                }]

            client_statement_verbal = remove_text_in_parentheses(client_statement)
            counselor_statement = self.counselor_agent.respond(
                client_statement=client_statement_verbal
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

    def run_session(self):
        history, total_tokens = self._exchange_statements()
        return {
            "idx": self.example["idx"],
            "cbt_technique": getattr(self.counselor_agent, "cbt_technique", None),
            "cbt_plan": getattr(self.counselor_agent, "cbt_plan", None),
            "history": history,
            "total_tokens": total_tokens,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run therapy sessions")
    parser.add_argument(
        "--input_data",
        type=str,
        default="test.csv",
    )
    parser.add_argument("--output_dir", type=str, default="results_v3")
    parser.add_argument(
        "--counselor_model_path", type=str, default="/home/model/camel-llama3"
    )
    parser.add_argument(
        "--client_model_name",
        type=str,
        default="gpt-3.5-turbo",
        choices=["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
        help="Type of LLM to use.",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=20,
        help="Maximum number of turns for the session.",
    )
    args = parser.parse_args()

    data_df = pd.read_csv(args.input_data)
    os.makedirs(args.output_dir, exist_ok=True)

    assert len(data_df) == 800
    
    output_path = pjoin(
        args.output_dir,
        f"{args.counselor_model_path.split('/')[-1]}_{args.client_model_name}.jsonl",
    )

    if os.path.exists(output_path):
        prev_cache = [json.loads(q) for q in open(output_path, "r")]
        cache_ids = list(map(lambda x: x["idx"], prev_cache))
    else:
        cache_ids = []

    therapy = TherapySession(
        client_model_name=args.client_model_name,
        counselor_model_name=args.counselor_model_path,
        max_turns=args.max_turns,
    )

    for i, row in tqdm(
        data_df.iterrows(),
        total=len(data_df),
        desc=f"Counseling with {args.counselor_model_path.split('/')[-1]}",
    ):
        if row["idx"] in cache_ids:
            continue
        therapy.initialize_session(row.to_dict())
        result = therapy.run_session()

        with open(output_path, "a+", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")
