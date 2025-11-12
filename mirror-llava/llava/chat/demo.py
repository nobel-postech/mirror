import os
import re
import json
import argparse
import pandas as pd

from tqdm import tqdm
from ast import literal_eval
from os.path import join as pjoin
from llava.chat.vlm import *


def load_counselor(args):
    if "ec" in args.model_path and 'planning' in args.model_path:
        counselor_agent = LlavaMirrorEcPlanningCounselor(
            args=args
        )
    elif "ec" in args.model_path:
        counselor_agent = LlavaMirrorEcCounselor(
            args=args
        )
    elif "planning" in args.model_path:
        counselor_agent = LlavaMirrorPlanningCounselor(
            args=args
        )
    elif "base" in args.model_path.lower():
        counselor_agent = LlavaMirrorCounselor(
            args=args
        )
    elif args.model_path.split("/")[-1] in ["llava-v1.5-7b", "llava-v1.5-13b"]:
        counselor_agent = LlavaCounselor(
            args=args
        )
    else:
        raise NotImplementedError(args.model_path)
    return counselor_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run therapy sessions")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./checkpoints/llava-v1.5-7b-mirror_fer_planning-task-lora-epoch5",
    )
    parser.add_argument(
        "--model-base", type=str, default="/home/model/llava-v1.5-7b"
    )
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--ctx_len", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--data-path", type=str, default="./AnnoMI/prompt_df.csv")
    parser.add_argument("--annomi-dir", type=str, default="./AnnoMI/images")
    parser.add_argument("--output-path", type=str, default="results/output.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    counselor_agent = load_counselor(args)

    df = pd.read_csv(
        args.data_path,
        converters={"history": literal_eval},
    )
    output_file = open(args.output_path, "w", encoding="utf-8")
    for i, row in tqdm(df.iterrows(), total=len(df), desc="MI Demo"):
        personal_info = row["personal_info"]
        reason_counseling = row["reason_counseling"]

        history = row["history"]
        counselor_agent.set_client(
            client_information=personal_info,
            reason_counseling=reason_counseling,
        )
        counselor_agent.set_history(history)
        client_statement_verbal = row["client_statement_verbal"]
        image_file = pjoin(
            args.annomi_dir, f"{row['video_id']}/client", row["image_path"]
        )
        assert os.path.exists(image_file), image_file

        response = counselor_agent.respond(
            client_statement=client_statement_verbal, image_file=image_file
        )
        entry = row.to_dict()
        entry.update(response)

        json.dump(entry, output_file, ensure_ascii=False)
        output_file.write("\n")
