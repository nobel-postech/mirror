import argparse
import os
import json
import pandas as pd
from tqdm import tqdm

from os.path import join as pjoin
from llava.chat.therapy import TherapySession
from llava.utils import disable_torch_init

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run therapy sessions")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--ctx_len", type=int, default=16)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    parser.add_argument("--image_save_dir", type=str, default="./playground/data/eval/images")
    parser.add_argument(
        "--input_data",
        type=str,
        default="./mirror/annot_data/filtered/w_gender/test.csv",
    )
    parser.add_argument("--output_dir", type=str, default="results")
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

    output_path = pjoin(
        args.output_dir,
        f"{args.model_path.split('/')[-1]}_{args.client_model_name}.jsonl",
    )

    if os.path.exists(output_path):
        prev_cache = [json.loads(q) for q in open(output_path, "r")]
        cache_ids = list(map(lambda x: x["idx"], prev_cache))
    else:
        cache_ids = []

    disable_torch_init()
    therapy = TherapySession(
        args=args, 
    )
    for i, row in tqdm(
        data_df.iterrows(),
        total=len(data_df),
        desc=f"Counseling with {args.model_path.split('/')[-1]}",
    ):
        if row["idx"] in cache_ids:
            continue
        try:
            therapy.initialize_session(row.to_dict())
            result = therapy.run_session(dialog_idx=row['idx'])

            with open(output_path, "a+", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")
        except Exception as e:
            print(f"{e}")
            print(f"ERROR: Pass {row['idx']}")