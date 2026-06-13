import os
from accelerate import Accelerator
import paper_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from print_utils import pprint
import torch
from tqdm import tqdm
import sys
from grpo_utils import generate_responses
import numpy as np
import pandas as pd

try:
    from reasoning.env import load_environment
except ModuleNotFoundError:
    from env import load_environment

# model_name = "HuggingFaceTB/SmolLM-360M-Instruct"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "Qwen/Qwen2-Math-1.5B-Instruct"
# model_name = "unsloth/Qwen2.5-Math-1.5B-Instruct"
model_name = sys.argv[1]
# model_name = "Qwen/Qwen3-1.7B"
DATASET = "data/paper_instructions_300K-v2"
batch_size = 4
data_size = 20
temperature = 0.5

# Pre-compile regex for post-processing
def post_process(response):
    if "<think>" in response and "</think>" not in response:
        response = response + "</think>"
    return response

def extract_answer_fast(response):
    think_end = response.find("</think>")
    if think_end != -1:
        return response[think_end + len("</think>"):].strip()
    return response.strip()


def extract_thinking_fast(response):
    match = thinking_pattern.search(response)
    if match:
        return match.group(1)
    else:
        match = fallback_pattern.search(response)
        return match.group(1).strip() if match else None

def print_stats(scores_list, save_csv=False, csv_suffix=""):
    df = pd.DataFrame(scores_list)
    if save_csv:
        os.makedirs("results", exist_ok=True)
        model_id = model_name.replace("/", "-")
        csv_suffix = f"_{csv_suffix}" if csv_suffix else ""
        df.to_json(f"results/results_{model_id}{csv_suffix}.json", orient="records", indent=4)
        print("Outs saved in: ", f"results_{model_id}{csv_suffix}.json")
    stats_cols = [col for col in df.columns if col not in ["question", "response", "answer"]]
    stats_df = df[stats_cols]
    pprint(stats_df.describe().round(2))
    return stats_df

def append_scores(total_scores, scores_dict):
    for k, v in scores_dict.items():
        if k not in total_scores:
            total_scores[k] = []

        total_scores[k].extend(v)
    return total_scores


def run_inference(
    llm,
    tokenizer,
    dataloader,
    save_csv=False,
    max_new_tokens=200,
    csv_suffix="",
    reward_env=None,
):
    print("Starting inference...")
    reward_env = reward_env or load_environment()
    total_scores = {}
    for i, d in enumerate(tqdm(dataloader)):

        inputs = dict(
            input_ids = d["input_ids"],
            attention_mask = d["attention_mask"]
        )
        with torch.no_grad():
            outputs = generate_responses(
                llm,
                inputs,
                n_rollouts=1,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=temperature,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_length = d["input_ids"].shape[1]
        newly_generated_tokens = outputs[:, input_length:]
        out = tokenizer.batch_decode(newly_generated_tokens, skip_special_tokens=True)
        out = [post_process(o) for o in out] 
        model_num_tokens = (newly_generated_tokens != tokenizer.eos_token_id).to(torch.int32).sum(axis=-1).cpu().numpy()
        answers = [extract_answer_fast(o) for o in out]
        
        total_reward, rewards = paper_dataset.calculate_rewards(
            reward_env,
            out,
            d["item"],
            model_num_tokens,
            max_new_tokens=300, 
            soft_threshold_tokens=200
        )
        
        stats = {} 
        stats["question"] = [item["prompt"] for item in d["item"]]
        stats["response"] = out
        stats["answer"] = answers
        stats["num_tokens"] = model_num_tokens
        stats["total_reward"] = total_reward
        for k, v in rewards.items():
            stats[k] = v

        total_scores = append_scores(
            total_scores, stats
        )
    stats_df = print_stats(total_scores, save_csv=save_csv, csv_suffix=csv_suffix)
    return stats_df

if __name__ == "__main__":
    # Load model with optimizations
    accelerator = Accelerator()

    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataloader = paper_dataset.get_dataloader(
        DATASET,
        tokenizer=tokenizer,
        batch_size=batch_size,
        data_size=data_size,
        seed=4
    )
    
    llm, dataloader = accelerator.prepare(llm, dataloader)
    total_scores = run_inference(llm, tokenizer, dataloader, save_csv=True)
