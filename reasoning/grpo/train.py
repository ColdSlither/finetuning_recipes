from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
import paper_dataset
import torch
from prompt_utils import system_prompt 
from grpo_utils import (
    calculate_entropy,
    calculate_grpo_loss,
    calculate_kld_loss,
    generate_responses,
)
import numpy as np
from print_utils import pprint
from peft import get_peft_model, LoraConfig, AutoPeftModelForCausalLM
import sys
import yaml
from accelerate import Accelerator
import random
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from inference import run_inference
from tqdm import tqdm
import math
import gc

try:
    from reasoning.env import load_environment
except ModuleNotFoundError:
    from env import load_environment

config_file = sys.argv[1]
model_id = sys.argv[2]
# Load configuration
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

# Extract hyperparameters from config
model_name = config["model"]["name"]
rollout_batch_size = config["training"]["rollout_batch_size"]
batch_size = config["training"]["batch_size"]
n_rollouts = config["training"]["n_rollouts"]
max_new_tokens = config["model"]["max_new_tokens"]
soft_threshold_tokens = config["model"]["soft_threshold"]
dataset_name_or_path = config["data"]["dataset"]
gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
learning_rate = config["training"]["learning_rate"]
num_epochs = config["training"]["num_epochs"]
log_every = config["training"]["log_every"]
train_data_size = config["data"]["train_data_size"]
test_data_size = config["data"]["test_data_size"]
test_batch_size = config["data"]["test_batch_size"]
kld_weight = config["loss"]["kld_weight"]
entropy_weight = config["loss"]["entropy_weight"]
top_p = config["training"]["top_p"]
temperature = config["training"]["temperature"]
from_sft = config["model"].get("from_sft", False)
buffer_size = config["training"].get("buffer_size", 500)
num_repeats = config["training"].get("num_repeats", 5)

accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

if from_sft:
    # Load the existing LoRA model directly
    llm = AutoPeftModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        is_trainable=True,
    )
else:
    llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    lora_config = LoraConfig(
        task_type="CAUSAL_LM", 
        r=32, 
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    )

    llm = get_peft_model(llm, lora_config)

llm.print_trainable_parameters()
llm = accelerator.prepare(llm)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

train_dataset_seed = accelerator.process_index
test_dataset_seed = 16
reward_env = load_environment()

dataloader = paper_dataset.get_dataloader(
    dataset_name_or_path,
    tokenizer=tokenizer,
    batch_size=rollout_batch_size,
    split=config["data"].get("train_split", "train"),
    data_size=train_data_size,
    seed=train_dataset_seed,
)
test_dataloader = paper_dataset.get_dataloader(
    dataset_name_or_path,
    tokenizer=tokenizer,
    batch_size=test_batch_size,
    split=config["data"].get("test_split", "test"),
    data_size=test_data_size,
    seed=test_dataset_seed,
)

optimizer = torch.optim.Adam(llm.parameters(), lr=learning_rate)
optimizer, dataloader, test_dataloader = accelerator.prepare(
    optimizer, dataloader, test_dataloader
)

print("All loaded!!")

def save_model(postfix=""):
    unwrapped_llm = accelerator.unwrap_model(llm)
    unwrapped_llm.save_pretrained(f"models/{model_id}/llm{postfix}")
    tokenizer.save_pretrained(f"models/{model_id}/llm{postfix}")


def inference(csv_suffix=""):
    stats_df = run_inference(
        accelerator.unwrap_model(llm),
        tokenizer,
        test_dataloader,
        max_new_tokens=max_new_tokens,
        save_csv=True,
        csv_suffix=csv_suffix,
    )
    return stats_df["total_reward"].mean(), stats_df["total_reward"].std()


def write_responses_to_file(responses, batch_idx, items):
    with open(f"responses_{model_id}.txt", "a") as f:
        for i, response in enumerate(responses):
            item = items[i]
            answer = item["answer"]
            answer = f"Ground Truth: {answer}" if answer is not None else ""

            f.write(
                f"Batch {batch_idx}, Response {i}:\n{response}\n{'='*50}\n{answer}\n{'='*50}\n"
            )


def write_logs(log):
    log_str = " ".join([f"{k}={v}" for k, v in log.items()])
    with open(f"logs_{model_id}.txt", "a") as f:
        f.write(log_str + "\n")


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


class GRPO:
    def __init__(self):
        self.reset()
        self.num_experiences = 0
        self.num_training = 0

    def reset(self):
        self.buffer = []
        self.losses = []
        self.rewards = []
        self.individual_rewards = defaultdict(list)

    def log(self):
        mean_loss = np.mean(self.losses)
        mean_reward = np.mean(self.rewards)
        std_reward = np.std(self.rewards)

        individual_rewards = {k: np.mean(v) for k, v in self.individual_rewards.items()}
        rewards_breakdown_str = "\n".join(
            [
                f"[blue]{k}: [/blue] [bold]{v:.3f}[/bold]"
                for k, v in individual_rewards.items()
            ]
        )
        pprint(
            f"""
[blue]training steps:[/blue]: [bold]{self.num_training}[/bold]
[blue]num experiences:[/blue]: [bold]{self.num_experiences}[/bold]
[blue]loss:[/blue]: [bold]{mean_loss:.3f}[/bold]
[blue]reward:[/blue]: [bold]{mean_reward:.3f} +- {std_reward:.3f}[/bold]
{rewards_breakdown_str}
[green]inference scores:[/green]: [bold]{self.mean_inference_score:.3f} +- {self.std_inference_score:.3f}[/bold]
""",
            title="",
        )
        write_logs(
            dict(
                mean_loss=mean_loss,
                mean_reward=mean_reward,
                std_reward=std_reward,
                inference_score=self.mean_inference_score,
                **individual_rewards,
            )
        )

    def train(self):
        optimizer.zero_grad()
        i = 0
        num_train_events = 0
        best_model_id = 0
        best_inference_score = -np.inf
        for epoch in range(num_epochs):
            for batch in dataloader:
                experiences = self.collect_experiences(batch, i)
                self.num_experiences += len(experiences)
                self.buffer.extend(experiences)

                i += 1
                if i % log_every == 0 and len(self.buffer) > 0:
                    pprint(
                        f"{accelerator.process_index}, [bold green]batch {i=}, buffer length: {len(self.buffer)}, Rewards: {np.mean(self.rewards):.3f} +- {np.std(self.rewards):.3f}[/bold green]"
                    )

                if len(self.buffer) >= buffer_size:
                    print("Will be training now!")
                    self.train_on_buffer()

                    clear_gpu_memory()

                    num_train_events += 1
                    self.buffer = []

                    if accelerator.is_main_process:
                        self.mean_inference_score, self.std_inference_score = inference(
                            num_train_events
                        )

                        save_model()
                        if self.mean_inference_score > best_inference_score:
                            pprint(
                                f"New best inference score: {self.mean_inference_score:.3f}"
                            )
                            save_model(f"_best_{best_model_id}")
                            best_model_id += 1
                            best_inference_score = self.mean_inference_score

                        self.log()

                    clear_gpu_memory()

                    self.reset()

    def calculate_logits(self, full_responses, attention_masks):
        logits = llm(input_ids=full_responses, attention_mask=attention_masks).logits

        log_probs = torch.log_softmax(logits, dim=-1)

        token_log_probs = torch.gather(
            log_probs, dim=2, index=full_responses.unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probs

    def collect_experiences(self, batch, i):
        llm.eval()
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        num_examples, input_size = inputs["input_ids"].shape
        item = batch["item"]

        with torch.no_grad():
            full_responses = generate_responses(
                accelerator.unwrap_model(llm),
                inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                n_rollouts=n_rollouts,
                top_p=top_p,
                temperature=temperature,
                do_sample=True,
            )

            responses = full_responses[:, input_size:]
            attention_masks = torch.cat(
                [
                    torch.repeat_interleave(batch["attention_mask"], n_rollouts, dim=0),
                    (responses != tokenizer.pad_token_id).to(torch.int64),
                ],
                dim=1,
            )
            token_log_probs = self.calculate_logits(full_responses, attention_masks)

        decoded_responses = tokenizer.batch_decode(responses, skip_special_tokens=True)

        model_num_tokens = (
            (responses != tokenizer.eos_token_id)
            .to(torch.int32)
            .sum(axis=-1)
            .cpu()
            .numpy()
        )

        # model_responses = [num_examples * n_rollouts, max_new_tokens]
        rewards, reward_breakdown = paper_dataset.calculate_rewards(
            reward_env,
            decoded_responses,
            np.repeat(item, n_rollouts),
            model_num_tokens,
            max_new_tokens=max_new_tokens,
            soft_threshold_tokens=soft_threshold_tokens,
        )

        reward_breakdown = {k: np.mean(v) for k, v in reward_breakdown.items()}
        # rewards = [num_examples, n_rollouts]

        # advantages = [num_examples, n_rollouts]
        rewards = rewards.reshape(num_examples, n_rollouts)
        advantages = (rewards - np.mean(rewards, axis=1, keepdims=True)) / (
            np.std(rewards, axis=1, keepdims=True) + 1e-8
        )
        advantages = advantages.reshape(-1, 1)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        self.rewards.extend(rewards.flatten().tolist())
        for reward_type, reward_value in reward_breakdown.items():
            self.individual_rewards[reward_type].extend(reward_value.flatten().tolist())

        #        if i % 5 == 0:
        #            write_responses_to_file(
        #                decoded_responses.flatten(),
        #                i,
        #                items=np.repeat(item, n_rollouts).tolist(),
        #            )

        padded_responses = (full_responses != tokenizer.pad_token_id).int()
        start_of_response = torch.argmax(padded_responses, dim=1)
        end_of_response = padded_responses.shape[1] - torch.argmax(
            torch.flip(padded_responses, dims=[1]), dim=1
        )

        response_mask = torch.zeros_like(full_responses)
        for i in range(len(full_responses)):
            response_mask[i, input_size : end_of_response[i]] = 1

        full_responses = full_responses.cpu()

        advantages = advantages.cpu()
        token_log_probs = token_log_probs.cpu()

        experiences = [
            (
                full_responses[i][start_of_response[i] : end_of_response[i]],
                response_mask[i][start_of_response[i] : end_of_response[i]],
                token_log_probs[i][start_of_response[i] : end_of_response[i]],
                advantages[i],
            )
            for i in range(num_examples * n_rollouts)
            # if (advantages[i].abs() > 0.01)
        ]
        return experiences

    def train_on_buffer(self):
        accelerator.wait_for_everyone()
        llm.train()
        random.shuffle(self.buffer)
        self.buffer = self.buffer[:buffer_size]
        total_examples = len(self.buffer)
        optimizer.zero_grad()
        needs_training = False
        num_steps = 0

        total_steps = math.ceil(total_examples / batch_size) * num_repeats
        progress_bar = tqdm(
            range(total_steps), desc="Training", disable=not accelerator.is_main_process
        )
        for _ in range(num_repeats):
            for i in range(0, total_examples, batch_size):
                with accelerator.accumulate(llm):
                    training_batch = self.buffer[i : i + batch_size]
                    self.num_training += 1
                    num_steps += 1
                    loss = self.train_on_batch(training_batch)
                    accelerator.backward(loss)
                    needs_training = True
                    optimizer.step()
                    optimizer.zero_grad()
                    needs_training = False
                    if accelerator.is_main_process:
                        progress_bar.set_description(
                            f"loss: {np.mean(self.losses):.3f}"
                        )
                        progress_bar.update(1)
        if needs_training:
            optimizer.step()
            optimizer.zero_grad()
        accelerator.wait_for_everyone()
        llm.eval()

    def train_on_batch(self, batch):
        input_ids = pad_sequence(
            [x[0] for x in batch],
            batch_first=True,
            padding_side="left",
            padding_value=tokenizer.pad_token_id,
        ).to(accelerator.device)

        attention_masks = pad_sequence(
            [torch.ones_like(x[0]) for x in batch],
            batch_first=True,
            padding_side="left",
            padding_value=0,
        ).to(accelerator.device)

        response_masks = pad_sequence(
            [x[1] for x in batch],
            batch_first=True,
            padding_side="left",
            padding_value=0,
        ).to(accelerator.device)

        old_log_probs = pad_sequence(
            [x[2] for x in batch],
            batch_first=True,
            padding_side="left",
            padding_value=0,
        ).to(accelerator.device)

        advantages = (
            torch.cat([x[3] for x in batch], dim=0).unsqueeze(-1).to(accelerator.device)
        )

        log_probs = self.calculate_logits(input_ids, attention_masks)
        reasoning_loss = calculate_grpo_loss(
            log_probs, old_log_probs, advantages, response_masks
        )

        total_loss = reasoning_loss

        if kld_weight > 0:
            kld_loss = calculate_kld_loss(log_probs, old_log_probs)
            total_loss = total_loss + kld_weight * kld_loss

        if entropy_weight > 0:
            entropy = calculate_entropy(log_probs, response_masks)
            total_loss = total_loss - entropy_weight * entropy

        self.losses.append(total_loss.item())
        return total_loss


if __name__ == "__main__":
    GRPO().train()
