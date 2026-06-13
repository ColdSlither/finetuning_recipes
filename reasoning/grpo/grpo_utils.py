import torch
import numpy as np

FORMAT_REWARD_WEIGHT = 0.2
CORRECTNESS_REWARD_WEIGHT = 0.8
LENGTH_REWARD_WEIGHT = 0
MAX_TOKENS = 500

def generate_responses(
    llm,
    inputs,
    max_new_tokens,
    eos_token_id,
    n_rollouts=4,
    top_p=0.95,
    temperature=0.5,
    do_sample=True,
):

    generated_response = llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        num_return_sequences=n_rollouts,
        temperature=temperature,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
    )
    return generated_response


def extract_answer(response):
    think_end = response.find("</think>")
    if think_end != -1:
        answer = response[think_end + len("</think>"):].strip()
        return answer
    return response.strip()


def calculate_length_reward(num_tokens, max_new_tokens, soft_threshold_tokens):
    if num_tokens < soft_threshold_tokens:
        return 0
    elif soft_threshold_tokens < num_tokens < max_new_tokens:
        return -(max_new_tokens - num_tokens) / (max_new_tokens - soft_threshold_tokens)
    else:
        return -1


def calculate_format_reward(response):
    has_think_open = "<think>" in response
    has_think_close = "</think>" in response
    think_count = response.count("<think>")
    think_close_count = response.count("</think>")

    if not has_think_open and not has_think_close:
        return -1

    if has_think_open and has_think_close and think_count == 1 and think_close_count == 1:
        after_think = response.split("</think>", 1)[1].strip() if "</think>" in response else ""
        if after_think:
            return 1.0
        return 0.5

    if has_think_open and has_think_close:
        return 0.3

    return 0.0


def calculate_correctness_reward(response, validation_object):
    raise RuntimeError("correctness rewards have been replaced by paper_dataset")


def calculate_rewards(
    model_responses,
    validation_objects,
    model_token_counts,
    max_new_tokens,
    soft_threshold_tokens,
):
    format_rewards = np.array(
        [calculate_format_reward(response) for response in model_responses]
    )
    length_rewards = np.array(
        [
            calculate_length_reward(num_token, max_new_tokens, soft_threshold_tokens)
            for num_token in model_token_counts
        ]
    )

    correctness_rewards = np.array(
        [
            calculate_correctness_reward(
                extract_answer(response), validation_objects[i]
            )
            for i, response in enumerate(model_responses)
        ]
    )
    # Calculate final rewards
    rewards = (
        FORMAT_REWARD_WEIGHT * format_rewards
        + CORRECTNESS_REWARD_WEIGHT * correctness_rewards
        + LENGTH_REWARD_WEIGHT * length_rewards
    )

    # Calculate average rewards using the pre-calculated components
    avg_rewards = {
        "total": (rewards),
        "format": (format_rewards),
        "correctness": (correctness_rewards),
        "length": (length_rewards),
    }

    return rewards, avg_rewards


def calculate_grpo_loss(
    log_probs,
    old_log_probs,
    advantages,
    full_response_mask,
    loss_implementation="grpo",
    clip_epsilon_low=0.2,
    clip_epsilon_high=0.3,
):
    importance_sampling_ratio = torch.exp(log_probs - old_log_probs)

    unclipped = advantages * importance_sampling_ratio
    clipped = advantages * torch.clamp(
        importance_sampling_ratio, 1 - clip_epsilon_low, 1 + clip_epsilon_high
    )
    loss = -torch.min(unclipped, clipped)
    loss = loss * full_response_mask
    
    if loss_implementation == "grpo":
        response_mask_sum = full_response_mask.sum(dim=1).clamp(min=1.0)
        return (loss.sum(dim=1) / response_mask_sum).mean()

    elif loss_implementation == "dr_grpo":
        return loss.sum() / MAX_TOKENS # MAX_TOKENS = 500

    elif loss_implementation == "bnpo":
        return loss.sum() / full_response_mask.sum().clamp(min=1.0)

def calculate_kld_loss(log_probs, old_log_probs):
    log_probs_flat = log_probs.view(-1, log_probs.shape[-1])
    old_log_probs_flat = old_log_probs.view(-1, old_log_probs.shape[-1]).detach()
    return torch.nn.functional.kl_div(
        log_probs_flat, old_log_probs_flat, log_target=True, reduction="batchmean"
    )


def calculate_entropy(log_probs, response_mask):
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    entropy = entropy * response_mask
    response_mask_sum = response_mask.sum(dim=1).clamp(min=1.0)
    return (entropy.sum(dim=1) / response_mask_sum).mean()
