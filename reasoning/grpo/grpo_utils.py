import torch

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

def calculate_kld_loss(log_probs, ref_log_probs, response_mask):
    kl = torch.exp(log_probs) * (log_probs - ref_log_probs.detach())
    kl = kl.sum(dim=-1) * response_mask
    response_mask_sum = response_mask.sum(dim=1).clamp(min=1.0)
    return (kl.sum(dim=1) / response_mask_sum).mean()


def calculate_entropy(log_probs, response_mask):
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    entropy = entropy * response_mask
    response_mask_sum = response_mask.sum(dim=1).clamp(min=1.0)
    return (entropy.sum(dim=1) / response_mask_sum).mean()
