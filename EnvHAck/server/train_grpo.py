import os
import torch
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", "unsloth/Qwen2.5-7B-Instruct-vllm-optimized") # Optimizes RL logic

from trl import GRPOTrainer, GRPOConfig
from server.env import StratifiedEpidemicEnv, EpidemicAction

# 1. Load Model with Unsloth (4-bit for memory efficiency)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-7B-Instruct",
    max_seq_length = 512,
    load_in_4bit = True,
    fast_inference = True,
)

# 2. Define Reward Functions (Process-Aware Feedback) [cite: 118, 137]
def reward_env_performance(prompts, completions, **kwargs):
    """Bridge between LLM output and your Environment math."""
    rewards = []
    env = StratifiedEpidemicEnv(task_level=3) # The "Hard" Ethical Dilemma [cite: 77]
    
    for completion in completions:
        try:
            # Parse the LLM's JSON output
            import json
            data = json.loads(completion)
            action = EpidemicAction(
                reasoning=data["reasoning"], 
                policy_choice=data["policy_choice"]
            )
            
            # Run one step in the environment
            _, reward_obj, _, _ = env.step(action)
            rewards.append(reward_obj.step_reward / 100.0) # Normalized
        except:
            rewards.append(-1.0) # Format Penalty [cite: 84]
    return rewards

def reward_reasoning_length(prompts, completions, **kwargs):
    """Ensures the model actually provides a Chain-of-Thought[cite: 95]."""
    return [0.5 if len(c) > 50 else -0.5 for c in completions]

# 3. Configure Training [cite: 186, 187]
training_args = GRPOConfig(
    output_dir = "outputs/EpidemicAI-GRPO",
    learning_rate = 5e-6,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    num_generations = 8, # Number of variations to try per prompt [cite: 135]
    max_steps = 100,      # Short run for the 24h hackathon
    logging_steps = 1,
)

# 4. Initialize Trainer
trainer = GRPOTrainer(
    model = model,
    reward_funcs = [reward_env_performance, reward_reasoning_length],
    args = training_args,
    train_dataset = None, # You can use a list of starter state prompts here
    tokenizer = tokenizer,
)

# 5. Execute Training [cite: 239]
if __name__ == "__main__":
    trainer.train()
    model.save_pretrained_merged("final_model", tokenizer, save_method = "merged_16bit") [cite: 204]