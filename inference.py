import os
from server.env import StratifiedEpidemicEnv, EpidemicAction
from server.llm_agent import MultiAgentPolicySystem

# --- MANDATORY ENVIRONMENT VARIABLES (Per Screenshot Requirements) ---
# Defaults are set ONLY for Base URL and Model Name. NO default for HF_TOKEN.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") 

def run_evaluation():
    # Initialize the agent
    agent = MultiAgentPolicySystem()
    
    # We will run the 3 tasks (Levels 1, 2, and 3)
    for task_level in [1, 2, 3]:
        env = StratifiedEpidemicEnv(task_level=task_level, max_days=60)
        obs = env.reset()
        history = []
        prev_action = None
        all_rewards = []
        
        # [START] format: task, env, and model
        print(f"[START] task=task_{task_level} env=EpidemicAI model={MODEL_NAME}", flush=True)
        
        done = False
        step_num = 1
        
        while not done:
            action_int = agent.get_action(obs, history, prev_action)
            action = EpidemicAction(policy_choice=action_int)
            
            next_obs, reward_obj, done, info = env.step(action)
            
            step_reward = reward_obj.step_reward
            all_rewards.append(step_reward)
            
            # [STEP] format: reward to 2 decimals, done is lowercase
            done_str = str(done).lower()
            print(f"[STEP] step={step_num} action=policy_{action_int} reward={step_reward:.2f} done={done_str} error=null", flush=True)
            
            history.append({'day': obs.day, 'action': action_int, 'obs': next_obs})
            obs = next_obs
            prev_action = action_int
            step_num += 1
            
        # [END] format: success lowercase, score to 3 decimals
        score = float(reward_obj.task_score)
        success = score >= 0.5
        success_str = str(success).lower()
        rewards_csv = ",".join(f"{r:.2f}" for r in all_rewards)
        
        print(f"[END] success={success_str} steps={step_num-1} score={score:.3f} rewards={rewards_csv}", flush=True)

if __name__ == "__main__":
    run_evaluation()