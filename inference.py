import os
from server.env import StratifiedEpidemicEnv, EpidemicAction
from server.llm_agent import MultiAgentPolicySystem

def run_evaluation():
    agent = MultiAgentPolicySystem()
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    
    for task_level in [1, 2, 3]:
        env = StratifiedEpidemicEnv(task_level=task_level, max_days=60)
        obs = env.reset()
        history = []
        prev_action = None
        
        all_rewards = []
        
        # ---------------------------------------------------------
        # STRICT OPENENV FORMAT: [START]
        # ---------------------------------------------------------
        print(f"[START] task=task_{task_level} env=EpidemicAI model={model_name}", flush=True)
        
        done = False
        step_num = 1
        
        while not done:
            action_int = agent.get_action(obs, history, prev_action)
            action = EpidemicAction(policy_choice=action_int)
            
            next_obs, reward_obj, done, info = env.step(action)
            
            step_reward = reward_obj.step_reward
            all_rewards.append(step_reward)
            
            # ---------------------------------------------------------
            # STRICT OPENENV FORMAT: [STEP]
            # ---------------------------------------------------------
            action_str = f"policy_{action_int}"
            done_str = str(done).lower()
            print(f"[STEP] step={step_num} action={action_str} reward={step_reward:.2f} done={done_str} error=null", flush=True)
            
            history.append({'day': obs.day, 'action': action_int, 'obs': next_obs})
            obs = next_obs
            prev_action = action_int
            step_num += 1
            
        # ---------------------------------------------------------
        # STRICT OPENENV FORMAT: [END]
        # ---------------------------------------------------------
        score = float(reward_obj.task_score)
        success = score >= 0.5 # Deem it a success if score is 50% or higher
        success_str = str(success).lower()
        rewards_csv = ",".join(f"{r:.2f}" for r in all_rewards)
        
        print(f"[END] success={success_str} steps={step_num-1} score={score:.2f} rewards={rewards_csv}", flush=True)

if __name__ == "__main__":
    run_evaluation()