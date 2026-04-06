import os
import json
from server.env import StratifiedEpidemicEnv, EpidemicAction
from server.llm_agent import MultiAgentPolicySystem

def run_evaluation():
    """
    OpenEnv Compliant Baseline Inference Script.
    Executes the environment across 3 difficulty tasks and logs strict structured stdout.
    """
    # Instantiate the OpenAI-compliant Multi-Agent System
    agent = MultiAgentPolicySystem()
    
    # Run through Easy (1), Medium (2), and Hard (3) tasks
    for task_level in [1, 2, 3]:
        env = StratifiedEpidemicEnv(task_level=task_level, max_days=60)
        
        # Pydantic Observation
        obs = env.reset()
        history = []
        prev_action = None
        total_reward = 0.0
        
        # ---------------------------------------------------------
        # MANDATORY LOG: [START]
        # ---------------------------------------------------------
        print(f"[START] Task {task_level} Initialized | State: {obs.model_dump_json()}")
        
        done = False
        step_num = 1
        
        while not done:
            # 1. Agent Logic (Passes Pydantic model to our OpenAI agent)
            action_int = agent.get_action(obs, history, prev_action)
            
            # 2. Convert raw integer to Pydantic Action model
            action = EpidemicAction(policy_choice=action_int)
            
            # 3. Step the Environment
            next_obs, reward_obj, done, info = env.step(action)
            
            # 4. Track total shaping reward
            total_reward += reward_obj.step_reward
            
            # ---------------------------------------------------------
            # MANDATORY LOG: [STEP]
            # ---------------------------------------------------------
            step_log = {
                "step": step_num,
                "action": action.model_dump(),
                "reward": reward_obj.model_dump(),
                "observation": next_obs.model_dump()
            }
            print(f"[STEP] {json.dumps(step_log)}")
            
            # 5. Append to history for the 3-day trend analysis
            history.append({
                'day': obs.day,
                'action': action_int,
                'obs': next_obs
            })
            
            obs = next_obs
            prev_action = action_int
            step_num += 1
            
        # ---------------------------------------------------------
        # MANDATORY LOG: [END]
        # ---------------------------------------------------------
        end_log = {
            "task": task_level,
            "final_score": round(reward_obj.task_score, 4),
            "total_step_reward": round(total_reward, 4)
        }
        print(f"[END] Task {task_level} Complete | Results: {json.dumps(end_log)}\n")

if __name__ == "__main__":
    # Ensure keys are loaded before running (The bot handles this automatically, 
    # but good to have a guardrail if you run it locally)
    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN environment variable is not set!")
    
    run_evaluation()