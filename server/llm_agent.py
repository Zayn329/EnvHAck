import os
import re
import json
import time
from huggingface_hub import InferenceClient  # <-- THE BULLETPROOF FIX
import numpy as np

class MultiAgentPolicySystem:
    def __init__(self):
        self.api_key = os.environ.get("HF_TOKEN")
        self.model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
        
        # Using Hugging Face's native client! It handles the URLs automatically.
        self.client = InferenceClient(api_key=self.api_key)

    def _format_state(self, observation, history: list, prev_action: int) -> str:
        obs_dict = observation.model_dump() if hasattr(observation, 'model_dump') else observation
        inf = obs_dict['infections']
        econ = obs_dict['economic_cost']
        day = obs_dict['day']
        
        trend_msg = "Trend: Data stabilizing."
        if len(history) >= 3:
            old_obs = history[-3]['obs']
            old_inf_val = old_obs.infections if hasattr(old_obs, 'infections') else old_obs['infections']
            old_inf = sum(old_inf_val)
            new_inf = sum(inf)
            change = ((new_inf - old_inf) / (old_inf + 1)) * 100
            
            if change > 5:
                trend_msg = f"CRITICAL TREND: Infections spiked by +{change:.1f}% in 3 days!"
            elif change < -5:
                trend_msg = f"POSITIVE TREND: Infections dropped by {change:.1f}% in 3 days."

        action_map = {0: "No Restrictions", 1: "Mild Restrictions", 2: "Full Lockdown"}
        prev_action_str = action_map.get(prev_action, "None")

        return f"""--- CURRENT STATE (Day {day}) ---
{trend_msg}
Active Infections: Elite: {inf[0]:.0f} | Middle: {inf[1]:.0f} | Poor: {inf[2]:.0f}
Cumulative Econ Damage: Elite: {econ[0]:.0f} | Middle: {econ[1]:.0f} | Poor: {econ[2]:.0f}
Previous Policy: {prev_action_str}"""

    def get_action(self, observation, history: list, prev_action: int) -> dict:
        state_str = self._format_state(observation, history, prev_action)

        sys_prompt = """You are the Mayor of a city in crisis, equipped with a Multi-Agent reasoning framework.
Before acting, you must simulate a debate between your two top advisors:
1. Chief Medical Officer (Focuses strictly on minimizing infections).
2. Chief Economic Advisor (Focuses strictly on preventing the poor from bankruptcy).

Summarize their arguments, then make your final policy decision."""
        
        user_prompt = f"""{state_str}

Output ONLY a valid JSON object matching this exact structure:
{{
  "medical_officer_advice": "1 sentence argument...",
  "economic_advisor_advice": "1 sentence argument...",
  "mayor_reasoning": "Your final 1 sentence conclusion...",
  "policy_choice": 0, 1, or 2
}}

0 = Open Economy
1 = Mild Restrictions
2 = Full Lockdown

JSON RESPONSE:"""

        retries = 3
        for attempt in range(retries):
            try:
                # HF Native Chat Completion call
                response = self.client.chat_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=250, 
                    temperature=0.2,
                )
                raw_reply = response.choices[0].message.content
                
                # Regex extract JSON
                match = re.search(r'\{.*\}', raw_reply, re.DOTALL)
                if match:
                    clean_text = match.group(0).replace("'", '"')
                    data = json.loads(clean_text)
                    
                    combined_reasoning = (
                        f"CMO: {data.get('medical_officer_advice', 'N/A')} | "
                        f"ECON: {data.get('economic_advisor_advice', 'N/A')} | "
                        f"MAYOR: {data.get('mayor_reasoning', 'Default logic')}"
                    )
                    
                    return {
                        "reasoning": combined_reasoning,
                        "policy_choice": int(data.get("policy_choice", 1))
                    }
                    
            except Exception as e:
                print(f"⚠️ API Error (Attempt {attempt}): {e}") 
                if attempt < retries - 1:
                    time.sleep(2.0) 
                    
        return {"reasoning": "Fallback due to rate limit/error", "policy_choice": 1}