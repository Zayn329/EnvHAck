import os
import re
import json
import time
from openai import OpenAI
import numpy as np

class MultiAgentPolicySystem:
    def __init__(self):
        self.base_url = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
        self.api_key = os.environ.get("HF_TOKEN")
        self.model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct") # Or Gemma when you load it
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

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

        sys_prompt = "You are the Crisis Management Committee. Analyze medical and economic data. Provide reasoning first, then a policy choice in JSON."
        
        user_prompt = f"""{state_str}

Output ONLY a valid JSON object:
{{
  "reasoning": "A concise explanation...",
  "policy_choice": 0, 1, or 2
}}

0 = Open Economy
1 = Mild Restrictions
2 = Full Lockdown

JSON RESPONSE:"""

        retries = 3
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=150, 
                    temperature=0.1,
                )
                raw_reply = response.choices[0].message.content
                
                # Regex extract JSON
                match = re.search(r'\{.*\}', raw_reply, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    return {
                        "reasoning": str(data.get("reasoning", "Default logic")),
                        "policy_choice": int(data.get("policy_choice", 1))
                    }
            except Exception:
                if attempt < retries - 1:
                    time.sleep(2.0) 
                    
        return {"reasoning": "Fallback due to rate limit", "policy_choice": 1}