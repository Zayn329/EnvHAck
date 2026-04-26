import os
import re
import json
import time
from openai import OpenAI

class MultiAgentPolicySystem:
    def __init__(self):
        # Qwen 2.5 72B Instruct is fully supported on the HF Serverless API
        self.model_id = "Qwen/Qwen2.5-72B-Instruct"
        
        # Securely pull the token from Hugging Face Space Secrets!
        self.hf_token = os.environ.get("HF_TOKEN")
        if not self.hf_token:
            print("WARNING: HF_TOKEN environment variable not set.")
            
        # Use the official OpenAI SDK for the Chat Completions endpoint
        self.client = OpenAI(
            base_url=f"https://api-inference.huggingface.co/models/{self.model_id}/v1/",
            api_key=self.hf_token or "dummy_token"
        )
        print(f"Connected to HF Serverless API for {self.model_id}")

    def _format_state(self, observation, history: list, prev_action: int) -> str:
        obs_dict = observation.model_dump() if hasattr(observation, 'model_dump') else observation
        inf, econ, day = obs_dict['infections'], obs_dict['economic_cost'], obs_dict['day']
        
        trend_msg = "Trend: Data stabilizing."
        if len(history) >= 3:
            past_record = history[-3]
            old_inf = past_record.get('total_infections', sum(inf))
            change = ((sum(inf) - old_inf) / (old_inf + 1)) * 100
            if change > 5: trend_msg = f"CRITICAL TREND: Infections spiked by +{change:.1f}%!"
            elif change < -5: trend_msg = f"POSITIVE TREND: Infections dropped by {change:.1f}%."

        action_map = {0: "Open", 1: "Mild", 2: "Lockdown"}
        return f"--- Day {day} ---\n{trend_msg}\nInfections: {int(sum(inf))} | Econ Damage: {int(sum(econ))}\nPrevious Policy: {action_map.get(prev_action, 'None')}"

    def get_action(self, observation, history: list, prev_action: int) -> dict:
        state_str = self._format_state(observation, history, prev_action)

        prompt = f"""You are the Mayor's Cabinet. Analyze the data and decide policy.
Cabinet Members: 
- CMO (Minimize infections)
- Economic Advisor (Prevent poor class bankruptcy)

{state_str}

Output JSON: {{"medical_advice": "...", "econ_advice": "...", "policy_choice": 0, 1, or 2}}"""
        
        # --- SMART RETRY LOOP FOR COLD STARTS ---
        raw_reply = None
        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that outputs JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.3
                )
                raw_reply = response.choices[0].message.content
                break # Success! Break the loop.
                
            except Exception as e:
                print(f"API Error (Attempt {attempt+1}): {e}")
                time.sleep(5) # Wait 5 seconds for cold starts
        
        # If API totally failed after all retries
        if not raw_reply:
            return {"reasoning": "Cabinet Offline (API timeout). Enacting emergency mild restrictions.", "policy_choice": 1}

        # --- ROBUST PARSING ---
        cleaned_reply = raw_reply.strip()
        if cleaned_reply.startswith("```json"): cleaned_reply = cleaned_reply[7:]
        elif cleaned_reply.startswith("```"): cleaned_reply = cleaned_reply[3:]
        if cleaned_reply.endswith("```"): cleaned_reply = cleaned_reply[:-3]

        match = re.search(r'\{.*\}', cleaned_reply, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                reasoning = f"CMO: {data.get('medical_advice', 'Health first')} | ECON: {data.get('econ_advice', 'Protect jobs')}"
                return {"reasoning": reasoning, "policy_choice": int(data.get("policy_choice", 1))}
            except Exception:
                try:
                    import ast
                    data = ast.literal_eval(json_str)
                    reasoning = f"CMO: {data.get('medical_advice', 'Health first')} | ECON: {data.get('econ_advice', 'Protect jobs')}"
                    return {"reasoning": reasoning, "policy_choice": int(data.get("policy_choice", 1))}
                except: pass
            
        # Recovery logic
        if "lockdown" in raw_reply.lower(): choice = 2
        elif "open" in raw_reply.lower(): choice = 0
        else: choice = 1
            
        return {"reasoning": f"Cabinet Discussion: {raw_reply[:150]}...", "policy_choice": choice}

    def interpret_anomaly(self, text_description: str) -> dict:
        fallback = {"target": "beta", "multiplier": 1.0}
        prompt = f"You are an Epidemic Simulator Engine. Read the following real-world event: '{text_description}'. Output a JSON dict containing exactly two keys: 'target' (either 'beta' for transmission or 'economy' for financial damage) and 'multiplier' (a float like 0.5 for a vaccine, or 2.0 for a super-spreader event)."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            raw_reply = response.choices[0].message.content
            
            # Robust parsing
            cleaned_reply = raw_reply.strip()
            if cleaned_reply.startswith("```json"): cleaned_reply = cleaned_reply[7:]
            elif cleaned_reply.startswith("```"): cleaned_reply = cleaned_reply[3:]
            if cleaned_reply.endswith("```"): cleaned_reply = cleaned_reply[:-3]
            
            match = re.search(r'\{.*\}', cleaned_reply, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    data = json.loads(json_str)
                    if 'target' in data and 'multiplier' in data:
                        return {"target": str(data['target']), "multiplier": float(data['multiplier'])}
                except Exception:
                    try:
                        import ast
                        data = ast.literal_eval(json_str)
                        if 'target' in data and 'multiplier' in data:
                            return {"target": str(data['target']), "multiplier": float(data['multiplier'])}
                    except: pass
        except Exception as e:
            print(f"NLP Interpretation Error: {e}")
            
        return fallback