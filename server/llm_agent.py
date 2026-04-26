import os
import re
import json
import time
import requests

class MultiAgentPolicySystem:
    def __init__(self):
        self.model_id = "zain329/EpidemicAI-Gemma2B-GRPO"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        
        # Securely pull the token from Hugging Face Space Secrets!
        self.hf_token = os.environ.get("HF_TOKEN", "")
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}
        print(f"☁️ Connected to HF Serverless API for {self.model_id}")

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

Output JSON: {{"medical_advice": "...", "econ_advice": "...", "policy_choice": 0, 1, or 2}}\nJSON RESPONSE:"""
        
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 200, "temperature": 0.3, "return_full_text": False}
        }
        
        # --- SMART RETRY LOOP FOR COLD STARTS ---
        raw_reply = None
        for attempt in range(5):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
                
                # Handle HF "Model is Loading" Cold Start
                if response.status_code == 503:
                    est_time = response.json().get('estimated_time', 20)
                    print(f"⏳ Model is waking up. Waiting {est_time:.0f} seconds...")
                    time.sleep(min(est_time, 20)) # Wait and try again
                    continue
                    
                response.raise_for_status()
                raw_reply = response.json()[0]['generated_text']
                break # Success! Break the loop.
                
            except Exception as e:
                print(f"⚠️ API Error (Attempt {attempt+1}): {e}")
                time.sleep(2)
        
        # If API totally failed after all retries
        if not raw_reply:
            return {"reasoning": "Cabinet Offline (API timeout). Enacting emergency mild restrictions.", "policy_choice": 1}

        # --- ROBUST PARSING ---
        match = re.search(r'\{.*\}', raw_reply, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0).replace("'", '"'))
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
        api_url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"
        prompt = f"You are an Epidemic Simulator Engine. Read the following real-world event: '{text_description}'. Output a JSON dict containing exactly two keys: 'target' (either 'beta' for transmission or 'economy' for financial damage) and 'multiplier' (a float like 0.5 for a vaccine, or 2.0 for a super-spreader event)."
        
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 100, "temperature": 0.1, "return_full_text": False}
        }
        
        try:
            response = requests.post(api_url, headers=self.headers, json=payload, timeout=10)
            response.raise_for_status()
            raw_reply = response.json()[0]['generated_text']
            
            # Robust parsing
            match = re.search(r'\{.*\}', raw_reply, re.DOTALL)
            if match:
                data = json.loads(match.group(0).replace("'", '"'))
                if 'target' in data and 'multiplier' in data:
                    return {"target": str(data['target']), "multiplier": float(data['multiplier'])}
        except Exception as e:
            print(f"⚠️ NLP Interpretation Error: {e}")
            
        return fallback