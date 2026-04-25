import os
from server.llm_agent import MultiAgentPolicySystem

# Make sure your HF token is set locally!
os.environ["HF_TOKEN"] = "your_hf_token_here" 

print("🧠 Booting up the Multi-Agent Cabinet...")
agent = MultiAgentPolicySystem()

# Mock environment data (A crisis scenario)
mock_obs = {
    'day': 15,
    'infections': [500, 2500, 8000], # High poor infections
    'economic_cost': [100, 1500, 45000] # Poor economy is near 50k bankruptcy
}

print("⚖️ Asking the Mayor for a decision...")
# We pass empty history and '1' as previous action for the test
response = agent.get_action(mock_obs, history=[], prev_action=1)

print("\n🎯 THE VERDICT:")
print(f"Policy Chosen: {response['policy_choice']}")
print(f"Reasoning Log:\n{response['reasoning']}")