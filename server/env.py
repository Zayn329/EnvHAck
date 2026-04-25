import numpy as np
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any

# ==========================================
# 1. OPENENV SPEC: PYDANTIC MODELS
# ==========================================
class EpidemicObservation(BaseModel):
    day: int
    susceptible: List[float]
    exposed: List[float]
    infections: List[float]
    recoveries: List[float]
    deaths: List[float]
    economic_cost: List[float]

class EpidemicAction(BaseModel):
    reasoning: str = Field(..., description="Multi-Agent Cabinet Debate & Final Conclusion")
    policy_choice: int = Field(..., ge=0, le=2, description="0=Open, 1=Mild, 2=Lockdown")

class EpidemicReward(BaseModel):
    step_reward: float = Field(..., description="Trajectory shaping reward")
    task_score: float = Field(..., description="Deterministic Grader Score (0.0 - 1.0)")
    reasoning_score: float = Field(..., description="Score for Chain-of-Thought quality")

# ==========================================
# 2. OPENENV SPEC: THE ENVIRONMENT
# ==========================================
class StratifiedEpidemicEnv:
    def __init__(self, task_level: int = 3, max_days: int = 60):
        self.task_level = task_level
        self.max_days = max_days
        self.num_classes = 3
        
        # Demographics: 10k Elite, 40k Middle, 50k Poor
        self.N = np.array([10000, 40000, 50000], dtype=np.float32)
        self.sigma = 0.2
        self.gamma = 0.1
        self.mortality_rate = np.array([0.005, 0.015, 0.03], dtype=np.float32)
        
        # Base transmission rate (can be modified by anomalies)
        self.base_beta = 0.4
        
        # Keywords for Heuristic Judge
        self.HEALTH_KEYWORDS = ["surge", "infection", "cases", "death", "hospital", "medical"]
        self.ECON_KEYWORDS = ["economy", "poor", "paycheck", "bankrupt", "wealth", "economic"]
        
        self.reset()

    def reset(self) -> EpidemicObservation:
        self.current_day = 0
        self.E = np.array([0, 50, 100], dtype=np.float32)
        self.I = np.array([0, 10, 20], dtype=np.float32)
        self.R = np.array([0, 0, 0], dtype=np.float32)
        self.D = np.array([0, 0, 0], dtype=np.float32)
        self.S = self.N - self.E - self.I - self.R - self.D
        self.economy_hit = np.array([0, 0, 0], dtype=np.float32)
        self.prev_action = None
        self.public_trust = 100.0
        self.days_in_lockdown = 0
        self.base_beta = 0.4 # Reset to default
        return self.state()

    def state(self) -> EpidemicObservation:
        return EpidemicObservation(
            day=self.current_day,
            susceptible=self.S.tolist(),
            exposed=self.E.tolist(),
            infections=self.I.tolist(),
            recoveries=self.R.tolist(),
            deaths=self.D.tolist(),
            economic_cost=self.economy_hit.tolist()
        )

    def inject_anomaly(self, anomaly_type: str):
        """
        New Feature: Stress-test the AI with mid-game shocks.
        """
        if anomaly_type == "vaccine":
            self.base_beta = 0.15 # Permanent drop in transmission
        elif anomaly_type == "variant":
            self.mortality_rate *= 1.5 # Virus becomes deadlier
        elif anomaly_type == "stimulus":
            # Relief for the poor tier
            self.economy_hit[2] = max(0, self.economy_hit[2] - 5000)

    def _verify_reasoning(self, action: EpidemicAction) -> float:
        reasoning = action.reasoning.lower()
        score = 0.0
        
        # Reward for referencing both pillars in task 3
        has_health = any(word in reasoning for word in self.HEALTH_KEYWORDS)
        has_econ = any(word in reasoning for word in self.ECON_KEYWORDS)
        
        if has_health: score += 0.4
        if has_econ: score += 0.4
        if has_health and has_econ: score += 0.2
        
        return score

    def _grade(self) -> float:
        MAX_TOTAL_DEATHS = 2150.0
        MAX_POOR_ECON = 6000.0 
        death_score = max(0.0, 1.0 - (np.sum(self.D) / MAX_TOTAL_DEATHS))
        econ_score = max(0.0, 1.0 - (self.economy_hit[2] / MAX_POOR_ECON))
        return float((death_score * 0.5) + (econ_score * 0.5))

    def step(self, action: EpidemicAction) -> Tuple[EpidemicObservation, EpidemicReward, bool, Dict[str, Any]]:
        act = action.policy_choice
        
        # Policy impact logic
        if act == 0: # Open
            beta_multipliers = np.array([1.0, 1.0, 1.0])
            econ_penalties = np.array([0.0, 0.0, 0.0])
            self.public_trust = min(100.0, self.public_trust + 5.0)
            self.days_in_lockdown = 0
        elif act == 1: # Mild
            beta_multipliers = np.array([0.5, 0.7, 0.9])
            econ_penalties = np.array([0.0, 5.0, 15.0])
            self.public_trust -= 1.0
            self.days_in_lockdown = max(0, self.days_in_lockdown - 1)
        elif act == 2: # Lockdown
            beta_multipliers = np.array([0.1, 0.4, 0.7]) 
            econ_penalties = np.array([0.0, 20.0, 100.0]) 
            self.days_in_lockdown += 1
            self.public_trust -= (2.0 * self.days_in_lockdown)

        # Handle Social Unrest (Trust Failure)
        social_unrest = False
        if self.public_trust <= 20.0:
            social_unrest = True
            beta_multipliers = np.array([2.0, 2.0, 2.0]) # People ignore the rules

        # SIR Simulation Step
        noise = np.random.normal(loc=0.0, scale=0.05, size=self.num_classes)
        beta = np.maximum(0, (self.base_beta * beta_multipliers) + noise)

        prev_D = self.D.copy()
        prev_econ = self.economy_hit.copy()

        new_E = beta * self.S * self.I / self.N
        new_I = self.sigma * self.E
        resolved_I = self.gamma * self.I
        delta_D = resolved_I * self.mortality_rate
        new_R = resolved_I * (1 - self.mortality_rate)

        self.S = np.maximum(0, self.S - new_E)
        self.E = np.maximum(0, self.E + new_E - new_I)
        self.I = np.maximum(0, self.I + new_I - resolved_I)
        self.R += new_R
        self.D += delta_D
        self.economy_hit += econ_penalties
        self.current_day += 1

        done = self.current_day >= self.max_days or self.economy_hit[2] > 10000

        # --- ADVANCED REWARD CALCULATION ---
        # 1. Mortality & Economy Penalties
        step_reward = -(np.sum(delta_D) * 15.0) - (econ_penalties[2] * 2.5)
        
        # 2. Public Trust Penalty
        trust_penalty = (100.0 - self.public_trust) * 0.2
        step_reward -= trust_penalty
        
        # 3. Oscillation Penalty
        if self.prev_action is not None and act != self.prev_action:
            step_reward -= 5.0
        self.prev_action = act

        # 4. Reasoning Quality Bonus
        reasoning_score = self._verify_reasoning(action)
        step_reward += (reasoning_score * 12.0)

        reward_obj = EpidemicReward(
            step_reward=float(step_reward), 
            task_score=self._grade(),
            reasoning_score=reasoning_score
        )

        return self.state(), reward_obj, done, {
            "public_trust": float(self.public_trust),
            "social_unrest": social_unrest,
            "days_in_lockdown": self.days_in_lockdown
        }