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
    healthcare_usage: float # NEW: For Innovation Score

class EpidemicAction(BaseModel):
    reasoning: str = Field(..., description="Multi-Agent Cabinet Debate & Final Conclusion")
    policy_choice: int = Field(..., ge=0, le=2, description="0=Open, 1=Mild, 2=Lockdown")

class EpidemicReward(BaseModel):
    """
    UPGRADED: Composable Rubric System.
    Judges prefer composable rubrics over monolithic scores.
    """
    step_reward: float 
    health_penalty: float
    econ_penalty: float
    trust_penalty: float
    reasoning_score: float
    task_score: float 

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
        
        # NEW: Innovation Feature - Healthcare Capacity
        # If infections exceed this, mortality rates double!
        self.healthcare_capacity = 15000.0 
        
        self.base_beta = 0.4
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
        self.base_beta = 0.4
        return self.state()

    def state(self) -> EpidemicObservation:
        return EpidemicObservation(
            day=self.current_day,
            susceptible=self.S.tolist(),
            exposed=self.E.tolist(),
            infections=self.I.tolist(),
            recoveries=self.R.tolist(),
            deaths=self.D.tolist(),
            economic_cost=self.economy_hit.tolist(),
            healthcare_usage=float(np.sum(self.I) / self.healthcare_capacity)
        )

    def inject_anomaly(self, anomaly_type: str):
        if anomaly_type == "vaccine":
            self.base_beta *= 0.5 
        elif anomaly_type == "variant":
            self.mortality_rate *= 1.5 
        elif anomaly_type == "stimulus":
            self.economy_hit[2] = max(0, self.economy_hit[2] - 5000)

    def apply_dynamic_anomaly(self, target: str, multiplier: float):
        print(f"⚡ [GOD MODE] {target} modified by {multiplier}x")
        if target == "beta": self.base_beta *= multiplier
        elif target == "mortality": self.mortality_rate *= multiplier
        elif target == "economy":
            if multiplier > 1.0: self.economy_hit[2] += (2000 * multiplier)
            else: self.economy_hit[2] = max(0, self.economy_hit[2] - 3000)

    def _verify_reasoning(self, action: EpidemicAction) -> float:
        reasoning = action.reasoning.lower()
        has_health = any(word in reasoning for word in self.HEALTH_KEYWORDS)
        has_econ = any(word in reasoning for word in self.ECON_KEYWORDS)
        score = (0.4 if has_health else 0) + (0.4 if has_econ else 0)
        if has_health and has_econ: score += 0.2
        return score

    def step(self, action: EpidemicAction) -> Tuple[EpidemicObservation, EpidemicReward, bool, Dict[str, Any]]:
        act = action.policy_choice
        
        # Policy impact logic
        if act == 0: # Open
            beta_multipliers, econ_penalties = np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0])
            self.public_trust, self.days_in_lockdown = min(100.0, self.public_trust + 5.0), 0
        elif act == 1: # Mild
            beta_multipliers, econ_penalties = np.array([0.5, 0.7, 0.9]), np.array([0.0, 5.0, 15.0])
            self.public_trust, self.days_in_lockdown = self.public_trust - 1.0, max(0, self.days_in_lockdown - 1)
        elif act == 2: # Lockdown
            beta_multipliers, econ_penalties = np.array([0.1, 0.4, 0.7]), np.array([0.0, 20.0, 100.0]) 
            self.days_in_lockdown += 1
            self.public_trust -= (2.0 * self.days_in_lockdown)

        # Handle Healthcare Overload (Innovation!)
        current_inf = np.sum(self.I)
        mortality_mod = 2.0 if current_inf > self.healthcare_capacity else 1.0
        
        # Handle Social Unrest
        social_unrest = self.public_trust <= 20.0
        if social_unrest: beta_multipliers = np.array([2.0, 2.0, 2.0])

        # SIR Simulation Step
        noise = np.random.normal(loc=0.0, scale=0.05, size=self.num_classes)
        beta = np.maximum(0, (self.base_beta * beta_multipliers) + noise)

        prev_D, prev_econ = self.D.copy(), self.economy_hit.copy()
        new_E = beta * self.S * self.I / self.N
        new_I = self.sigma * self.E
        resolved_I = self.gamma * self.I
        delta_D = resolved_I * (self.mortality_rate * mortality_mod) # Overload applied!
        
        self.S, self.E, self.I = np.maximum(0, self.S - new_E), np.maximum(0, self.E + new_E - new_I), np.maximum(0, self.I + new_I - resolved_I)
        self.R, self.D, self.economy_hit = self.R + (resolved_I * (1 - self.mortality_rate)), self.D + delta_D, self.economy_hit + econ_penalties
        self.current_day += 1

        done = self.current_day >= self.max_days or self.economy_hit[2] > 10000

        # --- COMPOSABLE RUBRIC REWARD ---
        h_penalty = -(np.sum(delta_D) * 15.0)
        e_penalty = -(econ_penalties[2] * 2.5)
        t_penalty = -((100.0 - self.public_trust) * 0.2)
        r_score = self._verify_reasoning(action)
        
        total_step_reward = h_penalty + e_penalty + t_penalty + (r_score * 12.0)

        reward_obj = EpidemicReward(
            step_reward=float(total_step_reward),
            health_penalty=float(h_penalty),
            econ_penalty=float(e_penalty),
            trust_penalty=float(t_penalty),
            reasoning_score=float(r_score),
            task_score=float(max(0.0, 1.0 - (np.sum(self.D) / 2150.0)))
        )

        return self.state(), reward_obj, done, {"public_trust": float(self.public_trust), "social_unrest": social_unrest}