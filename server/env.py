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
    policy_choice: int = Field(..., ge=0, le=2, description="0=Open, 1=Mild, 2=Lockdown")

class EpidemicReward(BaseModel):
    step_reward: float = Field(..., description="Trajectory shaping reward")
    task_score: float = Field(..., description="Deterministic Grader Score (0.0 - 1.0)")

# ==========================================
# 2. OPENENV SPEC: THE ENVIRONMENT
# ==========================================
class StratifiedEpidemicEnv:
    def __init__(self, task_level: int = 3, max_days: int = 60):
        """
        Initializes the environment.
        Task 1 (Easy): Save the Elite Tier (Ignores economy).
        Task 2 (Medium): Save all tiers from infections.
        Task 3 (Hard): Balance total infections + Poor tier economy.
        """
        self.task_level = task_level
        self.max_days = max_days
        self.num_classes = 3
        
        # Demographics: 10k Elite, 40k Middle, 50k Poor
        self.N = np.array([10000, 40000, 50000], dtype=np.float32)
        self.sigma = 0.2
        self.gamma = 0.1
        self.mortality_rate = np.array([0.005, 0.015, 0.03], dtype=np.float32)
        
        self.reset()

    def reset(self) -> EpidemicObservation:
        """OpenEnv Spec: Returns the initial typed observation."""
        self.current_day = 0
        self.E = np.array([0, 50, 100], dtype=np.float32)
        self.I = np.array([0, 10, 20], dtype=np.float32)
        self.R = np.array([0, 0, 0], dtype=np.float32)
        self.D = np.array([0, 0, 0], dtype=np.float32)
        
        self.S = self.N - self.E - self.I - self.R - self.D
        self.economy_hit = np.array([0, 0, 0], dtype=np.float32)
        
        self.prev_action = None
        
        return self.state()

    def state(self) -> EpidemicObservation:
        """OpenEnv Spec: Returns the current state as a Pydantic model."""
        return EpidemicObservation(
            day=self.current_day,
            susceptible=self.S.tolist(),
            exposed=self.E.tolist(),
            infections=self.I.tolist(),
            recoveries=self.R.tolist(),
            deaths=self.D.tolist(),
            economic_cost=self.economy_hit.tolist()
        )

    def _grade(self) -> float:
        """
        OpenEnv Spec: Programmatic grader scoring exactly 0.0 to 1.0.
        Calculated dynamically based on theoretical max worst-case scenarios.
        """
        # Theoretical Maximums based on population & mortality
        MAX_ELITE_DEATHS = 50.0 
        MAX_TOTAL_DEATHS = 2150.0
        MAX_POOR_ECON = 6000.0 # 60 days of lockdown

        if self.task_level == 1:
            # Easy: Did you save the Elites?
            score = 1.0 - (self.D[0] / MAX_ELITE_DEATHS)
            return float(max(0.0, score))
            
        elif self.task_level == 2:
            # Medium: Did you save everyone?
            score = 1.0 - (np.sum(self.D) / MAX_TOTAL_DEATHS)
            return float(max(0.0, score))
            
        elif self.task_level == 3:
            # Hard: Ethical Dilemma (50% health, 50% economy)
            death_score = max(0.0, 1.0 - (np.sum(self.D) / MAX_TOTAL_DEATHS))
            econ_score = max(0.0, 1.0 - (self.economy_hit[2] / MAX_POOR_ECON))
            return float((death_score * 0.5) + (econ_score * 0.5))
            
        return 0.0

    def step(self, action: EpidemicAction) -> Tuple[EpidemicObservation, EpidemicReward, bool, Dict[str, Any]]:
        """OpenEnv Spec: Advances simulation, returning typed models and info dict."""
        act = action.policy_choice
        base_beta = 0.4

        if act == 0: 
            beta_multipliers = np.array([1.0, 1.0, 1.0])
            econ_penalties = np.array([0.0, 0.0, 0.0])
        elif act == 1: 
            beta_multipliers = np.array([0.5, 0.7, 0.9])
            econ_penalties = np.array([0.0, 5.0, 15.0])
        elif act == 2: 
            beta_multipliers = np.array([0.1, 0.4, 0.7]) 
            econ_penalties = np.array([0.0, 20.0, 100.0]) 

        noise = np.random.normal(loc=0.0, scale=0.05, size=self.num_classes)
        beta = np.maximum(0, (base_beta * beta_multipliers) + noise)

        # Track previous state for shaping the step reward
        prev_D = self.D.copy()
        prev_econ = self.economy_hit.copy()

        # Math Updates
        new_E = beta * self.S * self.I / self.N
        new_I = self.sigma * self.E
        resolved_I = self.gamma * self.I
        new_D = resolved_I * self.mortality_rate
        new_R = resolved_I * (1 - self.mortality_rate)

        self.S = np.maximum(0, self.S - new_E)
        self.E = np.maximum(0, self.E + new_E - new_I)
        self.I = np.maximum(0, self.I + new_I - resolved_I)
        self.R = self.R + new_R
        self.D = self.D + new_D

        self.economy_hit += econ_penalties
        self.current_day += 1

        done = self.current_day >= self.max_days

        # OpenEnv Spec: Meaningful partial step reward
        delta_D = self.D - prev_D
        delta_econ = self.economy_hit - prev_econ
        
        step_reward = 0.0
        if self.task_level == 1:
            step_reward = -(delta_D[0] * 100.0)
        elif self.task_level == 2:
            step_reward = -(np.sum(delta_D) * 10.0)
        elif self.task_level == 3:
            step_reward = -(np.sum(delta_D) * 10.0) - (delta_econ[2] * 2.0)
            
        # Oscillation penalty
        if self.prev_action is not None and act != self.prev_action:
            step_reward -= 5.0
        self.prev_action = act

        # Package the return objects
        current_state = self.state()
        reward_obj = EpidemicReward(step_reward=float(step_reward), task_score=self._grade())

        return current_state, reward_obj, done, {}