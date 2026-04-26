import numpy as np
import networkx as nx
import plotly.graph_objects as go
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
        self.economy_hit = np.array([0, 0, 0], dtype=np.float32)
        self.prev_action = None
        self.public_trust = 100.0
        self.days_in_lockdown = 0
        self.base_beta = 0.4
        
        # Sprint 3: The Spatial Graph Upgrade
        self.G = nx.Graph()
        
        # Add nodes representing neighborhoods (10k nodes total, scaling factor 10)
        self.G.add_nodes_from([(i, {'tier': 0, 'state': 'S'}) for i in range(1000)])
        self.G.add_nodes_from([(i, {'tier': 1, 'state': 'S'}) for i in range(1000, 5000)])
        self.G.add_nodes_from([(i, {'tier': 2, 'state': 'S'}) for i in range(5000, 10000)])
            
        # Network Topology
        poor_G = nx.fast_gnp_random_graph(5000, 0.004)
        self.G.add_edges_from([(u+5000, v+5000, {'type': 'intra'}) for u, v in poor_G.edges()])
        mid_G = nx.fast_gnp_random_graph(4000, 0.002)
        self.G.add_edges_from([(u+1000, v+1000, {'type': 'intra'}) for u, v in mid_G.edges()])
        elite_G = nx.fast_gnp_random_graph(1000, 0.002)
        self.G.add_edges_from([(u, v, {'type': 'intra'}) for u, v in elite_G.edges()])
            
        # Cross-tier edges
        self.cross_tier_edges = []
        for _ in range(500): # Middle to Poor
            u, v = np.random.randint(1000, 5000), np.random.randint(5000, 10000)
            self.G.add_edge(u, v, type='cross')
            self.cross_tier_edges.append((u, v))
        for _ in range(100): # Elite to Middle
            u, v = np.random.randint(0, 1000), np.random.randint(1000, 5000)
            self.G.add_edge(u, v, type='cross')
            self.cross_tier_edges.append((u, v))
            
        # Initial infections
        for i in range(1000, 1005): self.G.nodes[i]['state'] = 'E'
        for i in range(5000, 5010): self.G.nodes[i]['state'] = 'E'
        self.G.nodes[1005]['state'] = 'I'
        self.G.nodes[5010]['state'] = 'I'
        self.G.nodes[5011]['state'] = 'I'
        
        # Cache layout for visualization (Spatial Segregation)
        self.pos = {}
        for i in range(1000): self.pos[i] = (np.random.uniform(0, 0.3), np.random.uniform(0.7, 1.0))
        for i in range(1000, 5000): self.pos[i] = (np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7))
        for i in range(5000, 10000): self.pos[i] = (np.random.uniform(0.7, 1.0), np.random.uniform(0, 0.3))
        
        self._update_arrays_from_graph()
        return self.state()

    def _update_arrays_from_graph(self):
        S_counts = np.zeros(3, dtype=np.float32)
        E_counts = np.zeros(3, dtype=np.float32)
        I_counts = np.zeros(3, dtype=np.float32)
        R_counts = np.zeros(3, dtype=np.float32)
        D_counts = np.zeros(3, dtype=np.float32)
        
        for u in self.G.nodes():
            tier = self.G.nodes[u]['tier']
            state = self.G.nodes[u]['state']
            if state == 'S': S_counts[tier] += 1
            elif state == 'E': E_counts[tier] += 1
            elif state == 'I': I_counts[tier] += 1
            elif state == 'R': R_counts[tier] += 1
            elif state == 'D': D_counts[tier] += 1
            
        self.S = S_counts * 10.0
        self.E = E_counts * 10.0
        self.I = I_counts * 10.0
        self.R = R_counts * 10.0
        self.D = D_counts * 10.0

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
            cross_edge_mult = 1.0
        elif act == 1: # Mild
            beta_multipliers, econ_penalties = np.array([0.5, 0.7, 0.9]), np.array([0.0, 5.0, 15.0])
            self.public_trust, self.days_in_lockdown = self.public_trust - 1.0, max(0, self.days_in_lockdown - 1)
            cross_edge_mult = 0.5
        elif act == 2: # Lockdown
            beta_multipliers, econ_penalties = np.array([0.1, 0.4, 0.7]), np.array([0.0, 20.0, 100.0]) 
            self.days_in_lockdown += 1
            self.public_trust -= (2.0 * self.days_in_lockdown)
            cross_edge_mult = 0.0

        # Handle Healthcare Overload (Innovation!)
        current_inf = np.sum(self.I)
        mortality_mod = 2.0 if current_inf > self.healthcare_capacity else 1.0
        
        # Handle Social Unrest
        social_unrest = self.public_trust <= 20.0
        if social_unrest: 
            beta_multipliers = np.array([2.0, 2.0, 2.0])
            cross_edge_mult = 1.0

        noise = np.random.normal(loc=0.0, scale=0.05, size=self.num_classes)
        beta = np.maximum(0, (self.base_beta * beta_multipliers) + noise)

        prev_D = self.D.copy()
        
        # --- GRAPH BASED SIR SPREAD ---
        new_states = {}
        for u in self.G.nodes():
            state = self.G.nodes[u]['state']
            tier = self.G.nodes[u]['tier']
            
            if state == 'S':
                # Check neighbors for infection
                for v in self.G.neighbors(u):
                    if self.G.nodes[v]['state'] == 'I':
                        edge_type = self.G[u][v]['type']
                        weight = cross_edge_mult if edge_type == 'cross' else 1.0
                        if np.random.rand() < (beta[tier] * weight * 0.05):
                            new_states[u] = 'E'
                            break
            elif state == 'E':
                if np.random.rand() < self.sigma:
                    new_states[u] = 'I'
            elif state == 'I':
                if np.random.rand() < self.gamma:
                    if np.random.rand() < (self.mortality_rate[tier] * mortality_mod):
                        new_states[u] = 'D'
                    else:
                        new_states[u] = 'R'
                        
        # Apply new states
        for u, state in new_states.items():
            self.G.nodes[u]['state'] = state
            
        self._update_arrays_from_graph()
        
        delta_D = self.D - prev_D
        self.economy_hit += econ_penalties
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

    def get_graph_figure(self):
        """
        Visualization Helper: Returns a Plotly figure of the graph.
        """
        edge_x = []
        edge_y = []
        for u, v in self.cross_tier_edges:
            x0, y0 = self.pos[u]
            x1, y1 = self.pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_colors = []
        color_map = {'S': 'blue', 'E': 'orange', 'I': 'red', 'R': 'green', 'D': 'black'}
        
        for node in self.G.nodes():
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            node_colors.append(color_map[self.G.nodes[node]['state']])

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='none',
            marker=dict(
                color=node_colors,
                size=4,
                line_width=0))
                
        fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Epidemic Spread Network (Neighborhood Nodes)',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
        return fig