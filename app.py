import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time

# Import your custom backend
from server.env import StratifiedEpidemicEnv, EpidemicAction
from server.llm_agent import MultiAgentPolicySystem

# --- 1. PAGE CONFIG & PREMIUM UX ---
st.set_page_config(page_title="EpidemicAI Command Center", page_icon="🌍", layout="wide")

# --- 2. SESSION STATE INITIALIZATION ---
if "env" not in st.session_state:
    st.session_state.env = StratifiedEpidemicEnv()
if "history" not in st.session_state:
    st.session_state.history = []
if "scenario_text" not in st.session_state:
    st.session_state.scenario_text = ""

env = st.session_state.env
# Always recreate the agent so hot-reloads work instantly (no state is stored in the agent anyway)
agent = MultiAgentPolicySystem()
history = st.session_state.history

# --- 3. THE SIDEBAR (CONTROLS & GOD MODE) ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Red_Cross_icon.svg/1024px-Red_Cross_icon.svg.png", width=50) # Optional logo
st.sidebar.title("Simulation Controls")

# Core Action
if st.sidebar.button("▶️ Run Next Day", type="primary", use_container_width=True, disabled=env.current_day >= env.max_days):
    with st.spinner("AI Cabinet is debating..."):
        obs = env.state()
        prev_act = env.prev_action if env.prev_action is not None else 0
        
        # 1. AI decides
        action_dict = agent.get_action(obs, history, prev_act)
        action_model = EpidemicAction(
            reasoning=action_dict.get("reasoning", "Fallback logic"),
            policy_choice=action_dict.get("policy_choice", 1)
        )
        
        # 2. Environment steps
        next_obs, reward, done, info = env.step(action_model)
        
        # 3. Save to history for charts
        total_I = sum(env.I)
        delta_I = total_I - history[-1]["total_infections"] if len(history) > 0 else total_I
        
        history.append({
            "day": env.current_day,
            "total_infections": total_I,
            "delta_infections": max(0, delta_I),
            "poor_economy": env.economy_hit[2],
            "public_trust": env.public_trust,
            "reasoning": action_model.reasoning,
            "policy": action_model.policy_choice
        })

st.sidebar.divider()

# God Mode Features
st.sidebar.markdown("### ⚡ God Mode: Reality Injection")
st.sidebar.caption("Test the AI with real-world shocks.")

# 1-Click Demo Buttons
col1, col2 = st.sidebar.columns(2)
if col1.button("🦠 Slum Variant"): 
    st.session_state.scenario_text = "A deadlier, highly contagious variant mutates in the poor tier."
if col2.button("📉 Market Crash"): 
    st.session_state.scenario_text = "A sudden global market crash wipes out the poor tier's remaining wealth."

user_anomaly = st.sidebar.text_input("Describe an event:", value=st.session_state.scenario_text)

if st.sidebar.button("Inject Event", use_container_width=True):
    if user_anomaly:
        with st.spinner("AI Translating NLP to Math..."):
            effect = agent.interpret_anomaly(user_anomaly)
            env.apply_dynamic_anomaly(effect['target'], effect['multiplier'])
            st.toast(f"System Shocked! {effect['target'].upper()} altered by {effect['multiplier']}x", icon="🚨")

# --- 4. MAIN DASHBOARD (METRICS) ---
st.title("🏛️ EpidemicAI Command Center")

m1, m2, m3, m4 = st.columns(4)
current_trust = env.public_trust

delta_inf = None
delta_econ = None
if len(history) > 0:
    delta_inf = int(history[-1]["delta_infections"])
    if len(history) >= 2:
        delta_econ = int(history[-1]["poor_economy"] - history[-2]["poor_economy"])
    else:
        delta_econ = int(history[-1]["poor_economy"])

m1.metric("Current Day", f"{env.current_day} / {env.max_days}")
m2.metric("Total Active Infections", f"{int(sum(env.I)):,}", 
          delta=f"{delta_inf:,} new infections" if delta_inf is not None else None, 
          delta_color="inverse")
m3.metric("Poor Tier Wealth (Damage)", f"${int(env.economy_hit[2]):,}", 
          delta=f"${delta_econ:,} damage" if delta_econ is not None else None, 
          delta_color="inverse")
m4.metric("Public Trust", f"{current_trust:.1f}%", 
          delta="RIOT WARNING" if current_trust <= 20 else "Stable", 
          delta_color="inverse" if current_trust <= 20 else "normal")

# --- 5. THE SOCIAL PULSE ---
if current_trust > 70:
    st.success("📱 @Citizen123: The Mayor is handling this perfectly. We feel safe! #FlattenTheCurve")
elif current_trust > 40:
    st.info("📱 @CityWorker: It's tough, but we are holding on. Hoping the economy opens soon.")
elif current_trust > 20:
    st.warning("📱 @LocalOwner: I can't pay rent. If they don't open up, we lose everything! #OpenUp")
else:
    st.error("📱 @AngryMob: WE ARE IGNORING THE LOCKDOWN. NO MORE RULES! 🛑🔥 #Riot")

st.divider()

# --- 6. INTERACTIVE DELTA CHARTS ---
if len(history) > 0:
    c1, c2 = st.columns(2)
    
    # Chart 1: Daily New Infections (Delta)
    days = [h["day"] for h in history]
    deltas = [h["delta_infections"] for h in history]
    fig_inf = px.bar(x=days, y=deltas, labels={"x": "Day", "y": "New Infections"}, title="Daily New Infections (Spikes)")
    fig_inf.update_traces(marker_color='crimson')
    c1.plotly_chart(fig_inf, use_container_width=True)
    
    # Chart 2: Poor Economy
    econ = [h["poor_economy"] for h in history]
    fig_econ = px.line(x=days, y=econ, labels={"x": "Day", "y": "Economic Damage ($)"}, title="Poor Economy Impact vs Day")
    fig_econ.update_traces(line_color='orange')
    fig_econ.add_hline(y=3000, line_dash="dash", line_color="red", annotation_text="Bankruptcy")
    c2.plotly_chart(fig_econ, use_container_width=True)

    # --- 6.5. SPATIAL GRAPH UPGRADE ---
    st.subheader("🌐 City Spread Topology")
    st.caption("Real-time node transmission. Red = Infected, Blue = Susceptible, Black = Deaths.")
    fig_spatial = env.get_graph_figure()
    st.plotly_chart(fig_spatial, use_container_width=True)

    # --- 7. THE AI CABINET DEBATE ---
    st.subheader("🧠 Live AI Cabinet Debate")
    latest_reasoning = history[-1]["reasoning"]
    policy_choice = history[-1]["policy"]
    
    # Parse reasoning string
    cmo_text = ""
    econ_text = ""
    
    if "|" in latest_reasoning:
        parts = latest_reasoning.split("|")
        for part in parts:
            if "CMO:" in part:
                cmo_text = part.replace("CMO:", "").strip()
            elif "ECON:" in part:
                econ_text = part.replace("ECON:", "").strip()
    else:
        # Fallback if no "|"
        if "CMO:" in latest_reasoning and "ECON:" in latest_reasoning:
            cmo_part = latest_reasoning.split("ECON:")[0]
            econ_part = latest_reasoning.split("ECON:")[1]
            cmo_text = cmo_part.replace("CMO:", "").strip()
            econ_text = econ_part.strip()
        else:
            cmo_text = latest_reasoning
            econ_text = ""

    # Render with Streamlit Chat UI
    if cmo_text:
        with st.chat_message("CMO", avatar="🩺"): 
            st.write(cmo_text)
            
    if econ_text:
        with st.chat_message("ECON", avatar="💰"): 
            st.write(econ_text)
            
    with st.chat_message("MAYOR", avatar="🏛️"): 
        st.write(f"The Mayor has decided to implement Policy Level {policy_choice}.")
else:
    st.info("👈 Click 'Run Next Day' in the sidebar to begin the simulation.")