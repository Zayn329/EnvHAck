# EpidemicAI Command Center - Master Context

## 1. Project Overview
This is a hackathon submission for the OpenEnv competition. We are building a "Strategic Resource Management World" (Theme #2). 
The app is a Streamlit dashboard simulating an epidemic across three socio-economic tiers (Elite, Middle, Poor). An AI Mayor (powered by our fine-tuned Gemma-2B GRPO model) makes daily policy decisions (Open, Mild, Lockdown) based on a simulated Multi-Agent debate between a Chief Medical Officer and an Economic Advisor.

## 2. The Final Architecture (Monolithic GPU)
- **Frontend:** Streamlit, Plotly (for charts).
- **Backend Environment:** Custom OpenEnv Python environment (`server/env.py`).
- **AI Brain:** `server/llm_agent.py` loads `zain329/EpidemicAI-Gemma2B-GRPO` directly into VRAM using `bitsandbytes` 4-bit quantization natively on a Hugging Face T4 GPU.
- **CRITICAL RULE:** `server/env.py` MUST remain compliant with the OpenEnv spec. `StratifiedEpidemicEnv` must have `reset()` and `step()` functions returning `(observation, reward, done, info)`.

## 3. File Map
- `app.py`: Streamlit UI, chart rendering, game loop.
- `server/env.py`: The mathematical SIR/Graph model.
- `server/llm_agent.py`: Formats the environment state, queries the LLM for the Cabinet Debate, and extracts the policy choice.

## 4. Current Sprints
- **Sprint 1 (Frontend):** Convert raw text outputs into a beautiful Streamlit Chat UI.
- **Sprint 2 (NLP):** Upgrade the "God Mode" anomaly injector to use zero-shot LLM parsing instead of hardcoded keywords.
- **Sprint 3 (Core Math):** Replace basic array math in the environment with a NetworkX spatial graph to simulate real city neighborhoods.