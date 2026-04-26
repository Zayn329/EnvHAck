---
title: EpidemicAI OpenEnv
emoji: 🦠
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
---

# Epidemic AI - OpenEnv Submission
🏛️ EpidemicAI: Multi-Agent Policy Command Center
Balancing Health, Wealth, and Public Trust through LLM-Driven Governance
Team: Neural Nexus

Lead Developer: Zain Pawle

Built For: Meta OpenEnv Hackathon 2026

🌟 Executive Summary
EpidemicAI is a next-generation epidemic simulation environment that moves beyond static mathematical models. By utilizing Generative Reinforcement Learning from Public Opinion (GRPO) and a Multi-Agent Cabinet Framework, the system simulates the socio-economic friction of a city in crisis. It forces a Large Language Model to navigate the "Impossible Triangle" of governance: minimizing mortality, preventing economic collapse of the vulnerable, and maintaining public trust to prevent social unrest.

🚀 Key Features
🧠 Multi-Agent Cabinet Intelligence
Every policy decision is the result of a simulated internal debate between specialized AI agents:

Chief Medical Officer (CMO): Optimized for infection suppression and hospital capacity.

Chief Economic Advisor (ECON): Optimized for the wealth preservation of the working class.

The Mayor (Decision Engine): Synthesizes conflicting advice based on the current "Social Pulse".

🏙️ Stratified Socio-Economic Environment
Unlike standard SIR models, our environment is socially aware:

Class Dynamics: Simulates three distinct tiers (Elite, Middle, Poor) with unique mortality risks and financial buffers.

Economic Realities: Lockdowns disproportionately damage the Poor tier; failing to provide stimulus can lead to bankruptcy and simulation failure.

🚨 Dynamic Trust & Social Unrest
Public Trust Metric: A live indicator of government legitimacy.

Riot Logic: If Trust drops below 20%, citizens enter "Social Unrest" mode, ignoring lockdown protocols and causing a second infection wave.

⚡ God Mode: Reality Injection
The dashboard includes an NLP-powered "Scenario Injector":

NLP to Math: Users can type "A vaccine was discovered" or "A deadlier variant emerged," and the AI translates these human events into environmental multipliers like beta or mortality_rate.

🛠️ Technical Architecture
Hybrid Inference Engine
The system utilizes a resilient hybrid backend to ensure 100% uptime:

Local Edge: Support for quantized Gemma-2B-GRPO running natively on consumer GPUs (GTX 1060) via 4-bit NF4 quantization.

Cloud Failover: Bulletproof integration with Hugging Face’s InferenceClient for high-speed serverless reasoning.

The Stack
Simulation: Python, NumPy, Pydantic.

UI/UX: Streamlit with Plotly Interactive Charts.

Graph Math: NetworkX city-map topography.

LLM Logic: Hugging Face Inference API.

🔧 Installation & Setup
Clone the repository:

Bash
git clone https://github.com/Zayn329/EnvHAck.git
cd EnvHAck
Set up your environment:

Bash
pip install -r requirements.txt
Set your API token:

Bash
# Windows PowerShell
$env:HF_TOKEN="your_huggingface_token"

# Linux/Mac
export HF_TOKEN="your_huggingface_token"
Run the Command Center:

Bash
python -m streamlit run app.py
