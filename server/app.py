import uvicorn
from fastapi import FastAPI
from server.env import StratifiedEpidemicEnv, EpidemicAction

app = FastAPI(title="Epidemic AI OpenEnv")

# Create a global environment instance for the validator bot
env = StratifiedEpidemicEnv(task_level=3)

@app.get("/")
def health_check():
    """HF Space Ping Test: Must return 200 OK"""
    return {"status": "healthy", "message": "Epidemic AI OpenEnv is running."}

@app.post("/reset")
def reset_env():
    """OpenEnv Spec: Reset endpoint"""
    obs = env.reset()
    return obs

@app.post("/step")
def step_env(action: EpidemicAction):
    """OpenEnv Spec: Step endpoint"""
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def get_state():
    """OpenEnv Spec: State endpoint"""
    return env.state()

def main():
    """Mandatory entry point for the OpenEnv validator."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()