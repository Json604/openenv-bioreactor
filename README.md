# Bioreactor Control RL Environment

Task family: `bioreactor-control`  
Environment: `openenv-bioreactor`

This environment models a simplified bioreactor control system, where an agent learns to maintain optimal oxygen and mixing levels - a core challenge in real-world fermentation and CFD systems.

## Environment

The agent controls a reactor with two hidden actuators: stirrer speed and oxygen input. The observation exposes the process variables an operator would monitor:

```text
[oxygen_level, mixing_uniformity, nutrient_concentration]
```

Each value is clamped to `[0.0, 1.0]`. The dynamics are lightweight and deterministic under a fixed seed:

```text
oxygen_level += 0.1 * oxygen_input - 0.05 * consumption
mixing_uniformity += 0.1 * stirrer_speed - 0.05 * decay
nutrient_concentration -= 0.03 * mixing_uniformity
```

Small seeded noise and task-specific disturbances make the medium and hard tasks less trivial without requiring CFD, training loops, or heavy simulation packages.

## Action Space

```text
0: increase stirrer speed
1: decrease stirrer speed
2: increase oxygen input
3: decrease oxygen input
4: do nothing
```

## Tasks And Graders

The repo defines three tasks in `tasks.py`:

```text
batch-startup-easy        easy    warm-started stabilization
fed-batch-shift-medium    medium  feed and viscosity disturbance recovery
high-density-hard         hard    high oxygen demand with foam/overdrive penalties
```

Every task has a deterministic programmatic grader in `graders.py`. Per-step reward is clamped to `[0.0, 1.0]` and combines:

```text
oxygen tracking score
mixing tracking score
nutrient balance score
stability bonus
actuator and foam penalties
```

The final trajectory score is also clamped to `[0.0, 1.0]` and combines average reward, stable-step fraction, survival length, and collapse penalty.

## OpenEnv API

The server exposes:

```text
GET  /
GET  /health
GET  /tasks
POST /reset
POST /step
GET  /state
```

`/reset` accepts an optional JSON body:

```json
{"task_id": "fed-batch-shift-medium", "seed": 23}
```

`/step` accepts:

```json
{"action": 2}
```

## Local Run

```bash
python3 -m pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
curl -s -X POST http://localhost:8000/step -H "Content-Type: application/json" -d '{"action":2}'
curl -s http://localhost:8000/state
```

## Baseline Inference

The required baseline script is in the repo root:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4.1-mini"
export HF_TOKEN="your-token-if-using-a-HF-router"
export OPENAI_API_KEY="your-openai-key-if-using-openai-directly"
python inference.py
```

It uses the OpenAI client, runs all three tasks, defaults invalid model output to action `4`, catches step errors, and always prints `[END]`.

Expected output shape:

```text
[START] task=batch-startup-easy env=openenv-bioreactor model=gpt-4.1-mini
[STEP] step=1 action=2 reward=0.76 done=false error=null
[END] success=true task=batch-startup-easy steps=50 score=0.71 rewards=0.76,...
```

## Docker

```bash
docker build -t openenv-bioreactor .
docker run --rm -p 8000:7860 openenv-bioreactor
```

Then:

```bash
curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
```

## Validation

```bash
python3 -m compileall bioreactor_env.py models.py graders.py tasks.py inference.py client.py app.py server
openenv validate --verbose
./scripts/validate-submission.sh https://YOUR-SPACE.hf.space .
```

See `SUBMISSION_INSTRUCTIONS.md` for the exact no-reading checklist.
