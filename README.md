---
title: openenv-bioreactor
emoji: "🧪"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Bioreactor Control RL Environment

Task family: `bioreactor-control`  
Environment: `openenv-bioreactor`

This environment models a simplified but process-realistic bioreactor control system, where an agent must protect dissolved oxygen and mixing while maximizing biomass growth and limiting byproduct formation during fermentation.

## Environment

The agent controls a reactor with three direct actuators: stirrer speed, oxygen input, and feed rate. This turns the benchmark from pure setpoint tracking into a production optimization problem with explicit process tradeoffs. The observation exposes the process variables an operator would monitor:

```text
[oxygen_level, mixing_uniformity, nutrient_concentration, biomass_concentration, byproduct_load, feed_rate]
```

Each value is clamped to `[0.0, 1.0]`. The dynamics are lightweight and deterministic under a fixed seed:

```text
oxygen_level += oxygen_transfer - oxygen_demand
mixing_uniformity += stirrer_effect - viscosity_decay - foam_drag
nutrient_concentration += feed_rate - biomass_consumption
biomass_concentration += growth_rate * oxygen_health * mixing_health
byproduct_load += oxygen_stress + overfeed_penalty - mixing_cleanup
```

Hidden state includes `foam_risk` and `shear_damage`. This creates real tradeoffs:

- more oxygen input improves dissolved oxygen but increases foam risk and gas cost
- more stirring improves mixing and oxygen transfer but can damage cells through shear
- more feed can accelerate growth but also raises oxygen demand and overflow-metabolite risk

The simulator stays lightweight: pure Python, short episodes, seeded noise, and no heavy numerical dependencies.

## Action Space

```text
0: increase stirrer speed
1: decrease stirrer speed
2: increase oxygen input
3: decrease oxygen input
4: do nothing
5: increase feed rate
6: decrease feed rate
```

## Tasks And Graders

The repo defines three tasks in `tasks.py`:

```text
startup-stabilization-easy      easy    recover startup and build clean biomass
fed-batch-optimization-medium   medium  survive feed pulses without overflow
oxygen-limited-recovery-hard    hard    finish a high-density run under oxygen stress
```

Every task has a deterministic programmatic grader in `graders.py`. Per-step reward is clamped to `[0.0, 1.0]` and combines:

```text
oxygen tracking score
mixing tracking score
nutrient balance score
biomass production score
byproduct purity score
safety score
growth bonus
actuator and foam penalties
```

The final trajectory score is also clamped to `[0.0, 1.0]` and combines average reward, final biomass achievement, time in the safe operating region, efficiency, survival length, and collapse penalty.

This makes the benchmark less like simple setpoint holding and more like an actual process-control problem: the agent has to grow biomass through changing operating phases without paying for that growth through oxygen starvation, overflow metabolites, foaming, or excessive shear.

Each task also has a terminal objective. A strong controller should finish the batch with:

```text
enough biomass
low byproduct accumulation
residual nutrient in the desired finishing window
oxygen above a task-specific floor
```

That means the agent is judged both on how it runs the trajectory and on how well it lands the end-of-batch operating target.

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
{"task_id": "fed-batch-optimization-medium", "seed": 23}
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
[START] task=startup-stabilization-easy env=openenv-bioreactor model=gpt-4.1-mini
[STEP] step=1 action=2 reward=0.76 done=false error=null
[END] success=true task=startup-stabilization-easy steps=50 score=0.71 rewards=0.76,...
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
