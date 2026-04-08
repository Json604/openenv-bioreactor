# Exact Submission Instructions

Follow this file only. Replace placeholders in angle brackets.

## 1. Local Smoke Test

```bash
cd /Users/kartikey/Desktop/meta_env
python3 -m pip install -r requirements.txt
python3 -m compileall bioreactor_env.py models.py graders.py tasks.py inference.py client.py app.py server
python3 inference.py
```

If `python3 inference.py` prints OpenAI/auth errors, set credentials and rerun:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4.1-mini"
export OPENAI_API_KEY="<YOUR_OPENAI_KEY>"
python3 inference.py
```

For Hugging Face router-based inference:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="<HF_MODEL_ID>"
export HF_TOKEN="<YOUR_HF_TOKEN>"
python3 inference.py
```

## 2. Run The OpenEnv Server Locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_id":"fed-batch-optimization-medium"}'
curl -s -X POST http://localhost:8000/step -H "Content-Type: application/json" -d '{"action":2}'
curl -s http://localhost:8000/state
curl -s http://localhost:8000/tasks
```

All commands should return JSON and `/reset` must return HTTP 200.

## 3. Docker Test

```bash
docker build -t openenv-bioreactor .
docker run --rm -p 8000:7860 openenv-bioreactor
```

In a second terminal:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
```

Expected output:

```text
200
```

## 4. OpenEnv Validate

```bash
python3 -m pip install openenv-core
openenv validate --verbose
```

If your organizer gave you their validator script, run:

```bash
bash scripts/validate-submission.sh https://<HF_USERNAME>-openenv-bioreactor.hf.space .
```

Use your actual deployed Space URL. Hugging Face may also use a URL like:

```text
https://<HF_USERNAME>-<SPACE_NAME>.hf.space
```

## 5. Create The Hugging Face Space

Create a new Hugging Face Space:

```text
Owner: <HF_USERNAME or TEAM>
Space name: openenv-bioreactor
SDK: Docker
Visibility: Public unless the competition explicitly says private
Tag: openenv
```

The repo you push must be the contents of:

```text
/Users/kartikey/Desktop/meta_env
```

Do not push only a subfolder. The Space repo root must contain:

```text
Dockerfile
openenv.yaml
inference.py
models.py
bioreactor_env.py
graders.py
tasks.py
server/app.py
requirements.txt
README.md
SUBMISSION_INSTRUCTIONS.md
```

## 6. Commit And Push

If this folder is not a git repo yet:

```bash
cd /Users/kartikey/Desktop/meta_env
git init
git add .
git commit -m "Add OpenEnv bioreactor control environment"
git branch -M main
git remote add origin https://huggingface.co/spaces/<HF_USERNAME>/openenv-bioreactor
git push -u origin main
```

If the Hugging Face Space repo already exists locally as a git repo:

```bash
cd /Users/kartikey/Desktop/meta_env
git status
git add .
git commit -m "Add OpenEnv bioreactor control environment"
git push
```

## 7. Add Space Secrets

In the Hugging Face Space settings, add these environment variables:

```text
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4.1-mini
OPENAI_API_KEY=<YOUR_OPENAI_KEY>
HF_TOKEN=<YOUR_HF_TOKEN if using HF router, otherwise omit>
```

If using the Hugging Face router instead of OpenAI direct:

```text
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=<HF_MODEL_ID>
HF_TOKEN=<YOUR_HF_TOKEN>
```

## 8. Final Pre-Submission Check

After the Space builds, run:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST https://<HF_USERNAME>-openenv-bioreactor.hf.space/reset -H "Content-Type: application/json" -d '{}'
bash scripts/validate-submission.sh https://<HF_USERNAME>-openenv-bioreactor.hf.space .
```

Expected:

```text
200
All 3/3 checks passed
```

## 9. What To Submit

Submit the Hugging Face Space URL:

```text
https://huggingface.co/spaces/<HF_USERNAME>/openenv-bioreactor
```

If the submission form asks for a live endpoint too, submit:

```text
https://<HF_USERNAME>-openenv-bioreactor.hf.space
```
