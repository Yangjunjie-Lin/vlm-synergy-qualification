# Isolated Worker Environments

Each frozen family executes in its own virtual environment. The main process communicates
with workers only through the versioned JSONL contract in `workers/contract.schema.json`.

Create the environments from the repository root on Windows:

```powershell
uv venv --python 3.11 envs/qwen/.venv
uv pip sync --python envs/qwen/.venv/Scripts/python.exe envs/qwen/requirements.lock
uv venv --python 3.11 envs/glm/.venv
uv pip sync --python envs/glm/.venv/Scripts/python.exe envs/glm/requirements.lock
uv python install 3.10
uv venv --python 3.10 envs/phi/.venv
uv pip sync --python envs/phi/.venv/Scripts/python.exe envs/phi/requirements.lock
```

The `.venv` directories are runtime state and are ignored by Git. Exact package pins and
environment policy remain tracked. Dependency preflight is not a model-load or scientific
attempt.

