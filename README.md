# ⚖ JudgeGPT

Multi-model LLM benchmarking with GPU-accelerated inference and an AI judge that scores every response.

## Features

- **Side-by-side model benchmarking** — TPS, TTFT, and response quality across any Ollama model
- **GPU-native inference** — runs directly on Apple Metal (M1/M2/M3), not inside Docker
- **AI Judge** — `qwen2.5:7b` scores responses on Accuracy, Clarity, Depth, Concision, Examples
- **Configurable judge** — edit the system prompt and model at runtime from the UI
- **Live token streaming** — watch models generate tokens simultaneously
- **OpenAI-compat playground** — test any OpenAI-compatible endpoint side by side
- **Leaderboard** — combined speed + quality ranking with human override

## Requirements

- Docker Desktop (Compose v2)
- macOS Apple Silicon (M1/M2/M3)
- Ollama installed natively (`brew install ollama`)

## Quick Start

```bash
./start.sh up
```

Opens at **http://localhost:3000**

The script automatically starts native Ollama on port 11435 (Metal GPU) then brings up the Docker containers. The judge model (`qwen2.5:7b`) pulls automatically on first run.

## Commands

```bash
./start.sh up      # start everything (native Ollama + Docker containers)
./start.sh down    # stop Docker containers
./start.sh logs    # tail orchestrator logs
```

## Usage

1. Select 1–4 models to benchmark
2. Write your prompt (or use the default)
3. Configure runs, token limit, and judge settings
4. Hit **Run Benchmark**
5. Watch tokens stream in the header status pills
6. Results appear in **Metrics** — TPS, TTFT charts + judge radar & scores table
7. Read full responses + human star ratings in **Responses**
8. See the combined leaderboard in **Overall**
9. Export as JSON or CSV

## Architecture

```
~/.ollama  (shared model cache — host + containers)
    │
    ├── native ollama  :11435  (Metal GPU — benchmark models + judge)
    │
docker network: judgegpt-net
    ├── judgegpt-orchestrator  :8080  (FastAPI — benchmark engine, judge coordinator)
    └── judgegpt-frontend      :3000  (React — nginx)
```

All LLM inference (both benchmark models and the judge) routes through the native Ollama instance for full Metal GPU acceleration.
