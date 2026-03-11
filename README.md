# ⚖ JudgeGPT

Multi-model LLM benchmarking with GPU-accelerated inference and an AI judge that scores every response.

## Features

- **Side-by-side model benchmarking** — TPS, TTFT, and response quality across any Ollama model
- **GPU-accelerated on any platform** — Apple Metal, NVIDIA CUDA, AMD ROCm, or CPU fallback
- **AI Judge** — `qwen2.5:7b` scores responses on Accuracy, Clarity, Depth, Concision, Examples
- **Configurable judge** — edit the system prompt and model at runtime from the UI
- **Live token streaming** — watch models generate tokens simultaneously
- **OpenAI-compat playground** — test any OpenAI-compatible endpoint side by side
- **Leaderboard** — combined speed + quality ranking with human score override

## Requirements

- Docker Desktop with Compose v2
- Ollama installed (`brew install ollama` on Mac, or see [ollama.ai](https://ollama.ai) for others)
- GPU drivers appropriate for your platform (see below)

## Quick Start

```bash
./start.sh up
```

Auto-detects your GPU and starts everything. Opens at **http://localhost:3000**.

## GPU Support

| Platform | GPU | Command |
|---|---|---|
| macOS Apple Silicon | Metal (M1/M2/M3/M4) | `./start.sh up` |
| Linux / Windows WSL2 | NVIDIA CUDA | `./start.sh up` |
| Linux | AMD ROCm | `./start.sh up` |
| Any machine | CPU fallback | `./start.sh up` |

You can also force a specific platform:

```bash
./start.sh up --platform nvidia   # NVIDIA GPU
./start.sh up --platform amd      # AMD ROCm
./start.sh up --platform cpu      # CPU only
./start.sh up --platform mac      # macOS native Ollama
```

### Platform setup

**macOS (Apple Silicon)**
No extra setup. `start.sh` automatically starts Ollama natively on port 11435 for full Metal GPU access. Docker on Mac cannot pass through Metal, so this is the correct architecture.

**NVIDIA (Linux / Windows WSL2)**
Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), then configure Docker:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**AMD ROCm (Linux)**
Install [ROCm drivers](https://rocm.docs.amd.com/en/latest/deploy/linux/installer/install.html), then add your user to the required groups:
```bash
sudo usermod -aG video,render $USER
```
Supported GPUs: RX 5000+, RX 6000+, RX 7000+, Instinct MI series. If your GPU isn't detected automatically, set `HSA_OVERRIDE_GFX_VERSION` in `docker-compose.amd.yml`.

**CPU (any machine)**
Works out of the box on any OS. Expect 2–8 t/s for 7B models depending on your CPU.

## Commands

```bash
./start.sh up              # start (auto-detects GPU)
./start.sh down            # stop all containers
./start.sh logs            # tail orchestrator logs
./start.sh platform        # print detected platform
```

## Usage

1. Select 1–4 models to benchmark
2. Write your prompt (or use the default)
3. Configure runs, token limit, and judge settings (model + system prompt)
4. Hit **Run Benchmark**
5. Results appear in **Metrics** — TPS/TTFT charts + judge radar & scores table
6. Read full responses + human star ratings in **Responses**
7. See the combined leaderboard in **Overall**
8. Export as JSON or CSV

## Architecture

```
~/.ollama  (shared model cache)
    │
    ├── [mac]    native ollama :11435  (Metal GPU — host process)
    ├── [nvidia] judgegpt-ollama :11435  (NVIDIA CUDA — Docker container)
    ├── [amd]    judgegpt-ollama :11435  (AMD ROCm   — Docker container)
    └── [cpu]    judgegpt-ollama :11435  (CPU only   — Docker container)
    │
docker network: judgegpt-net
    ├── judgegpt-orchestrator :8080  (FastAPI — benchmark engine + judge)
    └── judgegpt-frontend     :3000  (React — nginx)
```

All LLM inference — benchmark models and the AI judge — routes through the same Ollama instance for maximum GPU utilization.
