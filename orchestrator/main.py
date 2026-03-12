"""
JudgeGPT Orchestrator
=====================
Dynamically spawns Ollama containers per model, runs benchmarks,
measures real TPS/TTFT, and coordinates LLM judge scoring.

M1 Mac: Ollama containers use Metal GPU automatically via the
        ollama/ollama image — no special GPU flags needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import aiohttp
import docker
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("orchestrator")

# ── Config ─────────────────────────────────────────────────────────────────────

JUDGE_OLLAMA_URL   = os.getenv("JUDGE_OLLAMA_URL",   "http://judge-ollama:11434")
JUDGE_MODEL        = os.getenv("JUDGE_MODEL",        "qwen2.5:7b")

def _effective_judge_url() -> str:
    """Native Ollama (GPU) when available, else the dedicated judge container."""
    return NATIVE_OLLAMA_HOST if NATIVE_OLLAMA_HOST else JUDGE_OLLAMA_URL
DOCKER_NETWORK     = os.getenv("DOCKER_NETWORK",     "judgegpt-net")
OLLAMA_IMAGE       = os.getenv("OLLAMA_IMAGE",       "ollama/ollama:latest")
OLLAMA_BASE_PORT   = int(os.getenv("OLLAMA_BASE_PORT", "11501"))
OLLAMA_MODELS_PATH = os.getenv("OLLAMA_MODELS_PATH", os.path.expanduser("~/.ollama"))
CONTAINER_PREFIX   = "judgegpt-model-"

# When set, skip Docker containers entirely and route all model inference
# to a native Ollama instance running on the host (e.g. with Metal GPU on Mac).
# Example: NATIVE_OLLAMA_HOST=http://host.docker.internal:11435
NATIVE_OLLAMA_HOST = os.getenv("NATIVE_OLLAMA_HOST", "").rstrip("/")

# ── Docker client ──────────────────────────────────────────────────────────────

try:
    docker_client = docker.from_env()
    docker_client.ping()
    log.info("Docker connected")
except Exception as e:
    log.error(f"Docker not available: {e}")
    docker_client = None

# ── In-memory state ────────────────────────────────────────────────────────────

# model_name → {"container_id", "port", "status", "spawned_at"}
_active_containers: dict[str, dict] = {}
_port_pool: set[int] = set(range(OLLAMA_BASE_PORT, OLLAMA_BASE_PORT + 20))
_benchmark_results: dict[str, dict] = {}
_judge_scores: dict[str, dict] = {}
_human_scores: dict[str, float] = {}
_run_events: list[dict] = []
_benchmark_history: list[dict] = []   # capped at 100 entries, newest at end

# ── Mutable judge config ────────────────────────────────────────────────────────
# Default system prompt — overridable at runtime via POST /judge/config
JUDGE_SYSTEM = """You are an expert LLM response evaluator. Score the response on these 5 criteria (1-5 each).

CALIBRATION:
Strong response (5s): accurate, clear, deep, concise, uses good examples
Weak response (1s): wrong facts, confusing, shallow, padded, no examples

CRITERIA:
- accuracy: Factual correctness (1=wrong, 5=fully correct)
- clarity: Ease of understanding (1=confusing, 5=crystal clear)
- depth: Topic coverage (1=superficial, 5=comprehensive)
- concision: Signal-to-noise (1=very padded, 5=tight)
- examples: Use of analogies/examples (1=none, 5=excellent)

Think through each criterion. Then output ONLY this JSON (no other text):
{"accuracy":N,"clarity":N,"depth":N,"concision":N,"examples":N,"reasoning":"2-3 sentence summary"}"""

_judge_system_prompt: str = JUDGE_SYSTEM
_judge_model_name: str = JUDGE_MODEL

# ── Preset model catalog ───────────────────────────────────────────────────────

PRESET_MODELS = {
    "mistral:7b":      {"color": "#00e5ff", "size_gb": 4.1, "desc": "Mistral 7B"},
    "llama3:8b":       {"color": "#b388ff", "size_gb": 4.7, "desc": "Llama 3 8B"},
    "gemma2:9b":       {"color": "#ffd740", "size_gb": 5.5, "desc": "Gemma 2 9B"},
    "deepseek-r1:7b":  {"color": "#ff4081", "size_gb": 4.7, "desc": "DeepSeek R1 7B"},
    "phi3:mini":       {"color": "#69ff47", "size_gb": 2.2, "desc": "Phi-3 Mini"},
    "qwen2:7b":        {"color": "#ff6e40", "size_gb": 4.4, "desc": "Qwen2 7B"},
    "llama3.2:3b":     {"color": "#80cbc4", "size_gb": 2.0, "desc": "Llama 3.2 3B"},
    "mistral:latest":  {"color": "#4fc3f7", "size_gb": 4.1, "desc": "Mistral Latest"},
}

def model_color(model_name: str) -> str:
    return PRESET_MODELS.get(model_name, {}).get("color", "#888888")

# ── Container management ───────────────────────────────────────────────────────

def _alloc_port() -> int:
    if not _port_pool:
        raise RuntimeError("No free ports in pool")
    return _port_pool.pop()

def _free_port(port: int):
    _port_pool.add(port)

def _container_name(model: str) -> str:
    safe = model.replace(":", "-").replace("/", "-")
    return f"{CONTAINER_PREFIX}{safe}"

async def spawn_container(model_name: str) -> dict:
    """Start an Ollama instance for the given model.

    Native mode  (NATIVE_OLLAMA_HOST set): pulls the model into the host
    Ollama process (Metal GPU on Mac) and returns the shared URL.
    Docker mode (default): spawns a per-model Ollama container.
    """
    if NATIVE_OLLAMA_HOST:
        return await _spawn_native(model_name)
    return await _spawn_docker(model_name)


async def _spawn_native(model_name: str) -> dict:
    """Use the host-native Ollama (GPU) instead of a Docker container."""
    if model_name in _active_containers:
        info = _active_containers[model_name]
        if info["status"] == "running":
            log.info(f"Native model {model_name} already ready at {NATIVE_OLLAMA_HOST}")
            return info

    log.info(f"Native mode: pulling/verifying {model_name} at {NATIVE_OLLAMA_HOST}")
    await wait_for_ollama(NATIVE_OLLAMA_HOST, model_name, timeout=600)

    info = {
        "container_id": None,
        "container_name": None,
        "port": None,
        "model": model_name,
        "status": "running",
        "spawned_at": time.time(),
        "url": NATIVE_OLLAMA_HOST,
        "internal_url": NATIVE_OLLAMA_HOST,
        "native": True,
    }
    _active_containers[model_name] = info
    log.info(f"Native {model_name} ready at {NATIVE_OLLAMA_HOST}")
    return info


async def _spawn_docker(model_name: str) -> dict:
    """Spawn a per-model Ollama Docker container (CPU-only on Mac)."""
    if not docker_client:
        raise RuntimeError("Docker not available")

    if model_name in _active_containers:
        info = _active_containers[model_name]
        if info["status"] == "running":
            log.info(f"Container for {model_name} already running on :{info['port']}")
            return info

    port = _alloc_port()
    cname = _container_name(model_name)

    try:
        old = docker_client.containers.get(cname)
        old.remove(force=True)
        log.info(f"Removed stale container {cname}")
    except docker.errors.NotFound:
        pass

    log.info(f"Spawning {model_name} on :{port}")
    container = docker_client.containers.run(
        OLLAMA_IMAGE,
        name=cname,
        detach=True,
        ports={"11434/tcp": port},
        volumes={OLLAMA_MODELS_PATH: {"bind": "/root/.ollama", "mode": "rw"}},
        environment={
            "OLLAMA_HOST": "0.0.0.0",
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_MAX_LOADED_MODELS": "1",
        },
        network=DOCKER_NETWORK,
        labels={"judgegpt": "model-instance", "judgegpt-model": model_name},
        remove=False,
    )

    info = {
        "container_id": container.id,
        "container_name": cname,
        "port": port,
        "model": model_name,
        "status": "starting",
        "spawned_at": time.time(),
        "url": f"http://localhost:{port}",
        "internal_url": f"http://{cname}:11434",
    }
    _active_containers[model_name] = info

    await wait_for_ollama(f"http://host.docker.internal:{port}", model_name, timeout=120)
    info["status"] = "running"
    log.info(f"{model_name} ready on :{port}")
    return info


async def wait_for_ollama(url: str, model: str, timeout: int = 60):
    """Poll /api/tags until Ollama is up."""
    deadline = time.time() + timeout
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(f"{url}/api/tags", timeout=aiohttp.ClientTimeout(total=3)) as r:
                    if r.status == 200:
                        # Also ensure model is loaded
                        await pull_model_if_needed(url, model)
                        return
            except Exception:
                pass
            await asyncio.sleep(1)
    raise TimeoutError(f"Ollama for {model} did not start within {timeout}s")


async def pull_model_if_needed(base_url: str, model: str):
    """Pull the model if it's not already available in this Ollama instance."""
    async with aiohttp.ClientSession() as session:
        # Check if model is in the local list
        try:
            async with session.get(f"{base_url}/api/tags") as r:
                tags = await r.json()
                local_models = [m["name"] for m in tags.get("models", [])]
                # Normalize: ollama may list "mistral:7b" or "mistral:latest"
                model_base = model.split(":")[0]
                if any(model_base in lm for lm in local_models):
                    log.info(f"Model {model} already available")
                    return
        except Exception:
            pass

    log.info(f"Pulling {model}…")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/api/pull",
            json={"name": model, "stream": False},
            timeout=aiohttp.ClientTimeout(total=600),
        ) as r:
            result = await r.json()
            log.info(f"Pull result for {model}: {result.get('status')}")


async def kill_container(model_name: str):
    """Stop and remove the container for a model.
    In native mode the host Ollama keeps running — we just clear the cache entry."""
    info = _active_containers.pop(model_name, None)
    if not info:
        return
    if info.get("native"):
        log.info(f"Native mode: releasing {model_name} from cache (Ollama keeps running)")
        return
    _free_port(info["port"])
    try:
        container = docker_client.containers.get(info["container_id"])
        container.remove(force=True)
        log.info(f"Killed container for {model_name}")
    except Exception as e:
        log.warning(f"Could not kill container for {model_name}: {e}")


async def kill_all_model_containers():
    """Kill all dynamically spawned model containers."""
    models = list(_active_containers.keys())
    for m in models:
        await kill_container(m)

# ── Benchmark runner ───────────────────────────────────────────────────────────

async def run_single(
    model_name: str,
    container_info: dict,
    prompt: str,
    n_tokens: int,
) -> dict:
    """Run one inference pass, return TPS + TTFT + response text."""
    url = container_info["internal_url"]
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": n_tokens, "temperature": 0.7},
    }

    t_start = time.perf_counter()
    t_first = None
    tokens = 0
    response_text = ""

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{url}/api/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                # Try to extract the Ollama error message from JSON
                try:
                    detail = json.loads(body).get("error", body[:200])
                except Exception:
                    detail = body[:200]
                raise RuntimeError(f"Ollama returned {resp.status}: {detail}")
            async for line in resp.content:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if t_first is None:
                    t_first = time.perf_counter()
                token_text = chunk.get("response", "")
                response_text += token_text
                if token_text:
                    tokens += 1
                if chunk.get("done"):
                    # Ollama gives us eval_count and eval_duration
                    eval_count    = chunk.get("eval_count", tokens)
                    eval_duration = chunk.get("eval_duration", 1) / 1e9  # ns → s
                    prompt_duration = chunk.get("prompt_eval_duration", 0) / 1e9
                    break

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    ttft_ms = round((t_first - t_start) * 1000, 1) if t_first else 0
    tps = round(eval_count / eval_duration, 1) if eval_duration > 0 else 0

    return {
        "tps": tps,
        "ttft_ms": ttft_ms,
        "tokens": eval_count,
        "duration_ms": round(elapsed * 1000, 1),
        "response": response_text.strip(),
    }


async def benchmark_model(
    model_name: str,
    prompt: str,
    n_tokens: int,
    n_runs: int,
    event_queue: asyncio.Queue,
) -> dict:
    """Spawn container → warm up → run N times → aggregate."""
    await event_queue.put({"event": "spawn_start", "model": model_name})

    try:
        container_info = await spawn_container(model_name)
    except Exception as e:
        await event_queue.put({"event": "error", "model": model_name, "error": str(e)})
        return None

    await event_queue.put({"event": "spawn_done", "model": model_name, "port": container_info["port"]})

    # Warm-up pass (not counted)
    try:
        await event_queue.put({"event": "warmup", "model": model_name})
        await run_single(model_name, container_info, "Hello", 10)
    except Exception as e:
        log.warning(f"Warmup failed for {model_name}: {e}")

    runs = []
    for i in range(n_runs):
        await event_queue.put({"event": "run_start", "model": model_name, "run": i + 1, "total": n_runs})
        last_err: Exception | None = None
        for attempt in range(2):  # 1 retry on transient errors
            try:
                run = await run_single(model_name, container_info, prompt, n_tokens)
                run["run"] = i + 1
                runs.append(run)
                await event_queue.put({"event": "run_done", "model": model_name, "run": i + 1,
                                       "tps": run["tps"], "ttft_ms": run["ttft_ms"]})
                last_err = None
                break
            except Exception as e:
                last_err = e
                log.warning(f"Run {i+1} attempt {attempt+1} failed for {model_name}: {e}")
                if attempt == 0:
                    await asyncio.sleep(2)  # brief pause before retry
        if last_err is not None:
            log.error(f"Run {i+1} failed for {model_name} after retries: {last_err}")
            await event_queue.put({"event": "run_error", "model": model_name, "run": i + 1, "error": str(last_err)})

    if not runs:
        return None

    tps_vals  = [r["tps"]     for r in runs]
    ttft_vals = [r["ttft_ms"] for r in runs]
    result = {
        "model":     model_name,
        "color":     model_color(model_name),
        "runs":      runs,
        "tps_mean":  round(sum(tps_vals) / len(tps_vals), 1),
        "tps_max":   round(max(tps_vals), 1),
        "tps_min":   round(min(tps_vals), 1),
        "ttft_mean": round(sum(ttft_vals) / len(ttft_vals), 1),
        "ttft_min":  round(min(ttft_vals), 1),
        "response":  runs[-1]["response"],
        "prompt":    prompt,
        "timestamp": time.time(),
    }
    _benchmark_results[model_name] = result
    await event_queue.put({"event": "model_complete", "model": model_name, "result": result})
    return result

# ── LLM Judge ─────────────────────────────────────────────────────────────────

async def judge_response(model_name: str, question: str, response: str) -> dict:
    """Score a response using the (GPU-native) Ollama judge instance."""
    word_count = len(response.split())
    user_content = (
        f"Question asked: {question}\n\n"
        f"Response to evaluate ({word_count} words):\n{response}\n\n"
        f"Return ONLY the JSON scores."
    )
    judge_url = _effective_judge_url()

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{judge_url}/api/chat",
            json={
                "model": _judge_model_name,
                "messages": [
                    {"role": "system", "content": _judge_system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 400},
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Judge Ollama returned {resp.status}")
            data = await resp.json()
            raw = data.get("message", {}).get("content", "{}")

    try:
        parsed = json.loads(raw)
        criteria = ["accuracy", "clarity", "depth", "concision", "examples"]
        for c in criteria:
            parsed[c] = max(1, min(5, int(parsed.get(c, 3))))
        parsed["word_count"]  = word_count
        parsed["judge_model"] = _judge_model_name
        parsed["mock"]        = False
        return parsed
    except (json.JSONDecodeError, ValueError) as e:
        log.error(f"Judge parse error: {e} — raw: {raw[:200]}")
        raise RuntimeError(f"Judge returned unparseable response: {raw[:100]}")


def compute_leaderboard() -> list:
    if not _benchmark_results:
        return []
    all_tps  = [r["tps_mean"]  for r in _benchmark_results.values()]
    all_ttft = [r["ttft_mean"] for r in _benchmark_results.values()]
    max_tps  = max(all_tps)  if all_tps  else 1
    min_ttft = min(all_ttft) if all_ttft else 1
    criteria = ["accuracy", "clarity", "depth", "concision", "examples"]
    entries = []
    for name, r in _benchmark_results.items():
        tps_score  = (r["tps_mean"]  / max_tps)  * 100
        ttft_score = (min_ttft / r["ttft_mean"]) * 100
        j = _judge_scores.get(name)
        h = _human_scores.get(name)
        judge_norm = (sum(j[c] for c in criteria) / (len(criteria) * 5)) * 100 if j else None
        quality_sources = [s for s in [judge_norm, (h / 5) * 100 if h else None] if s]
        quality = sum(quality_sources) / len(quality_sources) if quality_sources else None
        combined = round(tps_score * 0.35 + ttft_score * 0.15 + quality * 0.50, 1) if quality else None
        entries.append({
            "model": name, "color": r["color"],
            "tps_mean": r["tps_mean"], "ttft_mean": r["ttft_mean"],
            "tps_score": round(tps_score, 1), "ttft_score": round(ttft_score, 1),
            "judge_score": round(judge_norm, 1) if judge_norm else None,
            "human_score": round((h / 5) * 100, 1) if h else None,
            "quality_score": round(quality, 1) if quality else None,
            "combined_score": combined,
        })
    entries.sort(key=lambda x: (x["combined_score"] or 0), reverse=True)
    for i, e in enumerate(entries):
        e["rank"] = i + 1
    return entries

# ── App lifecycle ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("JudgeGPT orchestrator starting…")
    # Pre-pull judge model in background
    asyncio.create_task(ensure_judge_model())
    yield
    log.info("Shutting down — cleaning up model containers…")
    await kill_all_model_containers()


async def ensure_judge_model():
    """Pull judge model into the Ollama instance on startup."""
    await asyncio.sleep(3)
    url = _effective_judge_url()
    try:
        await pull_model_if_needed(url, _judge_model_name)
        log.info(f"Judge model {_judge_model_name} ready at {url}")
    except Exception as e:
        log.warning(f"Could not pre-pull judge model: {e}")


app = FastAPI(title="JudgeGPT Orchestrator", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Pydantic models ────────────────────────────────────────────────────────────

class BenchmarkRequest(BaseModel):
    model_names: list[str]
    prompt: str = "Explain Einstein's theory of relativity in simple terms."
    n_tokens: int = Field(default=256, ge=32, le=2048)
    n_runs: int = Field(default=3, ge=1, le=10)
    auto_judge: bool = True
    keep_alive: bool = False  # if False, kill containers after benchmark

class HumanScoreRequest(BaseModel):
    model_name: str
    score: float = Field(ge=1, le=5)

class CustomModelRequest(BaseModel):
    model_name: str  # any valid Ollama model tag e.g. "llama3.2:3b"

class PullModelRequest(BaseModel):
    model_name: str

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "JudgeGPT Orchestrator", "version": "1.0.0", "status": "ok"}

@app.get("/models/catalog")
def model_catalog():
    return [
        {
            "name": k,
            "color": v["color"],
            "size_gb": v["size_gb"],
            "desc": v["desc"],
            "running": k in _active_containers and _active_containers[k]["status"] == "running",
        }
        for k, v in PRESET_MODELS.items()
    ]

@app.get("/models/running")
def running_models():
    return list(_active_containers.values())

@app.post("/models/spawn/{model_name}")
async def spawn_model(model_name: str):
    """Manually spawn a container for a model."""
    info = await spawn_container(model_name)
    return info

@app.delete("/models/{model_name}")
async def stop_model(model_name: str):
    """Kill a model container."""
    await kill_container(model_name)
    return {"killed": model_name}

@app.post("/benchmark/run")
async def run_benchmark_endpoint(req: BenchmarkRequest):
    """Run benchmark across all selected models, stream SSE progress."""
    queue: asyncio.Queue = asyncio.Queue()
    results = []

    async def stream() -> AsyncGenerator[str, None]:
        # Run all models concurrently
        tasks = [
            benchmark_model(name, req.prompt, req.n_tokens, req.n_runs, queue)
            for name in req.model_names
        ]

        async def run_tasks():
            done = await asyncio.gather(*tasks, return_exceptions=True)
            await queue.put(None)  # sentinel
            return done

        task_runner = asyncio.create_task(run_tasks())

        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

        task_results = await task_runner
        for r in task_results:
            if isinstance(r, dict) and r:
                results.append(r)

        # Auto-judge
        if req.auto_judge and results:
            yield f"data: {json.dumps({'event': 'judging_start'})}\n\n"
            for r in results:
                try:
                    score = await judge_response(r["model"], req.prompt, r["response"])
                    _judge_scores[r["model"]] = score
                    yield f"data: {json.dumps({'event': 'judge_done', 'model': r['model'], 'score': score})}\n\n"
                except Exception as e:
                    log.error(f"Judge failed for {r['model']}: {e}")
                    yield f"data: {json.dumps({'event': 'judge_error', 'model': r['model'], 'error': str(e)})}\n\n"

        # Kill containers unless keep_alive
        if not req.keep_alive:
            for name in req.model_names:
                asyncio.create_task(kill_container(name))

        lb = compute_leaderboard()

        # Persist to history
        if results:
            model_names_in_run = [r["model"] for r in results]
            _benchmark_history.append({
                "id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "prompt": req.prompt,
                "n_tokens": req.n_tokens,
                "n_runs": req.n_runs,
                "models": model_names_in_run,
                "results": {r["model"]: r for r in results},
                "judge_scores": {m: s for m, s in _judge_scores.items() if m in model_names_in_run},
                "leaderboard": lb,
            })
            if len(_benchmark_history) > 100:
                _benchmark_history.pop(0)

        yield f"data: {json.dumps({'event': 'done', 'leaderboard': lb})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/results")
def get_results():
    return list(_benchmark_results.values())

@app.get("/responses")
def get_responses():
    return {name: r["response"] for name, r in _benchmark_results.items()}

@app.get("/judge/scores")
def get_judge_scores():
    return _judge_scores

@app.get("/judge/status")
def judge_status():
    return {
        "url": _effective_judge_url(),
        "model": _judge_model_name,
        "mode": "native" if NATIVE_OLLAMA_HOST else "docker",
    }

class JudgeConfigRequest(BaseModel):
    system_prompt: Optional[str] = None
    model: Optional[str] = None

@app.get("/judge/config")
def get_judge_config():
    return {"model": _judge_model_name, "system_prompt": _judge_system_prompt}

@app.post("/judge/config")
async def set_judge_config(req: JudgeConfigRequest):
    global _judge_system_prompt, _judge_model_name
    if req.system_prompt is not None:
        _judge_system_prompt = req.system_prompt
        log.info("Judge system prompt updated")
    if req.model is not None and req.model != _judge_model_name:
        _judge_model_name = req.model
        # Pre-pull the new model in background
        asyncio.create_task(pull_model_if_needed(_effective_judge_url(), _judge_model_name))
        log.info(f"Judge model changed to {_judge_model_name}")
    return {"model": _judge_model_name, "system_prompt": _judge_system_prompt}

@app.post("/judge/run/{model_name}")
async def run_judge(model_name: str):
    r = _benchmark_results.get(model_name)
    if not r:
        raise HTTPException(404, f"No benchmark result for {model_name}")
    prompt = r.get("prompt", "Explain Einstein's theory of relativity in simple terms.")
    score = await judge_response(model_name, prompt, r["response"])
    _judge_scores[model_name] = score
    lb = compute_leaderboard()
    return {"score": score, "leaderboard": lb}

@app.post("/judge/human-score")
def human_score(req: HumanScoreRequest):
    _human_scores[req.model_name] = req.score
    return {"model": req.model_name, "score": req.score, "leaderboard": compute_leaderboard()}

@app.get("/leaderboard")
def leaderboard():
    return compute_leaderboard()

@app.get("/export/json")
def export_json():
    return {
        "results": _benchmark_results,
        "judge_scores": _judge_scores,
        "human_scores": _human_scores,
        "leaderboard": compute_leaderboard(),
        "exported_at": time.time(),
    }

@app.get("/export/csv")
def export_csv():
    from fastapi.responses import PlainTextResponse
    rows = ["model,tps_mean,ttft_mean,accuracy,clarity,depth,concision,examples,human_score,combined_score"]
    for e in compute_leaderboard():
        j = _judge_scores.get(e["model"], {})
        rows.append(",".join(str(x) for x in [
            e["model"], e["tps_mean"], e["ttft_mean"],
            j.get("accuracy",""), j.get("clarity",""), j.get("depth",""),
            j.get("concision",""), j.get("examples",""),
            _human_scores.get(e["model"],""),
            e.get("combined_score",""),
        ]))
    return PlainTextResponse("\n".join(rows), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=judgegpt-results.csv"})

@app.get("/metrics")
def metrics():
    from fastapi.responses import PlainTextResponse
    lines = ["# JudgeGPT metrics"]
    for name, r in _benchmark_results.items():
        lines.append(f'judgegpt_tps{{model="{name}"}} {r["tps_mean"]}')
        lines.append(f'judgegpt_ttft_ms{{model="{name}"}} {r["ttft_mean"]}')
    if docker_client:
        lines.append(f"judgegpt_active_containers {len(_active_containers)}")
    return PlainTextResponse("\n".join(lines))


# ── Live Token Streaming ───────────────────────────────────────────────────────

class StreamLiveRequest(BaseModel):
    model_names: list[str]
    prompt: str
    n_tokens: int = Field(default=256, ge=32, le=2048)
    keep_alive: bool = False


@app.post("/benchmark/stream-live")
async def stream_live_endpoint(req: StreamLiveRequest):
    """Stream tokens live from all models simultaneously — one SSE event per token."""
    queue: asyncio.Queue = asyncio.Queue()

    async def stream_model(model_name: str):
        try:
            container_info = await spawn_container(model_name)
            await queue.put({"event": "model_ready", "model": model_name})
        except Exception as e:
            await queue.put({"event": "model_error", "model": model_name, "error": str(e)})
            return

        url = container_info["internal_url"]
        t_start = time.perf_counter()
        t_first = None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{url}/api/generate",
                    json={
                        "model": model_name, "prompt": req.prompt, "stream": True,
                        "options": {"num_predict": req.n_tokens, "temperature": 0.7},
                    },
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    async for raw in resp.content:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if t_first is None:
                            t_first = time.perf_counter()
                        text = chunk.get("response", "")
                        if text:
                            await queue.put({"event": "token", "model": model_name, "text": text})
                        if chunk.get("done"):
                            eval_count    = chunk.get("eval_count", 0)
                            eval_duration = chunk.get("eval_duration", 1) / 1e9
                            tps  = round(eval_count / eval_duration, 1) if eval_duration else 0
                            ttft = round((t_first - t_start) * 1000, 1) if t_first else 0
                            await queue.put({"event": "model_done", "model": model_name,
                                           "tps": tps, "ttft_ms": ttft, "tokens": eval_count})
                            break
        except Exception as e:
            await queue.put({"event": "model_error", "model": model_name, "error": str(e)})

    async def sse_stream() -> AsyncGenerator[str, None]:
        async def run_all():
            await asyncio.gather(*[stream_model(n) for n in req.model_names], return_exceptions=True)
            await queue.put(None)

        asyncio.create_task(run_all())

        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

        if not req.keep_alive:
            for name in req.model_names:
                asyncio.create_task(kill_container(name))

        yield f"data: {json.dumps({'event': 'stream_done'})}\n\n"

    return StreamingResponse(sse_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── OpenAPI Proxy ──────────────────────────────────────────────────────────────

class ProxyChatRequest(BaseModel):
    endpoint: str
    model: str
    messages: list[dict]
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 512
    system_prompt: str = ""


@app.post("/proxy/chat")
async def proxy_chat(req: ProxyChatRequest):
    """
    Proxy any OpenAI-compatible /v1/chat/completions request.
    Normalises streaming output into { event:'token', text } / { event:'done' } SSE.
    Supports: OpenAI, Anthropic OpenAI-compat, local Ollama, any OpenAI-spec endpoint.
    """
    msgs: list[dict] = []
    if req.system_prompt:
        msgs.append({"role": "system", "content": req.system_prompt})
    msgs.extend(req.messages)

    hdrs = {"Content-Type": "application/json"}
    if req.api_key:
        hdrs["Authorization"] = f"Bearer {req.api_key}"

    payload = {
        "model": req.model,
        "messages": msgs,
        "stream": True,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }

    async def sse_stream() -> AsyncGenerator[str, None]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    req.endpoint, json=payload, headers=hdrs,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        yield f"data: {json.dumps({'event':'error','error':f'HTTP {resp.status}: {err[:300]}'})}\n\n"
                        return

                    async for raw_line in resp.content:
                        raw = raw_line.decode("utf-8", errors="replace").strip()
                        if not raw:
                            continue
                        if raw == "data: [DONE]":
                            yield f"data: {json.dumps({'event':'done'})}\n\n"
                            return

                        line = raw[6:] if raw.startswith("data: ") else raw
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        text = ""
                        finished = False
                        # OpenAI / Ollama OpenAI-compat (/v1/chat/completions)
                        if "choices" in chunk:
                            choice = chunk["choices"][0] if chunk["choices"] else {}
                            text = choice.get("delta", {}).get("content", "") or ""
                            finished = bool(choice.get("finish_reason"))
                        # Ollama /api/chat
                        elif "message" in chunk:
                            text = chunk["message"].get("content", "") or ""
                            finished = chunk.get("done", False)
                        # Ollama /api/generate
                        elif "response" in chunk:
                            text = chunk.get("response", "") or ""
                            finished = chunk.get("done", False)

                        if text:
                            yield f"data: {json.dumps({'event':'token','text':text})}\n\n"
                        if finished:
                            yield f"data: {json.dumps({'event':'done'})}\n\n"
                            return
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'event':'error','error':'Request timed out'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event':'error','error':str(e)})}\n\n"

    return StreamingResponse(sse_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Local model list & pull ─────────────────────────────────────────────────

@app.get("/models/local")
async def list_local_models():
    """Return models already downloaded in the Ollama instance."""
    url = _effective_judge_url()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/api/tags",
                                   timeout=aiohttp.ClientTimeout(total=5)) as resp:
                data = await resp.json()
                return data.get("models", [])
    except Exception:
        return []


@app.post("/models/pull")
async def pull_model_endpoint(req: PullModelRequest):
    """Pull a model from the Ollama registry, streaming progress as SSE.

    Each SSE event mirrors the Ollama /api/pull chunk:
      { "status": "pulling layer", "total": N, "completed": N }
    Final success event: { "status": "success" }
    """
    url = _effective_judge_url()
    log.info(f"Pulling model {req.model_name} from {url}")

    async def sse_stream() -> AsyncGenerator[str, None]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{url}/api/pull",
                    json={"name": req.model_name, "stream": True},
                    timeout=aiohttp.ClientTimeout(total=3600),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        try:
                            detail = json.loads(body).get("error", body[:200])
                        except Exception:
                            detail = body[:200]
                        yield f"data: {json.dumps({'status':'error','error':detail})}\n\n"
                        return
                    async for raw in resp.content:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            chunk = json.loads(raw)
                            yield f"data: {json.dumps(chunk)}\n\n"
                            if chunk.get("status") == "success":
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"data: {json.dumps({'status':'error','error':str(e)})}\n\n"

    return StreamingResponse(sse_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Benchmark history ───────────────────────────────────────────────────────

@app.get("/history")
def get_history():
    """Return benchmark history (newest first)."""
    return list(reversed(_benchmark_history))

@app.delete("/history")
def clear_history():
    _benchmark_history.clear()
    return {"cleared": True}
