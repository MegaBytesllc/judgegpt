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
import io
import json
import logging
import os
import platform
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

# ── GPU platform detection ─────────────────────────────────────────────────────

def _detect_gpu_platform() -> str:
    """Return 'metal', 'nvidia', 'rocm', or 'cpu'."""
    if NATIVE_OLLAMA_HOST and platform.system() == "Darwin":
        return "metal"
    if any(os.getenv(k) for k in ("HSA_OVERRIDE_GFX_VERSION", "ROCM_VERSION", "HIP_VISIBLE_DEVICES")):
        return "rocm"
    if any(os.getenv(k) for k in ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")):
        return "nvidia"
    return "cpu"

GPU_PLATFORM: str = _detect_gpu_platform()

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

# ── GPU helpers ────────────────────────────────────────────────────────────────

def _poll_gpu_from_container_sync(container_name: str) -> dict:
    """Exec GPU-monitoring tools inside an Ollama Docker container (sync, run in executor)."""
    if not docker_client:
        return {}
    try:
        container = docker_client.containers.get(container_name)
        # Try nvidia-smi first
        try:
            r = container.exec_run(
                "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total"
                " --format=csv,noheader,nounits",
                demux=True,
            )
            if r.exit_code == 0 and r.output[0]:
                parts = r.output[0].decode().strip().split(",")
                if len(parts) >= 3:
                    return {
                        "gpu_util_pct":  round(float(parts[0].strip()), 1),
                        "vram_used_mb":  round(float(parts[1].strip()), 1),
                        "vram_total_mb": round(float(parts[2].strip()), 1),
                        "gpu_type": "nvidia",
                    }
        except Exception:
            pass
        # Try rocm-smi
        try:
            r = container.exec_run("rocm-smi --showuse --showmeminfo vram --json", demux=True)
            if r.exit_code == 0 and r.output[0]:
                data = json.loads(r.output[0])
                utils, used, total = [], [], []
                for card_data in data.values():
                    if isinstance(card_data, dict):
                        u  = card_data.get("GPU use (%)")
                        vu = card_data.get("VRAM Total Used Memory (B)")
                        vt = card_data.get("VRAM Total Memory (B)")
                        if u  is not None: utils.append(float(u))
                        if vu is not None: used.append(float(vu) / 1024 / 1024)
                        if vt is not None: total.append(float(vt) / 1024 / 1024)
                out: dict = {"gpu_type": "rocm"}
                if utils: out["gpu_util_pct"]  = round(sum(utils) / len(utils), 1)
                if used:  out["vram_used_mb"]  = round(sum(used), 1)
                if total: out["vram_total_mb"] = round(sum(total), 1)
                if out.get("gpu_util_pct") is not None:
                    return out
        except Exception:
            pass
    except Exception:
        pass
    return {}


async def _get_vram_from_ps(ollama_url: str, model_name: str) -> dict:
    """Query Ollama /api/ps for VRAM used by a model (works on all platforms)."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{ollama_url}/api/ps",
                timeout=aiohttp.ClientTimeout(total=2),
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    model_base = model_name.split(":")[0]
                    for m in data.get("models", []):
                        if model_base in m.get("name", ""):
                            size_vram = m.get("size_vram", 0)
                            if size_vram > 0:
                                return {"vram_used_mb": round(size_vram / 1024 / 1024, 1)}
    except Exception:
        pass
    return {}

# ── Benchmark runner ───────────────────────────────────────────────────────────

async def run_single(
    model_name: str,
    container_info: dict,
    prompt: str,
    n_tokens: int,
) -> dict:
    """Run one inference pass, return TPS + TTFT + GPU perf + response text."""
    url = container_info["internal_url"]
    container_name = container_info.get("container_name")  # None in native mode
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

    # Done-chunk telemetry
    eval_count        = 0
    eval_duration_s   = 1.0
    load_duration_ms  = 0.0
    total_duration_ms = 0.0
    prompt_eval_count = 0
    prompt_tps        = 0.0

    # GPU polling: collect samples during inference
    gpu_samples: list[dict] = []

    async def _gpu_poll():
        while True:
            stats: dict = {}
            if container_name and docker_client and GPU_PLATFORM in ("nvidia", "rocm"):
                loop = asyncio.get_event_loop()
                stats = await loop.run_in_executor(
                    None, _poll_gpu_from_container_sync, container_name
                )
            if not stats:
                stats = await _get_vram_from_ps(url, model_name)
            if stats:
                gpu_samples.append(stats)
            await asyncio.sleep(1.5)

    gpu_task = asyncio.create_task(_gpu_poll())

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
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
                        eval_count        = chunk.get("eval_count", tokens)
                        eval_duration_s   = chunk.get("eval_duration", 1) / 1e9
                        load_duration_ms  = round(chunk.get("load_duration", 0) / 1e6, 1)
                        total_duration_ms = round(chunk.get("total_duration", 0) / 1e6, 1)
                        prompt_eval_count = chunk.get("prompt_eval_count", 0)
                        _prompt_dur_s     = chunk.get("prompt_eval_duration", 0) / 1e9
                        prompt_tps        = round(prompt_eval_count / _prompt_dur_s, 1) if _prompt_dur_s > 0 else 0
                        break
    finally:
        gpu_task.cancel()
        try:
            await gpu_task
        except asyncio.CancelledError:
            pass

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    ttft_ms = round((t_first - t_start) * 1000, 1) if t_first else 0
    tps = round(eval_count / eval_duration_s, 1) if eval_duration_s > 0 else 0

    # Aggregate GPU samples
    gpu_result: dict = {}
    if gpu_samples:
        utils = [s["gpu_util_pct"] for s in gpu_samples if "gpu_util_pct" in s]
        vraam = [s["vram_used_mb"] for s in gpu_samples if "vram_used_mb" in s]
        if utils:
            gpu_result["gpu_util_pct"] = round(max(utils), 1)
        if vraam:
            gpu_result["vram_used_mb"] = round(max(vraam), 1)
        vram_total = next((s.get("vram_total_mb") for s in gpu_samples if s.get("vram_total_mb")), None)
        if vram_total:
            gpu_result["vram_total_mb"] = vram_total
        gpu_type = next((s.get("gpu_type") for s in gpu_samples if s.get("gpu_type")), None)
        if gpu_type:
            gpu_result["gpu_type"] = gpu_type

    return {
        "tps": tps,
        "ttft_ms": ttft_ms,
        "tokens": eval_count,
        "duration_ms": round(elapsed * 1000, 1),
        "load_duration_ms": load_duration_ms,
        "total_duration_ms": total_duration_ms,
        "prompt_eval_count": prompt_eval_count,
        "prompt_tps": prompt_tps,
        "response": response_text.strip(),
        **gpu_result,
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

    tps_vals        = [r["tps"]              for r in runs]
    ttft_vals       = [r["ttft_ms"]          for r in runs]
    load_vals       = [r.get("load_duration_ms",  0) for r in runs]
    prompt_tps_vals = [r.get("prompt_tps",        0) for r in runs]
    result = {
        "model":           model_name,
        "color":           model_color(model_name),
        "runs":            runs,
        "tps_mean":        round(sum(tps_vals)        / len(tps_vals),        1),
        "tps_max":         round(max(tps_vals),                               1),
        "tps_min":         round(min(tps_vals),                               1),
        "ttft_mean":       round(sum(ttft_vals)       / len(ttft_vals),       1),
        "ttft_min":        round(min(ttft_vals),                              1),
        "load_ms_mean":    round(sum(load_vals)       / len(load_vals),       1),
        "prompt_tps_mean": round(sum(prompt_tps_vals) / len(prompt_tps_vals), 1),
        "response":        runs[-1]["response"],
        "prompt":          prompt,
        "timestamp":       time.time(),
    }
    # GPU stats — peak across runs
    gpu_utils = [r["gpu_util_pct"] for r in runs if r.get("gpu_util_pct") is not None]
    vram_used = [r["vram_used_mb"] for r in runs if r.get("vram_used_mb")  is not None]
    if gpu_utils: result["gpu_util_pct_peak"] = round(max(gpu_utils), 1)
    if vram_used: result["vram_used_mb"]       = round(max(vram_used), 1)
    for field in ("vram_total_mb", "gpu_type"):
        v = next((r[field] for r in runs if r.get(field)), None)
        if v is not None: result[field] = v
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
                "options": {"temperature": 0.1, "num_predict": 2048},
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
    n_tokens: int = Field(default=512, ge=-1)  # -1 = unlimited (Ollama num_predict: -1)
    n_runs: int = Field(default=3, ge=1, le=10)
    auto_judge: bool = True
    keep_alive: bool = False  # if False, kill containers after benchmark
    sequential: bool = False  # if True, run one model at a time (saves VRAM)

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
        tasks = [
            benchmark_model(name, req.prompt, req.n_tokens, req.n_runs, queue)
            for name in req.model_names
        ]

        async def run_tasks():
            if req.sequential:
                done = []
                for coro in tasks:
                    r = await coro
                    done.append(r)
            else:
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

class ExportPDFRequest(BaseModel):
    prompt: str = ""
    n_tokens: int = 512
    n_runs: int = 3
    results: dict = {}
    judge_scores: dict = {}
    leaderboard: list = []


@app.post("/export/pdf")
def export_pdf(req: ExportPDFRequest):
    """Generate a PDF benchmark report and return it as a download."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    # ── Colour palette ────────────────────────────────────────────────────────
    C_DARK    = colors.HexColor('#0d1117')
    C_SURFACE = colors.HexColor('#f5f7fa')
    C_BORDER  = colors.HexColor('#d0d7de')
    C_ROW_ALT = colors.HexColor('#eaf0f6')
    C_CYAN    = colors.HexColor('#0077aa')
    C_PURPLE  = colors.HexColor('#6e40c9')
    C_MUTED   = colors.HexColor('#7d8590')
    C_TEXT    = colors.HexColor('#1a1d23')
    C_WHITE   = colors.white
    C_GOLD    = colors.HexColor('#b08800')

    buf = io.BytesIO()
    W, H = letter
    margin = 0.75 * inch
    usable_w = W - 2 * margin

    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        leftMargin=margin, rightMargin=margin,
    )

    # ── Reusable styles ───────────────────────────────────────────────────────
    def section_hdr(text):
        return Paragraph(text, ParagraphStyle(
            'SH', fontName='Courier-Bold', fontSize=8, textColor=C_MUTED,
            spaceBefore=14, spaceAfter=5,
        ))

    def note(text):
        return Paragraph(text, ParagraphStyle(
            'Note', fontName='Courier', fontSize=7, textColor=C_MUTED,
            spaceAfter=4,
        ))

    story = []

    # ── Title banner ─────────────────────────────────────────────────────────
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    banner = Table(
        [['JudgeGPT  Benchmark Report', f'Generated: {ts}']],
        colWidths=[usable_w * 0.6, usable_w * 0.4],
    )
    banner.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), C_DARK),
        ('TEXTCOLOR',     (0, 0), (0,  0),  C_CYAN),
        ('TEXTCOLOR',     (1, 0), (1,  0),  C_MUTED),
        ('FONTNAME',      (0, 0), (0,  0),  'Courier-Bold'),
        ('FONTNAME',      (1, 0), (1,  0),  'Courier'),
        ('FONTSIZE',      (0, 0), (0,  0),  16),
        ('FONTSIZE',      (1, 0), (1,  0),  8),
        ('ALIGN',         (0, 0), (0,  0),  'LEFT'),
        ('ALIGN',         (1, 0), (1,  0),  'RIGHT'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 14),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 14),
        ('TOPPADDING',    (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
    ]))
    story.append(banner)

    # Config strip
    n_tok_label = 'unlimited' if req.n_tokens == -1 else str(req.n_tokens)
    config_strip = Table(
        [[f'Runs per model: {req.n_runs}',
          f'Max tokens: {n_tok_label}',
          f'Models benchmarked: {len(req.results)}']],
        colWidths=[usable_w / 3] * 3,
    )
    config_strip.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), C_SURFACE),
        ('FONTNAME',      (0, 0), (-1, -1), 'Courier'),
        ('FONTSIZE',      (0, 0), (-1, -1), 8),
        ('TEXTCOLOR',     (0, 0), (-1, -1), C_MUTED),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID',          (0, 0), (-1, -1), 0.5, C_BORDER),
    ]))
    story.append(config_strip)
    story.append(Spacer(1, 14))

    # ── Prompt ───────────────────────────────────────────────────────────────
    if req.prompt:
        story.append(section_hdr('BENCHMARK PROMPT'))
        words = len(req.prompt.split())
        story.append(note(f'{words} words · {len(req.prompt)} characters'))
        prompt_box = Table(
            [[req.prompt]],
            colWidths=[usable_w],
        )
        prompt_box.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), C_SURFACE),
            ('FONTNAME',      (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE',      (0, 0), (-1, -1), 11),
            ('TEXTCOLOR',     (0, 0), (-1, -1), C_TEXT),
            ('LEFTPADDING',   (0, 0), (-1, -1), 12),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 12),
            ('TOPPADDING',    (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LINEABOVE',     (0, 0), (-1,  0),  3, C_PURPLE),
            ('BOX',           (0, 0), (-1, -1), 0.5, C_BORDER),
        ]))
        story.append(prompt_box)
        story.append(Spacer(1, 14))

    # ── Performance results ───────────────────────────────────────────────────
    if req.results:
        story.append(section_hdr('PERFORMANCE RESULTS'))
        hdr = [['Model', 'Avg TPS', 'TTFT (ms)', 'Tokens', 'Duration (ms)']]
        rows = sorted(
            [
                [
                    name,
                    f"{r.get('tps_mean', '—')} t/s",
                    f"{r.get('ttft_mean', '—')}",
                    str(r.get('tokens_mean', '—')),
                    str(r.get('duration_mean', '—')),
                ]
                for name, r in req.results.items()
            ],
            key=lambda x: float(str(x[1]).split()[0]) if str(x[1]).split()[0].replace('.','').isdigit() else 0,
            reverse=True,
        )
        perf_t = Table(
            hdr + rows,
            colWidths=[usable_w * 0.36, usable_w * 0.16, usable_w * 0.16, usable_w * 0.16, usable_w * 0.16],
        )
        perf_t.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1,  0),  C_DARK),
            ('TEXTCOLOR',     (0, 0), (-1,  0),  C_WHITE),
            ('FONTNAME',      (0, 0), (-1,  0),  'Courier-Bold'),
            ('FONTNAME',      (0, 1), (-1, -1),  'Courier'),
            ('FONTSIZE',      (0, 0), (-1, -1),  9),
            ('TEXTCOLOR',     (0, 1), (-1, -1),  C_TEXT),
            ('ALIGN',         (1, 0), (-1, -1),  'CENTER'),
            ('ALIGN',         (0, 0), (0,  -1),  'LEFT'),
            ('LEFTPADDING',   (0, 0), (0,  -1),  8),
            ('TOPPADDING',    (0, 0), (-1, -1),  6),
            ('BOTTOMPADDING', (0, 0), (-1, -1),  6),
            ('GRID',          (0, 0), (-1, -1),  0.5, C_BORDER),
            ('ROWBACKGROUNDS',(0, 1), (-1, -1),  [C_WHITE, C_ROW_ALT]),
        ]))
        story.append(perf_t)
        story.append(Spacer(1, 14))

    # ── Judge scores ─────────────────────────────────────────────────────────
    CRITERIA = ['accuracy', 'clarity', 'depth', 'concision', 'examples']
    CRIT_LABELS = ['Accuracy', 'Clarity', 'Depth', 'Concision', 'Examples']

    if req.judge_scores:
        story.append(section_hdr('JUDGE SCORES  (out of 5)'))
        model_names = list(req.judge_scores.keys())
        col_w = (usable_w - usable_w * 0.22) / max(len(model_names), 1)
        j_hdr = [['Criterion'] + [m.split(':')[0] for m in model_names]]
        j_rows = []
        for key, label in zip(CRITERIA, CRIT_LABELS):
            row = [label]
            for m in model_names:
                v = req.judge_scores[m].get(key)
                row.append(f"{v}/5" if v is not None else '—')
            j_rows.append(row)
        # Average row
        avg_row = ['Average']
        for m in model_names:
            sc = req.judge_scores[m]
            avg = sum(sc.get(k, 0) for k in CRITERIA) / len(CRITERIA)
            avg_row.append(f"{avg:.2f}/5")
        j_rows.append(avg_row)

        judge_t = Table(
            j_hdr + j_rows,
            colWidths=[usable_w * 0.22] + [col_w] * len(model_names),
        )
        judge_t.setStyle(TableStyle([
            ('BACKGROUND',    (0,  0), (-1,  0),  C_DARK),
            ('TEXTCOLOR',     (0,  0), (-1,  0),  C_WHITE),
            ('FONTNAME',      (0,  0), (-1,  0),  'Courier-Bold'),
            ('FONTNAME',      (0,  1), (-1, -1),  'Courier'),
            ('FONTSIZE',      (0,  0), (-1, -1),  9),
            ('TEXTCOLOR',     (0,  1), (-1, -2),  C_TEXT),
            ('ALIGN',         (1,  0), (-1, -1),  'CENTER'),
            ('ALIGN',         (0,  0), (0,  -1),  'LEFT'),
            ('LEFTPADDING',   (0,  0), (0,  -1),  8),
            ('TOPPADDING',    (0,  0), (-1, -1),  5),
            ('BOTTOMPADDING', (0,  0), (-1, -1),  5),
            ('GRID',          (0,  0), (-1, -1),  0.5, C_BORDER),
            ('ROWBACKGROUNDS',(0,  1), (-1, -2),  [C_WHITE, C_ROW_ALT]),
            # Average row styling
            ('FONTNAME',      (0, -1), (-1, -1),  'Courier-Bold'),
            ('BACKGROUND',    (0, -1), (-1, -1),  colors.HexColor('#e8e4f3')),
            ('LINEABOVE',     (0, -1), (-1, -1),  1, C_BORDER),
        ]))
        story.append(judge_t)
        story.append(Spacer(1, 8))

        # Reasoning snippets
        for m in model_names:
            r_text = req.judge_scores[m].get('reasoning', '')
            if r_text:
                snippet = r_text[:280] + ('...' if len(r_text) > 280 else '')
                story.append(Paragraph(
                    f'<i><b>{m.split(":")[0]}:</b>  {snippet}</i>',
                    ParagraphStyle('Rsn', fontName='Helvetica-Oblique', fontSize=9,
                                   textColor=C_MUTED, leftIndent=8, spaceAfter=3),
                ))
        story.append(Spacer(1, 10))

    # ── Leaderboard ───────────────────────────────────────────────────────────
    if req.leaderboard:
        story.append(section_hdr('LEADERBOARD'))
        story.append(note('Combined score = TPS x35% + TTFT x15% + Quality x50%'))
        medals = {1: '#1', 2: '#2', 3: '#3'}
        lb_hdr = [['Rank', 'Model', 'TPS Score', 'TTFT Score', 'Quality', 'Combined']]
        lb_rows = []
        for e in req.leaderboard:
            rank = e.get('rank', '—')
            lb_rows.append([
                medals.get(rank, f'#{rank}'),
                e.get('model', '—'),
                f"{e['tps_score']:.1f}"     if e.get('tps_score')     is not None else '—',
                f"{e['ttft_score']:.1f}"    if e.get('ttft_score')    is not None else '—',
                f"{e['quality_score']:.1f}" if e.get('quality_score') is not None else '—',
                f"{e['combined_score']:.1f}" if e.get('combined_score') is not None else '—',
            ])
        lb_t = Table(
            lb_hdr + lb_rows,
            colWidths=[usable_w * 0.08, usable_w * 0.32, usable_w * 0.14,
                       usable_w * 0.14, usable_w * 0.14, usable_w * 0.14],
        )
        lb_t.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1,  0), C_DARK),
            ('TEXTCOLOR',     (0, 0), (-1,  0), C_WHITE),
            ('FONTNAME',      (0, 0), (-1,  0), 'Courier-Bold'),
            ('FONTNAME',      (0, 1), (-1, -1), 'Courier'),
            ('FONTSIZE',      (0, 0), (-1, -1), 9),
            ('TEXTCOLOR',     (0, 1), (-1, -1), C_TEXT),
            ('ALIGN',         (0, 0), (0,  -1), 'CENTER'),
            ('ALIGN',         (1, 0), (1,  -1), 'LEFT'),
            ('ALIGN',         (2, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING',   (1, 0), (1,  -1), 8),
            ('TOPPADDING',    (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID',          (0, 0), (-1, -1), 0.5, C_BORDER),
            ('ROWBACKGROUNDS',(0, 1), (-1, -1), [C_WHITE, C_ROW_ALT]),
            # Gold for rank-1
            ('TEXTCOLOR',     (0, 1), (0,  1),  C_GOLD),
            ('FONTNAME',      (0, 1), (-1,  1), 'Courier-Bold'),
        ]))
        story.append(lb_t)
        story.append(Spacer(1, 14))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width=usable_w, thickness=0.5, color=C_BORDER))
    story.append(Paragraph(
        f'Generated by JudgeGPT  |  {ts}',
        ParagraphStyle('Footer', fontName='Courier', fontSize=7,
                       textColor=C_MUTED, spaceBefore=6, alignment=1),
    ))

    doc.build(story)
    buf.seek(0)

    fname = f'judgegpt-report-{time.strftime("%Y%m%d-%H%M%S")}.pdf'
    return StreamingResponse(
        buf,
        media_type='application/pdf',
        headers={'Content-Disposition': f'attachment; filename="{fname}"'},
    )


@app.get("/gpu/info")
async def gpu_info():
    """Return GPU platform and live Ollama /api/ps VRAM data."""
    url = NATIVE_OLLAMA_HOST if NATIVE_OLLAMA_HOST else JUDGE_OLLAMA_URL
    models_loaded = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/api/ps", timeout=aiohttp.ClientTimeout(total=2)) as r:
                if r.status == 200:
                    data = await r.json()
                    models_loaded = data.get("models", [])
    except Exception:
        pass
    return {
        "platform": GPU_PLATFORM,
        "models_loaded": models_loaded,
    }


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
    n_tokens: int = Field(default=512, ge=-1)  # -1 = unlimited (Ollama num_predict: -1)
    keep_alive: bool = False
    sequential: bool = False  # if True, stream one model at a time


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
        container_name = container_info.get("container_name")
        t_start = time.perf_counter()
        t_first = None

        # GPU polling: emit gpu_stats events every ~2s during streaming
        async def _gpu_poll():
            while True:
                stats: dict = {}
                if container_name and docker_client and GPU_PLATFORM in ("nvidia", "rocm"):
                    loop = asyncio.get_event_loop()
                    stats = await loop.run_in_executor(
                        None, _poll_gpu_from_container_sync, container_name
                    )
                if not stats:
                    stats = await _get_vram_from_ps(url, model_name)
                if stats:
                    await queue.put({"event": "gpu_stats", "model": model_name, **stats})
                await asyncio.sleep(2.0)

        gpu_task = asyncio.create_task(_gpu_poll())

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
                            eval_count     = chunk.get("eval_count", 0)
                            eval_dur_s     = chunk.get("eval_duration", 1) / 1e9
                            tps            = round(eval_count / eval_dur_s, 1) if eval_dur_s else 0
                            ttft           = round((t_first - t_start) * 1000, 1) if t_first else 0
                            load_ms        = round(chunk.get("load_duration", 0) / 1e6, 1)
                            total_ms       = round(chunk.get("total_duration", 0) / 1e6, 1)
                            prompt_count   = chunk.get("prompt_eval_count", 0)
                            _prompt_dur_s  = chunk.get("prompt_eval_duration", 0) / 1e9
                            prompt_tps_val = round(prompt_count / _prompt_dur_s, 1) if _prompt_dur_s > 0 else 0
                            await queue.put({
                                "event": "model_done",
                                "model": model_name,
                                "tps": tps,
                                "ttft_ms": ttft,
                                "tokens": eval_count,
                                "load_duration_ms": load_ms,
                                "total_duration_ms": total_ms,
                                "prompt_tps": prompt_tps_val,
                            })
                            break
        except Exception as e:
            await queue.put({"event": "model_error", "model": model_name, "error": str(e)})
        finally:
            gpu_task.cancel()
            try:
                await gpu_task
            except asyncio.CancelledError:
                pass

    async def sse_stream() -> AsyncGenerator[str, None]:
        async def run_all():
            if req.sequential:
                for name in req.model_names:
                    await stream_model(name)
            else:
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
    max_tokens: int = 4096
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
