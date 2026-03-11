#!/usr/bin/env bash
# JudgeGPT — Start
# Usage: ./start.sh [up|down|logs|platform] [--platform mac|nvidia|amd|cpu]

set -euo pipefail

CMD="${1:-up}"
PLATFORM_OVERRIDE="${2:-}"

CYN='\033[0;36m'
GRN='\033[0;32m'
YLW='\033[1;33m'
PRP='\033[0;35m'
RED='\033[0;31m'
RST='\033[0m'

DOCKER="PATH=/usr/local/bin:/usr/bin:/bin docker"

# ── Platform detection ──────────────────────────────────────────────────────

detect_platform() {
  # Manual override: ./start.sh up --platform nvidia
  if [[ -n "$PLATFORM_OVERRIDE" ]]; then
    echo "${PLATFORM_OVERRIDE#--platform=}"
    return
  fi

  local os
  os="$(uname -s)"

  # macOS — always use native Ollama (Metal GPU, no Docker GPU passthrough on Mac)
  if [[ "$os" == "Darwin" ]]; then
    echo "mac"
    return
  fi

  # NVIDIA GPU — Linux or Windows WSL2
  if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q .; then
    echo "nvidia"
    return
  fi

  # AMD ROCm — Linux with /dev/kfd present
  if [[ -e /dev/kfd ]]; then
    echo "amd"
    return
  fi

  # CPU fallback — works everywhere
  echo "cpu"
}

PLATFORM=$(detect_platform)

compose_cmd() {
  case "$PLATFORM" in
    mac)    echo "$DOCKER compose -f docker-compose.yml -f docker-compose.mac.yml" ;;
    nvidia) echo "$DOCKER compose -f docker-compose.yml -f docker-compose.nvidia.yml" ;;
    amd)    echo "$DOCKER compose -f docker-compose.yml -f docker-compose.amd.yml" ;;
    cpu)    echo "$DOCKER compose -f docker-compose.yml -f docker-compose.cpu.yml" ;;
    *)      echo "$DOCKER compose -f docker-compose.yml -f docker-compose.cpu.yml" ;;
  esac
}

COMPOSE=$(compose_cmd)

# ── Commands ────────────────────────────────────────────────────────────────

case "$CMD" in
  up)
    echo -e "${PRP}⚖  JudgeGPT — platform: ${YLW}${PLATFORM}${RST}"
    echo ""

    # ── Mac: start native Ollama for Metal GPU ──────────────────────────────
    if [[ "$PLATFORM" == "mac" ]]; then
      if ! curl -sf http://localhost:11435 > /dev/null 2>&1; then
        echo -e "${YLW}Starting native Ollama on :11435 (Metal GPU)…${RST}"
        OLLAMA_HOST=0.0.0.0:11435 \
        OLLAMA_NUM_PARALLEL=4 \
        OLLAMA_MAX_LOADED_MODELS=4 \
        OLLAMA_FLASH_ATTENTION=1 \
        /usr/local/bin/ollama serve > /tmp/judgegpt-ollama.log 2>&1 &
        sleep 2
        echo -e "${GRN}✓ Ollama started${RST}"
      else
        echo -e "${GRN}✓ Native Ollama already running on :11435${RST}"
      fi

    # ── NVIDIA: verify container toolkit ───────────────────────────────────
    elif [[ "$PLATFORM" == "nvidia" ]]; then
      if ! $DOCKER info 2>/dev/null | grep -q "nvidia"; then
        echo -e "${YLW}⚠  nvidia-container-toolkit not detected in Docker.${RST}"
        echo -e "   Install guide: ${CYN}https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html${RST}"
        echo ""
      fi
      GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
      echo -e "${GRN}✓ NVIDIA GPU: ${GPU_NAME}${RST}"

    # ── AMD: verify ROCm devices ────────────────────────────────────────────
    elif [[ "$PLATFORM" == "amd" ]]; then
      if [[ ! -e /dev/kfd ]]; then
        echo -e "${RED}⚠  /dev/kfd not found — ROCm may not be installed.${RST}"
        echo -e "   Install guide: ${CYN}https://rocm.docs.amd.com/en/latest/deploy/linux/installer/install.html${RST}"
        echo ""
      else
        # Show GPU name from rocminfo if available
        GPU_NAME=$(rocminfo 2>/dev/null | grep -m1 "Marketing Name" | sed 's/.*Marketing Name: *//' || echo "")
        GFX_TARGET=$(rocminfo 2>/dev/null | grep -m1 "gfx" | grep -o 'gfx[0-9]*' | head -1 || echo "")
        if [[ -n "$GPU_NAME" ]]; then
          echo -e "${GRN}✓ AMD GPU: ${GPU_NAME}${RST}"
          [[ -n "$GFX_TARGET" ]] && echo -e "   Target: ${CYN}${GFX_TARGET}${RST}"
        else
          echo -e "${GRN}✓ AMD ROCm devices found${RST}"
        fi
        if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
          echo -e "   Override: ${YLW}HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}${RST}"
        fi
      fi

    # ── CPU fallback ────────────────────────────────────────────────────────
    else
      echo -e "${YLW}⚠  CPU mode — no GPU detected. Expect ~2-8 t/s on 7B models.${RST}"
      echo ""
    fi

    eval "$COMPOSE up --build -d"

    echo ""
    echo -e "${GRN}✓ JudgeGPT running${RST}"
    echo -e "  Dashboard  : ${CYN}http://localhost:3000${RST}"
    echo -e "  API docs   : ${CYN}http://localhost:8080/docs${RST}"
    echo -e "  Platform   : ${YLW}${PLATFORM}${RST}"
    echo ""
    echo -e "${YLW}First run: judge model (qwen2.5:7b ~4.7 GB) pulls automatically.${RST}"
    echo -e "${YLW}Benchmark models pull on first selection — only once each.${RST}"
    ;;

  down)
    echo -e "${YLW}Stopping JudgeGPT (${PLATFORM})…${RST}"
    eval "$COMPOSE down --remove-orphans"
    ;;

  logs)
    eval "$COMPOSE logs -f orchestrator"
    ;;

  gpu-info)
    # Detect AMD GPU details from inside an ROCm container — useful for troubleshooting
    echo -e "${PRP}⚖  JudgeGPT — AMD GPU detection${RST}"
    echo ""
    if [[ ! -e /dev/kfd ]]; then
      echo -e "${RED}✗ /dev/kfd not found — ROCm drivers not installed.${RST}"
      exit 1
    fi
    echo -e "${YLW}Running rocminfo inside ollama/ollama:rocm…${RST}"
    echo ""
    $DOCKER run --rm \
      --device=/dev/kfd:/dev/kfd \
      --device=/dev/dri:/dev/dri \
      --group-add video \
      --group-add render \
      ollama/ollama:rocm \
      sh -c "rocminfo 2>/dev/null | grep -E '(Name|gfx[0-9]|Architecture|Compute Unit|Max Clock)' | head -30" \
      || echo -e "${RED}rocminfo failed — ROCm may not recognise this GPU.${RST}"
    echo ""
    echo -e "${YLW}If your GPU appears above but Ollama still uses CPU, set:${RST}"
    echo -e "   ${CYN}RDNA 2 (RX 6000):  export HSA_OVERRIDE_GFX_VERSION=10.3.0${RST}"
    echo -e "   ${CYN}RDNA 3 (RX 7000):  export HSA_OVERRIDE_GFX_VERSION=11.0.0${RST}"
    echo -e "   ${CYN}RDNA 4 (RX 9000):  export HSA_OVERRIDE_GFX_VERSION=12.0.1${RST}"
    echo -e "   …then re-run:  ${YLW}./start.sh up${RST}"
    ;;

  platform)
    echo "$PLATFORM"
    ;;

  *)
    echo -e "Usage: ./start.sh ${CYN}[up|down|logs|platform|gpu-info]${RST} ${YLW}[--platform mac|nvidia|amd|cpu]${RST}"
    echo ""
    echo -e "Detected platform: ${YLW}$(detect_platform)${RST}"
    echo ""
    echo "Platform overrides:"
    echo "  ./start.sh up --platform mac     macOS Apple Silicon (native Ollama / Metal)"
    echo "  ./start.sh up --platform nvidia  NVIDIA GPU (Linux / Windows WSL2)"
    echo "  ./start.sh up --platform amd     AMD ROCm (Linux)"
    echo "  ./start.sh up --platform cpu     CPU only (any machine)"
    echo ""
    echo "AMD diagnostics:"
    echo "  ./start.sh gpu-info              Detect AMD GPU + gfx target inside ROCm container"
    ;;
esac
