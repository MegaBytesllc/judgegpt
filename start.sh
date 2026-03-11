#!/usr/bin/env bash
# JudgeGPT — Start
# Usage: ./start.sh [up|down|logs]

set -euo pipefail
CMD="${1:-up}"

CYN='\033[0;36m'; GRN='\033[0;32m'; YLW='\033[1;33m'; PRP='\033[0;35m'; RST='\033[0m'

case "$CMD" in
  up)
    echo -e "${PRP}⚖  Starting JudgeGPT…${RST}"
    echo ""

    # Ensure native Ollama is running (Metal GPU)
    if ! curl -sf http://localhost:11435 > /dev/null 2>&1; then
      echo -e "${YLW}Starting native Ollama on port 11435 (Metal GPU)…${RST}"
      OLLAMA_HOST=0.0.0.0:11435 \
      OLLAMA_NUM_PARALLEL=4 \
      OLLAMA_MAX_LOADED_MODELS=4 \
      OLLAMA_FLASH_ATTENTION=1 \
      /usr/local/bin/ollama serve > /tmp/ollama-native.log 2>&1 &
      sleep 2
    else
      echo -e "${GRN}✓ Native Ollama already running on :11435${RST}"
    fi

    PATH="/usr/local/bin:/usr/bin:/bin" docker compose up --build -d
    echo ""
    echo -e "${GRN}✓ JudgeGPT running${RST}"
    echo -e "  Dashboard : ${CYN}http://localhost:3000${RST}"
    echo -e "  API docs  : ${CYN}http://localhost:8080/docs${RST}"
    echo ""
    echo -e "${YLW}First run: the judge model (qwen2.5:7b ~4.7GB) pulls automatically into native Ollama.${RST}"
    echo -e "${YLW}Benchmark models pull on first selection — only once each.${RST}"
    ;;
  down)
    echo -e "${YLW}Stopping JudgeGPT…${RST}"
    docker compose down
    ;;
  logs)
    docker compose logs -f orchestrator
    ;;
  *)
    echo "Usage: ./start.sh [up|down|logs]"
    ;;
esac
