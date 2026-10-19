#!/bin/zsh
# ============================================================
#  WeaverAgent 一键启动脚本 ./start.sh
# ============================================================

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "${BLUE}        WeaverAgent 启动中...               ${NC}"
echo "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# ── 检查端口是否已占用，若占用则杀掉旧进程 ──────────────
kill_port() {
  local port=$1
  local pids
  pids=$(lsof -ti tcp:"$port" 2>/dev/null)
  if [ -n "$pids" ]; then
    echo "${YELLOW}⚠ 端口 $port 已被占用 (PID $(echo $pids | tr '\n' ' '))，正在释放...${NC}"
    echo "$pids" | xargs kill -9 2>/dev/null
    sleep 1
  fi
}

kill_port 5001
kill_port 3000

# ── 启动后端 ─────────────────────────────────────────────
echo "${GREEN}▶ 启动后端 (port 5001)...${NC}"
cd "$BACKEND" || exit 1

# 激活虚拟环境（优先用 .venv，其次用 conda/系统 python）
if [ -f "$BACKEND/.venv/bin/activate" ]; then
  source "$BACKEND/.venv/bin/activate"
  PYTHON="$BACKEND/.venv/bin/python"
elif command -v python3 &>/dev/null; then
  PYTHON="python3"
else
  PYTHON="python"
fi

# ── 检查关键依赖 graphrag 是否可用（editable 安装偶尔失效，自动修复） ──
if ! $PYTHON -c "import graphrag" 2>/dev/null; then
  echo "${YELLOW}⚠ graphrag 模块不可用，正在自动修复 editable 安装...${NC}"
  if command -v uv &>/dev/null; then
    uv pip install --no-deps \
      -e "$ROOT/graphrag/packages/graphrag-common" \
      -e "$ROOT/graphrag/packages/graphrag-cache" \
      -e "$ROOT/graphrag/packages/graphrag-storage" \
      -e "$ROOT/graphrag/packages/graphrag-input" \
      -e "$ROOT/graphrag/packages/graphrag-chunking" \
      -e "$ROOT/graphrag/packages/graphrag-vectors" \
      -e "$ROOT/graphrag/packages/graphrag-llm" \
      -e "$ROOT/graphrag/packages/graphrag" 2>&1 | tail -3
    if $PYTHON -c "import graphrag" 2>/dev/null; then
      echo "${GREEN}  ✓ graphrag 已修复${NC}"
    else
      echo "${RED}  ✗ graphrag 修复失败，请手动检查 ../graphrag/packages/${NC}"
    fi
  else
    echo "${RED}  ✗ 未找到 uv 命令，无法自动修复${NC}"
  fi
fi

$PYTHON run.py > "$BACKEND/logs/backend.log" 2>&1 &
BACKEND_PID=$!
echo "${GREEN}  后端 PID: $BACKEND_PID${NC}"

# ── 等待后端就绪 ─────────────────────────────────────────
echo "${YELLOW}  等待后端启动...${NC}"
for i in $(seq 1 20); do
  if curl -s http://localhost:5001/api/health > /dev/null 2>&1 || \
     curl -s http://localhost:5001/ > /dev/null 2>&1; then
    echo "${GREEN}  ✓ 后端已就绪${NC}"
    break
  fi
  sleep 1
  if [ "$i" -eq 20 ]; then
    echo "${YELLOW}  ⚠ 后端响应超时，请检查 backend/logs/backend.log${NC}"
  fi
done

# ── 启动前端 ─────────────────────────────────────────────
echo "${GREEN}▶ 启动前端 (port 3000)...${NC}"
cd "$FRONTEND" || exit 1
npm run dev > "$BACKEND/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "${GREEN}  前端 PID: $FRONTEND_PID${NC}"

# ── 等待前端就绪 ─────────────────────────────────────────
echo "${YELLOW}  等待前端启动...${NC}"
for i in $(seq 1 15); do
  if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "${GREEN}  ✓ 前端已就绪${NC}"
    break
  fi
  sleep 1
done

echo ""
echo "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "${GREEN}  ✅ WeaverAgent 启动完成！${NC}"
echo ""
echo "  前端地址: ${BLUE}http://localhost:3000${NC}"
echo "  后端地址: ${BLUE}http://localhost:5001${NC}"
echo ""
echo "  日志文件:"
echo "    后端: backend/logs/backend.log"
echo "    前端: backend/logs/frontend.log"
echo ""
echo "  停止服务: ${YELLOW}Ctrl+C${NC} 或运行 ${YELLOW}./stop.sh${NC}"
echo "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# ── 等待退出信号 ─────────────────────────────────────────
trap "echo ''; echo '${YELLOW}正在停止服务...${NC}'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo '${GREEN}已停止。${NC}'; exit 0" INT TERM

wait
