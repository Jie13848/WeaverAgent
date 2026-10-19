#!/bin/zsh
# WeaverAgent 停止脚本

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "${YELLOW}正在停止 WeaverAgent 服务...${NC}"

kill_port() {
  local port=$1
  local pid
  pid=$(lsof -ti tcp:"$port" 2>/dev/null)
  if [ -n "$pid" ]; then
    kill -9 "$pid" 2>/dev/null
    echo "${GREEN}  ✓ 已停止端口 $port (PID $pid)${NC}"
  else
    echo "  端口 $port 无运行中服务"
  fi
}

kill_port 5001
kill_port 3000

echo "${GREEN}完成。${NC}"
