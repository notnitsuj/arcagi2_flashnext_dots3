#!/bin/bash
# Stop one vLLM server (by port) and the processes still holding its GPUs, leaving servers on other GPUs alone.
# Usage: kill_server.sh PORT GPUS
port=$1; gpus=$2
me=$$; parent=$PPID
for p in $(pgrep -u "$USER" -f "vllm serve .*--port $port" | grep -v -w -e $me -e $parent); do
  g=$(ps -o pgid= -p $p | tr -d ' ')
  [ -n "$g" ] && kill -9 -- -$g 2>/dev/null
  kill -9 $p 2>/dev/null
done
uuids=$(nvidia-smi -i $gpus --query-gpu=uuid --format=csv,noheader | tr '\n' '|' | sed 's/|$//')
for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep -E "$uuids" | cut -d, -f1); do
  [ "$(ps -o user= -p $p 2>/dev/null | tr -d ' ')" = "$USER" ] && [ "$p" != "$me" ] && kill -9 $p 2>/dev/null
done
for i in $(seq 1 60); do
  busy=$(nvidia-smi -i $gpus --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)
  [ "$busy" -eq 0 ] && exit 0; sleep 3
done
echo "GPUs $gpus still busy" >&2; exit 1
