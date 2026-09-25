#!/bin/bash
# usage: dl_loop.sh <repo_spec> <cache_dir_name> <log_name> <allow patterns...>
spec=$1; name=$2; log=$3; shift 3
cd /home/jackb/workspace/anatomiae
b=/data/jackb/krasis/huggingface/hub/models--$name/blobs
for i in $(seq 1 1000); do
  pre=$(find $b -name '*.incomplete' -printf "%s\n" 2>/dev/null | awk '{s+=$1} END {print s+0}')
  HF_HUB_DISABLE_XET=1 uv run python scripts/bulk_download.py "$spec" --allow-patterns "$@" > artifacts/logs/dl/$log.attempt.log 2>&1
  post=$(find $b -name '*.incomplete' -printf "%s\n" 2>/dev/null | awk '{s+=$1} END {print s+0}')
  echo "{\"event\":\"http_attempt\",\"repo\":\"$name\",\"attempt\":$i,\"ts\":$(date +%s),\"partial_bytes_before\":$pre,\"partial_bytes_after\":$post}" >> artifacts/logs/download_restart_provenance.jsonl
  if grep -q '"event": "done"' artifacts/logs/dl/$log.attempt.log; then echo "DONE $name after $i attempts $(date)" >> artifacts/logs/dl/$log.status; exit 0; fi
  sleep 60
done
echo "GAVE UP $name $(date)" >> artifacts/logs/dl/$log.status
