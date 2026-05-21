#!/usr/bin/env bash
# 批量补量脚本（v2）：在 v1 已产 132 条基础上补到 ≥200 条
# 用法：bash scripts/batch_generate_benchmark_qa_v2.sh
set -u
cd "$(dirname "$0")/.."

topics=(
  苗族 彝族 蒙古族 朝鲜族 傣族 白族
  清明节 端午节 元宵节
  太极拳 中医 瓷器
)

start_ts=$(date +%s)
echo "[batch-v2] start at $(date), ${#topics[@]} topics"
for t in "${topics[@]}"; do
  echo "===== $(date +%H:%M:%S) topic=$t ====="
  python -m benchmark.cli generate-samples \
    --topic "$t" --skill-ids benchmark_qa \
    --task-type document_to_xy --limit 12 \
    2>&1 | tail -20
done
end_ts=$(date +%s)
echo "[batch-v2] done in $((end_ts - start_ts)) seconds"
echo "===== final count ====="
wc -l data/samples/benchmark_qa/verified.jsonl
