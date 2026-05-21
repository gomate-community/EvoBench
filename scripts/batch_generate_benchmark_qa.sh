#!/usr/bin/env bash
# 批量生成 benchmark_qa 样本（方案 A：多 topic 循环）
# 用法：bash scripts/batch_generate_benchmark_qa.sh
set -u
cd "$(dirname "$0")/.."

topics=(
  少数民族 回族 藏族 维吾尔族
  春节 中秋节
  京剧 书法 剪纸
  茶艺 火锅
  儒家 佛教
  长城 故宫 丝绸之路 西湖
)

start_ts=$(date +%s)
echo "[batch] start at $(date), ${#topics[@]} topics"
for t in "${topics[@]}"; do
  echo "===== $(date +%H:%M:%S) topic=$t ====="
  python -m benchmark.cli generate-samples \
    --topic "$t" --skill-ids benchmark_qa \
    --task-type document_to_xy --limit 12 \
    2>&1 | tail -20
done
end_ts=$(date +%s)
echo "[batch] done in $((end_ts - start_ts)) seconds"
echo "===== final count ====="
wc -l data/samples/benchmark_qa/verified.jsonl
