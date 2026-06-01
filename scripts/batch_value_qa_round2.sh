#!/bin/bash
# round2 批量生成：5 个全新 keyword，用改后 prompt 验证 medium_pattern 质量提升。
# 用法：
#   nohup bash scripts/batch_value_qa_round2.sh > logs/value_qa_round2_batch.log 2>&1 &

set -u
cd "$(dirname "$0")/.."

KEYWORDS=(
  "二十四节气"        # 文化元素符合度
  "见义勇为"          # 价值观忠实度 / 行为规范
  "学术不端"          # 行为规范 / 伤害风险
  "算法歧视"          # 公平合规度
  "个人信息保护法"    # 隐私合规度
)

LIMIT=9
MAX_RETRY=3
TOTAL=${#KEYWORDS[@]}

# 记录起始行数，用于事后切出 round2 增量
START_LINES=$(wc -l < data/samples/value_qa/verified.jsonl)
echo "===== round2 batch start $(date '+%F %T') total=$TOTAL ====="
echo "verified.jsonl 起始行数: $START_LINES"

for i in "${!KEYWORDS[@]}"; do
  kw="${KEYWORDS[$i]}"
  idx=$((i+1))
  echo
  echo "===== [$idx/$TOTAL] $kw ====="
  for try in $(seq 1 $MAX_RETRY); do
    out=$(python -m benchmark.cli generate-samples --topic "$kw" --skill-ids value_qa --limit $LIMIT 2>&1 | tail -3)
    echo "[try $try] $out"
    if echo "$out" | grep -qE "verified [1-9]"; then
      break
    fi
    sleep 3
  done
done

END_LINES=$(wc -l < data/samples/value_qa/verified.jsonl)
NEW=$((END_LINES - START_LINES))
echo
echo "===== round2 batch end $(date '+%F %T') ====="
echo "verified.jsonl 结束行数: $END_LINES"
echo "本轮新增样本: $NEW"

# 切出本轮增量 → round2_raw.jsonl
tail -n +$((START_LINES + 1)) data/samples/value_qa/verified.jsonl > data/samples/value_qa/verified.round2_raw.jsonl
echo "round2 raw 已保存: data/samples/value_qa/verified.round2_raw.jsonl ($NEW 条)"

# 简短统计
python -c "
import json
from collections import Counter
with open('data/samples/value_qa/verified.round2_raw.jsonl') as f:
    rows = [json.loads(l) for l in f]
print(f'round2 total = {len(rows)}')
print('维度分布:')
for k, v in Counter(r.get('metadata',{}).get('evaluation_dimension','?') for r in rows).most_common():
    print(f'  {k}: {v}')
print('value_level 分布:')
for k, v in Counter(r.get('metadata',{}).get('value_level','?') for r in rows).most_common():
    print(f'  {k}: {v}')
print('medium_pattern 分布:')
for k, v in Counter(r.get('metadata',{}).get('medium_pattern','-') for r in rows if r.get('metadata',{}).get('value_level')=='medium').most_common():
    print(f'  {k}: {v}')
"
