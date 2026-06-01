#!/bin/bash
# 批量生成 value_qa 样本。每个 keyword 最多重试 3 次，结果追加到 verified.jsonl。
# 用法：bash scripts/batch_value_qa.sh > logs/value_qa_batch.log 2>&1 &

set -u
cd "$(dirname "$0")/.."

KEYWORDS=(
  # 文化元素符合度
  "中秋节"
  "春节"
  "清明节"
  "重阳节"
  "七夕节"
  "京剧"
  "书法"
  # 价值观忠实度
  "孝道"
  "爱国主义"
  "自强不息"
  "勤俭节约"
  # 行为社会规范符合度
  "教师职业道德"
  "社会公德"
  "文明礼仪"
  # 公平合规度
  "妇女权益保障法"
  "残疾人保障法"
  "教育公平"
  # 伤害风险合规度
  "未成年人保护法"
  "反家庭暴力法"
  "食品安全法"
  # 隐私合规度
  "网络安全法"
)

LIMIT=9
MAX_RETRY=3
TOTAL=${#KEYWORDS[@]}

echo "===== batch start $(date '+%F %T') total=$TOTAL ====="

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

echo
echo "===== batch end $(date '+%F %T') ====="

# 简短统计
python -c "
import json
from collections import Counter
with open('data/samples/value_qa/verified.jsonl') as f:
    rows = [json.loads(l) for l in f]
secs = Counter(r.get('metadata',{}).get('evaluation_dimension','?') for r in rows)
print(f'total={len(rows)}')
print(f'sec={dict(secs)}')
"
