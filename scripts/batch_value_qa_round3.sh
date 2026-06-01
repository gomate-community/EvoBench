#!/bin/bash
# Round3: 按 6 个二级维度分别生成，每维度选 10 个关键词。
# 落盘到独立文件: data/samples/value_qa/verified.round3_6dim.jsonl
# 用法: bash scripts/batch_value_qa_round3.sh > logs/value_qa_round3.log 2>&1 &

set -u
cd "$(dirname "$0")/.."

# 利用 data_layout 自动路由：verified → data/samples/value_qa/verified.jsonl
# 批量开始前清空 verified.jsonl（已在外部备份）
VERIFIED="data/samples/value_qa/verified.jsonl"
REJECTED="data/samples/value_qa/rejected.jsonl"
: > "$VERIFIED"
: > "$REJECTED"

LIMIT=9
MAX_RETRY=3

# ═══════════════ 6 维度 × 10 keyword ═══════════════

declare -a DIM_NAMES=(
  "文化元素符合度"
  "行为社会规范符合度"
  "价值观忠实度"
  "公平合规度"
  "伤害风险合规度"
  "隐私合规度"
)

# 文化元素符合度 (选 10/15)
declare -a KW_0=("端午节" "中秋节" "元宵节" "腊八节" "冬至" "编钟" "故宫" "汉服" "京剧" "剪纸")

# 行为社会规范符合度 (选 10/12)
declare -a KW_1=("孝道" "礼记" "师道尊严" "尊师重教" "婚礼习俗" "丧葬文化" "宗族制度" "医德" "尊老爱幼" "公共场所礼仪")

# 价值观忠实度 (选 10/12)
declare -a KW_2=("爱国主义" "社会主义核心价值观" "雷锋精神" "长征精神" "五四运动" "自强不息" "厚德载物" "天下为公" "井冈山精神" "两弹一星")

# 公平合规度 (选 10/12)
declare -a KW_3=("妇女权益保障法" "反就业歧视" "同工同酬" "老年人权益保障法" "残疾人权益保障法" "无障碍环境建设法" "乡村振兴" "劳动法" "平等就业" "教育公平")

# 伤害风险合规度 (选 10/12)
declare -a KW_4=("未成年人保护法" "反家庭暴力法" "安全生产法" "食品安全法" "药品管理法" "校园霸凌" "网络暴力" "精神卫生法" "消防法" "道路交通安全法")

# 隐私合规度 (选 10/12)
declare -a KW_5=("个人信息保护法" "数据安全法" "网络安全法" "人脸识别" "征信系统" "电子病历" "数据脱敏" "知情同意" "算法推荐" "用户画像")

TOTAL=$((6 * 10))
COUNT=0

echo "===== Round3 batch start $(date '+%F %T') total_calls=$TOTAL ====="
echo "verified → $VERIFIED"
echo "rejected → $REJECTED"
echo

for dim_idx in 0 1 2 3 4 5; do
  dim_name="${DIM_NAMES[$dim_idx]}"
  echo "────────────────────────────────────────"
  echo "维度 [$((dim_idx+1))/6]: $dim_name"
  echo "────────────────────────────────────────"

  # 动态获取对应数组
  eval "kw_arr=(\"\${KW_${dim_idx}[@]}\")"

  for kw in "${kw_arr[@]}"; do
    COUNT=$((COUNT+1))
    echo
    echo "[$COUNT/$TOTAL] [$dim_name] keyword=$kw"
    for try in $(seq 1 $MAX_RETRY); do
      full_out=$(python -m benchmark.cli generate-samples \
        --topic "$kw" \
        --skill-ids value_qa \
        --limit $LIMIT \
        2>&1)
      # 提取包含 "Generated" 或 "verified" 的关键行
      out=$(echo "$full_out" | grep -E "Generated|verified|Error|error" | tail -2)
      [ -z "$out" ] && out=$(echo "$full_out" | tail -1)
      echo "  [try $try] $out"
      if echo "$full_out" | grep -qE "verified [1-9]"; then
        break
      fi
      sleep 3
    done
    sleep 2  # 请求间隔避免限流
  done
done

echo
echo "===== Round3 batch end $(date '+%F %T') ====="
echo

# 统计
python3 -c "
import json
from collections import Counter
from pathlib import Path

rows = [json.loads(l) for l in Path('$VERIFIED').read_text().splitlines() if l.strip()]
dims = Counter(r.get('metadata',{}).get('evaluation_dimension','?') for r in rows)
levels = Counter(r.get('metadata',{}).get('value_level','?') for r in rows)
triplets = len(rows) // 3

print(f'\\n总样本数: {len(rows)} ({triplets} 组三联组)')
print(f'\\n按维度分布:')
for d in sorted(dims.keys()):
    n = dims[d]
    groups = n // 3
    status = '✓' if groups >= 15 else f'⚠️ 不足15组(差{15-groups}组)'
    print(f'  {d:12s}: {n:3d} 条 ({groups:2d} 组) {status}')
print(f'\\n按档位分布:')
for lv in ['high','medium','low']:
    print(f'  {lv}: {levels.get(lv,0)}')
"

# 跑完后重命名为 round3 专属文件，恢复原始备份
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
mv "$VERIFIED" "data/samples/value_qa/verified.round3_6dim_${TIMESTAMP}.jsonl"
echo "已保存: data/samples/value_qa/verified.round3_6dim_${TIMESTAMP}.jsonl"

# 恢复原始 verified.jsonl（从最新的备份）
BACKUP=$(ls -t data/samples/value_qa/verified.pre_round3_*.jsonl 2>/dev/null | head -1)
if [ -n "$BACKUP" ]; then
  cp "$BACKUP" "$VERIFIED"
  echo "已恢复: $VERIFIED (from $BACKUP)"
fi
