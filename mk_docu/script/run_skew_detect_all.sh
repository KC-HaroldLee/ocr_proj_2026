#!/bin/bash
# mk_docu/images_split 전체 문서에 대해 skew_detect.py(A1/A2/B1/C1)를 실행한다.
# 결과는 mk_docu/images_step1_skew/ (angles.csv + 방법별 5-step 디버그 이미지)에 누적 저장됨.
#
# 사용법:
#   bash run_skew_detect_all.sh
#   bash run_skew_detect_all.sh --methods A1,C1   (추가 인자는 skew_detect.py로 그대로 전달)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/kk4ever/anaconda3/envs/ocr/bin/python3"
INPUT_DIR="$SCRIPT_DIR/../images_split"
OUTPUT_DIR="$SCRIPT_DIR/../images_step1_skew"
CSV_PATH="$OUTPUT_DIR/angles.csv"

if [ -f "$CSV_PATH" ]; then
    echo "주의: $CSV_PATH 에 기존 결과(00939648 images_ori 테스트 8행 포함)가 있습니다."
    echo "누적 기록되며 덮어쓰지 않으니, 깨끗하게 새로 비교하려면 실행 전 직접 삭제하세요."
fi

START_TS=$(date +%s)

"$PYTHON_BIN" "$SCRIPT_DIR/skew_detect.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    "$@"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
printf "run_skew_detect_all.sh 전체 소요시간: %02d:%02d:%02d\n" \
    $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
