#!/bin/bash
# mk_docu/images_split 전체 문서에 대해 skew_detect.py를 C1(Projection Profile)만 돌린다.
# image_process.md 1.1절 [2026-07-22 최종 추천 업데이트]에서 C1을 1순위로 채택한 결과,
# 4개 메소드 비교용(run_skew_detect_all.sh)과 분리해서 C1 단독 결과만 따로 보고 싶을 때 사용.
# 결과는 mk_docu/images_step1_skew_only_C1/ (angles.csv + step_0~6 디버그 이미지)에 저장됨.
#
# 사용법:
#   bash run_skew_detect_c1_only.sh
#   bash run_skew_detect_c1_only.sh --doc_id 00939648   (추가 인자는 skew_detect.py로 그대로 전달)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/kk4ever/anaconda3/envs/ocr/bin/python3"
INPUT_DIR="$SCRIPT_DIR/../images_split"
OUTPUT_DIR="$SCRIPT_DIR/../images_step1_skew_only_C1"
CSV_PATH="$OUTPUT_DIR/angles.csv"

if [ -f "$CSV_PATH" ]; then
    echo "주의: $CSV_PATH 에 기존 결과가 있습니다."
    echo "누적 기록되며 덮어쓰지 않으니, 깨끗하게 새로 비교하려면 실행 전 직접 삭제하세요."
fi

START_TS=$(date +%s)

"$PYTHON_BIN" "$SCRIPT_DIR/skew_detect.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --methods C1 \
    "$@"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
printf "run_skew_detect_c1_only.sh 전체 소요시간: %02d:%02d:%02d\n" \
    $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
