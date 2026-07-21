#!/bin/bash
# mk_docu/images_split 전체 문서에 대해 ocr_test_trocr.py(EasyOCR detect + ko-trocr recognize)를 돌린다.
# 사전조건: mk_docu/images_step1_skew_only_C1/angles.csv 에 전체 문서의 C1 각도가 있어야 함
#          (없으면 run_skew_detect_c1_only.sh 먼저 실행)
# 결과는 mk_docu/images_step2_ocr_C1/ (results.csv + 문서/페이지별 step_1~2 디버그 이미지)에 누적 저장됨.
#
# 사용법:
#   bash run_ocr_test_trocr_all.sh
#   bash run_ocr_test_trocr_all.sh --doc_id 00939648   (추가 인자는 ocr_test_trocr.py로 그대로 전달)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/kk4ever/anaconda3/envs/ocr/bin/python3"
CSV_PATH="$SCRIPT_DIR/../images_step2_ocr_C1/results.csv"

if [ -f "$CSV_PATH" ]; then
    echo "주의: $CSV_PATH 에 기존 결과가 있습니다."
    echo "누적 기록되며 덮어쓰지 않으니, 깨끗하게 새로 비교하려면 실행 전 직접 삭제하세요."
fi

START_TS=$(date +%s)

"$PYTHON_BIN" "$SCRIPT_DIR/ocr_test_trocr.py" "$@"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
printf "run_ocr_test_trocr_all.sh 전체 소요시간: %02d:%02d:%02d\n" \
    $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
