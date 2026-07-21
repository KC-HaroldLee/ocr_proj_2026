"""angles.csv(파일/페이지 기준 롱포맷)를 메소드 기준으로 다시 묶어서 보여준다.

1) 메소드별 전체 통계 (건수/평균/표준편차/최소/최대)
2) 메소드별 전체 문서-페이지 각도 목록 (파일이 아니라 메소드가 1차 그룹)

실행 (WSL, conda env 'ocr'):
    /home/kk4ever/anaconda3/envs/ocr/bin/python3 summarize_angles.py
    /home/kk4ever/anaconda3/envs/ocr/bin/python3 summarize_angles.py --csv ../images_step1_skew/angles.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="angles.csv를 메소드 기준으로 요약")
    parser.add_argument("--csv", default=str(Path(__file__).resolve().parent.parent / "images_step1_skew" / "angles.csv"))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"{csv_path} 가 없습니다. skew_detect.py를 먼저 돌려주세요.")
        return
    rows = load_rows(csv_path)
    if not rows:
        print(f"{csv_path} 에 데이터가 없습니다.")
        return

    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    print("=" * 70)
    print("메소드별 전체 통계 (angle_deg / seconds 기준)")
    print("=" * 70)
    print(f"{'method':<38}{'n':>5}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}{'avg_sec':>9}{'total_sec':>11}")
    for method in sorted(by_method):
        angles = np.array([float(r["angle_deg"]) for r in by_method[method]])
        secs = np.array([float(r.get("seconds") or 0) for r in by_method[method]])
        print(f"{method:<38}{len(angles):>5}{angles.mean():>8.2f}{angles.std():>8.2f}"
              f"{angles.min():>8.2f}{angles.max():>8.2f}{secs.mean():>9.2f}{secs.sum():>11.1f}")

    print()
    print("=" * 70)
    print("메소드별 문서-페이지 각도 목록")
    print("=" * 70)
    for method in sorted(by_method):
        print(f"\n--- {method} ---")
        for row in sorted(by_method[method], key=lambda r: (r["doc_id"], r["page_file"])):
            sec = float(row.get("seconds") or 0)
            print(f"  {row['doc_id']}/{row['page_file']:<20} angle_deg={float(row['angle_deg']):>7.2f}"
                  f"  n_samples={row['n_samples']:<5}  {sec:>5.2f}s  note={row['note']}")


if __name__ == "__main__":
    main()
