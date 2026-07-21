"""mk_docu/images_ori/<문서ID>/*.jpg 에 대해 4가지 디스큐 각도 추정 방법(A1/A2/B1/C1)을
비교하는 진단용 스크립트. image_process.md ## 1.1 디스큐 참고.

방법별로 5단계 디버그 이미지를 mk_docu/images_step1_skew/method_.../<페이지파일명>/ 에 저장하고,
최종 각도는 공통 angles.csv에 누적 기록한다.

angle_deg는 "이 값을 그대로 rotate_image(img, angle_deg)에 넣으면 기울기가 보정된다"는
값으로 정의한다 (측정각을 구해서 부호를 뒤집는 방식은 쓰지 않음).

실행 (WSL, conda env 'ocr'):
    /home/kk4ever/anaconda3/envs/ocr/bin/python3 skew_detect.py --doc_id 00939648
    /home/kk4ever/anaconda3/envs/ocr/bin/python3 skew_detect.py --methods A1,C1 --doc_id 00939648
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_KEYS = {
    "A1": "method_A1_hough_line_transform",
    "A2": "method_A2_minimum_area_rectangle",
    "B1": "method_B1_easyocr_bounding_box",
    "C1": "method_C1_projection_profile",
}

CSV_FIELDS = ["doc_id", "page_file", "method", "angle_deg", "angle_min",
              "angle_max", "angle_mean", "n_samples", "seconds", "note"]


# ---------------- 공통 유틸 ----------------

def rotate_image(img, angle_deg, border_value=(255, 255, 255)):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2 - center[0]
    M[1, 2] += new_h / 2 - center[1]
    return cv2.warpAffine(img, M, (new_w, new_h), borderValue=border_value)


def edge_angle_deg(x1, y1, x2, y2):
    if x2 < x1:
        x1, y1, x2, y2 = x2, y2, x1, y1
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def normalize_minarearect_angle(rect):
    (_, _), (w, h), angle = rect
    if w < h:
        angle += 90
    if angle > 45:
        angle -= 90
    elif angle <= -45:
        angle += 90
    return float(angle)


def draw_grid(img, spacing=None):
    out = img.copy()
    h, w = out.shape[:2]
    spacing = spacing or max(20, w // 30)
    color = (180, 180, 180)
    for x in range(0, w, spacing):
        cv2.line(out, (x, 0), (x, h), color, 1)
    for y in range(0, h, spacing):
        cv2.line(out, (0, y), (w, y), color, 1)
    return out


def hconcat_resize(img1, img2, target_h=1400):
    """두 이미지를 같은 높이로 맞춰서 나란히(hconcat) 붙인다 (보정 전/후 비교용)."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    r1 = cv2.resize(img1, (max(1, round(w1 * target_h / h1)), target_h))
    r2 = cv2.resize(img2, (max(1, round(w2 * target_h / h2)), target_h))
    divider = np.full((target_h, 6, 3), (0, 0, 0), dtype=np.uint8)
    return cv2.hconcat([r1, divider, r2])


def stats_lines(mean, mn, mx, n):
    return [f"mean: {mean:.2f} deg", f"min : {mn:.2f} deg",
            f"max : {mx:.2f} deg", f"n   : {n}"]


def draw_stats_box(img, lines):
    out = img.copy()
    h, w = out.shape[:2]
    scale = max(w, h) / 1500
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(scale * 2))
    line_h = int(38 * scale)
    box_w = int(430 * scale)
    box_h = line_h * len(lines) + int(20 * scale)
    x0, y0 = w - box_w - 10, h - box_h - 10
    cv2.rectangle(out, (x0, y0), (w - 10, h - 10), (255, 255, 255), -1)
    cv2.rectangle(out, (x0, y0), (w - 10, h - 10), (0, 0, 0), 2)
    for i, line in enumerate(lines):
        y = y0 + int(28 * scale) + i * line_h
        cv2.putText(out, line, (x0 + int(15 * scale), y), font, scale * 0.7,
                    (0, 0, 0), thickness, cv2.LINE_AA)
    return out


def save_step(out_dir: Path, filename: str, img):
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / filename), img)


def filter_outliers(angles, hard_range=20.0, mad_floor=2.0, mad_k=3.0):
    """표 테두리 같은 수직선을 먼저 하드컷(±hard_range)하고, 남은 값에 median+MAD 필터."""
    angles = np.asarray(angles, dtype=float)
    if len(angles) == 0:
        return np.array([], dtype=bool)
    hard_keep = np.abs(angles) <= hard_range
    if not hard_keep.any():
        return hard_keep
    survivors = angles[hard_keep]
    median = np.median(survivors)
    mad = np.median(np.abs(survivors - median))
    thresh = max(mad_floor, mad * mad_k)
    keep = hard_keep.copy()
    keep[hard_keep] = np.abs(survivors - median) <= thresh
    return keep


def dominant_cluster_mask(angles, hard_range=20.0, bin_width=0.5):
    """median 기반이 아니라, 실제로 박스가 가장 많이 몰린 각도 구간(최빈 bin + 인접 bin)을
    다수 클러스터로 채택. median+MAD는 분포가 이봉(bimodal)이면 소수 쪽을 다수로 오인해
    엉뚱한 값을 "아웃라이어 아님"으로 통과시킬 수 있어서, 실제 빈도수 기준으로 직접 고른다."""
    angles = np.asarray(angles, dtype=float)
    if len(angles) == 0:
        return np.array([], dtype=bool)
    hard_keep = np.abs(angles) <= hard_range
    if not hard_keep.any():
        return hard_keep
    bins = np.round(angles / bin_width).astype(int)
    values, counts = np.unique(bins[hard_keep], return_counts=True)
    best_bin = values[np.argmax(counts)]
    keep = hard_keep.copy()
    keep[hard_keep] = np.abs(bins[hard_keep] - best_bin) <= 1  # 최빈 구간 + 양옆 인접 구간
    return keep


def append_csv_row(csv_path: Path, row: dict):
    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _selftest_rotation_convention():
    canvas = np.zeros((300, 500), dtype=np.uint8)
    cv2.line(canvas, (50, 100), (450, 156), 255, 5)
    angle = normalize_minarearect_angle(cv2.minAreaRect(cv2.findNonZero(canvas)))

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    rotated = rotate_image(canvas_bgr, angle, border_value=(0, 0, 0))
    rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, rotated_bin = cv2.threshold(rotated_gray, 127, 255, cv2.THRESH_BINARY)
    pts_r = cv2.findNonZero(rotated_bin)
    angle_r = normalize_minarearect_angle(cv2.minAreaRect(pts_r)) if pts_r is not None else 999.0

    ok = abs(angle_r) < 1.0
    print(f"[selftest] 합성 기울기={angle:.2f}deg -> 보정 후 재측정={angle_r:.2f}deg -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("[selftest] 회전 부호 규칙이 깨졌을 수 있습니다. rotate_image / minAreaRect 정규화를 확인하세요.")
    return ok


# ---------------- 방법별 구현 ----------------

def method_a1_hough(img, out_dir: Path):
    step0_grid = draw_grid(img)
    save_step(out_dir, "step_0_original_grid.jpg", step0_grid)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    save_step(out_dir, "step_1_edges.jpg", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=150,
                             minLineLength=img.shape[1] // 8, maxLineGap=20)

    all_vis = img.copy()
    raw_angles, segments = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(all_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            raw_angles.append(edge_angle_deg(x1, y1, x2, y2))
            segments.append((x1, y1, x2, y2))
    save_step(out_dir, "step_2_lines_all.jpg", all_vis)

    keep_mask = filter_outliers(raw_angles)
    kept_angles = [a for a, k in zip(raw_angles, keep_mask) if k]
    kept_segments = [s for s, k in zip(segments, keep_mask) if k]

    filt_vis = img.copy()
    for x1, y1, x2, y2 in kept_segments:
        cv2.line(filt_vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
    if kept_angles:
        mean_a, min_a, max_a = float(np.mean(kept_angles)), float(np.min(kept_angles)), float(np.max(kept_angles))
        angle_final = float(np.median(kept_angles))
    else:
        mean_a = min_a = max_a = angle_final = 0.0
    filt_vis = draw_stats_box(filt_vis, stats_lines(mean_a, min_a, max_a, len(kept_angles)))
    save_step(out_dir, "step_3_lines_filtered_angle.jpg", filt_vis)

    rotated = rotate_image(img, angle_final)
    step4_grid = draw_grid(rotated)
    save_step(out_dir, "step_4_rotated_grid.jpg", step4_grid)

    gray_r = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    edges_r = cv2.Canny(gray_r, 50, 150, apertureSize=3)
    lines_r = cv2.HoughLinesP(edges_r, 1, np.pi / 360, threshold=150,
                               minLineLength=rotated.shape[1] // 8, maxLineGap=20)
    reverify_vis = rotated.copy()
    if lines_r is not None:
        for x1, y1, x2, y2 in lines_r.reshape(-1, 4):
            cv2.line(reverify_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    save_step(out_dir, "step_5_reverify_lines.jpg", reverify_vis)

    save_step(out_dir, "step_6_before_after_grid.jpg", hconcat_resize(step0_grid, step4_grid))

    return {
        "angle_deg": angle_final, "angle_min": min_a, "angle_max": max_a,
        "angle_mean": mean_a, "n_samples": len(kept_angles),
        "note": f"HoughLinesP total={len(raw_angles)} kept={len(kept_angles)}",
    }


def method_a2_minarearect(img, out_dir: Path, kernel_w=None, kernel_h=None):
    step0_grid = draw_grid(img)
    save_step(out_dir, "step_0_original_grid.jpg", step0_grid)

    h0, w0 = img.shape[:2]
    kw = kernel_w or max(15, round(w0 / 40))
    kh = kernel_h or max(3, round(h0 / 500))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))

    def compute(gray_img):
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        merged = cv2.dilate(binary, kernel, iterations=1)
        n_contours = len(cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        pts = cv2.findNonZero(merged)
        rect = cv2.minAreaRect(pts) if pts is not None else ((0, 0), (1, 1), 0.0)
        return binary, merged, rect, n_contours

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary, merged, rect, n_contours = compute(gray)
    save_step(out_dir, "step_1_binarized.jpg", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
    save_step(out_dir, "step_2_text_mask.jpg", cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR))

    angle = normalize_minarearect_angle(rect)
    box_vis = img.copy()
    box_pts = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(box_vis, [box_pts], 0, (0, 200, 0), 3)
    box_vis = draw_stats_box(box_vis, stats_lines(angle, angle, angle, 1))
    save_step(out_dir, "step_3_minarearect_angle.jpg", box_vis)

    rotated = rotate_image(img, angle)
    step4_grid = draw_grid(rotated)
    save_step(out_dir, "step_4_rotated_grid.jpg", step4_grid)

    gray_r = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, _, rect_r, _ = compute(gray_r)
    angle_r = normalize_minarearect_angle(rect_r)
    reverify_vis = rotated.copy()
    box_pts_r = cv2.boxPoints(rect_r).astype(int)
    cv2.drawContours(reverify_vis, [box_pts_r], 0, (0, 200, 0), 3)
    reverify_vis = draw_stats_box(reverify_vis, stats_lines(angle_r, angle_r, angle_r, 1))
    save_step(out_dir, "step_5_reverify_minarearect.jpg", reverify_vis)

    save_step(out_dir, "step_6_before_after_grid.jpg", hconcat_resize(step0_grid, step4_grid))

    return {
        "angle_deg": angle, "angle_min": angle, "angle_max": angle,
        "angle_mean": angle, "n_samples": n_contours,
        "note": f"blob_contours={n_contours} kernel=({kw},{kh})",
    }


def method_b1_easyocr(img, out_dir: Path, reader):
    step0_grid = draw_grid(img)
    save_step(out_dir, "step_0_original_grid.jpg", step0_grid)

    def detect_boxes(bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h_list, f_list = reader.detect(rgb, slope_ths=0.01)
        return (h_list[0] if h_list else []), (f_list[0] if f_list else [])

    def quad_angle(quad):
        pts = np.array(quad, dtype=float)
        best_dx, best_angle = -1.0, 0.0
        for i in range(4):
            p1, p2 = pts[i], pts[(i + 1) % 4]
            dx = abs(p2[0] - p1[0])
            if dx > best_dx:
                best_dx = dx
                best_angle = edge_angle_deg(p1[0], p1[1], p2[0], p2[1])
        return best_angle

    horizontal_list, free_list = detect_boxes(img)
    free_angles = [quad_angle(q) for q in free_list]
    keep_mask = dominant_cluster_mask(free_angles)

    all_vis = img.copy()
    for xmin, xmax, ymin, ymax in horizontal_list:
        cv2.rectangle(all_vis, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (200, 120, 0), 2)
    for quad, keep in zip(free_list, keep_mask):
        pts = np.array(quad, dtype=int)
        color = (0, 200, 0) if keep else (0, 0, 255)
        cv2.polylines(all_vis, [pts], True, color, 2)
    save_step(out_dir, "step_1_boxes_all.jpg", all_vis)
    # 색 구분(초록=유지/빨강=아웃라이어/파랑=축정렬-각도정보없음)이 곧 아웃라이어 표시라 step_1과 동일 이미지 재사용
    save_step(out_dir, "step_2_boxes_outlier.jpg", all_vis)

    kept_angles = [a for a, k in zip(free_angles, keep_mask) if k]
    kept_quads = [q for q, k in zip(free_list, keep_mask) if k]

    angle_vis = img.copy()
    for quad, angle in zip(kept_quads, kept_angles):
        pts = np.array(quad, dtype=int)
        cv2.polylines(angle_vis, [pts], True, (0, 200, 0), 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(angle_vis, f"{angle:.1f}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2, cv2.LINE_AA)
    if kept_angles:
        mean_a, min_a, max_a = float(np.mean(kept_angles)), float(np.min(kept_angles)), float(np.max(kept_angles))
        angle_final = float(np.median(kept_angles))
    else:
        mean_a = min_a = max_a = angle_final = 0.0
    angle_vis = draw_stats_box(angle_vis, stats_lines(mean_a, min_a, max_a, len(kept_angles)))
    save_step(out_dir, "step_3_boxes_angle.jpg", angle_vis)

    rotated = rotate_image(img, angle_final)
    step4_grid = draw_grid(rotated)
    save_step(out_dir, "step_4_rotated_grid.jpg", step4_grid)

    h_list_r, f_list_r = detect_boxes(rotated)
    reverify_vis = rotated.copy()
    for xmin, xmax, ymin, ymax in h_list_r:
        cv2.rectangle(reverify_vis, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (200, 120, 0), 2)
    for quad in f_list_r:
        cv2.polylines(reverify_vis, [np.array(quad, dtype=int)], True, (0, 0, 255), 2)
    save_step(out_dir, "step_5_reverify_boxes.jpg", reverify_vis)

    save_step(out_dir, "step_6_before_after_grid.jpg", hconcat_resize(step0_grid, step4_grid))

    return {
        "angle_deg": angle_final, "angle_min": min_a, "angle_max": max_a,
        "angle_mean": mean_a, "n_samples": len(kept_angles),
        "note": (f"total={len(horizontal_list) + len(free_list)} horiz={len(horizontal_list)} "
                 f"free={len(free_list)} kept={len(kept_angles)} (dominant_cluster,bin=0.5deg)"),
    }


def method_c1_projection(img, out_dir: Path):
    def binarize(bgr_img):
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def score_at(binary_img, angle):
        rotated = rotate_image(binary_img, angle, border_value=0)
        row_sum = rotated.sum(axis=1).astype(np.float64)
        return float(np.var(row_sum))

    def sweep(binary_img, lo, hi, step):
        angles = np.arange(lo, hi + 1e-9, step)
        scores = [score_at(binary_img, float(a)) for a in angles]
        best_idx = int(np.argmax(scores))
        return angles, scores, float(angles[best_idx])

    out_dir.mkdir(parents=True, exist_ok=True)
    step0_grid = draw_grid(img)
    save_step(out_dir, "step_0_original_grid.jpg", step0_grid)

    h0, w0 = img.shape[:2]
    scale = min(1.0, 800 / w0)
    small_binary = cv2.resize(binarize(img), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    full_binary = binarize(img)

    angles_c, scores_c, best_c = sweep(small_binary, -15.0, 15.0, 0.5)
    angles_f, scores_f, best_f = sweep(full_binary, best_c - 1.0, best_c + 1.0, 0.05)
    angle_final = best_f
    n_samples = len(angles_c) + len(angles_f)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(angles_c, scores_c, label="coarse @small (-15~15, 0.5deg)")
    ax.axvline(angle_final, color="red", linestyle="--", label=f"best={angle_final:.2f} deg")
    ax.set_xlabel("angle (deg)")
    ax.set_ylabel("row-sum variance")
    ax.set_title("Angle vs Projection-profile Sharpness (coarse pass)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_dir / "step_1_angle_score_curve.jpg"))
    plt.close(fig)

    rotated_bin_full = rotate_image(full_binary, angle_final, border_value=0)
    rotated_color_full = rotate_image(img, angle_final)
    row_sum = rotated_bin_full.sum(axis=1)
    disp_w, disp_h = int(rotated_color_full.shape[1] * scale), int(rotated_color_full.shape[0] * scale)
    display_color = cv2.resize(rotated_color_full, (disp_w, disp_h))

    fig, axes = plt.subplots(1, 2, figsize=(8, 8), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].imshow(cv2.cvtColor(display_color, cv2.COLOR_BGR2RGB))
    axes[0].axis("off")
    axes[0].set_title(f"rotated @ {angle_final:.2f} deg")
    axes[1].plot(row_sum, np.arange(len(row_sum)))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("pixel sum")
    axes[1].set_title("row profile (full-res)")
    fig.tight_layout()
    fig.savefig(str(out_dir / "step_2_best_profile.jpg"))
    plt.close(fig)

    angle_vis = draw_stats_box(img.copy(), stats_lines(angle_final, angle_final, angle_final, n_samples))
    save_step(out_dir, "step_3_best_angle_annotated.jpg", angle_vis)
    step4_grid = draw_grid(rotated_color_full)
    save_step(out_dir, "step_4_rotated_grid.jpg", step4_grid)

    rotated_full_binary_re = binarize(rotated_color_full)
    small_r = cv2.resize(rotated_full_binary_re, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    angles_re, scores_re, best_re = sweep(small_r, -2.0, 2.0, 0.1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(angles_re, scores_re)
    ax.axvline(best_re, color="red", linestyle="--", label=f"best={best_re:.2f} deg")
    ax.set_xlabel("angle (deg)")
    ax.set_ylabel("row-sum variance")
    ax.set_title("Reverify (rotated image, narrow sweep)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_dir / "step_5_reverify_profile.jpg"))
    plt.close(fig)

    save_step(out_dir, "step_6_before_after_grid.jpg", hconcat_resize(step0_grid, step4_grid))

    return {
        "angle_deg": angle_final, "angle_min": angle_final, "angle_max": angle_final,
        "angle_mean": angle_final, "n_samples": n_samples,
        "note": f"coarse@small=-15~15/0.5 fine@full={best_c:.1f}+-1/0.05",
    }


# ---------------- 실행 ----------------

def process_page(image_path: Path, doc_id: str, reader, output_root: Path, csv_path: Path,
                  methods, a2_kernel):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  이미지 읽기 실패: {image_path}")
        return
    page_stem = image_path.stem

    dispatch = {
        "A1": lambda: method_a1_hough(img, output_root / METHOD_KEYS["A1"] / page_stem),
        "A2": lambda: method_a2_minarearect(img, output_root / METHOD_KEYS["A2"] / page_stem,
                                             kernel_w=a2_kernel[0], kernel_h=a2_kernel[1]),
        "B1": lambda: method_b1_easyocr(img, output_root / METHOD_KEYS["B1"] / page_stem, reader),
        "C1": lambda: method_c1_projection(img, output_root / METHOD_KEYS["C1"] / page_stem),
    }

    for key in methods:
        if key not in dispatch:
            print(f"  알 수 없는 메소드: {key}")
            continue
        print(f"  [{key}] 처리중...")
        t0 = time.perf_counter()
        result = dispatch[key]()
        elapsed = time.perf_counter() - t0
        append_csv_row(csv_path, {
            "doc_id": doc_id,
            "page_file": image_path.name,
            "method": METHOD_KEYS[key],
            "seconds": round(elapsed, 2),
            **result,
        })
        print(f"  [{key}] angle_deg={result['angle_deg']:.2f} (n={result['n_samples']}) [{elapsed:.2f}s]")


def main():
    parser = argparse.ArgumentParser(description="이미지 디스큐 각도 검출 진단 스크립트 (A1/A2/B1/C1 비교)")
    parser.add_argument("--input_dir", default=str(Path(__file__).resolve().parent.parent / "images_ori"))
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent.parent / "images_step1_skew"))
    parser.add_argument("--doc_id", default=None, help="특정 문서ID 폴더만 처리 (폴더명 기준, 예: 00939648)")
    parser.add_argument("--methods", default="A1,A2,B1,C1", help="쉼표로 구분된 메소드 키 (A1,A2,B1,C1)")
    parser.add_argument("--a2_kernel_w", type=int, default=None)
    parser.add_argument("--a2_kernel_h", type=int, default=None)
    args = parser.parse_args()

    if not _selftest_rotation_convention():
        print("자체 검증 실패 - 회전 부호 규칙을 확인하세요. 중단합니다.")
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "angles.csv"
    methods = [m.strip().upper() for m in args.methods.split(",") if m.strip()]

    doc_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if args.doc_id:
        doc_dirs = [d for d in doc_dirs if d.name == args.doc_id]
        if not doc_dirs:
            print(f"'{args.doc_id}' 폴더를 찾을 수 없습니다: {input_dir}")
            return

    reader = None
    if "B1" in methods:
        import easyocr
        print("EasyOCR 모델 로딩중...")
        t_load = time.perf_counter()
        reader = easyocr.Reader(["ko", "en"], gpu=True)
        print(f"EasyOCR 모델 로딩 완료 [{time.perf_counter() - t_load:.2f}s]")

    t_total = time.perf_counter()
    n_pages = 0
    for doc_dir in doc_dirs:
        image_paths = sorted(p for p in doc_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for image_path in image_paths:
            print(f"=== {doc_dir.name}/{image_path.name} ===")
            t_page = time.perf_counter()
            process_page(image_path, doc_dir.name, reader, output_dir, csv_path, methods,
                         (args.a2_kernel_w, args.a2_kernel_h))
            n_pages += 1
            print(f"  (페이지 처리 시간: {time.perf_counter() - t_page:.2f}s)")

    total_elapsed = time.perf_counter() - t_total
    avg = total_elapsed / n_pages if n_pages else 0.0
    print(f"\n총 {n_pages}페이지 처리 완료 - 전체 {total_elapsed:.1f}s (페이지당 평균 {avg:.2f}s)")


if __name__ == "__main__":
    main()
