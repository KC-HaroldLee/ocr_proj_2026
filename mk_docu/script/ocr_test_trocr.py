"""C1(Projection Profile)로 보정한 이미지에 EasyOCR(detection)+ddobokki/ko-trocr(recognition)를
돌려서 OCR 파이프라인이 실제로 잘 동작하는지 확인하는 사전검증 스크립트.
모델 참고: https://huggingface.co/ddobokki/ko-trocr

mk_docu/images_step1_skew_only_C1/angles.csv 에서 문서·페이지별 C1 보정각을 읽어와서,
mk_docu/images_split 의 원본 이미지에 그 각도로 회전 보정을 적용한 뒤 OCR을 수행한다.
지금은 "모델이 잘 동작하는지"만 확인하는 단계라 워터마크 제외, 표/그림 분류, 캡션 입력 같은
검수 UX 로직은 의도적으로 생략함 (README 6.1절 — DB/검수 화면 설계 시 반영 예정).

결과는 mk_docu/images_step2_ocr_C1/<문서ID>/<페이지파일명>/ 에 저장:
    step_1_detect_boxes.jpg       - EasyOCR이 찾은 박스
    step_2_recognition_result.jpg - 박스 + ko-trocr 인식 텍스트 라벨
공통 results.csv에 박스별 인식 결과와 처리 시간(detect/recognize)을 누적 기록한다.
박스 개수가 recognize 총 시간에 그대로 비례하므로, 나중에 EasyOCR 검출이 부실해서
강제 업스케일링을 도입할 때 이 시간 기록이 비교 기준이 된다.

실행 (WSL, conda env 'ocr'), 먼저 run_skew_detect_c1_only.sh로 C1 각도부터 만들어둘 것:
    /home/kk4ever/anaconda3/envs/ocr/bin/python3 ocr_test_trocr.py --doc_id 00939648
"""
import argparse
import csv
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from skew_detect import rotate_image

CSV_FIELDS = ["doc_id", "page_file", "box_index", "bbox", "recognized_text",
              "n_boxes", "detect_seconds", "recognize_seconds"]

KOREAN_FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


def load_c1_angles(csv_path: Path):
    angles = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["method"] != "method_C1_projection_profile":
                continue
            angles[(row["doc_id"], row["page_file"])] = float(row["angle_deg"])
    return angles


def _clip_box(xmin, xmax, ymin, ymax, w, h, margin):
    x1 = max(0, int(xmin) - margin)
    y1 = max(0, int(ymin) - margin)
    x2 = min(w, int(xmax) + margin)
    y2 = min(h, int(ymax) + margin)
    return (x1, y1, x2, y2)


def boxes_from_detect(horizontal_list, free_list, w, h, margin=4):
    """이미 C1로 페이지 전체를 보정했으므로, free_list(회전 quad)도 축정렬
    바운딩박스로 단순화해서 크롭한다 (사전검증 단계라 정밀 워프는 생략)."""
    boxes = []
    for xmin, xmax, ymin, ymax in horizontal_list:
        boxes.append(_clip_box(xmin, xmax, ymin, ymax, w, h, margin))
    for quad in free_list:
        pts = np.array(quad)
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        boxes.append(_clip_box(xmin, xmax, ymin, ymax, w, h, margin))
    return boxes


def draw_boxes(img, boxes):
    out = img.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return out


def draw_boxes_with_text(img, boxes, texts):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(KOREAN_FONT_PATH, 22)
    except OSError:
        font = None
    for (x1, y1, x2, y2), text in zip(boxes, texts):
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        if font is not None and text:
            label_w = max(10, 12 * len(text))
            ty = max(0, y1 - 24)
            draw.rectangle([x1, ty, x1 + label_w, ty + 24], fill=(255, 255, 0))
            draw.text((x1 + 2, ty), text, fill=(0, 0, 0), font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def recognize_boxes(img, boxes, processor, tokenizer, model, device):
    import torch
    texts, seconds = [], []
    for x1, y1, x2, y2 in boxes:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            texts.append("")
            seconds.append(0.0)
            continue
        t0 = time.perf_counter()
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pixel_values = processor(pil_crop, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=64)
        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text = unicodedata.normalize("NFC", text)
        seconds.append(time.perf_counter() - t0)
        texts.append(text)
    return texts, seconds


def append_csv_row(csv_path: Path, row: dict):
    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def process_page(image_path, doc_id, angle, reader, processor, tokenizer, model, device,
                  output_root, csv_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  이미지 읽기 실패: {image_path}")
        return
    corrected = rotate_image(img, angle)
    h, w = corrected.shape[:2]

    t0 = time.perf_counter()
    rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
    horizontal_list, free_list = reader.detect(rgb)
    horizontal_list = horizontal_list[0] if horizontal_list else []
    free_list = free_list[0] if free_list else []
    detect_seconds = time.perf_counter() - t0

    boxes = boxes_from_detect(horizontal_list, free_list, w, h)
    page_stem = image_path.stem
    out_dir = output_root / doc_id / page_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "step_1_detect_boxes.jpg"), draw_boxes(corrected, boxes))

    texts, recognize_seconds_list = recognize_boxes(corrected, boxes, processor, tokenizer, model, device)
    cv2.imwrite(str(out_dir / "step_2_recognition_result.jpg"),
                draw_boxes_with_text(corrected, boxes, texts))

    for i, (box, text, rsec) in enumerate(zip(boxes, texts, recognize_seconds_list)):
        append_csv_row(csv_path, {
            "doc_id": doc_id, "page_file": image_path.name, "box_index": i,
            "bbox": f"{box[0]},{box[1]},{box[2]},{box[3]}", "recognized_text": text,
            "n_boxes": len(boxes), "detect_seconds": round(detect_seconds, 3),
            "recognize_seconds": round(rsec, 3),
        })

    total_recognize = sum(recognize_seconds_list)
    if boxes:
        print(f"  박스 {len(boxes)}개 - detect {detect_seconds:.2f}s, recognize 합계 {total_recognize:.2f}s "
              f"(박스당 평균 {total_recognize / len(boxes):.2f}s)")
    else:
        print(f"  박스 0개 - detect {detect_seconds:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="EasyOCR+ko-trocr 사전 검증 (C1 보정 이미지 기준)")
    parser.add_argument("--split_dir", default=str(Path(__file__).resolve().parent.parent / "images_split"))
    parser.add_argument("--skew_csv", default=str(Path(__file__).resolve().parent.parent
                                                    / "images_step1_skew_only_C1" / "angles.csv"))
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent.parent / "images_step2_ocr_C1"))
    parser.add_argument("--doc_id", default=None, help="특정 문서ID 폴더만 처리 (폴더명 기준)")
    args = parser.parse_args()

    skew_csv = Path(args.skew_csv)
    if not skew_csv.exists():
        print(f"{skew_csv} 가 없습니다. 먼저 run_skew_detect_c1_only.sh를 돌려주세요.")
        return
    angles = load_c1_angles(skew_csv)

    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "results.csv"

    import torch
    import easyocr
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    print("EasyOCR 모델 로딩중...")
    reader = easyocr.Reader(["ko", "en"], gpu=(device == "cuda"))
    print("ko-trocr 모델 로딩중...")
    processor = TrOCRProcessor.from_pretrained("ddobokki/ko-trocr")
    tokenizer = AutoTokenizer.from_pretrained("ddobokki/ko-trocr")
    model = VisionEncoderDecoderModel.from_pretrained("ddobokki/ko-trocr").to(device)
    model.eval()

    doc_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if args.doc_id:
        doc_dirs = [d for d in doc_dirs if d.name == args.doc_id]
        if not doc_dirs:
            print(f"'{args.doc_id}' 폴더를 찾을 수 없습니다: {split_dir}")
            return

    t_total = time.perf_counter()
    n_pages = 0
    for doc_dir in doc_dirs:
        image_paths = sorted(p for p in doc_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for image_path in image_paths:
            key = (doc_dir.name, image_path.name)
            if key not in angles:
                print(f"=== {doc_dir.name}/{image_path.name} === C1 각도 없음, 건너뜀 "
                      f"(run_skew_detect_c1_only.sh 먼저 실행 필요)")
                continue
            angle = angles[key]
            print(f"=== {doc_dir.name}/{image_path.name} (angle={angle:.2f}) ===")
            t_page = time.perf_counter()
            process_page(image_path, doc_dir.name, angle, reader, processor, tokenizer, model, device,
                         output_dir, csv_path)
            n_pages += 1
            print(f"  (페이지 처리 시간: {time.perf_counter() - t_page:.2f}s)")

    total_elapsed = time.perf_counter() - t_total
    avg = total_elapsed / n_pages if n_pages else 0.0
    print(f"\n총 {n_pages}페이지 처리 완료 - 전체 {total_elapsed:.1f}s (페이지당 평균 {avg:.2f}s)")


if __name__ == "__main__":
    main()
