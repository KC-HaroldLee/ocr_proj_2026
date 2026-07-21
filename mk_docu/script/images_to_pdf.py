"""mk_docu/images/<문서ID>/*.jpg 를 문서ID별로 하나의 PDF로 묶어 mk_docu/pdf/<문서ID>.pdf 로 저장.

사용법:
    python images_to_pdf.py
    python images_to_pdf.py --input_dir ../images --output_dir ../pdf
"""
import argparse
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def build_pdf(doc_dir: Path, output_path: Path) -> int:
    image_paths = sorted(p for p in doc_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        return 0

    opened = [Image.open(p) for p in image_paths]
    # DPI를 보존해야 PDF 페이지 크기(포인트 단위)가 원본 실제 크기로 인코딩됨.
    # 안 하면 PIL이 72dpi로 가정해버려서, 나중에 다른 dpi로 재래스터화할 때 이미지가 부풀려짐.
    dpi = opened[0].info.get("dpi", (300, 300))[0] or 300
    images = [img.convert("RGB") for img in opened]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path, save_all=True, append_images=images[1:], resolution=dpi)
    return len(images)


def main():
    parser = argparse.ArgumentParser(description="문서ID별 이미지 폴더를 PDF로 변환")
    parser.add_argument("--input_dir", default=str(Path(__file__).resolve().parent.parent / "images_ori"))
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent.parent / "pdf"))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    doc_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    print(f"대상 문서 폴더: {len(doc_dirs)}개")

    for i, doc_dir in enumerate(doc_dirs, 1):
        output_path = output_dir / f"{doc_dir.name}.pdf"
        n_pages = build_pdf(doc_dir, output_path)
        if n_pages:
            print(f"[{i}/{len(doc_dirs)}] {doc_dir.name}: {n_pages}페이지 -> {output_path}")
        else:
            print(f"[{i}/{len(doc_dirs)}] {doc_dir.name}: 이미지 없음, 건너뜀")


if __name__ == "__main__":
    main()
