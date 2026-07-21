"""mk_docu/pdf/<문서ID>.pdf 를 페이지별 이미지로 분할해 mk_docu/images_split/<문서ID>/ 에 저장.

images_ori -> pdf (images_to_pdf.py) 로 합친 걸 다시 이미지로 잘 쪼갤 수 있는지
테스트하기 위한 스크립트. 파일명 규칙은 원본과 동일하게 <문서ID>_%04d.jpg 로 맞춰서
images_split/<문서ID> 폴더를 images_ori/<문서ID> 폴더와 바로 비교할 수 있게 함.

실행 (WSL, conda env 'ocr' — conda run은 이 환경에서 PATH 오염으로 깨지므로 직접 바이너리 호출):
    /home/kk4ever/anaconda3/envs/ocr/bin/python3 pdf_to_images.py
    /home/kk4ever/anaconda3/envs/ocr/bin/python3 pdf_to_images.py --input_dir ../pdf --output_dir ../images_split --dpi 300
"""
import argparse
from pathlib import Path

import fitz  # PyMuPDF


def split_pdf(pdf_path: Path, output_dir: Path, dpi: int) -> int:
    doc_id = pdf_path.stem
    doc_out_dir = output_dir / doc_id
    doc_out_dir.mkdir(parents=True, exist_ok=True)

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    doc = fitz.open(pdf_path)
    try:
        for page_index in range(len(doc)):
            pixmap = doc[page_index].get_pixmap(matrix=matrix)
            out_path = doc_out_dir / f"{doc_id}_{page_index + 1:04d}.jpg"
            pixmap.save(out_path)
        return len(doc)
    finally:
        doc.close()


def main():
    parser = argparse.ArgumentParser(description="PDF를 페이지별 이미지로 분할")
    parser.add_argument("--input_dir", default=str(Path(__file__).resolve().parent.parent / "pdf"))
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent.parent / "images_split"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    pdf_paths = sorted(input_dir.glob("*.pdf"))
    print(f"대상 PDF: {len(pdf_paths)}개")

    for i, pdf_path in enumerate(pdf_paths, 1):
        n_pages = split_pdf(pdf_path, output_dir, args.dpi)
        print(f"[{i}/{len(pdf_paths)}] {pdf_path.name}: {n_pages}페이지 -> {output_dir / pdf_path.stem}")


if __name__ == "__main__":
    main()
