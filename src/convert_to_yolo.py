"""
Конвертация JSON-аннотаций VQA в структуру и формат YOLO.

Ожидаемая структура входа:
data/
├── raw/
│   ├── image_001.jpg
│   └── image_002.png
└── auto_annotated/
    ├── image_001.json
    └── image_002.json

Результат:
data/yolo_format/
├── images/train/
├── images/val/
├── labels/train/
├── labels/val/
└── dataset.yaml

Запуск из корня проекта:
python src/convert_to_yolo.py \
  --images-dir data/raw \
  --annotations-dir data/auto_annotated \
  --output-dir data/yolo_format
"""

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


CLASS_TO_ID = {
    "with_helmet": 0,
    "without_helmet": 1,
}


def xyxy_to_yolo(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Переводит [x1, y1, x2, y2] в YOLO xywh с нормализацией."""
    if len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = map(float, bbox)
    except (TypeError, ValueError):
        return None

    x1 = max(0.0, min(x1, image_width - 1))
    y1 = max(0.0, min(y1, image_height - 1))
    x2 = max(0.0, min(x2, image_width - 1))
    y2 = max(0.0, min(y2, image_height - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    if not all(0.0 < value <= 1.0 for value in [x_center, y_center, width, height]):
        return None

    return x_center, y_center, width, height


def find_image(images_dir: Path, image_name: str, stem: str) -> Path | None:
    """Находит изображение по имени из JSON либо по stem."""
    direct_path = images_dir / image_name
    if direct_path.exists():
        return direct_path

    for extension in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate

    return None


def create_dataset_yaml(output_dir: Path) -> Path:
    """Создаёт конфигурационный файл для Ultralytics YOLO."""
    yaml_path = output_dir / "dataset.yaml"

    content = f"""path: {output_dir.resolve()}
train: images/train
val: images/val

names:
  0: with_helmet
  1: without_helmet
"""

    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def prepare_output_dirs(output_dir: Path, clear: bool) -> None:
    """Создаёт или очищает YOLO-структуру."""
    if clear and output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def convert_one_annotation(
    annotation_path: Path,
    source_images_dir: Path,
    output_dir: Path,
    split: str,
) -> tuple[bool, Counter]:
    """Конвертирует один JSON-файл и копирует соответствующее изображение."""
    class_counter = Counter()

    try:
        annotation: dict[str, Any] = json.loads(
            annotation_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as error:
        print(f"[ERROR] Cannot read {annotation_path.name}: {error}")
        return False, class_counter

    image_name = annotation.get("image_name", "")
    image_width = annotation.get("image_width")
    image_height = annotation.get("image_height")
    detections = annotation.get("detections", [])

    if not isinstance(image_width, int) or not isinstance(image_height, int):
        print(f"[ERROR] No valid image size in {annotation_path.name}")
        return False, class_counter

    if not isinstance(detections, list):
        print(f"[ERROR] 'detections' must be a list in {annotation_path.name}")
        return False, class_counter

    image_path = find_image(
        images_dir=source_images_dir,
        image_name=image_name,
        stem=annotation_path.stem,
    )

    if image_path is None:
        print(f"[ERROR] Image not found for {annotation_path.name}")
        return False, class_counter

    output_image_path = (
        output_dir / "images" / split / f"{annotation_path.stem}{image_path.suffix.lower()}"
    )
    output_label_path = (
        output_dir / "labels" / split / f"{annotation_path.stem}.txt"
    )

    lines = []

    for detection in detections:
        if not isinstance(detection, dict):
            continue

        label = detection.get("label")
        bbox = detection.get("bbox")

        if label not in CLASS_TO_ID or not isinstance(bbox, list):
            continue

        yolo_bbox = xyxy_to_yolo(
            bbox=bbox,
            image_width=image_width,
            image_height=image_height,
        )

        if yolo_bbox is None:
            continue

        class_id = CLASS_TO_ID[label]
        x_center, y_center, width, height = yolo_bbox

        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{width:.6f} {height:.6f}"
        )
        class_counter[label] += 1

    shutil.copy2(image_path, output_image_path)
    output_label_path.write_text("\n".join(lines), encoding="utf-8")

    return True, class_counter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/auto_annotated"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/yolo_format"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Удалить предыдущую YOLO-разметку перед конвертацией",
    )
    args = parser.parse_args()

    if not args.images_dir.exists():
        raise FileNotFoundError(f"Не найдена папка изображений: {args.images_dir}")

    if not args.annotations_dir.exists():
        raise FileNotFoundError(
            f"Не найдена папка аннотаций: {args.annotations_dir}"
        )

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio должен быть в интервале (0, 1)")

    annotation_paths = sorted(args.annotations_dir.glob("*.json"))

    if len(annotation_paths) < 2:
        raise ValueError("Нужно минимум два JSON-файла для train/val split.")

    random.Random(args.seed).shuffle(annotation_paths)

    val_size = max(1, round(len(annotation_paths) * args.val_ratio))
    val_paths = annotation_paths[:val_size]
    train_paths = annotation_paths[val_size:]

    prepare_output_dirs(args.output_dir, clear=args.clear_output)

    total_classes = Counter()
    success = Counter()

    for split, paths in (("train", train_paths), ("val", val_paths)):
        for annotation_path in paths:
            converted, class_counter = convert_one_annotation(
                annotation_path=annotation_path,
                source_images_dir=args.images_dir,
                output_dir=args.output_dir,
                split=split,
            )

            if converted:
                success[split] += 1
                total_classes.update(class_counter)

    yaml_path = create_dataset_yaml(args.output_dir)

    print("\nConversion statistics")
    print(f"Train images: {success['train']}/{len(train_paths)}")
    print(f"Validation images: {success['val']}/{len(val_paths)}")
    print(f"with_helmet boxes: {total_classes['with_helmet']}")
    print(f"without_helmet boxes: {total_classes['without_helmet']}")
    print(f"YOLO config: {yaml_path}")


if __name__ == "__main__":
    main()