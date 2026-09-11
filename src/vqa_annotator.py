"""
Автоматическая VQA-разметка изображений для задачи helmet detection.

Модель получает изображение через OpenAI-compatible API vLLM и возвращает
JSON с bbox каждого человека и меткой наличия защитного шлема.

Пример запуска из корня проекта:
python src/vqa_annotator.py \
  --data-dir data/raw \
  --output-dir data/auto_annotated \
  --model Qwen/Qwen2.5-VL-7B-Instruct
"""

import argparse
import base64
import json
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from tqdm import tqdm


VALID_LABELS = {"with_helmet", "without_helmet"}

SYSTEM_PROMPT = """
You are an expert industrial-safety image annotator.

Detect every visible real person in the image and determine whether the person
is wearing a protective hard hat.

Return ONLY a valid JSON object. Do not use Markdown. Do not add explanations.

Required output format:
{
  "detections": [
    {
      "bbox": [x_min, y_min, x_max, y_max],
      "label": "with_helmet"
    }
  ]
}

Rules:
- One detection corresponds to one person.
- Bounding boxes must cover the whole visible person, not only the head or helmet.
- Coordinates are absolute pixel coordinates for the input image.
- Use integer coordinates only.
- Valid labels: "with_helmet", "without_helmet".
- Use "with_helmet" only when a protective hard hat is clearly visible.
- Use "without_helmet" only when the head is visible and a hard hat is clearly absent.
- Omit a person when their head is strongly occluded, too small, blurred, cropped,
  or when helmet status is uncertain.
- Do not annotate posters, statues, mannequins, reflections, drawings,
  helmets without people, or background objects.
""".strip()


class VQAAnnotator:
    def __init__(
        self,
        api_url: str,
        model_name: str,
        timeout: int = 180,
        max_retries: int = 3,
    ) -> None:
        self.api_url = api_url
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries

    @staticmethod
    def image_to_base64(image_path: Path) -> str:
        """Конвертирует изображение в JPEG base64."""
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=95)

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def extract_json(text: str) -> dict[str, Any] | None:
        """Извлекает JSON, в том числе из ответа в markdown-code block."""
        text = text.strip()
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def sanitize_detection(
        detection: dict[str, Any],
        image_width: int,
        image_height: int,
    ) -> dict[str, Any] | None:
        """Валидирует и ограничивает bbox границами изображения."""
        if not isinstance(detection, dict):
            return None

        label = detection.get("label")
        bbox = detection.get("bbox")

        if label not in VALID_LABELS:
            return None

        if not isinstance(bbox, list) or len(bbox) != 4:
            return None

        try:
            x1, y1, x2, y2 = [round(float(value)) for value in bbox]
        except (TypeError, ValueError):
            return None

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        return {
            "bbox": [x1, y1, x2, y2],
            "label": label,
        }

    def annotate_image(self, image_path: Path) -> tuple[list[dict[str, Any]], str]:
        """
        Возвращает валидные детекции и исходный ответ модели.

        Сырой текст ответа сохраняется отдельно: он нужен для диагностики
        JSON-ошибок и для анализа качества VQA-разметки.
        """
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        image_base64 = self.image_to_base64(image_path)

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Image width: {image_width}px. "
                                f"Image height: {image_height}px. "
                                "Detect people and helmet status. Output JSON only."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
        }

        last_error = ""

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()

                data = response.json()
                raw_response = data["choices"][0]["message"]["content"]
                parsed = self.extract_json(raw_response)

                if parsed is None:
                    last_error = "JSON parsing failed"
                    time.sleep(1)
                    continue

                raw_detections = parsed.get("detections", [])

                if not isinstance(raw_detections, list):
                    last_error = "'detections' is not a list"
                    time.sleep(1)
                    continue

                detections = []
                for detection in raw_detections:
                    cleaned = self.sanitize_detection(
                        detection,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    if cleaned is not None:
                        detections.append(cleaned)

                return detections, raw_response

            except (
                requests.RequestException,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as error:
                last_error = str(error)
                time.sleep(2)

        return [], f"ERROR: annotation failed after {self.max_retries} attempts: {last_error}"

    def annotate_dataset(
        self,
        data_dir: Path,
        output_dir: Path,
        skip_existing: bool = True,
    ) -> dict[str, int]:
        """Создаёт по одному JSON-файлу разметки на изображение."""
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            path
            for path in data_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        stats = {
            "total": len(image_paths),
            "processed": 0,
            "skipped": 0,
            "images_with_detections": 0,
            "total_detections": 0,
            "empty_or_failed": 0,
        }

        for image_path in tqdm(image_paths, desc="VQA annotation"):
            output_path = output_dir / f"{image_path.stem}.json"

            if skip_existing and output_path.exists():
                stats["skipped"] += 1
                continue

            with Image.open(image_path) as image:
                image_width, image_height = image.size

            detections, raw_response = self.annotate_image(image_path)

            annotation = {
                "image_name": image_path.name,
                "image_width": image_width,
                "image_height": image_height,
                "detections": detections,
                "raw_vqa_response": raw_response,
            }

            with output_path.open("w", encoding="utf-8") as file:
                json.dump(annotation, file, ensure_ascii=False, indent=2)

            stats["processed"] += 1
            stats["total_detections"] += len(detections)

            if detections:
                stats["images_with_detections"] += 1
            else:
                stats["empty_or_failed"] += 1

        return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/auto_annotated"),
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/v1/chat/completions",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Перегенерировать уже существующие JSON-аннотации",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Не найдена папка с данными: {args.data_dir}")

    annotator = VQAAnnotator(
        api_url=args.api_url,
        model_name=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    stats = annotator.annotate_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        skip_existing=not args.overwrite,
    )

    print("\nAnnotation statistics")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()