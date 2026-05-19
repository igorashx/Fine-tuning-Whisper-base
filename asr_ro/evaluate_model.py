from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from asr_ro.audio_loading import load_audio_sample
from asr_ro.metrics import summarize_metrics
from asr_ro.training_dataset import read_manifest_rows

LOGGER = logging.getLogger(__name__)


def transcribe_dataset(
    model_name_or_path: str,
    rows: list[dict[str, str]],
    device: str,
    batch_size: int = 4,
) -> list[str]:
    LOGGER.info("Încarc modelul `%s` pentru transcriere pe `%s`.", model_name_or_path, device)
    processor = WhisperProcessor.from_pretrained(model_name_or_path, language="romanian", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
    model.generation_config.language = "romanian"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="romanian", task="transcribe")
    model.to(device)
    model.eval()

    predictions: list[str] = []
    total_batches = (len(rows) + batch_size - 1) // batch_size if rows else 0
    for batch_start in tqdm(range(0, len(rows), batch_size), total=total_batches, desc=f"Transcriere {Path(model_name_or_path).name}", unit="batch", leave=True):
        batch_rows = rows[batch_start : batch_start + batch_size]
        audio_items = [load_audio_sample(row["audio_path"]) for row in batch_rows]
        arrays = [item["array"] for item in audio_items]
        sampling_rate = audio_items[0]["sampling_rate"]
        features = processor.feature_extractor(arrays, sampling_rate=sampling_rate, return_tensors="pt")
        with torch.no_grad():
            generated_ids = model.generate(features.input_features.to(device), max_new_tokens=128)
        predictions.extend(processor.batch_decode(generated_ids, skip_special_tokens=True))
    return predictions


def compare_models(
    test_csv: Path,
    fine_tuned_model: str,
    baseline_model: str,
    output_path: Path,
    limit: int | None,
    batch_size: int,
) -> dict[str, object]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = read_manifest_rows(test_csv, limit=limit)
    LOGGER.info("Evaluez %s exemple din `%s` pe `%s`.", len(rows), test_csv, device)
    references = [row["text"] for row in rows]

    baseline_predictions = transcribe_dataset(baseline_model, rows, device=device, batch_size=batch_size)
    fine_tuned_predictions = transcribe_dataset(fine_tuned_model, rows, device=device, batch_size=batch_size)

    baseline_metrics = summarize_metrics(references, baseline_predictions)
    fine_tuned_metrics = summarize_metrics(references, fine_tuned_predictions)
    deltas = {
        metric_name: baseline_metrics[metric_name] - fine_tuned_metrics[metric_name]
        for metric_name in baseline_metrics
    }

    preview_count = min(10, len(references))
    comparison = {
        "device": device,
        "samples": len(references),
        "baseline_model": baseline_model,
        "fine_tuned_model": fine_tuned_model,
        "baseline_metrics": baseline_metrics,
        "fine_tuned_metrics": fine_tuned_metrics,
        "improvement": deltas,
        "examples": [
            {
                "reference": references[index],
                "baseline_prediction": baseline_predictions[index],
                "fine_tuned_prediction": fine_tuned_predictions[index],
            }
            for index in range(preview_count)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Evaluarea s-a încheiat. Rezultatele sunt în `%s`.", output_path)
    return comparison


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compară Whisper base cu modelul fine-tuned pe test.tsv.")
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--fine-tuned-model", required=True)
    parser.add_argument("--baseline-model", default="openai/whisper-base")
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/evaluation/comparison.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = build_argument_parser()
    args = parser.parse_args()
    comparison = compare_models(
        test_csv=args.test_csv,
        fine_tuned_model=args.fine_tuned_model,
        baseline_model=args.baseline_model,
        output_path=args.output_path,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
