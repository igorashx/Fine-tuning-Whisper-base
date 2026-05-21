from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from asr_ro.audio_loading import load_audio_sample
from asr_ro.console import configure_utf8_console
from asr_ro.metrics import summarize_metrics
from asr_ro.parallelism import resolve_worker_count
from asr_ro.training_dataset import read_manifest_rows

LOGGER = logging.getLogger(__name__)


class WhisperInferenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, str]], processor: WhisperProcessor) -> None:
        self.rows = rows
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        audio = load_audio_sample(row["audio_path"])
        input_features = self.processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        return {"input_features": input_features}


@dataclass
class InferenceCollator:
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        return {"input_features": batch.input_features}


def transcribe_dataset(
    model_name_or_path: str,
    rows: list[dict[str, str]],
    device: str,
    batch_size: int = 4,
    num_workers: int = 0,
) -> list[str]:
    LOGGER.info("Încarc modelul `%s` pentru transcriere pe `%s`.", model_name_or_path, device)
    processor = WhisperProcessor.from_pretrained(model_name_or_path, language="romanian", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
    resolved_num_workers = resolve_worker_count(num_workers, allow_zero=True)
    model.generation_config.language = "romanian"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="romanian", task="transcribe")
    model.to(device)
    model.eval()

    dataset = WhisperInferenceDataset(rows, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=resolved_num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=InferenceCollator(processor),
    )

    predictions: list[str] = []
    LOGGER.info("Transcrierea `%s` folosește `%s` worker-i pentru DataLoader.", model_name_or_path, resolved_num_workers)
    for batch in tqdm(dataloader, total=len(dataloader), desc=f"Transcriere {Path(model_name_or_path).name}", unit="batch", leave=True):
        with torch.no_grad():
            generated_ids = model.generate(batch["input_features"].to(device), max_new_tokens=128)
        predictions.extend(processor.batch_decode(generated_ids, skip_special_tokens=True))
    return predictions


def compare_models(
    test_csv: Path,
    fine_tuned_model: str,
    baseline_model: str,
    output_path: Path,
    limit: int | None,
    batch_size: int,
    num_workers: int,
) -> dict[str, object]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = read_manifest_rows(test_csv, limit=limit)
    LOGGER.info("Evaluez %s exemple din `%s` pe `%s`.", len(rows), test_csv, device)
    references = [row["text"] for row in rows]

    baseline_predictions = transcribe_dataset(baseline_model, rows, device=device, batch_size=batch_size, num_workers=num_workers)
    fine_tuned_predictions = transcribe_dataset(fine_tuned_model, rows, device=device, batch_size=batch_size, num_workers=num_workers)

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
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> None:
    configure_utf8_console()
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
        num_workers=args.num_workers,
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
