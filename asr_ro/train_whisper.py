from __future__ import annotations

import argparse
import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from asr_ro.audio_loading import load_audio_sample
from asr_ro.metrics import summarize_metrics


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def load_manifest(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return Dataset.from_list(rows)


def build_prepare_dataset(processor: WhisperProcessor):
    def prepare_dataset(batch: dict[str, Any]) -> dict[str, Any]:
        audio = load_audio_sample(batch["audio_path"])
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    return prepare_dataset


def compute_metrics_builder(processor: WhisperProcessor):
    def compute_metrics(prediction_output) -> dict[str, float]:
        prediction_ids = prediction_output.predictions
        if isinstance(prediction_ids, tuple):
            prediction_ids = prediction_ids[0]
        label_ids = prediction_output.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        prediction_texts = processor.tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
        label_texts = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        metrics = summarize_metrics(label_texts, prediction_texts)
        return {key: round(value, 4) for key, value in metrics.items()}

    return compute_metrics


def filter_samples(dataset, limit: int | None):
    if limit is None:
        return dataset
    max_limit = min(limit, len(dataset))
    return dataset.select(range(max_limit))


def train_model(args: argparse.Namespace) -> dict[str, float]:
    processor = WhisperProcessor.from_pretrained(args.model_name, language="romanian", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.generation_config.language = "romanian"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="romanian", task="transcribe")

    if args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    train_dataset = filter_samples(load_manifest(args.train_csv), args.max_train_samples)
    eval_dataset = filter_samples(load_manifest(args.dev_csv), args.max_eval_samples)

    prepare_dataset = build_prepare_dataset(processor)
    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names)

    training_arguments = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        logging_steps=args.logging_steps,
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_arguments,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics_builder(processor),
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    evaluation_metrics = trainer.evaluate(metric_key_prefix="eval")
    trainer.log_metrics("eval", evaluation_metrics)
    trainer.save_metrics("eval", evaluation_metrics)

    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "train_csv": str(args.train_csv),
                "dev_csv": str(args.dev_csv),
                "train_metrics": train_result.metrics,
                "eval_metrics": evaluation_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {key: float(value) for key, value in evaluation_metrics.items() if isinstance(value, (float, int))}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tuning Whisper base pe Common Voice ro.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--dev-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/whisper-base-ro"))
    parser.add_argument("--model-name", default="openai/whisper-base")
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=8.0)
    parser.add_argument("--generation-max-length", type=int, default=225)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = train_model(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
