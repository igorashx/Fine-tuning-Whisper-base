from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from asr_ro.console import configure_utf8_console
from asr_ro.metrics import summarize_metrics
from asr_ro.parallelism import resolve_worker_count
from asr_ro.training_dataset import WhisperTrainingDataset, read_manifest_rows

LOGGER = logging.getLogger(__name__)


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


class WhisperValidationTrainer(Seq2SeqTrainer):
    def __init__(self, *args, validation_num_workers: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.validation_num_workers = validation_num_workers

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Evaluation requires an eval_dataset.")

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=self._get_eval_sampler(eval_dataset),
            collate_fn=self.data_collator,
            drop_last=getattr(self.args, "dataloader_drop_last", False),
            num_workers=self.validation_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        return self.accelerator.prepare(eval_dataloader)

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

def train_model(args: argparse.Namespace) -> dict[str, float]:
    LOGGER.info("Încarc processor și modelul `%s`.", args.model_name)
    processor = WhisperProcessor.from_pretrained(args.model_name, language="romanian", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    use_accelerator = torch.cuda.is_available()
    validation_num_workers = resolve_worker_count(args.validation_num_workers, allow_zero=True)
    model.generation_config.language = "romanian"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="romanian", task="transcribe")

    if args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False
        LOGGER.info("Encoderul a fost înghețat.")

    train_rows = read_manifest_rows(args.train_csv, limit=args.max_train_samples)
    eval_rows = read_manifest_rows(args.dev_csv, limit=args.max_eval_samples)
    LOGGER.info("Am încărcat %s exemple de train și %s exemple de evaluare.", len(train_rows), len(eval_rows))
    LOGGER.info("Validarea pe `dev` va folosi `%s` worker-i pentru DataLoader.", validation_num_workers)
    train_dataset = WhisperTrainingDataset(train_rows, processor)
    eval_dataset = WhisperTrainingDataset(eval_rows, processor)

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
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        logging_steps=args.logging_steps,
        disable_tqdm=False,
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        dataloader_pin_memory=use_accelerator,
        remove_unused_columns=False,
    )

    trainer = WhisperValidationTrainer(
        args=training_arguments,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        processing_class=processor,
        compute_metrics=compute_metrics_builder(processor),
        validation_num_workers=validation_num_workers,
    )

    LOGGER.info("Pornesc fine-tuning-ul în `%s` pe `%s`.", args.output_dir, "cuda" if use_accelerator else "cpu")
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
    LOGGER.info("Training încheiat. Rezumatul este în `%s`.", summary_path)
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
    parser.add_argument("--validation-num-workers", type=int, default=0)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    return parser


def main() -> None:
    configure_utf8_console()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = build_argument_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = train_model(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
