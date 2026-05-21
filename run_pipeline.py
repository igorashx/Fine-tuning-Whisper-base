from __future__ import annotations

import argparse
from datetime import datetime
import logging
import subprocess
import sys
from pathlib import Path

from asr_ro.dataset_api import DEFAULT_API_KEY_ENV, DEFAULT_DATASET_ID, DEFAULT_DATASET_SLUG, obtain_dataset

LOGGER = logging.getLogger(__name__)


def configure_logging(artifacts_root: Path) -> Path:
    logs_dir = artifacts_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"run_pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    logging.captureWarnings(True)
    return log_path


def run_command(command: list[str], log_handle) -> None:
    LOGGER.info("Rulez comanda: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    assert process.stdout is not None
    while True:
        chunk = process.stdout.read(8192)
        if not chunk:
            break
        text = chunk.decode("utf-8", errors="replace")
        sys.stdout.write(text)
        sys.stdout.flush()
        log_handle.write(text)
        log_handle.flush()

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rulează pipeline-ul complet pentru Whisper base pe Common Voice ro.")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--dataset-cache-dir", type=Path, default=Path("artifacts/datasets"))
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--model-name", default="openai/whisper-base")
    parser.add_argument("--prep-limit", type=int)
    parser.add_argument("--prep-num-workers", type=int, default=1)
    parser.add_argument("--prep-chunksize", type=int, default=1)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--skip-audio-normalization", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    log_path = configure_logging(args.artifacts_root)

    LOGGER.info("Logul complet al execuției va fi salvat în `%s`.", log_path)

    LOGGER.info("Pornesc pipeline-ul complet pentru `%s`.", args.model_name)

    dataset_root = obtain_dataset(
        download_root=args.dataset_cache_dir,
        dataset_id=args.dataset_id,
        dataset_slug=args.dataset_slug,
        api_key_env=args.api_key_env,
        force_download=args.force_download,
        force_extract=args.force_extract,
    )

    manifests_root = args.artifacts_root / "cv_ro"
    model_output_dir = args.artifacts_root / "whisper-base-ro"
    evaluation_output = args.artifacts_root / "evaluation" / "comparison.json"

    LOGGER.info("Dataset root detectat: `%s`.", dataset_root)

    prep_command = [
        sys.executable,
        "-m",
        "asr_ro.data_prep",
        "--dataset-root",
        str(dataset_root),
        "--output-root",
        str(manifests_root),
    ]
    if args.skip_audio_normalization:
        prep_command.append("--skip-audio-normalization")
    if args.prep_limit:
        prep_command.extend(["--limit-per-split", str(args.prep_limit)])
    prep_command.extend(["--num-workers", str(args.prep_num_workers), "--chunksize", str(args.prep_chunksize)])
    LOGGER.info("Etapa 1/3: pregătire date.")
    with log_path.open("a", encoding="utf-8") as log_handle:
        run_command(prep_command, log_handle)

    if not args.skip_train:
        train_command = [
            sys.executable,
            "-m",
            "asr_ro.train_whisper",
            "--train-csv",
            str(manifests_root / "manifests" / "train.csv"),
            "--dev-csv",
            str(manifests_root / "manifests" / "dev.csv"),
            "--output-dir",
            str(model_output_dir),
            "--model-name",
            args.model_name,
        ]
        if args.train_limit:
            train_command.extend(["--max-train-samples", str(args.train_limit), "--max-eval-samples", str(args.train_limit)])
        if args.fp16:
            train_command.append("--fp16")
        if args.freeze_encoder:
            train_command.append("--freeze-encoder")
        if args.gradient_checkpointing:
            train_command.append("--gradient-checkpointing")
        LOGGER.info("Etapa 2/3: antrenare model.")
        with log_path.open("a", encoding="utf-8") as log_handle:
            run_command(train_command, log_handle)

    if not args.skip_eval:
        eval_command = [
            sys.executable,
            "-m",
            "asr_ro.evaluate_model",
            "--test-csv",
            str(manifests_root / "manifests" / "test.csv"),
            "--fine-tuned-model",
            str(model_output_dir),
            "--baseline-model",
            args.model_name,
            "--output-path",
            str(evaluation_output),
        ]
        if args.eval_limit:
            eval_command.extend(["--limit", str(args.eval_limit)])
        LOGGER.info("Etapa 3/3: evaluare comparativă.")
        with log_path.open("a", encoding="utf-8") as log_handle:
            run_command(eval_command, log_handle)

    LOGGER.info("Pipeline finalizat. Artefacte principale: `%s`, `%s`, `%s`.", manifests_root, model_output_dir, evaluation_output)
    LOGGER.info("Logul complet al rulării este disponibil în `%s`.", log_path)


if __name__ == "__main__":
    main()
