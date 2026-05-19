from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rulează pipeline-ul complet pentru Whisper base pe Common Voice ro.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-name", default="openai/whisper-base")
    parser.add_argument("--prep-limit", type=int)
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

    manifests_root = args.artifacts_root / "cv_ro"
    model_output_dir = args.artifacts_root / "whisper-base-ro"
    evaluation_output = args.artifacts_root / "evaluation" / "comparison.json"

    prep_command = [
        sys.executable,
        "-m",
        "asr_ro.data_prep",
        "--dataset-root",
        str(args.dataset_root),
        "--output-root",
        str(manifests_root),
    ]
    if args.skip_audio_normalization:
        prep_command.append("--skip-audio-normalization")
    if args.prep_limit:
        prep_command.extend(["--limit-per-split", str(args.prep_limit)])
    run_command(prep_command)

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
        run_command(train_command)

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
        run_command(eval_command)


if __name__ == "__main__":
    main()
