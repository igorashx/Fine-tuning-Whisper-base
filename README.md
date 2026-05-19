# Fine-tuning ASR română cu Whisper base

Acest proiect pregătește datasetul `Common Voice ro`, face fine-tuning pentru `openai/whisper-base` pe split-urile oficiale `train/dev/test` și compară rezultatele cu modelul preantrenat folosind `WER` și `CER`.

## Structură

- `asr_ro/data_prep.py` normalizează audio la `16 kHz` mono și exportă CSV-urile `audio_path,text`.
- `asr_ro/train_whisper.py` rulează fine-tuning pe `train.csv` și validează pe `dev.csv`.
- `asr_ro/evaluate_model.py` compară modelul fine-tuned cu modelul preantrenat pe `test.csv`.
- `run_pipeline.py` orchestrează pașii principali.
- `notebooks/whisper_base_ro_colab.ipynb` oferă o variantă Colab cu explicații.

## Date folosite

Datasetul este citit direct din `1774203787031-cv-corpus-25.0-2026-03-09-ro/ro`.
Se folosesc strict split-urile oficiale:

- `ro/train.tsv` pentru antrenare
- `ro/dev.tsv` pentru validare
- `ro/test.tsv` pentru evaluare finală

Fiecare rând din manifestul rezultat conține:

- `audio_path` - calea absolută către fișierul audio normalizat
- `text` - transcrierea corectată conservator

## Cerințe

- Python `3.12+`
- `ffmpeg` disponibil în `PATH` pentru conversia `mp3 -> wav`
- GPU recomandat pentru antrenare, ideal Google Colab

## Instalare

```powershell
"d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/.venv/Scripts/python.exe" -m pip install -r requirements.txt
```

## Rulare rapidă locală

### 1. Preprocesare și generare manifest

```powershell
& "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/.venv/Scripts/python.exe" -m asr_ro.data_prep --dataset-root "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/1774203787031-cv-corpus-25.0-2026-03-09-ro" --output-root "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/cv_ro"
```

### 2. Fine-tuning Whisper base

```powershell
& "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/.venv/Scripts/python.exe" -m asr_ro.train_whisper --train-csv "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/cv_ro/manifests/train.csv" --dev-csv "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/cv_ro/manifests/dev.csv" --output-dir "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/whisper-base-ro" --model-name openai/whisper-base --fp16 --freeze-encoder
```

### 3. Evaluare comparativă

```powershell
"d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/.venv/Scripts/python.exe" -m asr_ro.evaluate_model --test-csv "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/cv_ro/manifests/test.csv" --fine-tuned-model "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/whisper-base-ro" --baseline-model openai/whisper-base --output-path "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/evaluation/comparison.json"
```

### 4. Pipeline complet

```powershell
"d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/.venv/Scripts/python.exe" run_pipeline.py --dataset-root "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/1774203787031-cv-corpus-25.0-2026-03-09-ro" --fp16 --freeze-encoder
```

## Parametri recomandați pentru Colab

- `train_batch_size=8`
- `eval_batch_size=8`
- `gradient_accumulation_steps=2`
- `num_train_epochs=8`
- `learning_rate=1e-5`
- `save_strategy=epoch`
- `evaluation_strategy=epoch`
- `save_total_limit=2`

Dacă memoria GPU este limitată, redu `train_batch_size` la `4` și păstrează `gradient_accumulation_steps=4`.

## Fișiere rezultate

- `artifacts/cv_ro/manifests/train.csv`
- `artifacts/cv_ro/manifests/dev.csv`
- `artifacts/cv_ro/manifests/test.csv`
- `artifacts/whisper-base-ro/`
- `artifacts/evaluation/comparison.json`

## Raport și prezentare

Notebook-ul include secțiuni pentru:

- descrierea datelor folosite
- preprocesare audio și text
- fine-tuning Whisper base
- evaluare cu `WER` și `CER`
- comparație cu modelul preantrenat
- observații despre recunoașterea accentului românesc

Graficele comparative pot fi generate direct din `comparison.json` și din fișierele de metrici salvate de trainer.
