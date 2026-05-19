# Fine-tuning ASR română cu Whisper base

Acest proiect pregătește datasetul `Common Voice ro`, face fine-tuning pentru `openai/whisper-base` pe split-urile oficiale `train/dev/test` și compară rezultatele cu modelul preantrenat folosind `WER` și `CER`.

## Structură

- `asr_ro/data_prep.py` normalizează audio la `16 kHz` mono și exportă CSV-urile `audio_path,text`.
- `asr_ro/train_whisper.py` rulează fine-tuning pe `train.csv` și validează pe `dev.csv`, extrăgând feature-urile Whisper on-the-fly.
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

## Comportament memorie

În versiunea curentă, manifestele rămân ușoare (`audio_path`, `text`), iar extragerea feature-urilor Whisper se face on-the-fly la nivel de exemplu/batch. Astfel este evitată materializarea întregului split într-un fișier Arrow mare, care putea declanșa `ArrowMemoryError` pe Windows sau pe mașini cu RAM limitat.

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
& "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/.venv/Scripts/python.exe" -m asr_ro.train_whisper --train-csv "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/cv_ro/manifests/train.csv" --dev-csv "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/cv_ro/manifests/dev.csv" --output-dir "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/whisper-base-ro" --model-name openai/whisper-base --freeze-encoder
```

Pentru un smoke test local sau pentru RAM limitat:

```powershell
& "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/.venv/Scripts/python.exe" -m asr_ro.train_whisper --train-csv "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/cv_ro/manifests/train.csv" --dev-csv "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/cv_ro/manifests/dev.csv" --output-dir "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/whisper-base-ro-smoke" --model-name openai/whisper-base --max-train-samples 200 --max-eval-samples 500 --freeze-encoder
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

## Troubleshooting

- Dacă apare `ArrowMemoryError`, verifică mai întâi că folosești versiunea actuală a proiectului, unde feature-urile sunt extrase on-the-fly și nu sunt materializate într-un dataset Arrow mare.
- Dacă apare `TypeError` pentru `evaluation_strategy`, folosești o versiune de `transformers` unde argumentul corect este `eval_strategy`; proiectul a fost actualizat pentru acest API.
- Dacă apare `TypeError` pentru `tokenizer` în `Seq2SeqTrainer`, versiunea locală de `transformers` folosește `processing_class`; proiectul a fost actualizat și pentru această schimbare.
- Avertismentul despre `pin_memory` pe CPU este benign; proiectul setează acum automat `dataloader_pin_memory=False` când nu există accelerator CUDA.
- Dacă rulezi local fără CUDA, evită `--fp16`; folosește doar `--freeze-encoder` și eventual un subset prin `--max-train-samples` / `--max-eval-samples`.
- Dacă `audio_path` nu pointează spre fișiere `.wav` la `16 kHz`, regenerează manifestele cu `asr_ro.data_prep` fără `--skip-audio-normalization`.

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
