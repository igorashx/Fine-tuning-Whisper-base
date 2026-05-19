# Fine-tuning ASR română cu Whisper base

Acest proiect descarcă datasetul `Common Voice ro` prin API, îl pregătește pentru `openai/whisper-base`, face fine-tuning pe split-urile oficiale `train/dev/test` și compară modelul rezultat cu modelul preantrenat folosind `WER` și `CER`.
Sursa datasetului: `https://mozilladatacollective.com/datasets/cmn2e8rmi01l6mm07vxurptse`

## Structură

- `asr_ro/data_prep.py` normalizează audio la `16 kHz` mono și exportă CSV-urile `audio_path,text`.
- `asr_ro/dataset_api.py` descarcă arhiva datasetului prin API, o extrage și detectează automat rădăcina corpusului.
- `asr_ro/train_whisper.py` rulează fine-tuning pe `train.csv` și validează pe `dev.csv`, extrăgând feature-urile Whisper on-the-fly.
- `asr_ro/evaluate_model.py` compară modelul fine-tuned cu modelul preantrenat pe `test.csv`.
- `run_pipeline.py` orchestrează pașii principali.
- `notebooks/whisper_base_ro_colab.ipynb` oferă o variantă Colab cu explicații.

## Date folosite

Datasetul este obținut prin API-ul Mozilla Data Collective, apoi extras automat într-un cache local sau Colab. Se folosesc strict split-urile oficiale:

- `ro/train.tsv` pentru antrenare
- `ro/dev.tsv` pentru validare
- `ro/test.tsv` pentru evaluare finală

Fiecare rând din manifestul rezultat conține:

- `audio_path` - cale relativă, portabilă, către fișierul audio din artefactele generate
- `text` - transcrierea corectată conservator

## Cerințe

- Python `3.12+`
- `ffmpeg` în `PATH` sau fallback-ul `imageio-ffmpeg` disponibil prin dependențele Python pentru conversia `mp3 -> wav`
- GPU recomandat pentru antrenare, ideal Google Colab
- cheie API furnizată prin variabilă de mediu, nu prin fișiere versionate

## Comportament memorie

În versiunea curentă, manifestele rămân ușoare (`audio_path`, `text`) și folosesc căi relative, iar extragerea feature-urilor Whisper se face on-the-fly la nivel de exemplu/batch. Astfel este evitată materializarea întregului split într-un fișier Arrow mare, care putea declanșa `ArrowMemoryError` pe Windows sau pe mașini cu RAM limitat, iar artefactele rămân portabile între Windows și Colab.

## Securitate API

- Nu păstra cheia API în repo, notebook sau istoric shell.
- Setează cheia în variabila de mediu `MOZILLA_DATA_COLLECTIVE_API_KEY`.
- Dacă cheia a fost expusă anterior, rotește-o și actualizează secretul înainte de a rula proiectul.

## Instalare

```powershell
"d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/.venv/Scripts/python.exe" -m pip install -r requirements.txt
```

## Configurare cheie API

```powershell
$Env:MOZILLA_DATA_COLLECTIVE_API_KEY = "<cheia-ta-api>"
```

## Rulare rapidă locală

### 1. Descărcare prin API și rulare pipeline complet

```powershell
& "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/.venv/Scripts/python.exe" run_pipeline.py --dataset-cache-dir "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/datasets" --freeze-encoder
```
sau pentru cell din colab
```powershell
!MOZILLA_DATA_COLLECTIVE_API_KEY="95c0da9d9d3e5c78ca2452db640312f39e69efe82d9cc4cfbbf074f1ee1d45ed" python run_pipeline.py \
  --dataset-cache-dir "/content/artifacts/datasets" \
  --freeze-encoder \
  --fp16
```

Adaugă `--fp16` doar când rulezi pe GPU compatibil, de exemplu în Colab.

### 2. Doar descărcare și extracție prin API

```powershell
& "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/.venv/Scripts/python.exe" -m asr_ro.dataset_api --download-root "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/datasets"
```

Comanda afișează rădăcina extrasă a datasetului, pe care o poți folosi și manual în pașii următori.

### 3. Preprocesare și generare manifest

```powershell
& "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/.venv/Scripts/python.exe" -m asr_ro.data_prep --dataset-root "<dataset_root_afisat_de_dataset_api>" --output-root "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/cv_ro"
```

### 4. Fine-tuning Whisper base

```powershell
& "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/.venv/Scripts/python.exe" -m asr_ro.train_whisper --train-csv "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/cv_ro/manifests/train.csv" --dev-csv "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/cv_ro/manifests/dev.csv" --output-dir "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/whisper-base-ro" --model-name openai/whisper-base --freeze-encoder
```

Pentru un smoke test local sau pentru RAM limitat:

```powershell
& "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/.venv/Scripts/python.exe" -m asr_ro.train_whisper --train-csv "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/cv_ro/manifests/train.csv" --dev-csv "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/cv_ro/manifests/dev.csv" --output-dir "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/whisper-base-ro-smoke" --model-name openai/whisper-base --max-train-samples 500 --max-eval-samples 100 --max-steps 20 --train-batch-size 1 --eval-batch-size 1 --freeze-encoder
```

### 5. Evaluare comparativă

```powershell
& "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/.venv/Scripts/python.exe" -m asr_ro.evaluate_model --test-csv "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/cv_ro/manifests/test.csv" --fine-tuned-model "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/whisper-base-ro" --baseline-model openai/whisper-base --output-path "d:/Python/master/Anul 1/Sem 2/Deep Learning/Laborator 3/Video to text/artifacts/evaluation/comparison.json"
```

## Parametri recomandați pentru Colab

- `train_batch_size=8`
- `eval_batch_size=8`
- `gradient_accumulation_steps=2`
- `num_train_epochs=8`
- `learning_rate=1e-5`
- `save_strategy=epoch`
- `eval_strategy=epoch`
- `save_total_limit=2`

Dacă memoria GPU este limitată, redu `train_batch_size` la `4` și păstrează `gradient_accumulation_steps=4`.

## Troubleshooting

- Dacă apare `ArrowMemoryError`, verifică mai întâi că folosești versiunea actuală a proiectului, unde feature-urile sunt extrase on-the-fly și nu sunt materializate într-un dataset Arrow mare.
- Dacă apare `TypeError` pentru `evaluation_strategy`, folosești o versiune de `transformers` unde argumentul corect este `eval_strategy`; proiectul a fost actualizat pentru acest API.
- Dacă apare `TypeError` pentru `tokenizer` în `Seq2SeqTrainer`, versiunea locală de `transformers` folosește `processing_class`; proiectul a fost actualizat și pentru această schimbare.
- Avertismentul despre `pin_memory` pe CPU este benign; proiectul setează acum automat `dataloader_pin_memory=False` când nu există accelerator CUDA.
- Dacă rulezi local fără CUDA, evită `--fp16`; folosește doar `--freeze-encoder` și eventual un subset prin `--max-train-samples` / `--max-eval-samples`.
- Pentru un smoke test verificat local, comanda recomandată este cea cu `500/100`, `--max-steps 20` și batch size `1`.
- După `python -m asr_ro.dataset_api`, folosește exact calea afișată în terminal ca valoare pentru `--dataset-root` dacă rulezi manual `asr_ro.data_prep`.
- Dacă `run_pipeline.py` nu găsește cheia API, verifică existența variabilei de mediu `MOZILLA_DATA_COLLECTIVE_API_KEY` în sesiunea curentă.
- Dacă URL-ul presigned expiră sau download-ul e întrerupt, rerulează comanda cu `--force-download` și, dacă e necesar, cu `--force-extract`.
- Dacă `audio_path` nu pointează spre fișiere `.wav` la `16 kHz`, regenerează manifestele cu `asr_ro.data_prep` fără `--skip-audio-normalization`.

## Fișiere rezultate

- `artifacts/datasets/common-voice-scripted-speech-25-0-romani-701de4ae/`
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
