# Fine-tuning ASR în română cu Whisper base

Acest proiect descarcă `Common Voice Romanian` prin API-ul Mozilla Data Collective, pregătește datele pentru `openai/whisper-base`, face fine-tuning pe split-urile oficiale `train/dev/test` și compară modelul rezultat cu baseline-ul preantrenat folosind `WER` și `CER`.

Dataset sursă: `https://mozilladatacollective.com/datasets/cmn2e8rmi01l6mm07vxurptse`

## Scop

- obținerea unui pipeline reproductibil pentru ASR în română;
- folosirea strictă a split-urilor oficiale `train/dev/test`;
- rulare locală sau în Google Colab fără a păstra datasetul în repo;
- generarea automată de artefacte, metrici și loguri de execuție.

## Ce face proiectul

- descarcă arhiva datasetului prin API și o extrage în cache local;
- detectează automat rădăcina corpusului extras;
- normalizează audio la `16 kHz` mono WAV;
- generează manifeste CSV portabile cu coloanele `audio_path,text`;
- antrenează `Whisper base` cu feature extraction on-the-fly;
- evaluează modelul fine-tuned versus baseline pe `test.tsv`;
- salvează logul complet al fiecărei rulări din `run_pipeline.py` într-un fișier separat.

## Structura proiectului

- `run_pipeline.py` orchestrează download-ul, preprocesarea, antrenarea și evaluarea.
- `asr_ro/dataset_api.py` descarcă datasetul prin API, extrage arhiva și gestionează cache-ul.
- `asr_ro/data_prep.py` pregătește manifestele și audio normalizat.
- `asr_ro/train_whisper.py` rulează fine-tuning pentru `openai/whisper-base`.
- `asr_ro/evaluate_model.py` compară baseline-ul cu modelul fine-tuned.
- `asr_ro/training_dataset.py` citește manifestele și rezolvă căile relative la runtime.
- `notebooks/whisper_base_ro_colab.ipynb` oferă o variantă rapidă de rulare în Colab.
- `tests/` conține testele unitare pentru utilitarele principale.

## Cum sunt folosite datele

Se folosesc strict split-urile oficiale din corpus:

- `ro/train.tsv` pentru antrenare;
- `ro/dev.tsv` pentru validare;
- `ro/test.tsv` pentru evaluare finală.

Manifestele generate conțin:

- `audio_path` — cale relativă către fișierul audio din directorul de artefacte;
- `text` — transcript normalizat conservator.

Acest lucru face artefactele portabile între Windows, Linux și Colab.

## Cerințe

- Python `3.12+`;
- pachetele din `requirements.txt`;
- `Git LFS`, dacă vrei să descarci checkpoint-ul fine-tuned versionat în repo;
- `ffmpeg` în `PATH` sau fallback-ul oferit de `imageio-ffmpeg`;
- cheie API Mozilla Data Collective disponibilă prin variabilă de mediu;
- GPU recomandat pentru antrenare; CPU este suportat, dar mult mai lent.

## Instalare

### Varianta recomandată: mediu virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dacă folosești alt shell sau alt sistem de operare, activează mediul virtual cu comanda specifică platformei tale.

## Modelul fine-tuned din repo și Git LFS

Checkpoint-ul fine-tuned versionat în `artifacts/whisper-base-ro/model.safetensors` este stocat prin `Git LFS`, nu ca blob Git obișnuit.

Dacă ai clonat repo-ul și vrei să descarci și greutățile versionate, rulează în rădăcina proiectului:

```powershell
git lfs install
git lfs pull
```

Dacă rulezi training-ul de la zero și nu ai nevoie de checkpoint-ul deja versionat, poți ignora acest pas.

## Configurarea cheii API

În PowerShell:

```powershell
$Env:MOZILLA_DATA_COLLECTIVE_API_KEY = "<cheia-ta-api>"
```

În Colab sau într-un shell POSIX:

```bash
export MOZILLA_DATA_COLLECTIVE_API_KEY="<cheia-ta-api>"
```

## Securitate

- nu păstra cheia API în repo sau în notebook-ul versionat;
- nu comite fișiere de tip `.env`, `API.txt` sau istoric shell cu cheia expusă;
- dacă cheia a fost expusă anterior, rotește-o înainte de utilizare reală.

## Rulare rapidă locală

### Pipeline complet

Comanda de mai jos:

- descarcă sau reutilizează arhiva din cache;
- pregătește manifestele;
- antrenează modelul;
- rulează evaluarea comparativă;
- salvează logul complet al rulării în `artifacts/logs/`.

```powershell
python run_pipeline.py --dataset-cache-dir "artifacts/datasets" --freeze-encoder
```

Adaugă `--fp16` doar dacă rulezi pe GPU compatibil CUDA.

Pentru paralelizarea pregătirii datelor, poți adăuga:

```powershell
python run_pipeline.py --dataset-cache-dir "artifacts/datasets" --freeze-encoder --prep-num-workers 4 --prep-chunksize 2
```

Pentru folosirea automată a tuturor workerilor disponibili:

```powershell
python run_pipeline.py --dataset-cache-dir "artifacts/datasets" --freeze-encoder --prep-num-workers -1
```

Pentru accelerarea validării din training și a evaluării finale:

```powershell
python run_pipeline.py --dataset-cache-dir "artifacts/datasets" --freeze-encoder --validation-num-workers 2 --test-num-workers 2
```

### Pipeline complet pe subset mic

Util pentru verificare rapidă pe CPU sau pentru un smoke test:

```powershell
python run_pipeline.py --dataset-cache-dir "artifacts/datasets" --freeze-encoder --prep-limit 100 --train-limit 100 --eval-limit 50
```

### Doar download și extracție prin API

```powershell
python -m asr_ro.dataset_api --download-root "artifacts/datasets"
```

Comanda afișează rădăcina datasetului extras, pe care o poți folosi manual în etapele următoare.

## Rulare manuală pe etape

### 1. Preprocesare și generare manifest

```powershell
python -m asr_ro.data_prep --dataset-root "<dataset_root_afisat_de_dataset_api>" --output-root "artifacts/cv_ro"
```

Pentru paralelizare la nivel de pregătire audio:

```powershell
python -m asr_ro.data_prep --dataset-root "<dataset_root_afisat_de_dataset_api>" --output-root "artifacts/cv_ro" --num-workers 4 --chunksize 2
```

Pentru selectare automată a tuturor workerilor disponibili:

```powershell
python -m asr_ro.data_prep --dataset-root "<dataset_root_afisat_de_dataset_api>" --output-root "artifacts/cv_ro" --num-workers -1
```

### 2. Fine-tuning

```powershell
python -m asr_ro.train_whisper --train-csv "artifacts/cv_ro/manifests/train.csv" --dev-csv "artifacts/cv_ro/manifests/dev.csv" --output-dir "artifacts/whisper-base-ro" --model-name openai/whisper-base --freeze-encoder
```

Pentru accelerarea validării pe `dev` în timpul fine-tuning-ului:

```powershell
python -m asr_ro.train_whisper --train-csv "artifacts/cv_ro/manifests/train.csv" --dev-csv "artifacts/cv_ro/manifests/dev.csv" --output-dir "artifacts/whisper-base-ro" --model-name openai/whisper-base --freeze-encoder --validation-num-workers 2
```

### 3. Evaluare comparativă

```powershell
python -m asr_ro.evaluate_model --test-csv "artifacts/cv_ro/manifests/test.csv" --fine-tuned-model "artifacts/whisper-base-ro" --baseline-model openai/whisper-base --output-path "artifacts/evaluation/comparison.json"
```

Pentru accelerarea încărcării și preprocesării audio la evaluarea finală:

```powershell
python -m asr_ro.evaluate_model --test-csv "artifacts/cv_ro/manifests/test.csv" --fine-tuned-model "artifacts/whisper-base-ro" --baseline-model openai/whisper-base --output-path "artifacts/evaluation/comparison.json" --num-workers 2
```

## Google Colab

Notebook-ul `notebooks/whisper_base_ro_colab.ipynb` conține o variantă simplificată de rulare.

Fluxul actual din notebook este:

- clonează branch-ul `with-API`;
- intră în directorul repo-ului;
- instalează dependențele;
- rulează `run_pipeline.py`.

Dacă vrei să folosești în Colab checkpoint-ul fine-tuned deja versionat în repo, rulează suplimentar `git lfs install` și `git lfs pull` după clonare.

Înainte de a executa celula de pipeline din notebook, trebuie să introduci cheia API în locul șirului gol din comandă sau să setezi variabila de mediu în sesiunea Colab.

Exemplu de comandă Colab:

```bash
MOZILLA_DATA_COLLECTIVE_API_KEY="<cheia-ta-api>" python run_pipeline.py \
	--dataset-cache-dir "/content/artifacts/datasets" \
	--freeze-encoder \
	--fp16 \
	--validation-num-workers 2 \
	--prep-num-workers 4 \
	--prep-chunksize 2
```

`--fp16` este recomandat în Colab doar dacă instanța are GPU activ.

## Loguri și progres

Proiectul afișează acum loguri mai detaliate și bare de progres `tqdm` pentru:

- download-ul datasetului;
- verificarea și extracția arhivei;
- pregătirea split-urilor `train/dev/test`;
- batch-urile de evaluare;
- progresul antrenării prin `Trainer`.

Pregătirea datelor poate rula secvențial sau paralel:

- `--num-workers 1` păstrează comportamentul secvențial și este opțiunea cea mai sigură pentru debugging;
- `--num-workers > 1` activează worker-i multiproces pentru conversia/copierea audio;
- `--num-workers -1` selectează automat toți workerii CPU disponibili;
- `--chunksize` controlează câte task-uri sunt trimise unui worker într-un lot.

Validarea și evaluarea finală pot fi accelerate prin worker-i CPU pentru DataLoader:

- `--validation-num-workers` controlează câți worker-i sunt folosiți la validarea pe `dev` în timpul fine-tuning-ului;
- `--test-num-workers` controlează câți worker-i sunt folosiți la evaluarea finală pe `test`;
- aceste opțiuni accelerează încărcarea și preprocesarea batch-urilor, nu inferența GPU în sine.

La fiecare rulare a lui `run_pipeline.py` se creează automat un fișier de log unic:

- `artifacts/logs/run_pipeline_YYYY-MM-DD_HH-MM-SS.log`

Acest fișier conține:

- logurile orchestratorului `run_pipeline.py`;
- output-ul complet al etapelor lansate prin subprocese.

## GPU vs CPU

- antrenarea și evaluarea detectează automat dacă `torch.cuda.is_available()` este `True`;
- dacă există CUDA funcțional, modelul rulează pe GPU;
- dacă nu există GPU, proiectul cade automat pe CPU;
- `--fp16` nu trebuie folosit pe CPU.
- `--validation-num-workers` și `--test-num-workers` folosesc CPU pentru a pregăti mai repede batch-urile care ajung pe GPU.

## Artefacte generate

Cele mai importante rezultate apar în:

- `artifacts/datasets/common-voice-scripted-speech-25-0-romani-701de4ae/`
- `artifacts/logs/run_pipeline_YYYY-MM-DD_HH-MM-SS.log`
- `artifacts/cv_ro/manifests/train.csv`
- `artifacts/cv_ro/manifests/dev.csv`
- `artifacts/cv_ro/manifests/test.csv`
- `artifacts/cv_ro/manifests/summary.json`
- `artifacts/whisper-base-ro/`
- `artifacts/whisper-base-ro/model.safetensors` — checkpoint versionat prin `Git LFS`
- `artifacts/whisper-base-ro/training_summary.json`
- `artifacts/evaluation/comparison.json`

## Metrici

Evaluarea comparativă salvează:

- `WER` — word error rate;
- `CER` — character error rate;
- exemple comparative între referință, baseline și modelul fine-tuned.

## Teste

Pentru a valida utilitarele principale:

```powershell
python -m pytest tests -q
```

## Troubleshooting

- `ArrowMemoryError` — versiunea curentă folosește feature extraction on-the-fly și ar trebui să evite problema; dacă apare, verifică să nu rulezi cod mai vechi.
- `TypeError` pentru `evaluation_strategy` — proiectul folosește `eval_strategy`, compatibil cu versiunea curentă de `transformers` instalată.
- `TypeError` pentru `tokenizer` în `Seq2SeqTrainer` — proiectul folosește `processing_class`.
- avertismentul `pin_memory` pe CPU este benign; proiectul îl dezactivează automat când nu există CUDA.
- dacă URL-ul presigned expiră, rerulează cu `--force-download`.
- dacă directorul extras este corupt sau incomplet, rerulează cu `--force-extract`.
- dacă `run_pipeline.py` nu găsește cheia API, verifică variabila `MOZILLA_DATA_COLLECTIVE_API_KEY` în sesiunea curentă.
- dacă rulezi pe CPU, folosește subsete mici prin `--prep-limit`, `--train-limit` și `--eval-limit`.
- dacă paralelizarea nu ajută sau încetinește, redu `--prep-num-workers`; pe HDD prea mulți workeri pot satura I/O.
- dacă `artifacts/whisper-base-ro/model.safetensors` lipsește după clone, rulează `git lfs install` și `git lfs pull`.
- dacă vezi în `model.safetensors` un fișier text scurt care începe cu `version https://git-lfs.github.com/spec/v1`, checkout-ul a fost făcut fără a descărca obiectul LFS; rulează `git lfs pull`.
- pe Windows, paralelizarea folosește procese separate; evită valori exagerate pentru `--prep-num-workers` și începe cu `2-4`.
- pentru validare și evaluare pe un singur GPU, începe cu `--validation-num-workers 2` și `--test-num-workers 2`; valori mai mari nu garantează accelerare.
- nu paraleliza inferența pentru baseline și fine-tuned pe același GPU; proiectul paralelizează doar I/O-ul și pregătirea batch-urilor.

## Observații importante

- `run_pipeline.py` nu cere `--dataset-root`; îl obține automat prin API și cache.
- manifestele folosesc căi relative, nu absolute.
- scrierea manifestelor rămâne în procesul principal, chiar dacă pregătirea audio rulează în paralel.
- proiectul este orientat pe portabilitate între local și Colab.
- notebook-ul curent este o variantă rapidă de pornire, iar `README.md` descrie fluxul complet al proiectului.
