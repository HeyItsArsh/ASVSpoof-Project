# Lightweight Spectrogram-Based Deepfake Audio Detection for Secure Authentication

A university research pipeline for **ASVspoof 2019 Logical Access**, comparing MFCC/delta/delta-delta SVM and gradient boosting with a one-channel MobileNetV2 trained from scratch. This is a spoofing countermeasure, not a speaker verifier or a production authentication system.

## Setup

Run commands in this project folder. Python **3.11 or 3.12**, 64-bit, is required; the tested environment is recorded in `docs/VERIFICATION.md`. Windows is the primary platform; the model code also runs on Linux, while the supplied local synthesis script uses Windows voices.

One setup command:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1
```

All commands below explicitly use the virtual environment, so activation is unnecessary. Direct dependencies are pinned in `requirements.txt`; `requirements-lock.txt` records the complete installed environment when available. The default wheels use CPU here. For a small CUDA GPU, install a matched Torch/Torchvision pair using the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/), record the replacement versions, and keep the config otherwise unchanged.

## Download the dataset

1. Download the **Logical Access (LA)** archive from the [official ASVspoof 2019 dataset page at Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336). Use `LA.zip`; this project does not use the PA archive.
2. Read the [ASVspoof 2019 dataset paper](https://arxiv.org/abs/1911.01601) for dataset details and the [official evaluation plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf) for protocols and metrics.
3. From the project folder in PowerShell, install the project environment:

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1
   ```

4. Download, verify the publisher checksum, and extract the archive into the project's `data/` folder:

   ```powershell
   .\.venv\Scripts\python.exe -m secure_audio.download --extract
   ```

5. Verify the extracted files and official split protocols before training:

   ```powershell
   .\.venv\Scripts\python.exe -m secure_audio check-data --content-hash
   ```

Allow about 30 GiB of free disk space for the primary study; 60 GiB or more is recommended when running all ablations. The audio is downloaded locally and is not included in this GitHub repository.
## Data: official sources only

Download **LA.zip**, not PA.zip, from [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336). The archive is approximately 7.12 GB. The [dataset paper](https://arxiv.org/pdf/1911.01601) explains the dataset; it does not contain the audio. The [official evaluation plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf) defines the challenge.

Check missing data/layout and free space:

```powershell
.\.venv\Scripts\python.exe -m secure_audio check-data
```

Download, verify the publisher's checksum, and extract with a resumable command:

```powershell
.\.venv\Scripts\python.exe -m secure_audio.download --extract
```

If interrupted, run the same command again. Four bounded parallel HTTP ranges improve throughput; each range resumes separately. A server ignoring exact ranges stops with an explicit error; use `--connections 1` for a sequential transfer (which restarts cleanly if Range is ignored). The original archive is preserved. Checksum mismatch stops extraction. Do not run two downloaders concurrently. About 30 GiB free is recommended for a primary study; **60+ GiB** is safer for all feature caches and ablations. Download speed depends on the host and connection.

Expected structure (no renamed trials and no manually generated split):

```text
data/
  LA.zip
  download_receipt.json
  LA/
    ASVspoof2019_LA_cm_protocols/
      ASVspoof2019.LA.cm.train.trn.txt
      ASVspoof2019.LA.cm.dev.trl.txt
      ASVspoof2019.LA.cm.eval.trl.txt
    ASVspoof2019_LA_train/flac/LA_T_*.flac
    ASVspoof2019_LA_dev/flac/LA_D_*.flac
    ASVspoof2019_LA_eval/flac/LA_E_*.flac
    README.LA.txt
```

Keep the official ASV protocol/score folders too, if included, although this project's standalone CM metrics do not consume them. CM protocol fields are `speaker utterance - attack key`. Bona fide is encoded as 1; spoof as 0. Larger continuous scores mean more bona fide.

Dev/eval folders also contain official ASV enrollment recordings. The checker allows those files only when their IDs appear in the official male/female enrollment protocols; they are excluded from CM model fitting and evaluation.

Audit every file and decoded PCM cross-split content before training:

```powershell
.\.venv\Scripts\python.exe -m secure_audio check-data --content-hash
```

Correct counts:

| Split | Bona fide | Spoof | Total |
|---|---:|---:|---:|
| Train | 2,580 | 22,800 | 25,380 |
| Dev | 2,548 | 22,296 | 24,844 |
| Eval | 7,355 | 63,882 | 71,237 |

Audits reject overlapping speaker IDs, trial IDs, resolved paths, cross-split identical decoded waveforms, duplicate protocol entries, missing/extra FLAC files, wrong attack IDs, and unexpected sampling rate/channel count. The audit receipt records protocol hashes. Re-audit after replacing/modifying audio. Cache keys include paths, sizes and modification times; deliberately modified files with preserved metadata require clearing the cache and re-auditing. No dev/eval observations contribute to fitted normalization statistics or model updates. They are never augmented or reassigned to another split.

## Model and preprocessing choices

For this laptop's RTX 3050, use the isolated CUDA environment described in `docs/GPU_SETUP.md`. Setup command: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/setup_gpu.ps1`. GPU runs use `.venv-gpu/Scripts/python.exe`; the original `.venv` remains the CPU fallback. `deep.device: auto` selects the available device without changing preprocessing or model settings.

- Shared front end: mono 16 kHz, four seconds, deterministic center crop or right zero-padding. No peak normalization or silence trimming. Invalid audio stops a benchmark rather than dropping trials. Inference reports an explicit rejection for corrupt, nonfinite, silent or under-100 ms clips.
- Baselines: 20 MFCCs plus first and second derivatives; time mean and standard deviation produce 120 features. An RBF SVM uses train-only `StandardScaler`, balanced class weights and three C candidates. Histogram gradient boosting uses balanced class weights and three leaf-count candidates; internal random validation splitting is disabled.
- Deep model: 64-bin log-Mel power dB, 512-point FFT, 160-sample hop; input `[batch, 1, 64, 401]` at four seconds. MobileNetV2 uses one input channel and one output logit, weighted binary cross entropy and AdamW. Train BatchNorm statistics are used at evaluation; no ImageNet weights/downloads are involved. Dropout and shuffling use fixed seeds. Dev EER selects checkpoints; patience is five epochs, maximum twenty.
- Why MobileNetV2: its depthwise convolutions and smaller parameter budget fit the laptop goal. The single-channel, single-output variant's actual parameter count is recorded in each run. ResNet-18 is a valid alternative, but larger. Smaller FLOPs do not guarantee faster CPU inference: record measured times. [Torchvision architecture documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html)
- Primary comparison: identical official trials, clip duration, no augmentation, three seeds (13, 37, 73), identical score orientation and metrics. Selection budgets are three predefined baseline candidates versus up to twenty deep checkpoints; report this difference. Baselines may be deterministic and show zero spread across seeds. Do not manufacture variation.
- Class imbalance: always report balanced accuracy as well as requested ordinary accuracy; a majority-class classifier can have high ordinary accuracy.

## Tests and sanity checks

Run unit/integration tests (generated fixtures are clearly separate from LA):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover loading, resampling/downmixing, padding/cropping, malformed/silent/short/nonfinite audio, finite noise features, feature dimensions, distinct Mel/log-Mel transforms, reproducible augmentation and initialization, known-answer EERs (0%, 50%, 100%), reference DET accumulation, tie diagnostics, fixed thresholds, split/content leakage, caches, both baseline train/save/evaluate round trips, deep checkpoint/inference, freeze integrity and three-seed aggregation. The official-data overlap test is skipped until the archive is extracted.

Synthetic mechanics checks, independent of the archive:

```powershell
.\.venv\Scripts\python.exe -m secure_audio sanity
```

Actual training-data diagnostics:

```powershell
.\.venv\Scripts\python.exe -m secure_audio sanity --real-data
```

The overfit test memorizes four training trials per class, uses a fixed one-second diagnostic crop and no augmentation, and requires >=95% train accuracy and BCE <0.15 after 200 steps. Tiny-set BatchNorm recalibration is confined to this diagnostic and is reported explicitly. It is not held-out performance. Increase `--steps` only if inspecting a failed diagnostic warrants it; do not hide failures.

The shuffled-label controls fit SVM, boosting and MobileNet on permuted training labels and evaluate **untouched dev labels**, with no tuning. MobileNet uses the fixed epoch budget and logit-zero threshold, scoring dev only after training. Mean balanced accuracy across three seeds is expected in 0.45–0.55, not necessarily ordinary accuracy near 50%. Run the deep control alone using `sanity --only shuffle-deep --real-data`. A diagnostic failure stops the full runner for inspection.

Silence and very short audio must be rejected. White noise must yield finite features/outputs without a crash; inference emits a spectral-flatness noise warning. This heuristic does not establish that arbitrary input is human speech. No noise gate filters official evaluation trials.

## Train, freeze, evaluate, summarize

Train all three models for all three seeds using train/dev only:

```powershell
.\.venv\Scripts\python.exe -m secure_audio train
```

Individual model: append `--model svm`, `--model boosting` or `--model mobile`. A single debugging seed can be specified with `--seed 13`, but a final report requires all three. The command refuses to overwrite completed runs. Add `--resume` to skip completed runs whose saved config matches. A checkpoint without `run.json` is an incomplete run: it is preserved as `model.interrupted.*`, then training restarts from its original seed. Optimizer/epoch continuation is not implemented. The full runner stops on errors and uses this resume policy when restarted.

Lock model weights, hyperparameters, selected checkpoints and dev thresholds:

```powershell
.\.venv\Scripts\python.exe -m secure_audio freeze
```

Evaluate the frozen runs on official eval:

```powershell
.\.venv\Scripts\python.exe -m secure_audio evaluate
```

Summarize the per-run results:

```powershell
.\.venv\Scripts\python.exe -m secure_audio aggregate
```

Each run saves `run.json`, a model artifact, and `dev/`/`eval/` folders containing `scores.csv`, `metrics.json`, `roc.npz`, `roc.png` and `confusion_matrix.png`. `aggregate.json` reports mean, **sample standard deviation** (ddof=1), min/max and individual values across seeds. Metrics are stored as fractions: multiply EER/accuracy by 100 for percentages. The spread measures run variability, not a population confidence interval.

**Metric convention:** EER follows the ASVspoof reference stable-sort discrete DET rule, choosing the nearest FAR/FRR point and averaging them. An additional interpolated, tie-aware EER is saved because legacy tied-score behavior can be misleading. Scores should be continuous. [Organizers' reference implementation](https://github.com/asvspoof-challenge/2021/blob/main/eval-package/eval_metrics.py)

Accuracy/confusion matrices use an **attainable dev-selected equal-error threshold**, frozen for eval. Eval EER necessarily traces eval thresholds; that descriptive calculation is not a deployed threshold. Per-attack EER compares each spoof attack with **all bona fide trials in that same split**. Per-attack spoof acceptance uses the pooled dev threshold. A01–A06 are available in dev, A07–A19 in eval; absent rows are unavailable, never zero.

**Challenge terminology:** EER was an official secondary metric in ASVspoof 2019; minimum t-DCF was the primary metric. This project implements standalone CM EER, not t-DCF, and cannot claim full official challenge ranking equivalence. t-DCF also requires a specified ASV system, scores and costs.

## Ablations

Generate configs before inspecting eval results:

```powershell
.\.venv\Scripts\python.exe -m secure_audio ablations
```

Four one-factor contrasts against the default reference: linear Mel vs log-Mel, two vs four seconds, six vs four seconds, and training-only gain/noise augmentation on vs off. Augmented training also randomizes crop position; state this bundled change in the report. No per-clip z-scoring erases the representation comparison. Shared caches are reused where valid; output roots keep experiment models separate.

For each generated configuration (example below), run the same train/freeze/evaluate/aggregate steps:

```powershell
.\.venv\Scripts\python.exe -m secure_audio --config configs/ablation_mel.yaml train --model mobile
```

```powershell
.\.venv\Scripts\python.exe -m secure_audio --config configs/ablation_mel.yaml freeze
```

```powershell
.\.venv\Scripts\python.exe -m secure_audio --config configs/ablation_mel.yaml evaluate
```

```powershell
.\.venv\Scripts\python.exe -m secure_audio --config configs/ablation_mel.yaml aggregate
```

Repeat for `ablation_length_2s.yaml`, `ablation_length_6s.yaml`, `ablation_augmentation.yaml`. The reference default run is the fourth-second/off/log-Mel condition and is not retrained as a duplicate. These are planned contrasts, not a grid search using eval. Select any preferred system on dev and freeze it before viewing eval; do not tune against attack-level eval feedback.

## Local synthetic generalisation probe

Generate clips entirely locally with installed Windows SAPI voices:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/generate_tts.ps1
```

The script reads `scripts/tts_prompts.txt`, varies installed voices and rate, and records voice/text/engine metadata in `data/synthetic/manifest.json`. It refuses to overwrite existing clips. These are generic voices, not impersonations. SAPI is a modest supplementary test, not a modern neural-TTS benchmark. Check Windows voice redistribution terms before sharing recordings.

Score them with a frozen trained run:

```powershell
.\.venv\Scripts\python.exe -m secure_audio external --run artifacts/runs/mobile/seed_13
```

Repeat for all seeds/models for a fair external comparison. Output is `external/probe.json`, including rejected clips and a spoof-detection rate for scored clips. **No EER/ROC/accuracy claim is made for a spoof-only collection.** A matched, consented bona fide external collection is needed for a two-class external benchmark. Do not treat these TTS clips as A01–A19 or mix them into official splits.

Score an individual recording:

```powershell
.\.venv\Scripts\python.exe -m secure_audio infer --run artifacts/runs/mobile/seed_13 --audio path/to/clip.wav
```

Only load this project's trusted model files; serialized model artifacts are not a safe untrusted interchange format.

## Website

Start the local audio-analysis website from this project folder:

```powershell
.\.venv\Scripts\python.exe web_server.py
```

Visit **http://127.0.0.1:8765**. It provides audio upload/playback, a waveform preview, model/seed selection, actual detector scores, JSON export and live experiment status. Model analysis becomes available automatically after a run finishes training. The website does not start or modify training.

The [private hosted interface](https://voxguard-audio-lab-menon.indigotitan512.chatgpt.site) requires ChatGPT sign-in and previews the design; actual Python inference runs through the local app. See `website/README.md` for details. Website API verification: `.venv/Scripts/python.exe -m pytest website_tests -q`.

## Full unattended study

After setup, one command runs tests, mechanics checks, official acquisition/audit, real controls, the primary comparison and all planned deep ablations, freezes everything before eval, produces aggregate results and scores the external clips:

```powershell
.\.venv\Scripts\python.exe scripts/run_all.py
```

Resume completed stages after a transfer interruption using:

```powershell
.\.venv\Scripts\python.exe scripts/run_all.py --resume
```

Read `artifacts/pipeline/status.json` and per-step `.log` files for progress. A nonzero child exit stops the pipeline. Source/config fingerprints prevent silently resuming a changed study. This is substantial work on a CPU laptop: downloads and three-seed training/ablations may take many hours or days. The runner does not invent or pre-fill scores while waiting. It does not implement optimizer-state resume within an interrupted epoch.

An OS file lock prevents starting a second pipeline runner. JSON artifacts use unique temporary files and retry transient Windows sharing locks. During an audit, the log prints progress every 1,000 trials. Historical errors may remain earlier in append-only logs; use the latest status and timestamped recovery entries to distinguish them from the current run.

## Project files and reporting

```text
secure_audio/    # protocols, audio/features, caches, models, training, evaluation, CLI
configs/         # reference config and generated ablation configs
scripts/         # setup, full study runner, local TTS generation/prompts
tests/           # generated-fixture unit and integration tests
docs/            # results template, report outline, limitations, verification record
data/            # private local data, excluded from version control
artifacts/       # caches, scores, checkpoints, figures and logs, excluded from version control
```

Fill `docs/RESULTS_TEMPLATE.md` with actual saved outputs. Use `docs/REPORT_OUTLINE.md` for the report and `docs/LIMITATIONS.md` for the caveats. `docs/VERIFICATION.md` distinguishes executed checks from work blocked by data or compute. Cite the dataset/evaluation plan and the MobileNetV2 architecture in the report; retain the dataset's attribution/license materials.
