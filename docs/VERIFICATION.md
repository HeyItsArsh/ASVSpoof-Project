# Verification record

This file is updated from executed commands. No benchmark metrics are inferred from generated fixtures.

Status on **22 October 2026**: implementation and generated-fixture verification complete; the official-data study is running asynchronously. Consult `artifacts/pipeline/status.json` for its current step. Real benchmark results remain unmeasured until the corresponding stage finishes.

Pending external prerequisites: complete official LA.zip download, checksum verification and extraction; full official data audit; three-seed model training/evaluation and planned ablations.

Commands and correct behavior:

| Check | Command (use `.venv/Scripts/python.exe` as Python) | Correct behavior |
|---|---|---|
| Unit/integration suite | `python -m pytest -q` | All generated-fixture tests pass; official-data test may skip until extraction |
| Synthetic mechanics | `python -m secure_audio sanity` | overfit >=95%, BCE <0.15; shuffled balanced accuracy 0.45–0.55; edge checks pass |
| Official file/content audit | `python -m secure_audio check-data --content-hash` | exact official counts, no overlap/content collision, audit receipt written |
| Real data controls | `python -m secure_audio sanity --real-data` | real training subset overfit; shuffled dev balanced accuracy around 0.5 |
| Primary study | `python -m secure_audio train` | three completed runs per model; real dev scores/checkpoints saved |
| Freeze | `python -m secure_audio freeze` | all seeds present, immutable-artifact manifest written before eval |
| Official eval | `python -m secure_audio evaluate` | 71,237 aligned eval scores per run, finite metrics/plots, no trial dropped |
| Seed spread | `python -m secure_audio aggregate` | means and sample SD from all three actual seeds |
| Ablations | `python -m secure_audio ablations`, then train/freeze/evaluate/aggregate with each generated config | four planned contrasts, three seeds each, no eval-based tuning |
| Local synthesis | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/generate_tts.ps1` | local WAVs and metadata manifest; no cloud TTS |
| External probe | `python -m secure_audio external --run artifacts/runs/mobile/seed_13` | detection/rejection counts; EER null for spoof-only probe |

The complete ordered study can be run with `python scripts/run_all.py`; durable step logs are in `artifacts/pipeline/`. Every uncompleted benchmark stage must be described as unmeasured in the report.

## Executed checks

- Final source suite: **37 passed, 1 skipped, no warnings**, in 33.11 seconds. The skip was `test_official_dataset_when_present`, because LA.zip had not yet finished downloading/extracting. The runner retained the output in `artifacts/pipeline/unit_tests.log`.
- Both classical models completed fit/save/reload/three-seed aggregation on generated protocol-shaped fixtures. MobileNet completed one training epoch, saved/reloaded its checkpoint, and scored fixture audio. These are code-path tests, not ASVspoof performance estimates.
- Known-answer reference EER tests passed at 0, 0.5 and 1, with an independent DET-accumulation equivalence check and separate tied-score diagnostic.
- Leakage tests passed for speaker, trial ID, resolved path and exact decoded content. Missing-file and duplicate-protocol detection also passed on fixtures.
- HTTP range download tests passed offline, including existing-prefix preservation, interrupted assembly resume and refusal of a server ignoring ranges. The actual official transfer is underway; the final archive checksum cannot be verified before completion.
- Local Windows TTS generation completed: **30 WAV clips**, three installed voices, two rate settings, five prompts. The manifest is `data/synthetic/manifest.json`.
- `python -m pip check`: **No broken requirements found**. A complete `requirements-lock.txt` was saved after installation.
- All 14 production Python source files parse; all production functions have docstrings and return annotations. `compileall` also passed during verification. The source was read through for protocol indexing, label/score orientation, feature shapes, seed/worker behavior, loss weighting, checkpoint selection/reload, score alignment and threshold isolation.
- Model parameter count measured from the actual adapted model: **2,224,577**. Initial runtime: Windows, Python 3.12.0, Torch 2.7.1 CPU, Torchvision 0.22.1; 8 logical CPU cores. The CPU-only Torch build did not expose CUDA. Subsequent hardware inspection confirmed an RTX 3050; see the CUDA transition below.

The final synthetic mechanics command also passed, with the retained output in `artifacts/pipeline/synthetic_sanity.log` and `artifacts/sanity_synthetic.json`:

| Diagnostic (generated data only) | Observed outcome |
|---|---|
| Eight-example overfit, 200 steps | 100% training accuracy; inference BCE 0.0002513 after documented tiny-set BN calibration |
| Shuffled SVM, three seeds | Balanced accuracy mean 51.66%, sample SD 5.32 percentage points |
| Shuffled boosting, three seeds | Balanced accuracy mean 52.10%, sample SD 2.77 percentage points |
| Shuffled MobileNet, three seeds | Balanced accuracy mean 50.00%, sample SD 0.00 percentage points |
| Silence and very short input | Explicit rejection |
| White noise | Finite features; checkpoint inference test also confirmed a finite score and noise warning |

These small synthetic controls are **not real-data accuracy/EER results**. The SVM's individual synthetic seed values range from 48.39% to 57.80%; the diagnostic pass band applies to the three-seed mean, not every individual small-sample run. Real train/dev controls are still queued.

## Review corrections

The review added a completion marker written only after dev artifacts finish, atomic model saves, a freeze check inside the evaluation function, config-consistency checks across frozen runs, resume checks and preservation of incomplete checkpoints, and a disk-space check before allocating caches. Windows CPU discovery was bounded through `LOKY_MAX_CPU_COUNT`; the Pyparsing dependency was pinned to remove upstream compatibility warnings. Global spatial mean pooling replaces the equivalent adaptive pooling operation to avoid a potential deterministic-CUDA backward limitation; CUDA behavior remains untested here.

## Not completed at this checkpoint

**Subsequent acquisition update:** the full archive was downloaded, matched its publisher MD5 and extracted. The first real audit exposed a checker bug: official ASV enrollment recordings were incorrectly treated as extra CM files. The checker now whitelists enrollment IDs from the official enrollment protocols, without adding them to CM trials. A regression test was added. Pipeline state preserves the verified download and is prepared for `scripts/run_all.py --resume`; verification and the full content audit are being repeated. The earlier pending-download descriptions below reflect the initial delivery checkpoint.

1. **Official archive checksum/extraction and full file/content audit:** waiting for transfer. The official metadata reports 7,640,952,520 bytes and MD5 `30c98f11d8b2bc21f2c257bfd78bb5c5`. Exact commands and expected counts are above and in README. No receipt is claimed before verification.
2. **Real-data overfit and shuffled-label controls for SVM, boosting and MobileNet:** queued after the audit. Correct results are the documented diagnostic pass criteria, not pre-filled observations.
3. **Real three-seed primary training/eval, all four planned ablations, per-attack analysis and seed aggregates:** queued. Correct output is a complete aligned score table for each split/run and finite metrics; no particular EER/accuracy is guaranteed.
4. **External TTS scoring with real trained checkpoints:** queued. Correct output records every clip as scored or rejected, reports a spoof-only detection rate and leaves EER null.
5. **GPU execution/determinism and CPU-versus-GPU runtime comparisons:** CUDA execution was initially unverified because the installed wheels were CPU-only. This has now been superseded by the successful CUDA transition below. A controlled CPU-versus-GPU full epoch comparison has not been performed; no speedup ratio is claimed.

The unattended runner stops at any failed step and writes the failing command/log location. To continue a stopped transfer or completed-stage sequence, use `.venv/Scripts/python.exe scripts/run_all.py --resume`. Do not start a second runner while the current one is active. Optimizer-state resume within an epoch is not provided; incomplete training runs restart from their original seed.

## Audit save failure recovery

Two overlapping full content audits reached completion. One successfully wrote `data/LA/audit.json`; the other failed with Windows `PermissionError` while replacing the shared `audit.json.tmp`. The final receipt and leftover temporary receipt were compared and are identical. The stored counts match all three official splits, the receipt has `decoded_content_checked: true`, and every protocol SHA-256 was independently checked against the current file. No trained model runs existed at recovery time.

The JSON writer now uses a unique temporary filename per writer and retries transient Windows sharing violations. A pipeline-wide OS file lock prevents simultaneous runner instances; it is released automatically when a process exits. New regression tests exercise concurrent writers, simulated transient locks and duplicate-runner exclusion. Directory resolution and file metadata scans were also adjusted to avoid repeated slow per-file path resolution on OneDrive while retaining symlink-aware cross-split checks. Audit progress now prints every 1,000 trials.

The successful audit receipt is retained. A versioned status backup records the recovery; the resumed pipeline proceeds to real-data sanity controls after regression verification, without repeating a completed content scan. This recovery changes operational file handling, not model settings, splits, random seeds or metrics.

Recovery verification completed: **50 tests passed in 63.98 seconds**, including the official-dataset overlap test, JSON concurrency/retry tests, pipeline locking and the website API tests. The combined run exposed a website cleanup timing race; temporary uploads are now removed before sending the success response, and the inference lock is released even if cleanup raises. The full suite passed after that correction. Evidence of receipt validation and the preserved previous study fingerprint is saved in `artifacts/pipeline/audit_save_recovery.json`. The full pipeline was restarted with `--resume`; its new console logs are `recovery.stdout.log` and `recovery.stderr.log`.

## Website addition

The local site uses `web_server.py` and `website/dist/` without adding dependencies or changing the study's source/config fingerprint. All **8 website API tests passed** in 26.01 seconds. They cover actual HTTP status/static serving, missing checkpoints, model/seed validation, upload bounds, origin/Host validation and detector integration/temporary-file cleanup using a generated test-only estimator. JavaScript syntax and Python compilation checks passed. Browser inspection confirmed the desktop layout, navigation, live untrained-model status, experiment table and quick-start dialog. Browser file-picker upload, mobile viewport interaction and real trained-model predictions remain unverified; API upload behavior was exercised by tests.

The hosted private interface was successfully deployed at `https://voxguard-audio-lab-menon.indigotitan512.chatgpt.site`. The browser reached its ChatGPT sign-in gate; post-login hosted browser QA was unavailable without the user's sign-in. Hosted Python inference is not offered. Use the local app at `http://127.0.0.1:8765` for actual checkpoint access. Models are still unavailable while the real study is auditing/training, so the UI disables analysis rather than supplying fabricated scores.

## CUDA transition — 22 October 2026

The NVIDIA driver and hardware were independently inspected: **NVIDIA GeForce RTX 3050 6GB Laptop GPU**. An isolated `.venv-gpu` was created with **Torch 2.7.1+cu126**, **Torchvision 0.22.1+cu126** and all other dependencies matching the original lockfile. The first installer encountered a timeout on an unnecessary unpinned dependency; the CUDA wheel was retained in pip's cache and installation completed with pinned dependencies and longer timeouts. The CPU environment was not modified.

- `pip check` passed with no broken requirements.
- Actual CUDA forward/backward updates passed at shape `[16, 1, 64, 401]` with deterministic algorithms enabled. Repeated inference and GPU checkpoint reload produced identical outputs.
- Peak allocated CUDA memory during this check was **723.06 MiB** on a GPU reporting **6143.5 MiB** total. The measured synthetic optimizer update averaged **0.1002 seconds**; it is not a full epoch or an ASVspoof benchmark.
- **All 50 regression tests passed in 128.84 seconds** in the GPU environment. The JUnit report is `artifacts/gpu_setup/tests.xml`.
- All synthetic sanity controls passed again with the overfit loop explicitly reporting `device=cuda`. Previous CPU synthetic results were preserved in `artifacts/gpu_setup/sanity_synthetic_cpu.json`.
- Completed train/dev MFCC and spectrogram caches were retained. The CPU runner was stopped after both spectrogram caches completed; the interrupted real-data control suite restarts under the GPU environment.
- GPU environment versions are pinned separately in `requirements-gpu-lock.txt`. `artifacts/gpu_setup/migration.json` records the original/new source fingerprints, hardware, checks and preserved cache file sizes. Only device/timing/progress logging was added to the training/control code; model settings, splits, features and seeds did not change.
- A single GPU-environment pipeline runner was launched with `--resume`. The real-data log confirmed `Tiny-set overfit: device=cuda`. New runner console logs are `artifacts/pipeline/gpu.stdout.log` and `gpu.stderr.log`.

Neural training and inference use CUDA via the unchanged `deep.device: auto` configuration. SVM, gradient boosting, audio preprocessing and audits remain CPU operations. Full real-data evaluation, ablations and final results remain pending; no genuine benchmark scores are inferred from the CUDA checks.
