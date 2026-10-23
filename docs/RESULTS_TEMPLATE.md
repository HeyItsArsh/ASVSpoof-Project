# Results — fill only from actual run artifacts

Status: **not measured until the corresponding run completes**. Do not copy paper baseline numbers into this system's results.

Machine/OS: ____; CPU/RAM: ____; GPU: ____; Python: ____; dependencies/lockfile: ____
Dataset archive checksum/receipt: ____; protocol hashes/audit: ____
Commit or source snapshot: ____; seeds: 13, 37, 73; config paths: ____

All EER/accuracy entries below are **percent mean ± sample SD across three seeds**; retain raw seed values. A zero SD can be valid for deterministic baselines. Seed SD is not a confidence interval.

| System | Dev EER | Eval EER | Dev accuracy | Eval accuracy | Eval balanced accuracy | Eval AUC |
|---|---|---|---|---|---|---|
| MFCC + SVM | ____ | ____ | ____ | ____ | ____ | ____ |
| MFCC + histogram boosting | ____ | ____ | ____ | ____ | ____ | ____ |
| Log-Mel + MobileNetV2 | ____ | ____ | ____ | ____ | ____ | ____ |

Operational thresholds: chosen on dev, one per model/seed, saved in run.json. EER is reference discrete CM EER; attach supplementary tie-aware interpolated EER if ties matter. Minimum t-DCF is not implemented.

| Attack | Dev EER / spoof acceptance | Eval EER / spoof acceptance |
|---|---|---|
| A01 | ____ | N/A |
| A02 | ____ | N/A |
| A03 | ____ | N/A |
| A04 | ____ | N/A |
| A05 | ____ | N/A |
| A06 | ____ | N/A |
| A07 | N/A | ____ |
| A08 | N/A | ____ |
| A09 | N/A | ____ |
| A10 | N/A | ____ |
| A11 | N/A | ____ |
| A12 | N/A | ____ |
| A13 | N/A | ____ |
| A14 | N/A | ____ |
| A15 | N/A | ____ |
| A16 | N/A | ____ |
| A17 | N/A | ____ |
| A18 | N/A | ____ |
| A19 | N/A | ____ |

Repeat the attack table for every system, with mean ± SD and n_spoof. All attacks in a split share that split's bona fide comparison pool. Discuss A16/A19 as reference algorithms related to training A04/A06; the remaining eval systems test unseen algorithms.

| Planned deep contrast | Dev EER | Eval EER | Selected epoch per seed |
|---|---|---|---|
| Reference: log-Mel, 4s, augmentation off | ____ | ____ | ____ |
| Linear Mel, 4s, off | ____ | ____ | ____ |
| Log-Mel, 2s, off | ____ | ____ | ____ |
| Log-Mel, 6s, off | ____ | ____ | ____ |
| Log-Mel, 4s, gain/noise/random-crop augmentation | ____ | ____ | ____ |

Training runtime/model bytes/parameter count: ____
Inference runtime and whether cache creation was included: ____
Attach ROC curves and confusion matrices per seed; do not average incompatible raw thresholds or claim pooled scores are a single run.

Overfit real training subset: n=8; steps ____; train accuracy ____; BCE ____; pass/fail ____
Shuffled-label control: model ____; balanced accuracy seed values ____; mean/SD ____; pass/fail ____
Silence/short/noise behavior: ____

External TTS: engine/version/voices/rates/prompts ____; total/scored/rejected ____
Spoof detection per model/seed ____; no EER for this spoof-only set.

Unexecuted commands, reasons and expected outputs: ____
Main findings supported by real numbers: ____
