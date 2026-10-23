"""Create a self-contained review snapshot without changing ongoing experiments."""
from pathlib import Path
import csv
import hashlib
import io
import json
import statistics
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Archive completed runs only, plus source and clearly labelled partial results."""
    now = datetime.now(timezone.utc).isoformat()
    status = json.loads((ROOT / 'artifacts/pipeline/status.json').read_text())
    runs = sorted(p.parent for base in ('artifacts/runs', 'artifacts/ablations')
                  for p in (ROOT / base).rglob('run.json'))
    rows = []
    files: set[Path] = set()
    for folder in runs:
        metadata = json.loads((folder / 'run.json').read_text())
        metrics = json.loads((folder / 'dev/metrics.json').read_text())
        relative = folder.relative_to(ROOT).as_posix()
        condition = relative.split('/')[2] if '/ablations/' in relative else 'primary'
        rows.append(dict(condition=condition, model=metadata['model'], seed=metadata['seed'],
                         dev_eer_percent=100 * metrics['eer'],
                         dev_accuracy_percent=100 * metrics['accuracy'],
                         training_seconds=metadata['training_seconds'], evidence=relative))
        files.update(p for p in folder.rglob('*') if p.is_file() and '.tmp' not in p.name)
    for directory in ('secure_audio', 'configs', 'scripts', 'tests', 'website_tests', 'docs', 'website/dist'):
        files.update(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    for pattern in ('README*', 'requirements*.txt', 'pyproject.toml', 'pytest.ini', 'web_server.py'):
        files.update(ROOT.glob(pattern))
    for directory in ('artifacts/pipeline', 'artifacts/gpu_check', 'artifacts/gpu_setup'):
        files.update(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and p.suffix in ('.json', '.xml', '.log'))
    files.update(ROOT.glob('artifacts/sanity*.json'))
    files.update(ROOT.glob('data/synthetic/*'))
    files.update(ROOT.glob('data/external_human/*.wav'))
    files.add(ROOT / 'data/LA/audit.json')
    files.update(p for p in (ROOT / 'data/LA').rglob('*.txt') if 'protocol' in str(p).lower())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    summary = ['| Condition | Model | Seeds completed | Dev EER % mean ± sample SD | Dev accuracy % mean ± sample SD |',
               '|---|---|---|---|---|']
    for condition, model in sorted({(r['condition'], r['model']) for r in rows}):
        group = [r for r in rows if (r['condition'], r['model']) == (condition, model)]
        def fmt(key: str) -> str:
            values = [r[key] for r in group]
            return f'{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}' if len(values) > 1 else f'{values[0]:.4f} (one run)'
        summary.append(f"| {condition} | {model} | {len(group)}/3 | {fmt('dev_eer_percent')} | {fmt('dev_accuracy_percent')} |")
    handoff = f'''# Review handoff: Lightweight Spectrogram-Based Deepfake Audio Detection

Snapshot UTC: {now}. This is a partial-study snapshot, not a completed benchmark.
Read this file first, then docs/REPORT_OUTLINE.md and docs/LIMITATIONS.md.

## What is finished

{len(rows)} of 21 planned main/ablation training runs have completion markers.
The three main models have all three seeds completed. Main models can be demonstrated locally.
Completed pipeline stages: {', '.join(status['completed'])}.
At capture, pipeline state was {status['state']}; stage was {status.get('current_step')}.
The included status/logs are a snapshot and will not update on your computer.
Training on the original laptop continues independently of this ZIP.

## Actual preliminary results

These are DEVELOPMENT results, used for checkpoint/hyperparameter/threshold selection.
They are optimistic selection results, not independent test performance.
Values below are calculated from included dev/metrics.json files; SD uses n-1.
No eval result is claimed here. Do not replace missing eval results with dev numbers.

{chr(10).join(summary)}

See REVIEW_RESULTS.csv for each seed and its evidence directory.
Each completed run includes raw scores, ROC and confusion-matrix PNGs, metrics,
configuration/protocol provenance, and its saved model. Individual attack results are
inside dev/metrics.json. Lower EER is better. Accuracy is affected by class imbalance.

## Procedure used

1. Downloaded official ASVspoof 2019 Logical Access archive and checked publisher MD5.
2. Used official CM train/dev/eval protocols, never a random split. Counts: train
   25,380 (2,580 bonafide); dev 24,844 (2,548 bonafide); eval 71,237 (7,355 bonafide).
   Audited speaker IDs, file identities, and decoded audio hashes across splits.
   Extra official ASV enrollment files were verified separately and excluded from CM training.
3. Loaded mono audio at 16 kHz with explicit silence/short-clip handling. Main duration
   4 seconds; inspect secure_audio/audio.py for exact crop/pad and normalization rules.
4. Classical baseline: Librosa 20 MFCCs plus first/second deltas, time aggregation;
   standardised RBF SVM and histogram gradient boosting. Dev EER selected predefined
   SVM C values 0.1/1/10 and boosting leaves 7/15/31 (150 iterations).
5. Deep model: 64-bin log-Mel, FFT 512, hop 160, one-channel MobileNetV2 trained from
   scratch. MobileNetV2 was chosen for laptop resource use through depthwise separable
   convolutions. This is a design justification, not a measured comparison with ResNet-18.
   About 2.22M parameters; batch 16, learning rate 0.001, weight decay 0.0001;
   maximum 20 epochs, patience 5, best dev-EER checkpoint. See source for optimiser details.
6. Fixed seeds 13,37,73. Baselines ran on CPU; neural training used RTX 3050 6GB,
   Torch 2.7.1+cu126. Feature caches reused across runs. Separate CPU environment preserved.
7. Verified tiny-set overfit, shuffled-label negative controls and silence/short/noise
   handling. Actual receipts: artifacts/sanity_real.json and sanity_synthetic.json.
   Prior full GPU regression run: 50 tests passed; tests were not rerun for packaging.
8. Ablations change one setting relative to primary: linear Mel, 2 seconds, 6 seconds,
   or online augmentation (gain/noise/random crop; consult implementation).
9. Evaluation computes EER, accuracy, balanced accuracy, ROC AUC/curve, confusion matrix
   and attack breakdown. Decision threshold comes from dev and is frozen for eval.
   A01-A06 occur in train/dev; A07-A19 occur in eval, so not every attack exists in every split.
10. Final planned stages freeze all configurations before eval, evaluate all conditions,
    aggregate across seeds, and test local Windows SAPI synthetic clips.

## What remains / what not to claim

Augmentation training was unfinished at capture. Official eval and final aggregation
had not completed. No final unseen-attack accuracy/EER or completed external-test rate
is claimed. EER is a countermeasure metric; the 2019 challenge primary metric min t-DCF
is not implemented. Do not describe this as a complete official challenge submission.
Shuffled-label controls should be interpreted with balanced accuracy because of imbalance.
Three seeds are repeated runs, not cross-validation; deterministic baselines may agree.

## Observed real-world failure

In a manual website test, a human Open Speech Repository 8 kHz recording was classified
as spoof by MobileNet and bonafide by both baselines (user observation, not a controlled
benchmark). Domain/bandwidth mismatch is a hypothesis, not a proven cause. More epochs
do not ensure robustness. Do not tune on this example or on official eval data.
Source: https://www.voiptroubleshooter.com/open_speech/american.html
Attribution: Open Speech Repository. Included sample is separate from ASVspoof training.

## How to write the report

Introduction: authentication risk and project scope. Dataset: official LA protocols,
counts, imbalance, attack separation, leakage audit. Method: preprocessing and both
baseline/deep families. Experiments: seeds, dev selection, sanity checks, ablations,
hardware. Results: label the table here preliminary DEV results and cite evidence;
insert ROC/confusion PNGs with model/seed/split captions. Leave eval as pending.
Limitations: unseen generators, channels, short speech, language/domain shift, false
rejection and lack of ASV integration. Ethics: consent, privacy, no identity guarantee,
and human review rather than automatic denial. Conclusion: implemented pipeline and
measured preliminary findings, with remaining work explicitly named.

## Demonstrate on another laptop

Extract the ZIP, open PowerShell inside its folder, then follow README.md/setup.ps1.
CPU inference is sufficient: .\\.venv\\Scripts\\python.exe web_server.py
Open http://127.0.0.1:8765 and use included local TTS/human samples.
No dataset download is needed for inference with included completed checkpoints.
Training/evaluation requires the separately downloaded full dataset; follow README.
Do not launch run_all.py merely to demonstrate the site. GPU installation is optional.
Do not load model.joblib files from untrusted sources.

## Sources

https://www.asvspoof.org/
https://datashare.ed.ac.uk/handle/10283/3336
https://arxiv.org/abs/1911.01601

## Archive scope

Includes source, configs, docs, tests, completed model artifacts/dev results, protocol
files, audit/sanity receipts, progress logs and small demonstration clips.
Excludes full LA audio/archive, caches, Python environments, Git internals and incomplete
model directories. No metric was fabricated. SHA256_MANIFEST.json checks archived bytes.
'''
    output = ROOT / 'deliverables/review-handoff-2026-10-23.zip'
    output.parent.mkdir(exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for p in sorted(files):
            if not p.is_file():
                continue
            name = p.relative_to(ROOT).as_posix()
            data = p.read_bytes()
            z.writestr(name, data)
            manifest[name] = hashlib.sha256(data).hexdigest()
        for name, data in {'START_HERE_REPORT_HANDOFF.md': handoff,
                           'REVIEW_RESULTS.csv': buf.getvalue()}.items():
            z.writestr(name, data)
            manifest[name] = hashlib.sha256(data.encode()).hexdigest()
        z.writestr('SHA256_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for name, expected in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected, name
    (output.parent / 'START_HERE_REPORT_HANDOFF.md').write_text(handoff, encoding='utf-8')
    print(json.dumps({'zip': str(output), 'size_mb': round(output.stat().st_size / 1e6, 1),
                      'completed_runs': len(rows), 'verified_files': len(manifest)}))


if __name__ == '__main__':
    main()
