# Report outline

1. **Introduction:** voice authentication threat model; TTS/VC injection in logical access; purpose of a lightweight countermeasure; distinction from speaker identity verification.
2. **Dataset:** cite ASVspoof 2019 and the evaluation plan; exact official LA split/class/speaker counts; known versus unseen attacks; protocol fields; license and acquisition; overlap/content audits; no random train/dev/eval split.
3. **Method:** shared 16-kHz preprocessing and clip policy; invalid-input behavior; MFCC/delta/delta-delta statistics, train-only scaling, SVM and histogram boosting; log-Mel MobileNetV2, single-channel adaptation, from-scratch initialization, weighted BCE, deterministic training; justify the resource tradeoff.
4. **Experiments:** fixed three seeds and versioned configs; train-only fitting, dev-only selection/thresholds, pre-eval freeze; hyperparameter budgets and runtime; planned Mel/log-Mel, duration and augmentation contrasts; unit/integration tests, tiny real subset overfit, shuffled-label controls, input edge cases; independent local TTS probe design.
5. **Results:** actual dev/eval EER with mean and sample SD, ordinary/balanced accuracy, AUC, ROC curves and confusion matrices; per-attack EER and spoof acceptance; attack availability; parameter/model-size/runtime figures; ablations; external spoof-only detection/rejection rates. Distinguish reference discrete EER from interpolated diagnostic EER. Explain that t-DCF, the primary 2019 challenge metric, was not evaluated.
6. **Limitations:** unseen generators, codecs/microphones/languages, short clips, artificial padding/cropping, imbalance, class-versus-speaker confounds, determinism limits across hardware, budget and uncertainty; include missing experiments honestly.
7. **Ethics:** consent and privacy for genuine external recordings, no identity cloning, dataset attribution, local processing, restricted storage, false-positive harm, no security guarantees or autonomous authentication deployment.

Appendices: complete config/versions, seed-level tables, audit receipts/protocol hashes, exact commands, test/verification record and artifact inventory.
