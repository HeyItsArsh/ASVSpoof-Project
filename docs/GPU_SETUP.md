# GPU environment

The laptop has an NVIDIA GeForce RTX 3050 Laptop GPU with 6 GiB VRAM. The original `.venv` contains CPU-only Torch; GPU work uses an isolated `.venv-gpu`, preserving the CPU environment as a fallback.

From the project folder, one setup command installs the matched Torch 2.7.1 / Torchvision 0.22.1 CUDA 12.6 builds, pinned study dependencies and a real CUDA forward/backward/checkpoint check:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/setup_gpu.ps1
```

This first-time installation needs internet and downloads approximately 2.7 GB for the Torch wheel, plus dependencies. Once installation and caching are complete, training is local. Instructions follow the [official version-specific PyTorch installation page](https://pytorch.org/get-started/previous-versions/#v271). A separate CUDA Toolkit installation is not needed for these prebuilt wheels; a compatible NVIDIA driver is required.

When available, `requirements-gpu-lock.txt` captures the complete tested environment and the official CUDA index. Do not replace the CPU `requirements-lock.txt`: the running study fingerprint includes it. GPU and CPU use identical code, configurations, data and seeds. Run metadata records actual library versions and the selected device. Cross-device identical floating-point results are not guaranteed.

Check real GPU operation:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/check_gpu.py
```

The check uses the configured `[16, 1, 64, 401]` batch, deterministic-algorithm enforcement, finite optimizer updates, repeated inference and checkpoint restoration. It saves actual runtime/memory observations in `artifacts/gpu_check/verification.json`. Its synthetic batch timings are not full epoch timings or ASVspoof performance metrics.

Resume the single full-study runner:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/run_all.py --resume
```

First stop an existing CPU runner at a safe stage boundary. Do not run both pipeline instances. Completed caches and the official audit are reusable. An interrupted sanity-control component may restart because the full sanity command only writes its final report after all components finish.

`deep.device: auto` already selects CUDA when available, so no model or preprocessing config change is needed. MobileNet training, neural sanity controls and inference use the GPU. MFCC/spectrogram caching, audits, scikit-learn SVM and gradient boosting remain on CPU.

To run the local website with CUDA inference:

```powershell
.\.venv-gpu\Scripts\python.exe web_server.py
```

Watch actual GPU usage:

```powershell
nvidia-smi -l 2
```

Keep the laptop plugged in and awake for the long experiment. If a configured batch does not fit, choose and document a smaller batch before producing comparison results, using the same batch setting across the planned deep runs.
