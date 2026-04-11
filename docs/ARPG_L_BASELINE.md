# ARPG-L Baseline Reproduction

This repo now includes helper scripts for the vanilla ARPG-L checkpoint evaluation flow. The target recipe matches the repo-recommended setup from `GETTING_STARTED.md`:

- `ARPG-L`
- `sample-schedule arccos`
- `cfg-scale 5.0`
- `step 64`
- FID against OpenAI's ImageNet-256 reference batch

## 1. Remote host requirements

Use a Linux x86_64 machine with NVIDIA GPUs. The vanilla training and sampling entrypoints require CUDA, so the full run cannot be reproduced on Apple Silicon.

## 2. Create environments

Create the main runtime environment:

```sh
conda create -n arpg python=3.10 -y
conda activate arpg
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
python -m pip install -r requirements-repro.txt
./scripts/check_repro_env.sh
```

Create a separate evaluation environment for OpenAI's `guided-diffusion` evaluator:

```sh
conda create -n arpg-eval python=3.10 -y
conda activate arpg-eval
python -m pip install -r requirements-eval.txt
```

## 3. Download baseline assets

```sh
./scripts/download_baseline_assets.sh
```

This downloads:

- `weights/arpg_300m.pt`
- `weights/vq_ds16_c2i.pt`
- `eval/VIRTUAL_imagenet256_labeled.npz`
- `external/guided-diffusion`

## 4. Run the smoke test

Default path uses a single GPU for parity with the runbook and to keep the baseline simple:

```sh
conda activate arpg
./scripts/run_arpgl_smoke.sh
```

If `torch.compile` causes trouble on your host, rerun with:

```sh
ARPG_NO_COMPILE=1 ./scripts/run_arpgl_smoke.sh
```

## 5. Run the full 50k-sample baseline

```sh
conda activate arpg
./scripts/run_arpgl_full.sh
```

The helper script still defaults to `--nproc_per_node=1`. If you decide to scale up later, the sampler now builds class labels for the padded sample count so multi-GPU runs do not exhaust labels when `total_samples` exceeds `num_fid_samples`.

## 6. Evaluate the output batch

```sh
conda activate arpg-eval
./scripts/eval_arpgl_baseline.sh
```

By default the evaluator reads:

- reference batch: `eval/VIRTUAL_imagenet256_labeled.npz`
- sample batch: `samples/arpgl-baseline/ARPG-L-arpg_300m-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-5.0-cfg-schedule-linear-sample-schedule-arccos-step-64-seed-0.npz`

## 7. Acceptance target

Treat the run as reproduced if FID is near `2.38`. As a practical threshold:

- `2.2` to `2.6`: acceptable reproduction
- above `2.6`: investigate environment drift, broken weights, or compile/precision differences
