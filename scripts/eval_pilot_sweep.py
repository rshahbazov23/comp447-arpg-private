"""Batch FID/IS/precision/recall evaluation over a pilot-sweep output directory.

Finds every *.npz under `--pilot-dir`, runs OpenAI's guided-diffusion
evaluator against the reference batch, parses the output, and writes a
single CSV with one row per NPZ.

RESUMABLE: if the CSV already exists and already has a row for a given NPZ,
that NPZ is skipped. Safe to re-run.

Usage:
    python scripts/eval_pilot_sweep.py \\
        --pilot-dir /content/pilot-YYYYMMDD \\
        --reference-npz eval/VIRTUAL_imagenet256_labeled.npz \\
        --guided-diffusion external/guided-diffusion/evaluations/evaluator.py \\
        --out-csv /content/pilot-YYYYMMDD/results.csv
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys


RESULT_COLUMNS = ["npz", "mode", "metric", "tau", "cap", "fid", "is_score",
                  "sfid", "precision", "recall", "eval_seconds", "status"]


def parse_evaluator_output(text: str) -> dict:
    """Extract FID/IS/sFID/Precision/Recall from the evaluator stdout."""
    out = {"fid": None, "is_score": None, "sfid": None,
           "precision": None, "recall": None}
    patterns = {
        "fid": r"FID:\s*([\d.]+)",
        "is_score": r"Inception Score:\s*([\d.]+)",
        "sfid": r"sFID:\s*([\d.]+)",
        "precision": r"Precision:\s*([\d.]+)",
        "recall": r"Recall:\s*([\d.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                out[key] = None
    return out


def config_from_npz_name(npz_path: str):
    """Extract mode/metric/tau/cap from the NPZ filename (vanilla or rejection)."""
    name = os.path.basename(npz_path).replace(".npz", "")
    m = re.search(r"mode-rejection-metric-(\w+)-tau-([\d.]+)-cap-([\d.]+)", name)
    if m:
        return "rejection", m.group(1), float(m.group(2)), float(m.group(3))
    m = re.search(r"mode-refinement-metric-(\w+)-k-([\d.]+)", name)
    if m:
        return "refinement", m.group(1), None, float(m.group(2))
    return "vanilla", None, None, None


def load_existing_rows(csv_path: str) -> set:
    """Return the set of NPZ basenames already present in the CSV (for resume)."""
    if not os.path.exists(csv_path):
        return set()
    seen = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") in ("ok", "OK", "PASS") or row.get("fid"):
                seen.add(row["npz"])
    return seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", required=True,
                        help="Root directory containing NPZs from the sweep.")
    parser.add_argument("--reference-npz", required=True,
                        help="Path to the ImageNet-256 reference batch NPZ.")
    parser.add_argument("--guided-diffusion", required=True,
                        help="Path to guided-diffusion's evaluator.py.")
    parser.add_argument("--out-csv", required=True,
                        help="Output CSV path.")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip NPZs that already appear in the CSV (default: True).")
    args = parser.parse_args()

    import time

    if not os.path.exists(args.reference_npz):
        print(f"ERROR: reference NPZ not found: {args.reference_npz}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.guided_diffusion):
        print(f"ERROR: evaluator.py not found: {args.guided_diffusion}", file=sys.stderr)
        sys.exit(1)

    npzs = sorted(glob.glob(os.path.join(args.pilot_dir, "**", "*.npz"), recursive=True))
    if not npzs:
        print(f"No NPZ files found under {args.pilot_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(npzs)} NPZ files under {args.pilot_dir}")

    seen = load_existing_rows(args.out_csv) if args.skip_existing else set()
    if seen:
        print(f"Will skip {len(seen)} NPZs already present in {args.out_csv}")

    evaluator_dir = os.path.dirname(os.path.abspath(args.guided_diffusion))
    evaluator_script = os.path.basename(args.guided_diffusion)
    ref_abs = os.path.abspath(args.reference_npz)

    # Open CSV in append mode if it exists, else write mode with header.
    write_header = not os.path.exists(args.out_csv)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    f = open(args.out_csv, "a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(RESULT_COLUMNS)
        f.flush()

    n_ok = n_fail = n_skip = 0
    try:
        for i, npz in enumerate(npzs, start=1):
            basename = os.path.basename(npz)
            mode, metric, tau, cap = config_from_npz_name(npz)

            print(f"\n[{i}/{len(npzs)}] {basename}")
            print(f"    mode={mode}  metric={metric}  tau={tau}  cap={cap}")

            if basename in seen:
                print("    SKIP (already in CSV)")
                n_skip += 1
                continue

            t0 = time.perf_counter()
            try:
                result = subprocess.run(
                    [sys.executable, evaluator_script, ref_abs, os.path.abspath(npz)],
                    cwd=evaluator_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = result.stdout + "\n" + result.stderr
                metrics = parse_evaluator_output(output)
                elapsed = time.perf_counter() - t0

                if metrics["fid"] is None:
                    # Parsing failed → treat as failure, record the stderr tail.
                    status = "parse_error"
                    tail = output.strip().splitlines()[-5:]
                    print(f"    FAILED: could not parse FID from output")
                    for line in tail:
                        print(f"      | {line}")
                    n_fail += 1
                else:
                    status = "ok"
                    print(f"    FID={metrics['fid']:.3f}  "
                          f"IS={metrics['is_score']}  "
                          f"P={metrics['precision']}  "
                          f"R={metrics['recall']}  "
                          f"({elapsed:.1f}s)")
                    n_ok += 1

                writer.writerow([
                    basename, mode, metric,
                    tau if tau is not None else "",
                    cap if cap is not None else "",
                    metrics["fid"], metrics["is_score"], metrics["sfid"],
                    metrics["precision"], metrics["recall"],
                    round(elapsed, 2), status,
                ])
                f.flush()

            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"    FAILED: {type(e).__name__}: {e}")
                writer.writerow([
                    basename, mode, metric,
                    tau if tau is not None else "",
                    cap if cap is not None else "",
                    None, None, None, None, None,
                    round(elapsed, 2), f"error:{type(e).__name__}",
                ])
                f.flush()
                n_fail += 1
    finally:
        f.close()

    print(f"\n{'=' * 60}")
    print(f"Evaluation complete.")
    print(f"  OK:      {n_ok}")
    print(f"  Skipped: {n_skip}")
    print(f"  Failed:  {n_fail}")
    print(f"  CSV:     {args.out_csv}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
