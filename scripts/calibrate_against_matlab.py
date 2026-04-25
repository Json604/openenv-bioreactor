"""Calibrate the Python port against IndPenSim Octave output_5/.

Runs the Python plant for the same nominal conditions as Batch 1 of the
Octave reference, overlays curves on the same time axis, saves
`docs/calibration/python_vs_matlab.png`, and prints per-variable error
summaries. The corresponding pytest in `tests/test_plant_calibration.py`
asserts the bands documented in the design spec §3.4.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bioperator_env.plant.engine import Plant, PlantConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
REF = REPO_ROOT / "IndPenSim/output_5/IndPenSim_V2_export_V7.csv"
OUT = REPO_ROOT / "docs/calibration/python_vs_matlab.png"


def load_reference_batch(batch_no: int = 1, rows_per_batch: int = 1150) -> pd.DataFrame:
    """Load batch by row index (the CSV has duplicate 'Batch_ref' columns that
    collapse on parse, so we slice by the known fixed batch length)."""
    df = pd.read_csv(REF)
    df.columns = [c.strip() for c in df.columns]
    start = (batch_no - 1) * rows_per_batch
    end = start + rows_per_batch
    return df.iloc[start:end].reset_index(drop=True)


def run_python_batch(seed: int, T: float, fault_code: int = 0) -> pd.DataFrame:
    plant = Plant(PlantConfig(
        seed=seed, T_total_h=T, h_step=0.2,
        fault_code=fault_code, randomise_params=False,
    ))
    plant.reset()
    rows = []
    n = int(round(T / 0.2))
    for _ in range(n):
        rows.append({
            "Time (h)": plant.t_h + 0.2,  # MATLAB samples at end of step
            "S": plant.state[0],
            "DO2": plant.state[1],
            "P": plant.state[3],
            "V": plant.state[4],
            "pH": plant.pH(),
            "T": plant.state[7],
        })
        plant.step({})
    return pd.DataFrame(rows)


def overlay_plots(ref: pd.DataFrame, py: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    plot_specs = [
        ("S",   "Substrate (g/L)"),
        ("DO2", "Dissolved O2 (mg/L)"),
        ("P",   "Penicillin (g/L)"),
        ("V",   "Volume (L)"),
        ("pH",  "pH"),
        ("T",   "Temperature (K)"),
    ]
    for ax, (col, label) in zip(axes.ravel(), plot_specs):
        ax.plot(ref["Time (h)"], ref[col], label="MATLAB / Octave", linewidth=2.0, alpha=0.85)
        ax.plot(py["Time (h)"], py[col], label="Python port", linewidth=1.8, linestyle="--")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("BioOperatorEnv: Python port vs MATLAB IndPenSim (Batch 1, no faults)",
                 fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved overlay plot: {out_path}")


def report_errors(ref: pd.DataFrame, py: pd.DataFrame) -> dict:
    summary = {}
    for col in ["S", "DO2", "P", "V", "pH", "T"]:
        n = min(len(ref), len(py))
        ref_v = ref[col].iloc[:n].astype(float).values
        py_v = py[col].iloc[:n].astype(float).values
        diff = py_v - ref_v
        summary[col] = {
            "mean_abs_err": float(np.mean(np.abs(diff))),
            "max_abs_err": float(np.max(np.abs(diff))),
            "mean_rel_err": float(np.mean(np.abs(diff) / np.maximum(np.abs(ref_v), 1e-6))),
            "ref_range": (float(ref_v.min()), float(ref_v.max())),
            "py_range": (float(py_v.min()), float(py_v.max())),
        }
    return summary


if __name__ == "__main__":
    ref = load_reference_batch(batch_no=1)
    T = float(ref["Time (h)"].iloc[-1])
    print(f"Reference batch length: {T:.1f} h, {len(ref)} samples")

    py = run_python_batch(seed=42, T=T, fault_code=0)
    print(f"Python batch length: {py['Time (h)'].iloc[-1]:.1f} h, {len(py)} samples")

    overlay_plots(ref, py, OUT)

    print("\nPer-variable error summary:")
    summary = report_errors(ref, py)
    for col, stats in summary.items():
        print(f"  {col:>4s}  mean|err|={stats['mean_abs_err']:.4g}  "
              f"max|err|={stats['max_abs_err']:.4g}  "
              f"mean_rel={stats['mean_rel_err']:.3%}  "
              f"ref={stats['ref_range']}  py={stats['py_range']}")
