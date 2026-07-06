from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from builders.motifs import motif_W_from_edges
from core.simulate import Simulator, SimulationConfig
from metrics.info import entropy_base2, mutual_information_joint


def main() -> None:
    out_dir = Path("data/out_single_motif")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Example motif (swap later with thesis motifs)
    W = motif_W_from_edges(
        N=3,
        edges=[
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
        ],
    )

    sim = Simulator()

    sigmas = np.linspace(0.0, 10.0, 51)
    cfg_base = dict(T=100_000, noise_mode="same_input", seed=123, init="random")

    rows = []
    for sigma in sigmas:
        cfg = SimulationConfig(sigma=float(sigma), **cfg_base)
        res = sim.run_state_counts(W=W, cfg=cfg)
        Hz = entropy_base2(res.pz)
        MIz = mutual_information_joint(res.p_joint)
        rows.append({"sigma": float(sigma), "Hz": float(Hz), "MIz": float(MIz), "steps": int(res.steps)})

    np.save(out_dir / "sigmas.npy", sigmas)
    np.save(out_dir / "Hz.npy", np.array([r["Hz"] for r in rows], dtype=float))
    np.save(out_dir / "MIz.npy", np.array([r["MIz"] for r in rows], dtype=float))

    meta = {"W": W.tolist(), "cfg_base": cfg_base, "n_sigmas": len(sigmas)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    lines = ["sigma,Hz,MIz,steps"]
    lines += [f'{r["sigma"]},{r["Hz"]},{r["MIz"]},{r["steps"]}' for r in rows]
    (out_dir / "summary.csv").write_text("\n".join(lines))

    print(f"Wrote outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
