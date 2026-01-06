"""
Adapted from https://github.com/thijor/dp-cvep-speller (Thielen et al., 2021).
"""
import numpy as np
import pyntbci
from pathlib import Path

out_dir = Path(".\cvep_speller\codes")
out_path = out_dir / "mseq_32_shift.npz"

# Shifted m-sequence
code = pyntbci.stimulus.make_m_sequence(poly=[1, 0, 0, 0, 0, 1], base=2, seed=6 * [1])[
    0, :
]
n_keys = 32
codes = np.zeros((n_keys, code.size), dtype="uint8")
for i in range(n_keys):
    codes[i, :] = np.roll(code, 2*i)
np.savez(file=out_path, codes=codes)