import numpy as np
from metrics.info import mutual_information_joint

def test_mi_nonnegative_on_independent():
    P = np.ones((4,4), dtype=float)
    mi = mutual_information_joint(P)
    assert mi >= 0.0
    assert abs(mi - 0.0) < 1e-12
