import numpy as np
from core.accumulators import encode_state_bits

def test_encode_state_bits():
    assert encode_state_bits(np.array([0,0,0], dtype=np.uint8)) == 0
    assert encode_state_bits(np.array([1,0,0], dtype=np.uint8)) == 1
    assert encode_state_bits(np.array([0,1,0], dtype=np.uint8)) == 2
    assert encode_state_bits(np.array([1,1,0], dtype=np.uint8)) == 3
    assert encode_state_bits(np.array([1,1,1], dtype=np.uint8)) == 7
