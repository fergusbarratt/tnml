from xmps.fMPS import fMPS
import numpy as np
import math

def to_mps(vec, D=None):
    """to_mps.

    Args:
        vec:
        D:
    """
    vec = vec.reshape(-1)
    L = math.ceil(math.log(len(vec), 2))
    padded_vec = np.pad(vec, (0, 2 ** L - 784)).reshape(*[2] * L)
    return fMPS().left_from_state(padded_vec).left_canonicalise(D)
