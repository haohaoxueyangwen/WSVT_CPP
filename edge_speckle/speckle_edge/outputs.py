"""Versioned field contract and explicit detector-coordinate origins."""
import numpy as np

NAMES = ('displace_y', 'displace_x', 'DPC_y', 'DPC_x', 'phase', 'transmission', 'match_score_neg_ssd')
UNITS = dict(displace_y='detector pixel', displace_x='detector pixel', DPC_y='rad', DPC_x='rad',
             phase='rad', transmission='dimensionless', match_score_neg_ssd='negative SSD, arbitrary units')


def describe_outputs(outputs, shape, half=10):
    if set(outputs) != set(NAMES):
        raise ValueError('Seven output fields required')
    h, w = shape[-2:]
    result = {}
    for name in NAMES:
        array = outputs[name]
        expected = (h, w) if name == 'transmission' else (h - 2 * half, w - 2 * half)
        if array.shape != expected or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError('Invalid output: ' + name)
        result[name] = dict(shape=list(array.shape), dtype=str(array.dtype), unit=UNITS[name],
                            detector_origin_yx=[0, 0] if name == 'transmission' else [half, half])
    return result
