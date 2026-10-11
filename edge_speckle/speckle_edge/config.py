"""Configuration contracts. No torch import or silent algorithm fallback."""
from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class RunConfig:
    mode: str
    pixel_m: float
    distance_m: float
    energy_ev: float
    front: str = 'auto'
    spline: str = 'gpu'
    input_dtype: str = 'auto'
    reference_policy: str = 'each-pair'
    cpu_threads: int = 2
    device: int = 0
    warmup: int = 1
    repeats: int = 1
    allow_experimental: bool = False
    half: int = 10
    coarse_radius: int = 5
    fine_radius: int = 4
    iterations: int = 1
    cut: int = 1
    pyramid_levels: int = 2

    def __post_init__(self):
        if self.mode not in ('wxst', 'wsvt'):
            raise ValueError('Mode must be wxst or wsvt')
        for name in ('pixel_m', 'distance_m', 'energy_ev'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be explicitly supplied, positive and finite')
        for name in ('cpu_threads', 'device', 'warmup', 'repeats', 'half', 'coarse_radius', 'fine_radius', 'iterations', 'cut', 'pyramid_levels'):
            if type(getattr(self, name)) is not int:
                raise ValueError(name + ' must be an integer')
        if type(self.allow_experimental) is not bool:
            raise ValueError('allow_experimental must be a boolean')
        if self.cpu_threads < 1 or self.device < 0 or not 0 <= self.warmup <= 5 or not 1 <= self.repeats <= 5:
            raise ValueError('Positive CPU threads, nonnegative device, warmup0..5, repeats1..5 required')
        if (self.half, self.coarse_radius, self.fine_radius, self.iterations, self.cut, self.pyramid_levels) != (10, 5, 4, 1, 1, 2):
            raise ValueError('This backend implements half10/coarse5/fine4/I1/cut1/two levels; changing metadata cannot change kernels')
        if self.spline not in ('cpu', 'gpu') or self.reference_policy not in ('each-pair', 'static'):
            raise ValueError('Unsupported spline or reference policy')
        if self.input_dtype not in ('auto', 'float32', 'uint16'):
            raise ValueError('Unsupported input dtype')
        if self.resolved_front not in (('fused', 'materialized') if self.mode == 'wxst' else ('baseline', 'reordered')):
            raise ValueError('Front end does not belong to the selected method')
        if self.resolved_front == 'reordered' and not self.allow_experimental:
            raise ValueError('WSVT reordering is experimental; pass --allow-experimental explicitly')
        if self.input_dtype == 'uint16' and (self.mode != 'wsvt' or self.resolved_front != 'baseline'):
            raise ValueError('uint16 is supported only by original-order WSVT')

    @property
    def resolved_front(self):
        return self.front if self.front != 'auto' else ('fused' if self.mode == 'wxst' else 'baseline')

    @property
    def variant_id(self):
        return 'edge-{}-{}-{}'.format(self.mode, self.resolved_front, self.spline)

    def to_dict(self):
        result = asdict(self)
        result.update(front=self.resolved_front, frames=100 if self.mode == 'wsvt' else 1,
                      template=7 if self.mode == 'wxst' else 0, descriptor_depth=58 if self.mode == 'wxst' else 63,
                      variant_id=self.variant_id)
        return result


def validate_arrays(config, sample, reference):
    if sample.shape != reference.shape:
        raise ValueError('Sample/reference shapes differ')
    if (config.mode == 'wxst' and sample.ndim != 2) or (config.mode == 'wsvt' and (sample.ndim != 3 or sample.shape[0] != 100)):
        raise ValueError('WXST requires HxW; WSVT requires100xHxW')
    if min(sample.shape[-2:]) <= 28:
        raise ValueError('Image dimensions must exceed28 for the fixed search configuration')
    for array in (sample, reference):
        if array.dtype.kind not in 'uif':
            raise ValueError('Real numeric arrays required; complex/object/bool inputs are unsupported')
        if array.dtype.kind == 'f' and not np.isfinite(array).all():
            raise ValueError('Non-finite input')
    dtype = config.input_dtype
    if dtype == 'auto':
        dtype = 'uint16' if (config.mode == 'wsvt' and config.resolved_front == 'baseline' and
                            sample.dtype == reference.dtype == np.dtype('uint16')) else 'float32'
    if dtype == 'uint16' and not sample.dtype == reference.dtype == np.dtype('uint16'):
        raise ValueError('uint16 requires actual uint16 data; floating inputs are never quantized')
    with np.errstate(over='ignore', invalid='ignore'):
        sample = np.ascontiguousarray(sample, dtype=dtype)
        reference = np.ascontiguousarray(reference, dtype=dtype)
    if dtype == 'float32' and (not np.isfinite(sample).all() or not np.isfinite(reference).all()):
        raise ValueError('Input is not representable as finite float32')
    return sample, reference, dtype
