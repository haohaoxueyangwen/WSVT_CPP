"""Public factory; numerical runtime is imported only when a session is created."""
from pathlib import Path
from .config import RunConfig


def create_session(config, shape, input_dtype):
    if not isinstance(config, RunConfig):
        raise TypeError('Pass a validated RunConfig')
    if input_dtype not in ('float32', 'uint16'):
        raise ValueError('Resolve actual input dtype with validate_arrays before creating a session')
    if config.input_dtype != 'auto' and input_dtype != config.input_dtype:
        raise ValueError('Resolved dtype conflicts with explicit configuration')
    if len(shape) != 2 or any(type(v) is not int or v <= 28 for v in shape):
        raise ValueError('Session shape must contain two integer dimensions greater than28')
    native = Path(__file__).resolve().parent / '_native'
    needed = ['libedge_kernels.so'] + (['libedge_spline.so'] if config.spline == 'cpu' else [])
    missing = [name for name in needed if not (native / name).is_file()]
    if missing:
        raise RuntimeError('Native libraries missing: {}; build on Jetson with python3 tools/build.py'.format(', '.join(missing)))
    from .runtime import WXSTEdgeSession, WSVTEdgeSession
    session_type = WXSTEdgeSession if config.mode == 'wxst' else WSVTEdgeSession
    session = session_type(shape, front=config.resolved_front, spline=config.spline,
                          pixel=config.pixel_m, distance=config.distance_m, energy=config.energy_ev,
                          cpu_threads=config.cpu_threads, input_dtype=input_dtype, device=config.device,
                          half=config.half)
    return session
