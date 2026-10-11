"""Separate method entry points with common configuration, results and receipts."""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
import numpy as np
from .api import create_session
from .config import RunConfig, validate_arrays
from .outputs import describe_outputs
from .provenance import implementation_receipt, sha256, write_json


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='mode', required=True)
    commands.add_parser('doctor', help='Inspect dependencies; do not run reconstruction')
    for mode in ('wxst', 'wsvt'):
        p = commands.add_parser(mode, help='Single image pair' if mode == 'wxst' else '100 scan positions')
        p.add_argument('--config', type=Path)
        p.add_argument('--input', type=Path, required=True, help='NPZ: sample/reference keys')
        p.add_argument('--out', type=Path, required=True, help='A new result directory')
        p.add_argument('--front', choices=('auto', 'fused', 'materialized') if mode == 'wxst' else ('auto', 'baseline', 'reordered'))
        p.add_argument('--spline', choices=('cpu', 'gpu'))
        p.add_argument('--input-dtype', choices=('auto', 'float32', 'uint16'))
        p.add_argument('--wsvt-workspace', choices=('materialized','streamed'))
        p.add_argument('--spatial-chunk', type=int)
        copies = p.add_mutually_exclusive_group()
        copies.add_argument('--numpy-output-copy', action='store_true', default=None)
        copies.add_argument('--no-numpy-output-copy', dest='numpy_output_copy', action='store_false', default=None)
        p.add_argument('--reference-policy', choices=('each-pair', 'static'))
        for name in ('warmup', 'repeats', 'cpu-threads', 'device'):
            p.add_argument('--' + name, type=int)
        for name in ('pixel-m', 'distance-m', 'energy-ev'):
            p.add_argument('--' + name, type=float)
        p.add_argument('--allow-experimental', action='store_true', default=None)
    return root


def doctor():
    result = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                  native_libraries={name: (Path(__file__).parent / '_native' / name).is_file()
                                    for name in ('libedge_kernels.so', 'libedge_spline.so')})
    try:
        import torch
        result.update(torch=torch.__version__, cuda_available=torch.cuda.is_available(), cuda=torch.version.cuda)
        if torch.cuda.is_available():
            result['devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError as exc:
        result['torch_error'] = str(exc)
    print(json.dumps(result, indent=2))


def resolve_config(args):
    values = json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    if not isinstance(values, dict):
        raise ValueError('Config must be a JSON object')
    if values.get('mode', args.mode) != args.mode:
        raise ValueError('Config mode differs from CLI method')
    values['mode'] = args.mode
    for name in RunConfig.__dataclass_fields__:
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    return RunConfig(**values)


def run(args):
    config = resolve_config(args)
    if args.out.exists():
        raise FileExistsError('Choose a new result directory: ' + str(args.out))
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    run_started = time.perf_counter()
    with np.load(args.input, allow_pickle=False) as data:
        sample, reference, dtype = validate_arrays(config, data['sample'], data['reference'])
    read_s = time.perf_counter() - run_started
    input_hash = sha256(args.input)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'status.json', dict(state='running', variant_id=config.variant_id))
    session = None
    try:
        import torch
        torch.set_num_threads(config.cpu_threads)
        start = time.perf_counter()
        session = create_session(config, sample.shape[-2:], dtype)
        session.stream.synchronize()
        init_s = time.perf_counter() - start
        records = []
        initial_reference_s = session.prepare_reference(reference) if config.reference_policy == 'static' else None
        for index in range(config.warmup + config.repeats):
            start = time.perf_counter()
            ref_s = session.prepare_reference(reference) if config.reference_policy == 'each-pair' else 0.
            outputs, info = session.solve(sample)
            records.append(dict(index=index, warmup=index < config.warmup,
                                reference_prepare_this_call_s=ref_s, sample_solver_s=info['solver_s'],
                                solver_this_call_s=ref_s + info['solver_s'],
                                api_this_call_s=time.perf_counter() - start, runtime_detail=info))
        metadata = describe_outputs(outputs, sample.shape, config.half)
        start = time.perf_counter()
        np.savez_compressed(args.out / 'reconstruction.npz', **outputs)
        save_s = time.perf_counter() - start
        measured = [row for row in records if not row['warmup']]
        summary = {}
        for name in ('solver_this_call_s', 'api_this_call_s', 'sample_solver_s'):
            values = [row[name] for row in measured]
            summary[name] = dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values) if len(values) > 1 else None)
        report = dict(schema_version=1, package_version='0.1.1', configuration=config.to_dict(),
                      input=dict(path=str(args.input.resolve()), sha256=input_hash, shape=list(sample.shape), resolved_dtype=dtype),
                      hardware=dict(host=platform.node(), platform=platform.platform(), device=config.device,
                                    gpu=torch.cuda.get_device_name(config.device), torch=torch.__version__, cuda=torch.version.cuda),
                      file_read_and_conversion_s=read_s, initialization_s=init_s,
                      initial_static_reference_s=initial_reference_s, file_save_s=save_s,
                      run_wall_before_report_s=time.perf_counter() - run_started,
                      timing_summary=summary, records=records, outputs=metadata, implementation=implementation_receipt(),
                      timing_scope=dict(solver='sample GPU pipeline including input write; reference preparation added only for each-pair',
                                        api='reference/sample validation, preparation, computation and owned-output download; excludes I/O and initialization',
                                        run_wall='whole command before report writing, including warmup, all measured calls, input hashing and I/O; not a per-pair latency'),
                      numerical_policy='Original subpixel rule retained; localized CPU/GPU differences recorded in docs/VALIDATION.md',
                      geometry_source='Explicit configuration/CLI; propagation distance must be calibrated for the experiment')
        write_json(args.out / 'report.json', report)
        write_json(args.out / 'status.json', dict(state='complete', variant_id=config.variant_id, report='report.json'))
        print(json.dumps(dict(output=str(args.out.resolve()), variant=config.variant_id, timing_summary=summary), indent=2))
    except BaseException as exc:
        write_json(args.out / 'status.json', dict(state='failed', error_type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        if session is not None:
            session.close()


def main():
    args = parser().parse_args()
    if args.mode == 'doctor':
        doctor()
    else:
        run(args)
