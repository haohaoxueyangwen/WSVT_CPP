"""Short packaging regression against the unchanged reviewed edge runtime."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import threading
import numpy as np
import torch
from speckle_edge import RunConfig, create_session
from speckle_edge.config import validate_arrays
from speckle_edge.outputs import describe_outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-edge', type=Path, required=True)
    p.add_argument('--wxst-input', type=Path, required=True)
    p.add_argument('--wsvt-input', type=Path, required=True)
    p.add_argument('--u16-input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location('packaging_reference', args.original_edge / 'edge_runtime.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    torch.set_num_threads(2)
    cases = [('wxst', args.wxst_input, 'fused', 'gpu', False),
             ('wxst', args.wxst_input, 'materialized', 'cpu', True),
             ('wsvt', args.wsvt_input, 'baseline', 'gpu', False),
             ('wsvt', args.u16_input, 'baseline', 'gpu', False)]
    rows = []
    for mode, path, front, spline, rectangular in cases:
        config = RunConfig(mode, pixel_m=1.625e-6, distance_m=.4845, energy_ev=12400., front=front, spline=spline)
        with np.load(path, allow_pickle=False) as data:
            raw_s, raw_r = data['sample'], data['reference']
        if rectangular:
            raw_s, raw_r = raw_s[:127, :159], raw_r[:127, :159]
        sample, reference, dtype = validate_arrays(config, raw_s, raw_r)
        original_type = old.WXSTEdgeSession if mode == 'wxst' else old.WSVTEdgeSession
        with original_type(sample.shape[-2:], front=front, spline=spline, input_dtype=dtype) as baseline:
            baseline.prepare_reference(reference)
            expected, _ = baseline.solve(sample)
        with create_session(config, sample.shape[-2:], dtype) as session:
            rejected_unprepared = False
            try:
                session.solve(sample)
            except RuntimeError:
                rejected_unprepared = True
            session.prepare_reference(reference)
            pointers = {name: value.data_ptr() for name, value in session.b.items()}
            actual, info = session.solve(sample)
            metadata = describe_outputs(actual, sample.shape)
            delta = {name: float(np.max(np.abs(actual[name].astype(np.float64) - expected[name].astype(np.float64)))) for name in actual}
            exact = all(np.array_equal(actual[name], expected[name]) for name in actual)
            kept = actual['phase'].copy()
            session.solve(sample, download=False)
            reused = pointers == {name: value.data_ptr() for name, value in session.b.items()}
            owned = np.array_equal(actual['phase'], kept)
            errors = []
            def misuse():
                try:
                    session.solve(sample)
                except RuntimeError:
                    errors.append('cross_thread_rejected')
            thread = threading.Thread(target=misuse)
            thread.start()
            thread.join()
        closed_rejected = False
        try:
            session.solve(sample)
        except RuntimeError:
            closed_rejected = True
        session.close()  # idempotent close
        row = dict(mode=mode, front=front, spline=spline, dtype=dtype, shape=list(sample.shape),
                   exact_field_values=exact, max_abs_difference=delta, persistent_buffers_reused=reused,
                   owned_outputs_unchanged=bool(owned), unprepared_rejected=rejected_unprepared,
                   cross_thread_rejected=bool(errors), closed_rejected=closed_rejected, outputs=metadata)
        row['pass'] = bool(exact and reused and owned and rejected_unprepared and errors and closed_rejected)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(packaging_regression_pass=all(r['pass'] for r in rows), cases=rows,
                  scope='Short packaging and ownership regression; not a new performance or physical-accuracy experiment')
    (args.out / 'checks.json').write_text(json.dumps(report, indent=2))
    if not report['packaging_regression_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
