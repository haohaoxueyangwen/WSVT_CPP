"""Catch configuration/coordinate mistakes without CUDA dependencies."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from speckle_edge.cli import parser, resolve_config
from speckle_edge.config import RunConfig, validate_arrays
from speckle_edge.outputs import NAMES, describe_outputs

GEOMETRY = dict(pixel_m=1.625e-6, distance_m=.4845, energy_ev=12400.)


class Contracts(unittest.TestCase):
    def test_modes_and_fixed_kernel_configuration(self):
        self.assertEqual(RunConfig('wxst', **GEOMETRY).resolved_front, 'fused')
        self.assertEqual(RunConfig('wsvt', **GEOMETRY).resolved_front, 'baseline')
        for options in (dict(front='baseline'), dict(half=16), dict(fine_radius=2), dict(cpu_threads=0),
                        dict(pixel_m=0), dict(repeats=True), dict(input_dtype='uint16')):
            with self.subTest(options=options), self.assertRaises(ValueError):
                RunConfig('wxst', **dict(GEOMETRY, **options))
        with self.assertRaises(ValueError):
            RunConfig('wsvt', front='reordered', **GEOMETRY)
        self.assertEqual(RunConfig('wsvt', front='reordered', allow_experimental=True, **GEOMETRY).resolved_front, 'reordered')

    def test_dtype_preservation_and_input_errors(self):
        config = RunConfig('wsvt', **GEOMETRY)
        stack = np.ones((100, 32, 40), dtype=np.uint16)
        sample, reference, dtype = validate_arrays(config, stack, stack)
        self.assertEqual(dtype, 'uint16')
        self.assertIs(sample, stack)
        self.assertIs(reference, stack)
        with self.assertRaises(ValueError):
            validate_arrays(RunConfig('wsvt', input_dtype='uint16', **GEOMETRY), stack.astype(np.float32), stack)
        with self.assertRaises(ValueError):
            validate_arrays(RunConfig('wxst', **GEOMETRY), stack, stack)
        bad = np.ones((32, 40), dtype=np.float64)
        bad[0, 0] = 1e300
        with self.assertRaises(ValueError):
            validate_arrays(RunConfig('wxst', **GEOMETRY), bad, bad)
        with self.assertRaises(ValueError):
            validate_arrays(RunConfig('wxst', **GEOMETRY), bad.astype(np.complex128), bad.astype(np.complex128))

    def test_output_detector_origins_and_finite_validation(self):
        shape = (40, 50)
        output = {name: np.ones(shape if name == 'transmission' else (20, 30), np.float32) for name in NAMES}
        meta = describe_outputs(output, shape)
        self.assertEqual(meta['phase']['detector_origin_yx'], [10, 10])
        self.assertEqual(meta['transmission']['detector_origin_yx'], [0, 0])
        output['phase'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            describe_outputs(output, shape)

    def test_cli_mode_mismatch_and_explicit_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(dict(mode='wsvt', **GEOMETRY)))
            args = parser().parse_args(['wxst', '--config', str(path), '--input', 'input.npz', '--out', 'out'])
            with self.assertRaises(ValueError):
                resolve_config(args)
            path.write_text(json.dumps(dict(mode='wxst', **GEOMETRY)))
            self.assertEqual(resolve_config(args).mode, 'wxst')
            path.write_text(json.dumps(dict(mode='wxst', **GEOMETRY, invented_parameter=True)))
            with self.assertRaises(TypeError):
                resolve_config(args)


if __name__ == '__main__':
    unittest.main()
