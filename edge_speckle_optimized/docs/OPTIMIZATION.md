# Memory optimization candidate 0.1.1

Original 0.1.0 stays in ../edge_speckle, unchanged. New engineering controls:
- numpy_output_copy=False: remove redundant NumPy copy after independent device-to-host tensor creation.
- wsvt_workspace=streamed: omit full normalized100 stack; fuse normalization with spatial-x loads and fine DWT loads.
- spatial_chunk=16 (1..100): temporary spatial-x workspace channel count; no operator reordering.

Original SSD, subpixel fitting, spline, phase integration and numerical precision remain unchanged.
Normalization is computed in double and rounded to float before the unchanged filter/DWT arithmetic.
Use materialized + numpy_output_copy=True for a local old-path ablation.
Original docs/source_origins.json is the archival 0.1.0 ancestry; docs/optimization_origins.json records this engineering diff.
Benchmark validation must verify descriptor equality and seven outputs, including host ownership after reuse/close.
