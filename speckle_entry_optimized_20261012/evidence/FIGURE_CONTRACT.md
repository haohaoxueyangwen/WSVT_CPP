# Matched entry optimization figure contract

Backend: the existing Python/matplotlib benchmark figure workflow.
Conclusion: measured reductions in complete entry overhead, accompanied by
unchanged reconstructions under the frozen same-input protocol.

Evidence chain: per-cell warmup plus five formal array API measurements;
paired CPU and native GPU before/after comparisons; seven-field bitwise checks;
same-ROI phase images and common central profiles. Each timing bar shows
mean +/- sample standard deviation, with raw trial dots. No prior uint16/hot
session timing enters these panels.

Archetype: quantitative timing grid and separate image plate with profiles.
Export: PNG preview, SVG editable text, PDF; 180 mm width; no photographic
manipulation. Same before/after phase color limits, saved central 512-square ROI,
coordinate crop recorded in JSON. Rendering is illustrative; full-array metrics
carry the agreement test. Units and timing boundary in every timing figure.
No physical-truth or measured SIMD contribution claim beyond this validation.
