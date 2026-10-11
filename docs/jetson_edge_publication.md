# Jetson edge source publication — 2026-10-11

This update adds the completed `edge_speckle 0.1.0` engineering package to the existing source-sync branch. The root C++ source/configuration remains the committed snapshot recorded in `SOURCE_SNAPSHOT.json`; no workstation development files or manuscript edits are applied to it. The root README is extended with CPU/GPU navigation; its previous hash in the historical source snapshot describes the earlier restoration commit.

The edge package is copied from the source archive whose SHA256 is `348b068d4354e67c9a6bb7ff62da1d372d3b96f6342aade54626281fc587408d`. Its 67 content files plus release manifest are copied byte-for-byte. `.gitattributes` disables line-ending conversion for this subtree, so Git blobs preserve the reviewed hashes. Raw images, NPZ reconstructions, native binaries and build directories are excluded.

The package contains separate WXST/WSVT APIs and CLI commands, package-local CUDA includes, native build tooling, configuration templates, output coordinates/units, timing scopes, lifecycle checks, historical WXST prototype sources and lightweight JSON evidence. Its CPU comparison archive preserves the exact source used for the earlier isolated Orin benchmark, including source modules not enabled by the no-dark-field comparison. That archive is a historical comparison input, not an update to the maintained root C++ line; its fixed driver is not a general-purpose public CPU API.

Pre-publication validation rechecks the release hashes and portable input/output/configuration tests. Recorded device verification covers a CUDA11.4/SM87 rebuild, CPU spline build, four small-image packaging comparisons, two complete CLI executions and legacy native ABI loading. Full-size performance and cross-backend differences remain attached to their original batches. WSVT reordering is explicitly experimental and is not promoted by this publication.

GitHub Actions runs the portable contracts and source-manifest checks. It has no Orin device and does not claim CUDA numerical or performance validation. Original workspaces, server datasets and research results remain untouched.
