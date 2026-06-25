# Foundation Probe 10% Subset

This directory stores a local, stratified 10% Val/Test subset for diffusion
foundation probing.

Tracked:

```text
README.md
subset_summary.json
```

Ignored local artifacts:

```text
annotations/Val_10pct.json
annotations/Test_10pct.json
annotations/manifest.json
images/
```

The subset is sampled independently within each SkyFind source subset using a
fixed seed. Images are symlinked, hardlinked, copied, or omitted depending on
the `--image-mode` option used by the builder script.

