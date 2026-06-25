# Results

This directory stores lightweight result indexes and small JSON summaries.

Large artifacts such as attention tensors, reconstructed images, crops, and
visualizations should be stored outside git or under a separately ignored
artifact path.

Recommended structure:

```text
results/
  subsets/
  exp_1_sd_tiny_target_retention/
  exp_2_sd_cross_attention_response/
  exp_3_sd_self_attention_structure/
  exp_4_full_vs_crop/
```

Spatial-prior artifacts should be added only after the foundation probing stage
shows usable attention signal.
