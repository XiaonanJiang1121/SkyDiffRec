# Data

This directory is for local SkyFind probing subsets.

Generated annotation files and images are intentionally ignored by git because
SkyFind data should not be redistributed through this public repository.

Recreate the local 10% probing subset with:

```bash
python foundation_probing/tools/build_skyfind_10pct_subset.py \
  --skyfind-root ../BioLoc/data/SkyFind_data \
  --output-root data/foundation_probe_10pct \
  --image-mode symlink
```

