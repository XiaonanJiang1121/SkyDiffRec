# Training-Free Diffusion Foundation Probing Protocol

Date: 2026-06-25

Status: synced with the user-provided Notion `PaperDesign` content in this
thread.

## 0. Positioning

This protocol explores whether VLM, Stable Diffusion, and diffusion-style box
generation can provide training-free or light-training weak spatial grounding
signals for SkyFind.

The immediate target is not to fix the final model. It is to answer:

```text
As frozen or mostly frozen foundation models, can VLM / SD / diffusion-style
signals help with UAV tiny targets, long expressions, similar instances, and
coarse VLM boxes on SkyFind?
```

Do not assume in advance whether VLM or diffusion should be the coarse stage.
The coarse-to-fine role assignment should be decided from the probing results.

SAM is not part of this first protocol.

## 1. Problem Summary

SkyFind limitations:

```text
1. UAV tiny target localization is difficult; target area is often extremely
   small relative to the full image.
2. Multiple similar instances cause target confusion.
3. Expressions are longer and more complex than common RefCOCO-style text.
4. Each sample has only one image, one expression, and one target box.
5. There are no auxiliary object boxes, masks, scene graphs, or reference-object
   annotations, even though language often implies reference objects.
```

VLM limitations from `VLMSkyFind`:

```text
Qwen2.5-VL-7B can often produce approximate boxes, but still struggles with
tiny targets, long text, ordinal descriptions, and precise point/box selection.

InternVL / LLaVA / DeepSeek mostly fail through over-large boxes, center drift,
and scale/shape errors. LLaVA's median box area is much larger than the GT area.
```

Diffusion limitations to verify:

```text
1. Existing diffusion grounding methods often use simple image-text
   interaction.
2. They are not naturally friendly to tiny UAV targets.
3. They are usually tested on shorter RefCOCO/RefCOCOg-style expressions, not
   SkyFind-style long spatial descriptions.
```

## 2. Research Goal

The final architecture is expected to be two-stage:

```text
coarse -> fine
```

However, this protocol does not decide the module roles yet. It tests whether:

```text
VLM can be coarse or fine
DM can be coarse or fine
VLM and DM are complementary
attention-derived signals can later become spatial priors
```

The later spatial module idea is:

```text
VLM / DM attention maps
-> weak continuous spatial priors
-> replacement for missing object-level SpatialInfo
```

RSVG-ZeroOV and LazyMCoT motivate this direction: VLM cross-attention can be
globally aware but dispersed, while DM self-attention may provide more complete
object structure. The probing stage must first verify whether this complement
exists on SkyFind.

## 3. Dataset Scale

Formal probing scale:

```text
10% of SkyFind Val
10% of SkyFind Test
```

Before the 10% run, use a small smoke subset:

```text
20-50 samples
```

Use a stratified subset for both smoke and formal probing:

```text
target size: tiny / small / large
expression length: short / medium / long
relation keyword: relational / non-relational
ordinal keyword: ordinal / non-ordinal
source dataset
Qwen VLM behavior: correct / center miss / scale-shape failure / parse failure
```

Keep the subset manifest fixed:

```text
data/foundation_probe_10pct/annotations/Val_10pct.json
data/foundation_probe_10pct/annotations/Test_10pct.json
data/foundation_probe_10pct/annotations/manifest.json
data/foundation_probe_10pct/subset_summary.json
```

## 4. Shared Metrics

Attention metrics:

```text
Pointing Game
Top-k Hit at 1%, 5%, 10%
GT Attention Ratio
peak-to-GT center distance / GT diagonal
attention entropy / peakness
non-target energy ratio
```

Box metrics, when a heatmap is converted to a box:

```text
IoU@0.5
IoU@mean over 0.5 / 0.6 / 0.7 / 0.8 / 0.9
mIoU
area ratio
center error / GT diagonal
centered scale-shape failure
```

Use the same failure-mode definitions as `VLMSkyFind` where possible.

## 5. Validity Controls

These controls are not separate experiments, but they should be included when
claiming an attention map contains grounding signal:

```text
same image + wrong noun
same image + shuffled expression
same image + generic prompt
wrong image + same expression
random wrong crop
```

The useful signal should separate from controls in GT attention ratio,
pointing accuracy, or heatmap-to-box quality.

## 6. Experiment 1: SD Tiny-Target Retention

Question:

```text
Under the RSVG-ZeroOV / Stable Diffusion v1.4 default 512x512 setting, can SD
still visually retain SkyFind tiny targets?
```

Inputs:

```text
SkyFind image
expression
GT bbox
```

Operations:

```text
1. Resize the full image to 512x512.
2. Map the GT box to the 512x512 image.
3. Record resized target width, height, and area.
4. Run SD v1.4 inversion / reconstruction when available.
5. Save the reconstruction image.
6. Evaluate full-image and target-crop reconstruction quality.
```

Metrics:

```text
target_resized_width
target_resized_height
target_resized_area
target-crop MSE
PSNR / SSIM / LPIPS if practical
target min-side in 16 / 32 / 64 attention-grid cells
```

Extra diagnostic:

```text
Check whether reconstruction error sharply increases when the target becomes
smaller than roughly 4x4 pixels in the 512 image. This estimates the visual
"eyesight boundary" of SD v1.4 on SkyFind.
```

Resolution rule:

```text
Keep the 512x512 setting for this experiment.
Do not switch to dynamic resolution yet.
```

Reasons:

```text
dynamic resolution can cause OOM
dynamic resolution can destabilize inversion
dynamic resolution complicates VLM/DM feature alignment
```

Possible later solution:

```text
crop and resize
```

But crop-based processing is evaluated explicitly in Experiment 4.

Decision:

```text
If tiny targets disappear under 512 full-image processing, later attention
failures should be interpreted as scale-limited rather than simply a failure of
diffusion semantics.
```

## 7. Experiment 2: SD Cross-Attention Response

Question:

```text
Given a SkyFind expression, does SD cross-attention respond near the GT target?
```

Motivation:

```text
SD v1.4 CLIP tokenizer has a 77-token limit.
RSVG-ZeroOV also handles prompts around the 75-token boundary.
SkyFind expressions are longer and can exceed the comfortable prompt length.
```

Prompt variants:

```text
P1. SkyFind expression only
P2. object phrases extracted from the SkyFind expression
P3. "a remote sensing image of {object phrase}"
P4. VLM / RSVG-style prompt:
    "Locate it according to the following description.  {expression} The output
    format should be like [x1, y1, x2, y2] without any other text."
```

Settings:

```text
full image -> 512x512
extract SD cross-attention at 16 / 32 / 64
evaluate token-level and phrase-aggregated maps
```

Metrics:

```text
Pointing Game
Top-k Hit
GT Attention Ratio
peak-to-GT center distance / GT diagonal
positive-vs-control gap
```

Decision:

```text
If SD cross-attention responds near GT under some prompt variants, diffusion
can be considered a candidate coarse semantic signal.

If full expression fails but object phrases work, later methods should parse or
compress long expressions before SD conditioning.

If none of the prompt variants beat controls, SD cross-attention should not be
treated as a reliable coarse locator for SkyFind under this setting.
```

## 8. Experiment 3: SD Self-Attention Object Structure

Question:

```text
Does SD self-attention provide object-structure or object-extent information on
SkyFind?
```

Oracle diagnostic:

```text
Use the GT center only for analysis, not as a method input.
Find the corresponding attention cell.
Take the self-attention row for that cell.
Check whether the attended region covers the same target object.
```

Metrics:

```text
self-attention region IoU with GT
self-attention region bbox area ratio
centered scale-shape failure
peak / region center error
non-target energy ratio, especially background / grass / water / road regions
```

Interpretation:

```text
If self-attention spreads into background or similar instances, DM structural
prior may not be suitable for SkyFind tiny objects.

If oracle self-attention covers the target well, SD self-attention can be a
candidate object-structure signal, but it still needs a reliable seed from VLM,
SD cross-attention, or another coarse mechanism.
```

## 9. Experiment 4: Full Image vs Crop Image

Question:

```text
Is SD failure caused by full-image tiny target resolution, or by remote-sensing
domain / semantic mismatch?
```

Settings:

```text
A. full image -> 512x512
B. GT oracle crop with context -> 512x512
C. random wrong crop -> 512x512
```

Optional diagnostic settings after A-C:

```text
D. Qwen predicted box crop with context -> 512x512
E. diffusion-generated / attention-proposed crop -> 512x512
```

Evaluate:

```text
current implementation: SD cross-attention with full expression
current implementation: SD self-attention GT-center row probe
recorded diagnostic: simple reconstruction MSE / PSNR per context
later extension: cross/self compatibility score
heatmap-to-box only as a diagnostic conversion
```

Decision cases:

```text
full image poor + GT crop clearly better:
  SD is not useless; it is scale-limited.
  Later work can study coarse crop, VLM crop, or diffusion crop.

full image poor + Qwen crop clearly better:
  VLM and DM may be complementary, with VLM providing a useful crop seed.

GT crop still poor:
  SD v1.4 is likely mismatched to SkyFind target categories, aerial viewpoint,
  or tiny-object structure under the tested setup.
```

Role decision:

```text
This experiment is the first point where we can start deciding whether VLM or
diffusion should provide coarse localization, fine refinement, or only weak
spatial prior signals.
```

## 10. Deferred Stage: Attention-Derived Spatial Prior

This stage should start only after Experiments 1-4 show that at least one
attention source carries usable signal.

Candidate direction:

```text
parse target phrases, reference phrases, and spatial relation phrases
extract VLM / DM attention maps
construct weak continuous spatial fields
score or evolve candidate boxes by relation consistency
```

This is where VLM cross-attention and DM self-attention fusion can be tested as
a replacement for missing object-level SpatialInfo.

## 11. Reporting Template

Each experiment report should include:

```text
1. objective
2. exact subset manifest
3. model / checkpoint / prompt / resolution settings
4. output artifact paths
5. main metric table
6. control table
7. tiny / small / large and expression-length breakdowns
8. qualitative examples
9. failure interpretation
10. decision for the next experiment
```

Report paths:

```text
reports/exp_1_sd_tiny_target_retention.md
reports/exp_2_sd_cross_attention_response.md
reports/exp_3_sd_self_attention_structure.md
reports/exp_4_full_vs_crop.md
```

Result paths:

```text
results/exp_1_sd_tiny_target_retention/
results/exp_2_sd_cross_attention_val_full/
results/exp_3_sd_self_attention_structure_val_full/
results/exp_4_full_vs_crop_attention_val_full/
```

## 12. Immediate Next Step

Before running SD extraction:

```text
1. build the fixed 10% Val/Test subset manifest
2. implement the scale/grid audit script
3. run local CPU scale audit
4. configure server SD v1.4 / DDIM inversion dependencies
5. run reconstruction audit with PSNR / SSIM / LPIPS
```

Then run Experiment 1 on the smoke subset before launching the full 10% run.
