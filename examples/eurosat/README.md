# EuroSAT: the vision certification dataset

EuroSAT (Helber et al., 2019): 27,000 Sentinel-2 satellite patches at 64 x 64 px,
labelled with ten land-use classes: annual crop, forest, herbaceous vegetation,
highway, industrial, pasture, permanent crop, residential, river, sea or lake. One
95 MB zip from Zenodo, no account needed, MIT licensed per the Zenodo record.

```bash
python prepare.py            # downloads, verifies the checksum, writes images/ and data.csv
python prepare.py --per-class 500
```

**Why this is the certification dataset.** Satellite patches are as far from
ImageNet photographs as image classification gets while still being a task people
ship, and the overlap with ImageNet is zero by construction: Sentinel-2A launched in
2015, three years after ILSVRC-2012 was fixed. A pretrained backbone has never seen
this domain, so the gap between a frozen-feature probe and a fine-tune is real
headroom, and a gain here cannot be the backbone remembering the picture.

**Measured on an Apple M5 with MPS, resnet18, batch 64, 3 epochs, sealed 80/20
split of all 27,000 tiles:**

| px | linear probe on frozen features | fine-tune, 3 epochs | seconds per epoch |
|---|---|---|---|
| 64 (native) | 0.897 | 0.974 | 14 |
| 128 (upsampled) | 0.919 | 0.981 | 53 |

Two levers show in that table before an agent touches it: fine-tuning, and input
resolution. Published ImageNet ResNet-50 numbers on this dataset reach 0.986 when
fine-tuned at 224 px (Helber et al.).

The layout is flat on purpose: every image is `images/NNNNN.jpg` in shuffled order
and the label lives only in the CSV, so a holdout path can never tell a model its
class. The archive's own filenames repeat the class name, which is exactly why.

**Ceiling, measured by the eval sweep on 2026-09-16:** the baseline, a plain CNN
trained from zero at 64 px for 20 epochs, scores 0.950, and the best of thirteen
recipes 0.986, convnext_tiny fine-tuned for 3 epochs at 64 px: 3.6 points of
headroom, about 22 standard errors, since one is 0.16 points on this holdout. resnet50
and resnet18 at 128 px, both 0.985, tie the ceiling. The plain CNN beats every frozen
backbone here, 0.899 for the resnet18 probe and 0.944 for convnext_tiny's: on tiles
no ImageNet photograph resembles, features learned from the tiles themselves carry
most of the way, and a pretrained model has to be fine-tuned to pass them.

**Status:** working, and the dataset the v0.6 certification run uses. Since v0.6 an
image run goes end to end here: the plain CNN baseline, briefed experiments that
fine-tune through `fit()` or the agent's own torch code, a sealed holdout, and
`best.ipynb`.

```bash
iterate run --data data.csv --target label --metric accuracy
```

`--target label` is required on the CSV form. Pass `--metric accuracy` so the run's
numbers line up with the stored baseline and ceiling: left to pick for itself, the
metric step chose `f1_macro` here 2 times of 3. The run needs torch, which the harness
installs at the start with your consent, or `pip install 'iterate-ai[vision]'`.

First live run, 2026-09-18, gemma4:12b on an Apple M5 with MPS, 3 iterations, 43 min,
no `--metric`, so the metric step picked `f1_macro`: the baseline reproduced the stored
accuracy exactly, 0.9496, and `f1_macro` went from 0.9477 to 0.9848 (accuracy 0.9854)
on the first iteration, a convnext_tiny fine-tune at 128 px. The third iteration wrote
its own torch code, a timm efficientnet_b0, and scored 0.9478.

Certification run, 2026-09-21, on the released code (gemma4:12b, Apple M5 with MPS,
10 iterations allowed, stopped on patience after 4): the baseline reproduced the stored
0.9496 exactly, and iteration 1 reached 0.9876 with resnet18 fine-tuned at 128 px for 10
epochs. Iterations 2 to 4 tried convnext_tiny 0.9604, resnet50 0.9756 and a timm resnet50
0.9461, none of which beat it. No traceback. `iterate.vision.load(best_model.pt).predict()`
re-scored the winner at 0.9876 over all 5,400 sealed holdout images.

GPU compatible: it trains on an Apple GPU (MPS) or an NVIDIA GPU (CUDA), and falls back
to CPU.

The ceiling sweep still runs on its own:
`python -m evals.run ceilings --datasets eurosat`.

**License:** MIT, per the Zenodo record. Cite: Helber, Bischke, Dengel, Borth. "EuroSAT: A Novel Dataset and
Deep Learning Benchmark for Land Use and Land Cover Classification." IEEE JSTARS
2019. https://doi.org/10.1109/JSTARS.2019.2918242
