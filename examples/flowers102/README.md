# Flowers102: the vision example

Oxford Flowers102 (Nilsback and Zisserman, 2008): 8,189 photographs of 102 flower
species, 40 to 258 per class. One 329 MiB archive and a 502-byte label file from
Oxford's Visual Geometry Group, no account needed.

```bash
python prepare.py            # downloads, verifies both checksums, writes images/ and data.csv
python prepare.py --per-class 40
```

`data.csv` has two columns: `image`, a path relative to this folder, and `label`,
`class_001` to `class_102`. Oxford ships numbers, not names, and every name map in
circulation is community work, so the number is what the CSV carries.

**Why this dataset.** It is the classic transfer-learning benchmark that an
ImageNet-pretrained backbone has not already trained on: one of its 8,189 images
appears in the ImageNet-1k training set and none of its test images is a
near-duplicate of one (Kornblith et al. 2019, Table H.1; Kolesnikov et al. 2020,
Table 7). Imagenette and Imagewoof were rejected for this example because every one
of their images is an ILSVRC-2012 image.

**Measured on an Apple M5 with MPS, resnet18, batch 64, 3 epochs, sealed 80/20
split of all 8,189 images:**

| px | linear probe on frozen features | fine-tune, 3 epochs | seconds per epoch |
|---|---|---|---|
| 160 | 0.892 | 0.958 | 21 |
| 224 | 0.932 | 0.972 | 39 |

The layout is flat on purpose: every image is `images/NNNNN.jpg` in shuffled order
and the label lives only in the CSV, so a holdout path can never tell a model its
class. The script asserts that before writing anything.

**Ceiling, measured by the eval sweep on 2026-09-16:** the baseline, a plain CNN
trained from zero at 64 px for 20 epochs, scores 0.554, and the best of thirteen
recipes 0.974, convnext_tiny fine-tuned for 3 epochs at 160 px: 42 points of
headroom. One standard error on this holdout is 0.4 points, so resnet18 fine-tuned at
224 px, 0.972, ties it. The resnet18 probe on frozen features scores 0.892, so most of
the headroom is pretrained features, and the rest is fine-tuning them.

**Status:** working. Since v0.6 an image run goes end to end on this dataset: the plain
CNN baseline, briefed experiments that fine-tune through `fit()` or the agent's own
torch code, a sealed holdout, and `best.ipynb`.

```bash
iterate run --data data.csv --target label --metric accuracy
```

`--target label` is required on the CSV form. Pass `--metric accuracy` so the run's
numbers line up with the stored baseline and ceiling: left to pick for itself, the
metric step chose `f1_macro` here 3 times of 3. The run needs torch, which the harness
installs at the start with your consent, or `pip install 'iterate-ai[vision]'`.

Live on gemma4:12b, Apple M5 with MPS, 2026-09-20, on the keep-best code: the baseline
scored 0.5544, as stored. Iteration 1, resnet18 fine-tuned at 160 px, 0.9523.
Iteration 2, convnext_tiny fine-tuned at 224 px, 0.9811, the best: at about 150 s an
epoch the time budget gave it 2 epochs. Iteration 3 asked for 2, 4 and 3 epochs, got the
same 2 each time, and tied it. 3 iterations, 39 min, no traceback. 0.9811 is past the sweep's
0.974 by 0.7 points, under two standard errors: convnext_tiny at 224 px is a pairing the
sweep never ran, so the stored ceiling is the best of thirteen recipes and not a true
upper bound. An earlier run on main 6485be3 reached 0.9670 in 29 min.

RTX 4050 (CUDA, over WSL2), epoch time against MPS, VRAM peak and the out-of-memory
capture: SLOT_4050

The ceiling sweep still runs on its own:
`python -m evals.run ceilings --datasets flowers102`.

**License:** Oxford states no license for the images. The script downloads them for
local use and this repo redistributes nothing. Cite: Nilsback, M-E. and Zisserman,
A. "Automated flower classification over a large number of classes." ICVGIP 2008.
