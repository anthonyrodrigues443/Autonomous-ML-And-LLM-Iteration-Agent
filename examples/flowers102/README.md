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

**Status:** data ready (v0.6 Day 1). The `DLModelTarget` lands on Day 2 of the
v0.6 sprint and the `iterate run` switch for images on Day 3; until then this
folder is data only.

**License:** Oxford states no license for the images. The script downloads them for
local use and this repo redistributes nothing. Cite: Nilsback, M-E. and Zisserman,
A. "Automated flower classification over a large number of classes." ICVGIP 2008.
