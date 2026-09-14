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

**Status:** data ready (v0.6 Day 1). The `DLModelTarget` lands on Day 2 of the
v0.6 sprint and the `iterate run` switch for images on Day 3.

**License:** MIT, per the Zenodo record. Cite: Helber, Bischke, Dengel, Borth. "EuroSAT: A Novel Dataset and
Deep Learning Benchmark for Land Use and Land Cover Classification." IEEE JSTARS
2019. https://doi.org/10.1109/JSTARS.2019.2918242
