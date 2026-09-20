# Storm wind speed: the number-label vision example

NASA Tropical Cyclone Wind Estimation (Maskey et al., 2020): infrared GOES satellite
frames of 494 Atlantic and East Pacific storms from 2000 to 2019. Each frame is
labelled with the storm's maximum sustained wind in knots. Source Cooperative serves
the frames file by file, and no account is needed.

```bash
python prepare.py              # about 250 MB: downloads, verifies, writes images/, train.csv, holdout.csv
python prepare.py --every 12   # half as many frames
```

`train.csv` and `holdout.csv` each have two columns: `image`, a path relative to this
folder, and `label`, the wind speed in whole knots.

**Why this dataset.** It is a real number to predict from a picture, and the pictures
are infrared satellite frames, not the photographs ImageNet is built from. The label
is a property of the whole storm: the size, shape and coldness of its cloud mass,
which survives shrinking the frame to 64 px where a photo-quality score would not.
Above a small CNN there are published numbers, measured another way: a pretrained
ResNet-18 at 224 px, given the current frame and the two before it, is reported at
10.5 kt on the competition's test set (arXiv 2404.08325), and competition winners
reached 6.3 kt with stacks of past frames on that same test set, which continues 227 of
its 371 storms from training. Neither is measured the way this single-frame split by
storm is.

**The split is by storm.** Frames of one storm are 30 minutes apart and nearly
identical, so a frame split would score a model on storms it trained on. The script
keeps every 6th frame of every storm, puts one fifth of the storms in the holdout
with a fixed seed, and asserts no storm lands on both sides:

| | frames | storms |
|---|---|---|
| train.csv | 9,632 | 395 |
| holdout.csv | 2,276 | 99 |

The official test set is not used: 227 of its 371 storms are later frames of training
storms.

**Two kinds of file, and one odd frame.** 10,882 frames are 366 px grayscale. 1,025
frames, from 60 storms, are 1093 px files with three identical channels. One training
frame, from storm npf, is a 366 px colour file. Both main kinds sit on both sides of the
split. In training their wind speeds are spread the same way; the holdout's 153 large
frames come from 14 weaker storms, a mean of 38 kt against 49 for the rest. The harness
converts and resizes every image, so a run needs to do nothing about it.

**Measured on an Apple M5 with MPS, one seed per row unless marked:**

| model | holdout RMSE, knots | error bar by storm |
|---|---|---|
| guess the training mean | 26.6 | 1.7 |
| simple CNN from zero, 64 px, 20 epochs (the baseline) | 13.1, 13.5 with seed 43 | 0.7 |
| simple CNN from zero, 128 px, 20 epochs | 16.5 | 1.2 |
| resnet18 frozen probe, 160 px | 13.2 | 0.6 |
| resnet18 fine-tune, 3 epochs, 64 px | 10.6 | 0.5 |
| resnet18 fine-tune, 3 epochs, 128 px | 9.3 | 0.4 |
| resnet18 fine-tune, 3 epochs, 160 px | 9.1 | 0.4 |

The error bar resamples whole holdout storms, since frames of one storm move
together. A formula that treats every frame as independent gives an error bar two to
five times smaller, which would call real ties wins: 2.7 to 5 times smaller for
rmse / sqrt(2n), which assumes normal errors, and 2.1 to 3.6 times for a bootstrap over
frames, the smallest gaps at the fine-tunes.

**Ceiling, measured by the eval sweep on 2026-09-16:** the plain CNN baseline scores
13.12 knots and the best of twelve recipes 8.87, resnet18 fine-tuned for 3 epochs at
224 px: 4.25 knots of headroom, a third of the baseline's error. It is a tie, not a
winner: convnext_tiny fine-tuned at 160 px scored 8.87, and resnet18 and resnet50 at
160 px 9.11 and 9.13, all inside the storm-level error bar of about 0.4 knots at these
scores. A frozen backbone takes about a third of the headroom at best, 13.17 for the
resnet18 probe and 11.62 for convnext_tiny's; fine-tuning takes the rest.

**Status:** working, the number-label image example. Since v0.6 an image run goes end
to end here: the plain CNN baseline as a regressor, briefed experiments that fine-tune
through `fit()` or the agent's own torch code, a sealed holdout, and `best.ipynb`.

```bash
iterate run --train train.csv --holdout holdout.csv --target label --metric rmse
```

The split by storm is yours, so the holdout is sealed exactly as `prepare.py` wrote it.
`--target label` is required on the CSV form. The live run below passed no `--metric`
and the metric step picked rmse by itself; naming it makes the run repeatable. The run
needs torch, which the harness installs at the start with your consent, or
`pip install 'iterate-ai[vision]'`.

Live on gemma4:12b, Apple M5 with MPS (main 6485be3, SLOT_KEEP_BEST, machine under
memory pressure): the baseline scored 13.12 knots, as stored, iteration 1 9.16 and
iteration 2 9.20. Against the 8.87 ceiling that is about 93% of the headroom, in 2
iterations and 48.5 min. The time budget cut two fits to 9 and 8 of their 10 epochs.

RTX 4050 (CUDA, over WSL2), epoch time against MPS, VRAM peak and the out-of-memory
capture: SLOT_4050

The ceiling sweep still runs on its own:
`python -m evals.run ceilings --datasets cyclone_wind`.

**License:** CC-BY-4.0. Cite: M. Maskey, R. Ramachandran, I. Gurung, B. Freitag,
M. Ramasubramanian, J. Miller. "Tropical Cyclone Wind Estimation Competition Dataset",
Version 1.0, Radiant MLHub. https://doi.org/10.34911/rdnt.xs53up
