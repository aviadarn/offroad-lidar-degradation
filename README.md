# Off-road lidar perception under degraded sensing

How well does a published off-road 3D segmentation baseline hold up when the sensor
stops being clean? This repository audits the
[GOOSE 3D challenge](https://www.codabench.org/competitions/5745) baseline on data
recorded from a **Liebherr R924 tracked excavator**, and measures what happens as the
lidar is degraded the way a machine in the field degrades it: a sensor mast that shifts,
beams that die, dust, rain.

Everything here is measured on rented hardware, not estimated.

---

## Why this dataset

[GOOSE-Ex](https://arxiv.org/abs/2409.18788) is, as far as I can find, the only public
semantic segmentation dataset recorded from a working excavator. It ships 5,000 labelled
multimodal frames from a Liebherr R924 (`alice`) and a Boston Dynamics Spot (`spot`), in
landfill, quarry and construction environments, and combines with the original
[GOOSE](https://arxiv.org/abs/2310.16788) offroad-vehicle data into a single challenge
taxonomy of 9 superclasses.

What the validation split actually contains (`tools/inspect_goose.py`):

| | frames | points | points / frame |
|---|---|---|---|
| `alice` — Liebherr R924 excavator | 192 | 51.5 M | 268,173 |
| `spot` — quadruped | 215 | 18.0 M | 83,642 |
| `vehicle` — GOOSE offroad vehicle | 961 | 174.9 M | 181,989 |

The excavator returns **3.2x more points per frame than the quadruped**, from a mast
looking down at its own work area. Any claim that a model "generalizes across platforms"
has to survive that gap, which is why the platform slice is reported separately
everywhere below.

The class mix on the excavator frames is pure earthworks — soil 33%, low grass 20%,
asphalt 6%, debris 4.6% — and **19.4% of all returns are the machine itself**
(`ego_vehicle`): the arm and bucket swinging through the field of view, which is the
self-occlusion case an excavator has and a car does not.

## What is measured

1. **Reproduce the published baseline.** The official PTv3 checkpoint reports
   `mIoU/mAcc/allAcc 0.8096/0.8576/0.9197` on the challenge validation split. Step one
   is getting that number back on a pristine checkout, before touching anything.
2. **Slice it by platform.** One prediction pass, three numbers: excavator, quadruped,
   offroad vehicle. The headline number is dominated by the vehicle frames, which are
   72% of the validation points.
3. **Degrade the sensor and watch it fall.** Seeded, reproducible perturbations applied
   to the excavator frames only, scored with the same model and the same metric:

   | condition | what it models |
   |---|---|
   | `mount_drift` | the sensor mast shifting a fraction of a degree — a retrofit kit on a machine that vibrates all day |
   | `ring_dropout` | dead or blinded beams |
   | `dust` | particulate returns: beams terminating early, weakly, on airborne dust |
   | `rain` | range-dependent attenuation — far returns lost first, survivors dimmer |
   | `range_noise` | ranging jitter along the beam |
   | `intensity_gain` | reflectivity calibration drift |

   Mount drift is the one that matters most for an aftermarket autonomy kit, and it is
   the closest thing this data supports to the extrinsic-calibration drift a
   camera+lidar stack would suffer: the archives ship no camera extrinsics, so true
   early-fusion calibration studies are not possible on this release.

## The result so far

The published PTv3 checkpoint scores **0.7830 mIoU** over all 1,368 validation
frames here (single forward pass; the published 0.8096 uses 10-way test-time
augmentation — see the caveat below). Sliced by the platform the frames came from,
the same predictions read:

| platform | mIoU | mAcc | allAcc | points |
|---|---|---|---|---|
| `vehicle` — offroad vehicle | **0.7968** | 0.8632 | 0.9281 | 174.9 M |
| `spot` — quadruped | 0.7113 | 0.8091 | 0.8003 | 18.0 M |
| `alice` — **Liebherr R924 excavator** | **0.6040** | 0.6812 | 0.9010 | 51.5 M |

**The headline number is the vehicle talking.** It contributes 72% of the validation
points, and the excavator sits 19 mIoU points below it — on a checkpoint whose own
config trains on `["train", "trainEx"]`, so excavator frames were *in the training
set*. This is not a zero-shot transfer gap.

Two classes account for almost all of it:

| class | excavator IoU | excavator points | vehicle IoU |
|---|---|---|---|
| `obstacle` | **0.060** | 3,145,603 (6.1%) | 0.660 |
| `artificial_ground` | **0.013** | 783,447 (1.5%) | 0.737 |
| `natural_ground` | 0.882 | 31,077,410 (60.4%) | 0.797 |

Neither is starved of support — three million labelled obstacle points is not a
small-sample artifact. The confusion matrix says exactly what happens instead:

| from the excavator | predicted as |
|---|---|
| `obstacle` | **natural_ground 89%**, obstacle 7%, vegetation 2% |
| `artificial_ground` | **natural_ground 98%**, artificial_ground 1% |

From the same viewpoint the vehicle keeps 79% of its obstacle points and 81% of its
artificial ground. **Seen from an excavator's mast, the model flattens obstacles and
made ground into dirt** — it holds on to the dominant class (soil, 60% of returns, IoU
0.88) and loses the two classes a machine that digs has the most reason to care about.

Caveat on the anchor: this run uses a single forward pass, not the baseline's 10-way
TTA, so 0.7830 is not a like-for-like reproduction of 0.8096. The platform *gap* is
measured within one protocol and does not depend on that difference, but quantifying
the TTA delta is the next thing on the list.

## Status

| Phase | State |
|---|---|
| 0 · inventory the data, verify formats and label pairing | **done** |
| 1 · reproduce the published baseline, slice by platform | **done** — see above |
| 2 · degradation sweep on the excavator frames | running |
| 3 · a deployable model + TensorRT/Jetson latency budget | planned |

## Layout

```
tools/inspect_goose.py     split inventory: frames, points, class support, per platform
tools/eval_miou.py         confusion-matrix mIoU, sliced by platform and scenario
tools/degrade.py           seeded sensor degradations, labels carried through
provision/setup_box.sh     bare CUDA box -> data + challenge labels + baseline weights
provision/bootstrap_box.sh GPU arch check; rebuilds custom ops if the image mismatches
provision/run_baseline.sh  evaluate the baseline on val and slice the result
provision/run_sweep.sh     one data root, config and exp dir per degradation condition
results/                   measurements, as they land
```

## Notes for anyone repeating this

- The point clouds are `float32 [x, y, z, intensity]`; labels are `uint32` with the
  semantic id in the low 16 bits. The challenge labels are a **separate download**
  ([Zenodo](https://zenodo.org/records/16942462)) from the point clouds, and the
  dataset class expects them under `labels_challenge/`, not `labels/`.
- The baseline's Docker tag in `scripts/run_image.sh` does not exist on Docker Hub;
  the published tags are prefixed with a version
  (`pointcept/pointcept:v1.5.0-pytorch2.0.1-cuda11.7-cudnn8-devel`).
- That image is CUDA 11.7 and its custom ops are built for `sm_80`. Ada cards
  (RTX 4090, L40S at `sm_89`) cannot run it, and `sm_86` cards may need pointops
  rebuilt. `provision/bootstrap_box.sh` checks before a run is wasted.
- Pointcept's tester **reuses an existing `<name>_pred.npy`** instead of recomputing
  it. Every condition in a sweep needs its own experiment directory, or every curve
  comes out flat.
- **Pin `numpy<2`.** spconv 2.3.6 and cumm 0.4.11 are built against the numpy 1.x C
  ABI. Under numpy 2.x the first sparse convolution dies with a bare
  `Floating point exception` — SIGFPE inside `cumm.tensorview.from_numpy`, no Python
  traceback, no message. Installing `torch-geometric` or `torch-scatter` is enough to
  pull numpy 2 in, so the pin has to be re-asserted after the dependency install.
- Pointcept v1.5's config dump calls `yapf.FormatCode(..., verify=True)`; `verify` was
  removed in yapf 0.40, so a fresh environment needs `yapf==0.32.0`.
- The stock `pytorch/pytorch` images need `torch-scatter` (from the PyG wheel index),
  `spconv-cu120`, `torch-geometric` and `pointops` (compiled for the rented GPU's
  arch) before the tester will import.

## Credit

Dataset and baseline are the GOOSE authors' work (Fraunhofer IOSB / Universität der
Bundeswehr München): [goose-dataset.de](https://goose-dataset.de) ·
[devkit](https://github.com/FraunhoferIOSB/goose_dataset) ·
[Pointcept fork](https://github.com/FraunhoferIOSB/Pointcept/tree/goose).
Dataset licence CC BY-SA 4.0.
