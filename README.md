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

## Status

| Phase | State |
|---|---|
| 0 · inventory the data, verify formats and label pairing | **done** |
| 1 · reproduce the published baseline, slice by platform | running |
| 2 · degradation sweep on the excavator frames | queued behind phase 1 |
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

## Credit

Dataset and baseline are the GOOSE authors' work (Fraunhofer IOSB / Universität der
Bundeswehr München): [goose-dataset.de](https://goose-dataset.de) ·
[devkit](https://github.com/FraunhoferIOSB/goose_dataset) ·
[Pointcept fork](https://github.com/FraunhoferIOSB/Pointcept/tree/goose).
Dataset licence CC BY-SA 4.0.
