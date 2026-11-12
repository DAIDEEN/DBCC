# Domain-Balanced Candidate Calibration for False-Alarm Suppression in Dense Infrared Small Target Detection

We rescore candidate regions from UIU-Net, SCTransNet and DNANet using local contrast, region geometry and neighboring responses. The scorer gives equal weight to the sparse and dense calibration domains. Validation on both domains determines the retention threshold.

## Setup

Install the dependencies.

```sh
python -m pip install -r requirements.txt
```

Download [weights.zip from Baidu Netdisk](https://pan.baidu.com/s/1ljvSisqxW8btS3kwUA0eOw?pwd=a5d3) using extraction code `a5d3`. Extract it so that `uiu.pth`, `sctrans.pth`, and `dnanet.pth` are directly inside the repository's `weights/` directory.

`run.py` loads the architectures from `models/` and the checkpoints from `weights/`. Each checkpoint includes the normalization values used at training time. See the upstream README in each model directory for attribution.

## Repository layout

```text
run.py              # Inference and Dense463 evaluation
calibration.py      # Candidate generation, rescoring and matching
models/             # UIU-Net, SCTransNet and DNANet architectures
configs/            # Frozen calibration settings per detector and fold
splits/             # Dataset filenames, folds and exclusions
weights/            # Three final checkpoints and their checksums
```

Single-image inference was checked with Python 3.7, PyTorch 1.9.0, NumPy 1.21.6, SciPy 1.7.3 and OpenCV 4.12.0. Install a PyTorch build compatible with your hardware.

## Inference

```sh
python run.py --model uiu --image image.png
```

The output contains candidate centers in original image coordinates and confidence scores. The default uses the frozen fold-0 balanced configuration. Use `--device cuda` for GPU inference.

## Dense463 evaluation

Download the DenseSIRST images and target masks from the dataset authors. Use the filenames and fold assignments in `splits/dense463.json`.

```sh
python run.py --model uiu --dense-images /path/to/images --dense-masks /path/to/masks --policy mixed_tradeoff
```

Use `--policy native_025` to evaluate the same candidates with the detector scores. The script reports TP, FP, FN, precision, recall, and point AP using one-to-one matching within three original-image pixels. Each image uses the frozen configuration for its held-out fold.

The release covers inference and Dense463 point evaluation with frozen configurations. Detector retraining, calibrator fitting, box AP and the IST-A transfer analysis are outside the scope of this package.

## Citation

If you use this code, please cite the accompanying paper with the title above.
