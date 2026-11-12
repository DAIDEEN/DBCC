"""Run a final detector and its frozen candidate calibration."""
import argparse
from contextlib import redirect_stdout
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

from calibration import adjacency, apply, curve, match, proposals

ROOT = Path(__file__).resolve().parent


def detector(model, upstream):
    sys.path.insert(0, str(upstream.resolve()))
    if model == "uiu":
        from model import UIUNET
        return UIUNET(3, 1), 3
    if model == "sctrans":
        from model.Config import get_SCTrans_config
        from model.SCTransNet import SCTransNet
        return SCTransNet(get_SCTrans_config(), mode="train", deepsuper=True), 1
    spec = importlib.util.spec_from_file_location("dnanet", str(upstream / "model/DNANet/model_DNANet.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DNANet(mode="train", deep_supervision=True), 1


def gray(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["uiu", "sctrans", "dnanet"], required=True)
    parser.add_argument("--upstream", type=Path, help="Use an existing upstream checkout")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy", choices=["native_025", "native_050", "mixed_tradeoff", "mixed_strict"], default="mixed_tradeoff")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path)
    group.add_argument("--dense-images", type=Path)
    parser.add_argument("--dense-masks", type=Path)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    args = parser.parse_args()
    if args.dense_images and not args.dense_masks:
        parser.error("--dense-masks is required for Dense463 evaluation")
    upstream = args.upstream or ROOT / "models" / args.model
    if not upstream.is_dir():
        parser.error(f"Missing bundled architecture for {args.model}: {upstream}")
    with redirect_stdout(sys.stderr):
        model, channels = detector(args.model, upstream)
    checkpoint = torch.load(str(ROOT / "weights" / (args.model + ".pth")), map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(args.device).eval()
    torch.set_num_threads(4)
    configs = [json.loads((ROOT / "configs" / f"{args.model}_fold{i}.json").read_text()) for i in range(5)]

    def detect(path, fold):
        image = gray(path)
        if image is None:
            raise ValueError(f"Unable to decode {path}")
        h, w = image.shape
        x = (image.astype(np.float32) - checkpoint["normalization_mean"]) / checkpoint["normalization_std"]
        x = np.pad(x, ((0, (-h) % 32), (0, (-w) % 32)))
        x = np.repeat(x[None, None], channels, axis=1)
        with torch.no_grad():
            output = model(torch.from_numpy(x).to(args.device))
        if isinstance(output, (list, tuple)):
            output = output[0] if args.model == "uiu" else output[-1]
        score = output[0, 0, :h, :w].cpu().numpy()
        if args.policy.startswith("native"):
            return [dict(x=x, y=y, confidence=s) for y, x, s, _, _ in proposals(score, "cc" + args.policy[-3:])]
        return apply(image, score, configs[fold][args.policy])

    if args.image:
        print(json.dumps(detect(args.image, args.fold), indent=2))
        return
    images, scores = [], []
    tp = fp = total = 0
    for row in json.loads((ROOT / "splits/dense463.json").read_text()):
        candidates = detect(args.dense_images / row["image"], row["fold"])
        mask = gray(args.dense_masks / row["mask"])
        labels, count = ndimage.label(mask > 0, structure=np.ones((3, 3)))
        centers = ndimage.center_of_mass(mask > 0, labels, range(1, count + 1))
        edges = adjacency([(r["x"], r["y"]) for r in candidates],
                          np.array([(x, y) for y, x in centers]).reshape(-1, 2))
        items = [dict(overlap=edge) for edge in edges]
        hits = len(match(items, [True] * len(items)))
        tp += hits
        fp += len(items) - hits
        total += count
        images.append(dict(items=items, gt_count=count))
        scores.append([r["confidence"] for r in candidates])
    sweep = curve(images, scores)
    recall = np.asarray(sweep["TP"]) / total
    precision = np.asarray(sweep["TP"]) / np.maximum(1, np.asarray(sweep["TP"]) + sweep["FP"])
    ap = np.mean([max(precision[recall >= r], default=0) for r in np.linspace(0, 1, 101)])
    print(json.dumps(dict(images=len(images), targets=total, TP=tp, FP=fp, FN=total-tp,
                          precision=tp/max(1, tp+fp), recall=tp/total, point_AP=float(ap)), indent=2))


if __name__ == "__main__":
    main()
