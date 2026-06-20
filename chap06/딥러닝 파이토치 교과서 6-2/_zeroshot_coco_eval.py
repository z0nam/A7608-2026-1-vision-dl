"""
Zero-shot COCO Faster R-CNN 평가 (PennFudanPed 보행자 검출)

헤드 교체 없이 COCO 사전학습 가중치를 그대로 사용하고,
COCO 'person'(label==1) 예측만 골라 PennFudan 보행자 정답과 비교한다.
학습 없음 — 추론만. 기말 레포트 검출 실험용 실측치 생성.
"""
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as func
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)

DATA_ROOT = Path(__file__).resolve().parent  # PennFudanPed/ 가 있는 폴더
COCO_PERSON = 1  # COCO 클래스 인덱스에서 person == 1
IMAGE_SIZE = 320


class PennFudanDetection(Dataset):
    def __init__(self, root, image_size=IMAGE_SIZE):
        self.root = Path(root)
        self.image_size = image_size
        self.images = sorted((self.root / "PennFudanPed" / "PNGImages").glob("*.png"))
        self.masks = sorted((self.root / "PennFudanPed" / "PedMasks").glob("*.png"))

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])
        image = image.resize((self.image_size, self.image_size))
        mask = mask.resize((self.image_size, self.image_size),
                           resample=Image.Resampling.NEAREST)
        mask = np.array(mask)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        boxes = []
        for oid in obj_ids:
            pos = np.where(mask == oid)
            y0, y1 = np.min(pos[0]), np.max(pos[0])
            x0, x1 = np.min(pos[1]), np.max(pos[1])
            if x1 > x0 and y1 > y0:
                boxes.append([x0, y0, x1, y1])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        return func.to_tensor(image), {"boxes": boxes}


def box_iou(b1, b2):
    if len(b1) == 0 or len(b2) == 0:
        return torch.zeros((len(b1), len(b2)))
    a1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    a2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)
    lt = torch.max(b1[:, None, :2], b2[:, :2])
    rb = torch.min(b1[:, None, 2:], b2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = a1[:, None] + a2 - inter
    return inter / union.clamp(min=1e-6)


def evaluate(model, loader, device, score_threshold=0.5, iou_threshold=0.5):
    model.eval()
    tp = fp = fn = 0
    matched_ious = []
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                labels = out["labels"].cpu()
                scores = out["scores"].cpu()
                boxes = out["boxes"].cpu()
                # person + score 임계값
                keep = (labels == COCO_PERSON) & (scores >= score_threshold)
                pred = boxes[keep]
                gt = tgt["boxes"]
                if len(pred) == 0:
                    fn += len(gt); continue
                if len(gt) == 0:
                    fp += len(pred); continue
                ious = box_iou(pred, gt)
                matched = set()
                for pi in range(len(pred)):
                    best_iou, best_gt = ious[pi].max(dim=0)
                    best_gt = int(best_gt)
                    if best_iou >= iou_threshold and best_gt not in matched:
                        tp += 1; matched.add(best_gt); matched_ious.append(float(best_iou))
                    else:
                        fp += 1
                fn += len(gt) - len(matched)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return dict(TP=tp, FP=fp, FN=fn, Precision=precision, Recall=recall,
               F1=f1, MeanIoU=mean_iou, GT=tp + fn)


def main():
    # MPS는 torch 1.13에서 RPN의 topk(k>16) 미지원 → CPU 사용 (추론 170장, 수 분)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    ds = PennFudanDetection(DATA_ROOT)
    print(f"전체 이미지: {len(ds)} | 평가: zero-shot COCO (학습 없음)")
    loader = DataLoader(ds, batch_size=2, shuffle=False,
                        collate_fn=PennFudanDetection.collate_fn,
                        num_workers=0, pin_memory=False)
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device)
    print("모델: fasterrcnn_resnet50_fpn_v2 (COCO 사전학습, 헤드 교체 없음)")
    print("=" * 55)
    t0 = time.time()
    for th in (0.5, 0.7):
        m = evaluate(model, loader, device, score_threshold=th, iou_threshold=0.5)
        print(f"[score>={th}, IoU>=0.5] "
              f"TP {m['TP']} FP {m['FP']} FN {m['FN']} (GT {m['GT']}) | "
              f"P {m['Precision']:.3f} R {m['Recall']:.3f} "
              f"F1 {m['F1']:.3f} mIoU {m['MeanIoU']:.3f}")
    print(f"추론 시간: {time.time() - t0:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
