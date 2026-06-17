"""노트북(이미지_분할을_위한_신경망.ipynb)을 에폭 1로 자동 실행 → 예측 결과 PNG 저장.
수업 따라가기 전, 파이프라인이 도는지 + 결과 미리보기용."""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import OxfordIIITPet
import segmentation_models_pytorch as smp

torch.manual_seed(42); np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f'사용 장치: {device}', flush=True)

# ---- 하이퍼파라미터 (에폭만 1로) ----
IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS = 256, 8, 1
LEARNING_RATE, NUM_CLASSES = 1e-4, 3
MAX_TRAIN_BATCHES = 100   # 미리보기용: 일부만 학습 (전체=460)
MAX_EVAL_BATCHES = 40     # 미리보기용: 일부만 평가 (전체=459)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')

class PetSegmentationDataset(Dataset):
    def __init__(self, root, split='trainval', image_size=256, download=True):
        self.dataset = OxfordIIITPet(root=root, split=split,
                                     target_types='segmentation', download=download)
        self.image_size = image_size
        self.img_transform = T.Compose([
            T.Resize((image_size, image_size)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.mask_transform = T.Resize((image_size, image_size),
                                       interpolation=T.InterpolationMode.NEAREST)
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = torch.tensor(np.array(mask), dtype=torch.long) - 1
        return image, mask

print('데이터셋 준비 중... (첫 실행 시 ~800MB 다운로드)', flush=True)
train_dataset = PetSegmentationDataset(DATA_DIR, 'trainval', IMAGE_SIZE)
test_dataset = PetSegmentationDataset(DATA_DIR, 'test', IMAGE_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'학습 {len(train_dataset)} / 테스트 {len(test_dataset)}', flush=True)

model = smp.Unet(encoder_name='vgg16', encoder_weights='imagenet',
                 in_channels=3, classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def compute_iou(preds, labels, num_classes=NUM_CLASSES, ignore_index=255):
    preds, labels = preds.view(-1), labels.view(-1)
    valid = labels != ignore_index
    preds, labels = preds[valid], labels[valid]
    ious = []
    for cls in range(num_classes):
        pc, lc = preds == cls, labels == cls
        inter = (pc & lc).sum().float(); union = (pc | lc).sum().float()
        if union == 0: continue
        ious.append((inter / union).item())
    return np.mean(ious) if ious else 0.0

# ---- 학습 1 에폭 ----
print('=' * 50, flush=True)
print(f'U-Net 학습 시작 | {NUM_EPOCHS} 에폭 | {device}', flush=True)
model.train()
tot_loss = tot_iou = 0.0
for bi, (imgs, masks) in enumerate(train_loader):
    imgs, masks = imgs.to(device), masks.to(device)
    outputs = model(imgs)
    loss = criterion(outputs, masks)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    preds = outputs.argmax(1)
    tot_loss += loss.item(); tot_iou += compute_iou(preds.cpu(), masks.cpu())
    if (bi + 1) % 20 == 0:
        print(f'  배치 [{bi+1}/{MAX_TRAIN_BATCHES}] 손실 {loss.item():.4f} '
              f'IoU {tot_iou/(bi+1):.4f}', flush=True)
    if bi + 1 >= MAX_TRAIN_BATCHES:
        break
n_tr = bi + 1
print(f'학습 평균({n_tr}배치) -- 손실 {tot_loss/n_tr:.4f} mIoU {tot_iou/n_tr:.4f}', flush=True)

# ---- 테스트 평가 ----
model.eval(); te_loss = te_iou = 0.0; n_te = 0
with torch.no_grad():
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        te_loss += criterion(outputs, masks).item()
        te_iou += compute_iou(outputs.argmax(1).cpu(), masks.cpu())
        n_te += 1
        if n_te >= MAX_EVAL_BATCHES:
            break
te_loss /= n_te; te_iou /= n_te
print(f'테스트 -- 손실 {te_loss:.4f} mIoU {te_iou:.4f} ({te_iou*100:.1f}%)', flush=True)
torch.save(model.state_dict(), os.path.join(HERE, 'best_unet_pet.pth'))

# ---- 예측 결과 시각화 (셀 13) ----
COLORS = np.array([[255, 100, 80], [80, 160, 255], [255, 230, 50]], dtype=np.uint8)
CLASS_NAMES = ['전경 (동물)', '배경', '경계']
def denorm(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1)
def mask_rgb(m): return COLORS[np.clip(m.numpy(), 0, NUM_CLASSES - 1)]

imgs, masks = next(iter(test_loader))
with torch.no_grad():
    preds = model(imgs.to(device)).argmax(1).cpu()
n = min(4, BATCH_SIZE)
fig, axes = plt.subplots(n, 3, figsize=(12, n * 3.5))
fig.suptitle('U-Net 세그멘테이션 결과 (에폭 1, 테스트셋)', fontsize=13, fontweight='bold')
for ax, t in zip(axes[0], ['원본 이미지', '정답 마스크', '예측 마스크']):
    ax.set_title(t, fontsize=11, fontweight='bold')
for i in range(n):
    iou_v = compute_iou(preds[i].unsqueeze(0), masks[i].unsqueeze(0))
    axes[i, 0].imshow(denorm(imgs[i]).permute(1, 2, 0).numpy()); axes[i, 0].axis('off')
    axes[i, 1].imshow(mask_rgb(masks[i])); axes[i, 1].axis('off')
    axes[i, 2].imshow(mask_rgb(preds[i])); axes[i, 2].axis('off')
    axes[i, 2].set_title(f'IoU {iou_v:.3f}', fontsize=9)
patches = [mpatches.Patch(color=COLORS[c] / 255, label=CLASS_NAMES[c]) for c in range(NUM_CLASSES)]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=11, bbox_to_anchor=(0.5, 0.01))
plt.tight_layout(rect=[0, 0.04, 1, 1])
out = os.path.join(HERE, 'unet_pred_epoch1.png')
plt.savefig(out, dpi=110, bbox_inches='tight')
print(f'\n저장 완료: {out}', flush=True)
print('DONE', flush=True)
