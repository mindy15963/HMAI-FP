import os  # 파일·디렉터리 경로 조작을 위해 OS 모듈을 불러옵니다
import pandas as pd  # CSV 등을 읽고 데이터프레임으로 다루기 위해 pandas를 불러옵니다
import numpy as np  # 수치 계산 및 배열 처리를 위해 NumPy를 불러옵니다
from PIL import Image  # 이미지 파일을 열고 RGB 변환을 위해 Pillow의 Image를 불러옵니다

import torch  # PyTorch 프레임워크를 불러옵니다
import torch.nn as nn  # 신경망 구성 블록(레이어, 손실 등)을 위해 nn 서브모듈을 불러옵니다
from torch.utils.data import Dataset, DataLoader  # 데이터셋과 배치 처리를 위한 클래스를 불러옵니다
from torchvision import transforms  # 이미지 전처리(변형)를 위한 유틸리티를 불러옵니다
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights  # EfficientNet-B3 모델과 사전학습 가중치를 불러옵니다

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)  # 평가 지표 계산에 사용할 sklearn의 함수들을 불러옵니다

import matplotlib.pyplot as plt  # 학습 결과 및 시각화를 위해 matplotlib를 불러옵니다
import seaborn as sns  # 혼동행렬 등 시각화 스타일을 위해 seaborn을 불러옵니다
import cv2  # 이미지 후처리(컬러맵, 리사이즈)를 위해 OpenCV를 불러옵니다


# ============================================================
# Dataset
# ============================================================
class ORIGADataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)  # CSV 파일을 읽어 이미지 파일명과 라벨 정보를 데이터프레임으로 로드합니다
        self.img_dir = img_dir  # 이미지가 저장된 폴더 경로를 멤버 변수로 저장합니다
        self.transform = transform  # 학습/평가 시 적용할 torchvision 변형(transform)을 저장합니다

    def __len__(self):
        return len(self.data)  # 데이터셋의 샘플 수(길이)를 반환합니다

    def __getitem__(self, idx):
        img_name = os.path.basename(self.data.iloc[idx]["ImageName"])  # 데이터프레임에서 이미지 파일명을 읽어 basename을 얻습니다
        label = int(self.data.iloc[idx]["glaucoma"])  # 해당 인덱스의 glaucoma 라벨을 정수형으로 읽습니다

        img_path = os.path.join(self.img_dir, img_name)  # 이미지 디렉터리와 파일명을 결합해 전체 경로를 만듭니다
        image = Image.open(img_path).convert("RGB")  # 이미지를 열고 3채널 RGB로 변환합니다

        if self.transform:
            image = self.transform(image)  # transform이 정의되어 있으면 이미지에 전처리를 적용합니다

        return image, torch.tensor(label, dtype=torch.long)  # (이미지 텐서, 라벨 텐서)를 반환합니다


# ============================================================
# Transforms
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 모델 입력 크기에 맞게 이미지를 리사이즈합니다
    transforms.RandomHorizontalFlip(),  # 수평 반전을 무작위로 적용해 데이터 다양성을 늘립니다
    transforms.RandomRotation(15),  # ±15도 범위 내에서 무작위 회전을 적용합니다
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # 밝기와 대비를 무작위로 변경해 강건성을 향상합니다
    transforms.ToTensor(),  # PIL 이미지를 PyTorch 텐서로 변환하고 픽셀 범위를 [0,1]로 정규화합니다
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 평가 시에도 모델 입력 크기로 리사이즈합니다
    transforms.ToTensor(),  # 평가 이미지를 텐서로 변환합니다
])


# ============================================================
# Dataset / Loader
# ============================================================
train_dataset = ORIGADataset(
    "./ORIGA Retinal Fundus Image Dataset/ORIGA_train.csv",  # 학습용 CSV 경로
    "./ORIGA Retinal Fundus Image Dataset/ORIGA/train",  # 학습 이미지 디렉터리
    transform=train_transform  # 학습용 변형을 전달
)

test_dataset = ORIGADataset(
    "./ORIGA Retinal Fundus Image Dataset/ORIGA_test.csv",  # 테스트용 CSV 경로
    "./ORIGA Retinal Fundus Image Dataset/ORIGA/test",  # 테스트 이미지 디렉터리
    transform=test_transform  # 테스트용 변형을 전달
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 학습 데이터 로더: 배치 크기 16, 셔플 적용
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)  # 테스트 데이터 로더: 셔플 비적용


# ============================================================
# Model
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"  # 사용 가능한 경우 GPU(cuda)를 사용하도록 설정합니다

model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)  # ImageNet 사전학습 가중치를 가진 EfficientNet-B3 모델을 불러옵니다
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # 마지막 분류층을 이진 분류(2클래스)에 맞게 교체합니다
model = model.to(device)  # 모델을 선택한 디바이스로 이동시킵니다

criterion = nn.CrossEntropyLoss()  # 다중(이진 포함) 클래스 분류에 적합한 교차엔트로피 손실을 사용합니다
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Adam 옵티마이저로 모델 파라미터를 업데이트합니다


# ============================================================
# Training & Evaluation
# ============================================================
def train_one_epoch(model, loader):
    model.train()  # 모델을 훈련 모드로 전환합니다(드롭아웃/배치정규화 동작 변경)
    total_loss = 0  # 에포크 누적 손실을 저장할 변수를 초기화합니다

    for imgs, labels in loader:  # 데이터로더에서 배치 단위로 이미지를 불러옵니다
        imgs, labels = imgs.to(device), labels.to(device)  # 입력과 라벨을 학습 디바이스로 이동시킵니다

        optimizer.zero_grad()  # 이전 배치의 그래디언트를 초기화합니다
        outputs = model(imgs)  # 모델을 순전파하여 로짓(logits)을 얻습니다
        loss = criterion(outputs, labels)  # 출력과 정답 라벨로 손실을 계산합니다

        loss.backward()  # 역전파를 통해 그래디언트를 계산합니다
        optimizer.step()  # 옵티마이저가 파라미터를 갱신합니다

        total_loss += loss.item()  # 배치 손실을 누적합니다

    return total_loss / len(loader)  # 평균 손실을 반환합니다


def evaluate(model, loader):
    model.eval()  # 모델을 평가 모드로 전환합니다(드롭아웃/배치정규화 비활성화)
    preds, trues, probs = [], [], []  # 예측값, 실제값, 확률을 저장할 리스트를 초기화합니다

    with torch.no_grad():  # 평가 중에는 그래디언트를 계산하지 않아 메모리와 연산을 절약합니다
        for imgs, labels in loader:  # 평가 데이터 로더에서 배치 단위로 불러옵니다
            imgs = imgs.to(device)  # 입력을 디바이스로 이동시킵니다
            outputs = model(imgs)  # 모델로부터 로짓을 얻습니다

            softmax = torch.softmax(outputs, dim=1)[:, 1]  # 클래스 1(녹내장)일 확률을 소프트맥스로 계산합니다

            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())  # 가장 높은 확률을 가진 클래스 인덱스를 예측으로 저장합니다
            trues.extend(labels.numpy())  # 실제 라벨을 리스트에 추가합니다
            probs.extend(softmax.cpu().numpy())  # 클래스1의 확률을 리스트에 추가합니다

    acc       = accuracy_score(trues, preds)  # 정확도 계산
    f1        = f1_score(trues, preds)  # F1 점수 계산
    auc       = roc_auc_score(trues, probs)  # ROC AUC 계산 (확률 기반)
    precision = precision_score(trues, preds)  # 정밀도 계산
    recall    = recall_score(trues, preds)  # 재현율(민감도) 계산
    cm        = confusion_matrix(trues, preds)  # 혼동행렬 계산

    return acc, f1, auc, precision, recall, cm, preds, trues  # 평가 지표와 예측/실제값을 반환합니다


# ============================================================
# History Logging
# ============================================================
history = {"loss": [], "acc": [], "f1": [], "auc": [], "precision": [], "recall": []}  # 학습/평가 지표를 기록할 딕셔너리를 초기화합니다

num_epochs = 30  # 전체 학습 에포크 수를 설정합니다

for epoch in range(num_epochs):  # 에포크 루프를 실행합니다
    train_loss = train_one_epoch(model, train_loader)  # 한 에포크를 학습하고 평균 손실을 반환받습니다
    acc, f1, auc, precision, recall, cm, preds, trues = evaluate(model, test_loader)  # 에포크 후 테스트셋으로 평가합니다

    history["loss"].append(train_loss)  # 손실 기록을 업데이트합니다
    history["acc"].append(acc)  # 정확도 기록을 업데이트합니다
    history["f1"].append(f1)  # F1 기록을 업데이트합니다
    history["auc"].append(auc)  # AUC 기록을 업데이트합니다
    history["precision"].append(precision)  # 정밀도 기록을 업데이트합니다
    history["recall"].append(recall)  # 재현율 기록을 업데이트합니다

    print(f"[Epoch {epoch+1}] Loss: {train_loss:.4f} | Acc: {acc:.4f} | "
          f"F1: {f1:.4f} | AUC: {auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")  # 현재 에포크의 요약 결과를 출력합니다


# ============================================================
# Confusion Matrix Plot
# ============================================================
plt.figure(figsize=(6, 5))  # 플롯 크기를 지정합니다
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Glaucoma"],
            yticklabels=["Normal", "Glaucoma"])  # 혼동행렬을 히트맵으로 시각화합니다
plt.title("Confusion Matrix")  # 그래프 제목을 설정합니다
plt.xlabel("Predicted")  # x축 레이블을 설정합니다
plt.ylabel("True")  # y축 레이블을 설정합니다
plt.show()  # 혼동행렬 플롯을 화면에 표시합니다


# ============================================================
# Metric Graphs
# ============================================================
plt.figure(figsize=(16, 10))  # 여러 지표를 한 화면에 그리기 위한 큰 캔버스 크기를 지정합니다

plt.subplot(3, 2, 1)  # 3행 2열의 서브플롯 중 첫 번째 위치를 선택합니다
plt.plot(history["loss"], marker="o")  # 학습 손실 변화를 선으로 플롯합니다
plt.title("Training Loss")  # 서브플롯 제목을 설정합니다
plt.grid(True)  # 그리드(격자선)를 켭니다

plt.subplot(3, 2, 2)  # 두 번째 서브플롯 위치를 선택합니다
plt.plot(history["acc"], marker="o")  # 정확도 변화를 플롯합니다
plt.title("Accuracy")  # 제목 설정
plt.grid(True)  # 그리드 켜기

plt.subplot(3, 2, 3)  # 세 번째 서브플롯 위치를 선택합니다
plt.plot(history["f1"], marker="o")  # F1 점수 변화를 플롯합니다
plt.title("F1 Score")  # 제목 설정
plt.grid(True)  # 그리드 켜기

plt.subplot(3, 2, 4)  # 네 번째 서브플롯 위치를 선택합니다
plt.plot(history["auc"], marker="o")  # AUC 변화를 플롯합니다
plt.title("AUC")  # 제목 설정
plt.grid(True)  # 그리드 켜기

plt.subplot(3, 2, 5)  # 다섯 번째 서브플롯 위치를 선택합니다
plt.plot(history["precision"], marker="o")  # 정밀도 변화를 플롯합니다
plt.title("Precision")  # 제목 설정
plt.grid(True)  # 그리드 켜기

plt.subplot(3, 2, 6)  # 여섯 번째 서브플롯 위치를 선택합니다
plt.plot(history["recall"], marker="o")  # 재현율 변화를 플롯합니다
plt.title("Recall")  # 제목 설정
plt.grid(True)  # 그리드 켜기

plt.tight_layout()  # 서브플롯 간의 레이아웃 간격을 자동 조정합니다
plt.show()  # 지표 그래프들을 화면에 표시합니다


# ============================================================
# Grad-CAM
# ============================================================
def generate_gradcam(model, image_tensor):
    model.eval()  # Grad-CAM 계산을 위해 모델을 평가 모드로 전환합니다
    image_tensor = image_tensor.unsqueeze(0).to(device)  # 단일 이미지를 배치 차원(1, C, H, W)으로 확장하고 디바이스로 이동합니다

    features = None  # 순전파에서의 특징 맵을 저장할 변수 초기화
    gradients = None  # 역전파에서의 그래디언트를 저장할 변수 초기화

    # Forward Hook
    def forward_hook(module, input, output):
        nonlocal features
        features = output  # forward 시점의 출력(특징 맵)을 캡처합니다

    # Backward Hook
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]  # backward 시점의 그래디언트를 캡처합니다

    last_conv = model.features[-1]  # EfficientNet의 마지막 컨볼루션 블록을 선택합니다

    last_conv.register_forward_hook(forward_hook)  # 선택한 블록에 forward hook을 등록합니다
    last_conv.register_full_backward_hook(backward_hook)  # 선택한 블록에 backward hook을 등록합니다

    # Forward
    output = model(image_tensor)  # 이미지에 대해 순전파를 수행하여 로짓을 얻습니다
    pred_class = output.argmax(dim=1)  # 모델이 예측한 클래스 인덱스를 얻습니다
    output[0, pred_class].backward()  # 예측 클래스 스코어에 대해 역전파를 수행하여 그래디언트를 얻습니다

    weights = gradients.mean(dim=(2, 3), keepdim=True)  # 채널별 그래디언트의 전역 평균을 가중치로 계산합니다

    cam = (weights * features).sum(dim=1).squeeze()  # 가중합을 통해 클래스 활성화 지도를 계산합니다
    cam = cam.detach().cpu().numpy()  # 텐서를 NumPy 배열로 변환합니다

    cam = np.maximum(cam, 0)  # 음수 값을 제거하여 ReLU를 적용합니다
    cam = cam / cam.max()  # 0~1 범위로 정규화합니다

    return cam  # 정규화된 CAM을 반환합니다


# ============================================================
# Grad-CAM Visualization
# ============================================================
sample_img, _ = test_dataset[0]  # 테스트 데이터셋의 첫 샘플(이미지, 라벨)을 가져옵니다
cam = generate_gradcam(model, sample_img)  # 해당 샘플에 대해 Grad-CAM을 생성합니다

orig = sample_img.permute(1, 2, 0).numpy()  # (C,H,W) 텐서를 (H,W,C) 형태로 변환하여 NumPy 배열로 만듭니다

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # CAM을 컬러맵으로 변환하여 heatmap을 생성합니다
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # OpenCV의 BGR을 RGB로 변환합니다
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))  # 원본 이미지 크기에 맞게 heatmap을 리사이즈합니다

overlay = 0.5 * heatmap + 0.5 * (orig * 255)  # 원본 이미지와 heatmap을 반반씩 합성하여 오버레이를 만듭니다

plt.figure(figsize=(10, 5))  # 시각화 캔버스 크기를 지정합니다
plt.subplot(1, 2, 1)  # 좌측 서브플롯을 선택합니다
plt.title("Original")  # 원본 이미지 서브플롯 제목 설정
plt.imshow(orig)  # 원본 이미지를 화면에 표시합니다

plt.subplot(1, 2, 2)  # 우측 서브플롯을 선택합니다
plt.title("Grad-CAM")  # Grad-CAM 결과 서브플롯 제목 설정
plt.imshow(np.uint8(overlay))  # 오버레이 이미지를 화면에 표시합니다
plt.show()  # 시각화를 출력합니다