**CNN 기반 컴퓨터 비전 방법의 이해와 응용: ResNet을 중심으로 한 분류·검출·분할의 동일 관점 비교 분석**

# 서론

컴퓨터 비전은 이미지로부터 의미 있는 정보를 추출하는 분야이며, 그 출발점은 입력 이미지가 어떤 범주에 속하는지를 예측하는 **이미지 분류(image classification)** 다. 중간 레포트에서는 분류를 중심으로 AlexNet, VGG, ResNet, EfficientNet 네 모델을 비교하며 CNN의 구조와 학습 원리를 정리하였다. 기말 레포트는 그 연장선에서, 분류 분야의 **대표 모델 1개로 ResNet을 선정**하여 구조·원리를 심층 분석하고, 전이학습 실험 결과와 공개 성능을 함께 해석한다.

ResNet을 대표 모델로 고른 이유는 단순한 분류 정확도 때문이 아니라, ResNet이 도입한 **잔차 학습(residual learning)과 skip connection**이 이후 검출·분할 모델의 **표준 특징 추출 네트워크**(backbone)로 그대로 재사용되기 때문이다. 따라서 본 레포트는 ResNet을 분류의 대표 모델로 깊이 분석한 뒤, 같은 특징 추출 네트워크가 검출(Faster R-CNN)과 분할(U-Net)로 어떻게 확장되는지를 **동일 관점 비교표**로 정리한다. 이렇게 하면 "하나의 모델"에 집중하면서도, 그 모델이 분류·검출·분할을 관통하는 토대임을 비교·분석으로 보일 수 있다.

# 이미지 분류 문제와 CNN의 기본 구조

이미지 분류는 입력 이미지 *x*가 주어졌을 때 class label *y*를 예측하는 문제다. 모델은 각 class에 대한 score(또는 확률)를 출력하고 가장 높은 class를 예측값으로 택하며, 학습은 정답과 예측 확률의 차이(보통 softmax + cross-entropy loss)를 줄이도록 파라미터를 조정한다.

CNN은 이 문제에 특히 적합한 구조를 가진다. 핵심 성질은 세 가지다. **(1) 합성곱(convolution)**: 작은 filter(예: 3×3)를 이미지 전체에 같은 가중치로 미끄러뜨려 지역적 특징을 뽑는다. 이미지 전체를 연결하는 fully connected layer보다 파라미터가 적고(가중치 공유), 공간적 인접성을 활용하며, 물체가 어디에 있든 같은 filter가 반응한다(translation equivariance). **(2) 계층적 특징**: layer를 쌓으면 receptive field가 넓어져, 얕은 층은 edge·texture 같은 저수준 패턴을, 깊은 층은 그것들을 조합한 고수준 의미(눈·바퀴·얼굴)를 학습한다. **(3) 다운샘플링(downsampling)**: pooling이나 stride>1 합성곱으로 feature map의 H×W를 절반씩 줄여 맥락을 넓히고 연산을 줄인다. 다만 그 대가로 **위치 정보를 잃는데**, 이 손실을 어떻게 다루느냐가 뒤에서 검출·분할 설계의 핵심 쟁점이 된다.

요컨대 CNN의 발전은 단순히 layer 수가 늘어난 역사가 아니라, **더 깊고 안정적이며 효율적인 특징 학습 구조**를 찾는 과정으로 볼 수 있다.

# 분류 대표 모델의 발전 흐름

네 모델은 CNN 발전의 서로 다른 단계를 보여준다. 본 장은 흐름을 압축해 정리하고, 대표 모델 ResNet은 다음 장에서 따로 심층 분석한다.

- **AlexNet (2012)**: ImageNet 대회에서 깊은 CNN의 가능성을 입증한 출발점. 기여는 구조 자체보다 **학습을 가능케 한 요소들** — ReLU(기울기 소실 완화, 학습 가속), dropout(과적합 억제), data augmentation, GPU 학습 — 에 있다. 단, fully connected layer의 파라미터 비중이 크고 효율이 낮다.
- **VGG (2014)**: 큰 filter 대신 **3×3 작은 filter를 반복**해 깊이를 늘렸다. 작은 filter를 여러 번 쌓으면 큰 receptive field 효과를 내면서 비선형을 더 자주 적용해 표현력이 는다. 구조가 직관적이라 교육·전이학습에 유용하나, 파라미터·연산량이 많다.
- **ResNet (2016)**: **residual learning + skip connection**으로 "깊게 쌓아도 학습되는" 문제를 풀었다. 본 레포트의 대표 모델로 다음 장에서 상세히 다룬다.
- **EfficientNet (2019)**: depth·width·resolution을 정해진 비율로 **동시에** 키우는 **compound scaling**으로, 적은 연산·파라미터로 높은 정확도를 달성했다. CNN 설계가 "더 깊게"에서 "더 효율적으로" 이동했음을 보여준다.

이 흐름의 공통 메시지는, 깊이를 늘려 표현력을 키우되 그것을 **안정적으로 학습**할 구조적 장치가 필요했다는 점이며, 그 장치를 가장 분명하게 제시한 모델이 ResNet이다.

# 대표 모델 심층 분석: ResNet

## 핵심 문제와 해결: 열화(degradation)와 잔차 학습

ResNet의 출발점은 직관에 어긋나는 관찰이다. 평범한(plain) 망은 층을 더 쌓으면 **학습 오차마저 더 커진다.** 이는 과적합이 아니라 **최적화 자체가 어려워지는** 현상(degradation)으로, 너무 깊으면 기울기가 앞쪽 층까지 잘 전달되지 않기 때문이다.

해결책이 **잔차 학습**이다. 블록이 목표 함수 *H(x)* 를 통째로 학습하는 대신 **잔차 *F(x) = H(x) − x*** 만 학습하고, 출력은 입력을 그대로 더한 ***F(x) + x*** 로 만든다. 입력 *x*를 더해 주는 우회로가 **skip connection(shortcut)** 이다. 이 단순한 구조가 두 가지를 동시에 해결한다.

- **항등 함수 학습이 쉬워진다**: 어떤 층이 "아무것도 안 하는 것"이 최선이라면 plain 망은 여러 비선형 층으로 항등 함수를 흉내 내야 해 어렵지만, 잔차 구조는 *F(x)=0* 으로만 만들면 된다. 즉 "깊게 쌓아도 최소한 손해는 안 본다"가 보장된다.
- **기울기가 잘 흐른다**: 역전파 시 *+x* 경로가 **기울기 고속도로** 역할을 해, 깊은 층의 기울기가 감쇠 없이 앞으로 전달된다.

함께 쓰인 **배치 정규화(BatchNorm)** 가 각 층 입력 분포를 안정화하고, **bottleneck 블록**(1×1로 채널 축소 → 3×3 → 1×1로 복원)이 깊이를 늘리면서 연산량을 통제한다. ResNet의 의의는 정확도 향상을 넘어, residual connection이 컴퓨터 비전뿐 아니라 다양한 딥러닝 구조의 기본 아이디어가 되었고 **이후 검출·분할 모델의 표준 특징 추출 네트워크**가 되었다는 데 있다.

[그림 1] ResNet 잔차 블록 — 블록은 H(x)가 아니라 잔차 F(x)만 학습하고, 입력 x를 더해 출력한다.

## 입력/출력과 평가지표

- **입력**: RGB 이미지 1장. **출력**: 클래스 확률 벡터(예: ImageNet 1000차원), argmax가 예측 클래스.
- **평가지표**: **Top-1 Accuracy**(최상위 예측이 정답인 비율)와 **Top-5 Accuracy**(상위 5개 안에 정답이 포함된 비율). class가 많고 비슷할 때는 Top-5가 모델의 후보 예측력을 함께 보여준다. 실용적으로는 파라미터 수·FLOPs·추론 속도도 함께 고려된다.
- **공개 성능(참고)**: ResNet-50, ImageNet 검증셋 기준 Top-1 ≈ 76%, Top-5 ≈ 93%.

## 실험: 전이학습 기반 소규모 분류와 그 해석

**설정**: ImageNet 사전학습 특징 추출 네트워크를 소규모 **이진 분류** 데이터셋에 **전이학습(transfer learning)** 으로 미세조정했다(딥러닝 파이토치 교과서 6장 실습 기반). 짧은 학습(10 epoch 내외)에서 검증 정확도는 약 **58%** 에 머물렀다(class가 2개뿐이라 Top-5는 100%).

**분석**: 절반 남짓이면 사실상 잘 학습되지 않은 것이다. 원인은 ① 데이터셋이 작고, ② 학습 epoch이 짧아 특징 추출 네트워크가 새 도메인에 충분히 적응하지 못했으며, ③ 학습률·해동(freeze/unfreeze) 설정의 영향으로 보인다. 공개 ImageNet에서 Top-1 ~76%를 내는 동일 구조가 소규모 도메인에서는 부진했다는 점이 핵심이다. 교훈은 분명하다 — **사전학습이 만능은 아니며, 전이학습도 충분한 데이터·학습량·튜닝이 있어야 위력을 발휘한다.** 이 관찰은 뒤의 비교에서 검출·분할의 전이 결과와 나란히 놓일 때, "전이의 성패는 사전학습 도메인과의 일치도에 달렸다"는 일관된 메시지로 이어진다.

# 동일 관점 비교표: ResNet 특징 추출 네트워크가 검출·분할로 확장되는 방식

대표 모델 ResNet은 분류에만 쓰이지 않는다. 검출의 Faster R-CNN은 ResNet+FPN을 특징 추출 네트워크로 쓰고, 분할의 U-Net은 ResNet과 **같은 skip connection 아이디어**를 공간 정보 보존에 활용한다(그림 2). 본 장은 동일한 관점(입력/출력 형태, 핵심 구조, 장점, 단점, 평가지표)으로 세 과제를 비교한다.

[그림 2] 공유 특징 추출 네트워크와 head 분기 — 하나의 특징 추출 네트워크를 공유하고 head만 갈아 끼워 분류·검출·분할로 분기한다.

## 동일 관점 비교표

분류·검출·분할 동일 관점 비교

| 관점 | 분류 (ResNet) | 검출 (Faster R-CNN) | 분할 (U-Net) |
|---|---|---|---|
| **입력 형태** | RGB 이미지 1장 | RGB 이미지 1장 | RGB 이미지 1장 |
| **출력 형태** | 클래스 확률 벡터(이미지당 1개) | 객체별 (박스+클래스+점수) 목록(가변 개수) | 입력과 동일 해상도의 픽셀별 라벨 맵 |
| **위치 정보** | 없음(전역) | 박스 수준(거침) | 픽셀 수준(최고) |
| **핵심 구조** | residual block + skip connection | 특징 추출 네트워크(ResNet+FPN) + RPN + RoI head (2단계) | 인코더-디코더 + skip connection |
| **특징 추출 네트워크 관계** | 기준 특징 추출 네트워크 | ResNet 특징 추출 네트워크를 그대로 재사용 | 인코더가 분류 특징 추출 네트워크와 동형, skip은 ResNet과 같은 아이디어 |
| **장점** | 단순·안정적 학습, 특징 추출 네트워크 재사용성 | 다중 객체 위치 검출, 높은 정확도 | 픽셀 단위 정밀 경계, 적은 데이터에 강건 |
| **단점** | 위치 정보 없음 | 2단계라 무겁고 느림 | 출력 해상도 커 연산·메모리 비용 큼 |
| **평가지표** | Top-1 / Top-5 Accuracy | mAP (+ Precision/Recall/F1, IoU) | mIoU, 픽셀 정확도 |
| **결과(실험·공개)** | 전이학습 검증 ~58% / 공개 ImageNet Top-1 ~76% | zero-shot COCO: Recall 0.998, F1 0.762, 평균 IoU 0.895 | 사전학습 인코더 1 epoch 테스트 mIoU 69.1% |
주) 검출·분할의 수치는 본 과제에서 직접 수행한 실험 결과다. 검출은 COCO 사전학습 fasterrcnn_resnet50_fpn_v2를 추가 학습 없이 PennFudanPed 보행자(person)로 평가했고, 분할은 ImageNet 사전학습 VGG 인코더를 단 1 epoch 학습해 Oxford-IIIT Pet 3클래스를 평가했다.

[그림 3] U-Net 분할 실험 출력(직접 실행) — 왼쪽부터 원본·정답 마스크·예측. 샘플별 IoU 표기(0.568~0.736).

그림 3은 분할 실험의 실제 출력이다. 전경(반려동물)과 배경은 1 epoch만으로도 또렷이 분리되지만, 얇고 모호한 '경계' 픽셀에서 오차가 집중되어 샘플별 IoU 편차(0.568~0.736)를 만든다. 이는 아래 통합 분석 ③에서 skip connection이 보강하려는 바로 그 어려움(경계 복원)과 정확히 맞닿는다.

## 통합 분석: 세 과제를 관통하는 통찰

**① 하나의 특징 추출 네트워크, 갈아 끼우는 head.** 세 모델은 동일한 CNN 특징 추출 네트워크에 기반한다. ResNet 같은 분류 특징 추출 네트워크가 검출(Faster R-CNN)과 분할(U-Net 인코더)에 그대로 재사용되고, 차이는 출력 head뿐이다 — 분류는 전역 풀링+FC, 검출은 RPN+RoI head, 분할은 업샘플 디코더. **"표현 학습은 공유하고, 과제는 head로 분기한다"** 가 현대 비전의 설계 문법이며, ResNet을 대표 모델로 고른 이유가 바로 이 재사용성이다.

**② 평가지표는 모두 혼동행렬 한 뿌리에서 갈라진다.** Accuracy, Precision/Recall/F1, IoU·mAP·mIoU는 전부 TP/FP/FN의 조합이다. 단위만 다르다 — 분류는 **이미지** 단위, 검출은 **박스**(IoU≥0.5로 정답 판정) 단위, 분할은 **픽셀** 단위로 같은 셈을 한다. 지표를 따로 외울 게 아니라 "무엇을 한 단위로 세느냐"의 차이로 이해할 수 있다.

**③ 다운샘플이 만든 위치 정보 손실, 그리고 그 보충.** 앞서 본 대로 다운샘플은 맥락을 넓히는 대가로 위치를 잃는다. 세 과제는 이 손실을 각자 다룬다 — 분류는 위치가 필요 없어 **무시**, 검출은 앵커·RoI·박스 회귀로 위치를 **재구성**, 분할은 skip connection으로 얕은 층의 위치 정보를 **되살린다**. 흥미롭게도 ResNet의 skip이 "기울기 보존"이라면 U-Net의 skip은 "공간 정보 보존"으로, **같은 더하기/잇기 아이디어의 다른 활용**이다. 과제가 정밀해질수록 "잃어버린 위치를 어떻게 되찾느냐"가 설계의 본질이 된다.

**④ 전이의 성패는 도메인 일치도에 달렸다.** 세 결과를 한자리에 놓으면 일관된 메시지가 보인다. 검출(보행자=COCO person, 거의 동일 도메인)은 zero-shot으로도 Recall 0.998. 분할(Pet, 새 객체지만 사전학습 인코더 사용)은 1 epoch에 69.1%. 분류(이진, 새 도메인+짧은 학습)는 ~58%에 그쳤다. **사전학습된 표현이 강력할수록, 그리고 목표가 사전학습 도메인에 가까울수록, 적은 학습으로 큰 성능**을 얻는다.

# 결론 및 개인 의견

본 레포트는 분류 분야의 대표 모델로 **ResNet**을 선정해 잔차 학습과 skip connection의 원리를 분석하고, 전이학습 실험(~58%)과 공개 성능(Top-1 ~76%)을 해석했다. 나아가 같은 특징 추출 네트워크가 검출(Faster R-CNN)·분할(U-Net)로 확장되는 방식을 동일 관점 비교표로 정리해, 세 과제가 별개가 아니라 **같은 특징 표현 위에 과제별 head를 얹은 형제**임을 보였다.

개인적으로 한 학기를 관통하는 가장 중요한 아이디어는 **skip connection**이라고 생각한다. ResNet에서는 기울기를 보존해 "깊게 쌓는 것"을, U-Net에서는 공간 정보를 보존해 "정밀하게 복원하는 것"을 가능케 했다. 표면적으로는 단순한 "더하기/잇기"지만, **좋은 정보를 손실 없이 흘려보내는 통로**라는 본질은 같다. 결국 비전 성능의 핵심은 "더 화려한 head"가 아니라 "**좋은 표현을 안정적으로 학습·보존하는 구조**"이며, ResNet은 그 구조를 가장 분명하게 제시했기에 분류를 넘어 검출·분할의 토대가 되었다. 향후 연구는 이 토대 위에 Transformer 결합(ViT·DETR), 경량화, 실제 배치 효율을 더하는 방향으로 나아갈 것이다.

# 참고문헌

1. K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," IEEE CVPR, 2016.
2. A. Krizhevsky, I. Sutskever, G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," NeurIPS, 2012.
3. K. Simonyan, A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," ICLR, 2015 (arXiv:1409.1556).
4. M. Tan, Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," ICML, 2019.
5. S. Ren, K. He, R. Girshick, J. Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," NeurIPS, 2015.
6. O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI, 2015.
7. J. Long, E. Shelhamer, T. Darrell, "Fully Convolutional Networks for Semantic Segmentation," IEEE CVPR, 2015.
8. 서지영, 『딥러닝 파이토치 교과서』, 길벗, 2022. (6장 실습 코드 기반)
9. 데이터셋: PennFudanPed (cis.upenn.edu/~jshi/ped_html), Oxford-IIIT Pet (torchvision OxfordIIITPet).
10. PyTorch Contributors, "Torchvision Models," PyTorch Documentation.
