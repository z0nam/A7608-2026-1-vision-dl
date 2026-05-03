# Context

## 2026-04-28 중간리포트 준비 계획

### 기준 폴더

- 중간리포트 관련 작업은 이 `mid_report/` 폴더를 기준으로 진행한다.
- 기존 `chap03/` 폴더는 Chapter 3 발표/실습 자료용으로 유지한다.
- 과제 안내 PDF 원본 위치: `A7608_mid_report_CNN/00.Object/중간 레포트_수정본1.pdf`

### 선택 주제

- 과제명: CNN 기반 컴퓨터 비전 방법의 이해와 응용: 분류, 검출, 분할 관점에서의 비교 분석
- 선택 영역: 이미지 분류(Image Classification)
- 리포트 작업 제목 초안: 이미지 분류 모델의 발전을 통해 본 CNN 구조와 학습 원리 비교 분석
- 비교 대상 모델: AlexNet, VGG, ResNet, EfficientNet

### 이 주제를 선택한 이유

- CNN의 기본 구성요소를 가장 직접적으로 이해할 수 있다.
  - convolution
  - pooling
  - activation function
  - fully connected layer
  - feature hierarchy
- 모델 발전 흐름이 CNN 이해 흐름과 잘 맞는다.
  - AlexNet: 현대 CNN 기반 이미지 분류의 출발점
  - VGG: 작은 convolution filter를 깊게 쌓는 설계
  - ResNet: 깊은 네트워크의 학습 문제와 skip connection
  - EfficientNet: 정확도, 계산량, 모델 크기의 균형
- 객체 검출과 의미 분할은 응용 파이프라인 개념이 많아 CNN 자체를 먼저 이해하려는 목적에는 이미지 분류보다 부담이 크다.

### 리포트 핵심 질문

- CNN은 이미지에서 어떤 방식으로 특징을 추출하고 분류에 활용하는가?
- AlexNet, VGG, ResNet, EfficientNet은 CNN 구조를 어떤 방향으로 발전시켰는가?
- 깊이, 연결 방식, 파라미터 수, 계산량, 정확도 사이에는 어떤 trade-off가 있는가?
- 단순히 정확도가 높은 모델이 항상 좋은 모델이라고 볼 수 있는가?

### 전체 구성 초안

1. 서론
   - 컴퓨터 비전에서 이미지 분류 문제의 의미
   - CNN이 전통적인 특징 추출 방식보다 강점을 갖는 이유
   - 리포트의 비교 관점 제시: 구조, 핵심 아이디어, 성능, 장단점

2. 이미지 분류 문제 정의
   - 입력 이미지가 주어졌을 때 하나 이상의 class label을 예측하는 문제로 정의
   - 대표 데이터셋 예시: ImageNet, CIFAR-10
   - 분류 문제에서 CNN이 학습하는 특징의 계층성 설명

3. CNN 기본 구조 정리
   - convolution layer: 지역 패턴과 공간 구조 학습
   - pooling layer: 위치 변화에 대한 강건성, feature map 축소
   - activation: 비선형성 부여
   - fully connected/classifier: 추출된 특징을 class score로 변환
   - softmax와 cross entropy loss 간단 설명

4. 대표 모델 비교
   - AlexNet
     - ImageNet에서 CNN의 가능성을 보여준 모델
     - ReLU, dropout, data augmentation, GPU 학습을 강조
   - VGG
     - 3x3 convolution을 반복적으로 쌓아 깊이를 늘린 모델
     - 구조가 단순하고 이해하기 쉽지만 파라미터 수가 많음
   - ResNet
     - residual block과 skip connection을 통해 깊은 네트워크 학습 문제를 완화
     - CNN 구조 이해에서 가장 중요한 전환점으로 다룰 것
   - EfficientNet
     - compound scaling으로 depth, width, resolution을 균형 있게 확장
     - 정확도뿐 아니라 효율성 관점까지 포함

5. 평가 지표
   - Accuracy
   - Top-1 accuracy
   - Top-5 accuracy
   - Parameters
   - FLOPs 또는 계산량
   - Inference speed
   - 메모리 사용량

6. 비교 분석
   - 구조 비교: 깊이, filter 구성, skip connection 여부, scaling 방식
   - 성능 비교: 정확도와 계산량의 균형
   - 학습 난이도 비교: 깊어질수록 생기는 gradient 문제와 해결 방식
   - 활용성 비교: 교육용 이해, 실무 적용, 모바일/edge 환경 적합성

7. 본인 의견 및 향후 발전 방향
   - CNN 발전은 단순히 깊게 만드는 방향에서 효율적으로 설계하는 방향으로 이동했다는 관점 제시
   - ResNet은 깊은 CNN 학습의 병목을 해결한 핵심 모델로 평가
   - EfficientNet은 제한된 자원에서 좋은 성능을 내는 현대적 설계 방향을 보여줌
   - Vision Transformer와 CNN의 비교는 결론에서 짧게 언급하되, 본문 중심은 CNN에 둔다.

8. 결론
   - AlexNet에서 EfficientNet까지의 흐름을 통해 CNN의 핵심 원리와 발전 방향 요약
   - 이미지 분류 모델 비교가 CNN 구조 이해에 주는 의미 정리

### 모델별 비교 표 초안

| 모델 | 핵심 아이디어 | 장점 | 단점 | CNN 이해 포인트 |
| --- | --- | --- | --- | --- |
| AlexNet | 깊은 CNN, ReLU, dropout | CNN의 가능성 입증 | 구조가 크고 현대 기준 효율 낮음 | 기본 CNN 구조 |
| VGG | 3x3 convolution 반복 | 단순하고 직관적 | 파라미터 수 많음 | 깊이와 작은 filter |
| ResNet | residual learning, skip connection | 매우 깊은 네트워크 학습 가능 | 구조 이해에 residual 개념 필요 | gradient 문제와 해결 |
| EfficientNet | compound scaling | 정확도와 효율 균형 | 설계 원리가 상대적으로 복잡 | 효율적 CNN 설계 |

### 참고문헌 후보

- Krizhevsky, Sutskever, Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", 2012.
- Simonyan and Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", 2014.
- He et al., "Deep Residual Learning for Image Recognition", 2015.
- Tan and Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", 2019.
- PyTorch 또는 torchvision 공식 모델 문서.
- 수업 교재와 강의자료.

### 작업 순서

1. 완료: 과제 PDF 요구사항을 기준으로 목차를 확정한다.
2. 진행 중: 각 모델 원 논문에서 핵심 아이디어와 구조 그림/표현을 정리한다.
3. 완료: 모델별 장단점을 한 문단씩 작성한다.
4. 완료: 평가 지표를 이미지 분류 관점에서 간단히 설명한다.
5. 완료: 비교 표를 만들고, 본문에서는 표를 해석하는 방식으로 작성한다.
6. 완료: 본인 의견에서는 CNN 발전 방향을 "더 깊게"에서 "더 효율적으로"로 정리한다.
7. 다음 작업: A4 5페이지 이내에 맞춰 문장을 압축한다.

### 산출물

- 1차 본문 초안: `mid_report/draft.md`

### 주의 사항

- 인터넷 설명문 복사보다 원 논문, 교재, 공식 문서를 근거로 정리한다.
- 모델별 세부 수치에 집착하기보다 구조적 차이와 발전 흐름을 중심으로 쓴다.
- Vision Transformer는 비교 대상이 아니므로 결론의 향후 방향에서만 짧게 언급한다.
- 리포트의 목적은 최신 성능 조사보다 CNN 이해와 비교 분석이다.
