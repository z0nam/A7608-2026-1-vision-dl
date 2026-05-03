# 이미지 분류 모델의 발전을 통해 본 CNN 구조와 학습 원리 비교 분석

## 내가 해야 할 일

- [ ] 발제 일정표에서 내 담당 주차, 장, 발표 범위를 최종 확인한다.
- [ ] 이 중간리포트 주제가 수업 발제 내용과 겹치는지 확인한다.
  - 겹친다면 리포트의 비교 모델을 발제 내용과 연결한다.
  - 겹치지 않는다면 리포트는 독립 과제로 유지한다.
- [ ] 과제 안내문 기준으로 제출 조건을 다시 확인한다.
  - A4 5페이지 이내
  - 한글 문서 기본 여백
  - 10pt
  - 줄 간격 160%
  - 참고문헌 포함
  - 인터넷 자료 복사/붙여넣기 금지
- [ ] 원 논문 또는 공식 자료에서 각 모델의 핵심 근거를 확인한다.
  - AlexNet: ImageNet 성능 향상, ReLU, dropout, GPU 학습
  - VGG: 3x3 convolution 반복과 깊은 구조
  - ResNet: residual block, skip connection, 깊은 네트워크 학습 문제
  - EfficientNet: compound scaling, depth/width/resolution 균형
- [ ] 본문에 넣을 표 1개를 최종 확정한다.
  - 현재 초안의 모델 비교 표를 유지할지
  - 정확도/파라미터/FLOPs 비교 표로 바꿀지 결정
- [ ] 그림을 넣을지 결정한다.
  - 넣는다면 CNN 기본 구조 그림 또는 ResNet residual block 그림 1개 정도가 적절하다.
  - 5페이지 제한이 있으므로 그림은 최대 1개만 권장한다.
- [ ] 초안 문장을 한글 리포트 문체로 다듬는다.
  - 영어 용어는 처음 등장할 때만 병기한다.
  - 이후에는 가능한 한 한국어 용어를 사용한다.
- [ ] 본인 의견 부분을 더 개인적인 학습 관점으로 다듬는다.
  - "CNN 이해에 가장 도움이 된 모델은 무엇인가"
  - "정확도와 효율성 중 어떤 관점이 더 중요하다고 보는가"
- [ ] 참고문헌 형식을 통일한다.
  - 논문 4개는 반드시 유지한다.
  - PyTorch 문서는 필요하면 공식 자료로만 남긴다.
- [ ] 최종 제출 전 표절 위험이 있는 문장을 줄인다.
  - 논문 abstract나 인터넷 설명문과 비슷한 표현은 피한다.
  - 모델 설명은 내 말로 구조와 의미를 다시 설명한다.
- [ ] 한글 문서 또는 제출 형식으로 옮긴 뒤 5페이지 이내인지 확인한다.

## 1. 서론

컴퓨터 비전은 영상이나 이미지로부터 의미 있는 정보를 추출하는 인공지능 분야이다. 그중 이미지 분류(Image Classification)는 입력 이미지가 어떤 범주에 속하는지를 예측하는 가장 기본적인 문제이다. 예를 들어 고양이 사진을 고양이로, 자동차 사진을 자동차로 분류하는 작업이 이에 해당한다. 이미지 분류는 객체 검출, 의미 분할, 영상 인식 등 더 복잡한 컴퓨터 비전 문제의 기반이 되기 때문에 CNN을 이해하기 위한 출발점으로 적합하다.

합성곱 신경망(Convolutional Neural Network, CNN)은 이미지의 공간적 구조를 이용해 특징을 학습하는 딥러닝 모델이다. 전통적인 컴퓨터 비전에서는 사람이 직접 edge, corner, texture 같은 특징을 설계해야 했지만, CNN은 convolution layer를 통해 데이터로부터 특징을 자동으로 학습한다. 낮은 층에서는 선이나 색상 변화 같은 단순한 패턴을, 깊은 층에서는 물체의 부분이나 전체 형태처럼 더 추상적인 특징을 학습한다.

본 보고서에서는 이미지 분류를 중심으로 대표적인 CNN 모델인 AlexNet, VGG, ResNet, EfficientNet을 비교한다. 네 모델은 각각 CNN의 가능성, 깊은 구조의 효과, 매우 깊은 네트워크 학습 문제의 해결, 효율적 모델 설계라는 발전 흐름을 보여준다. 이를 통해 CNN의 기본 구조와 학습 원리, 그리고 정확도와 계산 효율 사이의 관계를 이해하는 것을 목표로 한다.

## 2. 이미지 분류 문제와 CNN의 기본 구조

이미지 분류는 입력 이미지 \(x\)가 주어졌을 때 해당 이미지의 class label \(y\)를 예측하는 문제이다. 모델은 각 class에 대한 score 또는 확률을 출력하고, 가장 높은 확률을 가진 class를 최종 예측값으로 선택한다. 학습 과정에서는 정답 label과 예측 확률 사이의 차이를 줄이도록 모델의 파라미터를 조정한다. 일반적으로 softmax 함수와 cross entropy loss가 사용된다.

CNN은 이미지 분류 문제에 적합한 구조를 가진다. 일반적인 CNN은 convolution layer, activation function, pooling layer, classifier로 구성된다. Convolution layer는 작은 filter를 이미지 위에서 이동시키며 지역적인 특징을 추출한다. 이 구조는 이미지 전체를 한 번에 연결하는 fully connected layer보다 파라미터 수가 적고, 이미지의 공간적 인접성을 잘 활용할 수 있다.

Activation function은 모델에 비선형성을 부여한다. ReLU(Rectified Linear Unit)는 입력이 0보다 작으면 0, 크면 그대로 출력하는 단순한 함수이지만, 깊은 신경망의 학습을 빠르게 만드는 데 중요한 역할을 했다. Pooling layer는 feature map의 크기를 줄여 계산량을 낮추고, 작은 위치 변화에 덜 민감한 표현을 만들도록 돕는다. 마지막 classifier는 CNN이 추출한 특징을 바탕으로 class score를 계산한다.

CNN의 핵심은 특징의 계층적 학습이다. 초기 layer는 edge나 색상 대비 같은 저수준 특징을 학습하고, 중간 layer는 texture나 object part를, 마지막 layer는 class 구분에 필요한 고수준 특징을 학습한다. 따라서 CNN 모델의 발전은 단순히 layer 수가 늘어난 역사라기보다, 더 깊고 안정적이며 효율적인 특징 학습 구조를 찾는 과정으로 볼 수 있다.

## 3. 대표 CNN 모델

### 3.1 AlexNet

AlexNet은 2012년 ImageNet Large Scale Visual Recognition Challenge에서 큰 성능 향상을 보이며 CNN의 가능성을 널리 알린 모델이다. AlexNet 이전에도 CNN은 존재했지만, 대규모 이미지 데이터셋과 GPU 학습을 활용해 깊은 CNN이 실제 이미지 분류 문제에서 강력하다는 점을 보여주었다는 데 의미가 있다.

AlexNet은 여러 개의 convolution layer와 fully connected layer로 구성된다. 구조적으로는 큰 convolution filter를 사용해 이미지의 지역 특징을 추출하고, 이후 layer를 거치며 점점 추상적인 특징을 학습한다. AlexNet의 중요한 특징은 ReLU activation, dropout, data augmentation, GPU 기반 학습이다. ReLU는 기존 sigmoid나 tanh보다 gradient가 잘 전달되어 학습 속도를 높였다. Dropout은 일부 뉴런을 무작위로 비활성화하여 과적합을 줄이는 데 사용되었다. Data augmentation은 이미지를 이동하거나 반전시키는 방식으로 학습 데이터를 다양화하여 일반화 성능을 높였다.

AlexNet의 장점은 CNN이 대규모 이미지 분류에서 효과적이라는 것을 명확하게 보여주었다는 점이다. 그러나 현대 기준으로 보면 구조가 단순하지 않고, fully connected layer의 파라미터 비중이 크며, 계산 효율도 높다고 보기는 어렵다. 그럼에도 AlexNet은 CNN 기반 이미지 분류 연구의 출발점으로 중요하다.

### 3.2 VGG

VGG는 네트워크 구조를 단순화하면서 깊이를 늘린 모델이다. VGG의 핵심 아이디어는 큰 convolution filter 대신 작은 3x3 convolution filter를 반복적으로 사용하는 것이다. 여러 개의 3x3 convolution을 쌓으면 더 큰 receptive field를 갖는 효과를 만들 수 있고, 동시에 비선형 activation을 여러 번 적용할 수 있어 표현력이 증가한다.

VGG의 장점은 구조가 매우 직관적이라는 점이다. 같은 크기의 작은 filter를 반복하고, pooling을 통해 feature map 크기를 줄이며, 깊이가 증가할수록 channel 수를 늘리는 방식은 이후 CNN 구조를 이해하는 기본 틀이 되었다. VGG는 CNN에서 깊이가 성능 향상에 중요한 요소임을 보여주었다.

반면 VGG는 파라미터 수와 계산량이 많다는 단점이 있다. 특히 fully connected layer 부분의 파라미터가 크고, 깊은 convolution stack도 계산 비용을 증가시킨다. 따라서 VGG는 이해와 전이학습에는 유용하지만, 제한된 자원에서 효율적으로 사용하기에는 부담이 있다. VGG는 "작은 filter를 깊게 쌓는 CNN"이라는 설계 원리를 이해하는 데 좋은 모델이다.

### 3.3 ResNet

ResNet은 매우 깊은 네트워크를 안정적으로 학습하기 위해 residual learning과 skip connection을 도입한 모델이다. CNN이 깊어질수록 표현력은 증가할 수 있지만, 실제 학습에서는 gradient vanishing이나 optimization 문제가 발생한다. 단순히 layer를 더 많이 쌓는다고 항상 성능이 좋아지는 것은 아니다.

ResNet의 핵심은 입력 \(x\)를 몇 개의 layer를 거친 출력 \(F(x)\)에 더해 \(F(x) + x\) 형태로 전달하는 residual block이다. 이 구조에서 네트워크는 전체 mapping을 직접 학습하기보다 입력과 출력 사이의 차이, 즉 residual을 학습한다. Skip connection은 gradient가 뒤쪽 layer에서 앞쪽 layer로 더 잘 전달되도록 도와 매우 깊은 네트워크의 학습을 가능하게 한다.

ResNet의 장점은 깊은 CNN 학습의 병목을 해결했다는 점이다. ResNet 이후로 수십, 수백 개 layer를 가진 네트워크가 실용적으로 사용될 수 있게 되었고, residual connection은 컴퓨터 비전뿐 아니라 다양한 딥러닝 구조에서도 기본 아이디어로 활용되었다. 단점은 AlexNet이나 VGG에 비해 구조적 개념이 조금 더 복잡하다는 점이다. 그러나 CNN을 깊이 있게 이해하려면 ResNet은 반드시 다루어야 하는 모델이다.

### 3.4 EfficientNet

EfficientNet은 CNN 모델을 효율적으로 확장하는 방법을 제안한 모델이다. 기존 연구에서는 모델의 depth를 늘리거나, channel width를 키우거나, 입력 image resolution을 높이는 방식으로 성능을 개선했다. 그러나 이 세 요소를 불균형하게 키우면 계산량은 크게 증가하지만 성능 향상은 제한적일 수 있다.

EfficientNet의 핵심 아이디어는 compound scaling이다. 이는 depth, width, resolution을 일정한 비율로 함께 확장하는 방식이다. 즉, 네트워크를 깊게만 만들거나 넓게만 만드는 것이 아니라, 세 가지 차원을 균형 있게 키워 정확도와 효율성을 함께 고려한다. EfficientNet은 기본 모델을 먼저 설계한 뒤, scaling coefficient를 이용해 여러 크기의 모델로 확장한다.

EfficientNet의 장점은 높은 정확도를 비교적 적은 파라미터와 계산량으로 달성할 수 있다는 점이다. 이는 모바일, edge device, 실시간 서비스처럼 자원이 제한된 환경에서 특히 중요하다. 단점은 AlexNet이나 VGG처럼 구조가 단순히 직관적이지는 않고, compound scaling이라는 설계 원리를 따로 이해해야 한다는 점이다. EfficientNet은 CNN 발전이 "더 깊게"에서 "더 효율적으로" 이동했음을 보여주는 대표적인 모델이다.

## 4. 평가 지표

이미지 분류 모델을 비교할 때 가장 기본적인 지표는 accuracy이다. Accuracy는 전체 이미지 중 모델이 올바르게 분류한 비율이다. 단일 정답을 예측하는 문제에서는 직관적이고 이해하기 쉬운 지표이다.

ImageNet과 같은 대규모 분류 문제에서는 Top-1 accuracy와 Top-5 accuracy가 자주 사용된다. Top-1 accuracy는 모델이 가장 높은 확률로 예측한 class가 정답인 비율이다. Top-5 accuracy는 모델이 예측한 상위 5개 class 중 정답이 포함된 비율이다. class 수가 많은 데이터셋에서는 Top-5 accuracy가 모델의 후보 예측 능력을 함께 보여준다.

정확도만으로 모델을 평가하는 것은 충분하지 않다. 실제 사용 환경에서는 파라미터 수, FLOPs, inference speed, 메모리 사용량도 중요하다. 파라미터 수가 많으면 모델 저장 공간과 학습 비용이 증가하고, FLOPs가 많으면 추론 속도가 느려질 수 있다. 따라서 CNN 모델 비교에서는 정확도와 계산 효율을 함께 고려해야 한다.

## 5. 모델 비교 분석

| 모델 | 핵심 아이디어 | 장점 | 단점 | CNN 이해 포인트 |
| --- | --- | --- | --- | --- |
| AlexNet | 깊은 CNN, ReLU, dropout | CNN의 가능성 입증 | 현대 기준 효율 낮음 | 기본 CNN 구조와 대규모 학습 |
| VGG | 3x3 convolution 반복 | 단순하고 직관적 | 파라미터 수와 계산량이 큼 | 깊이와 작은 filter의 효과 |
| ResNet | residual learning, skip connection | 매우 깊은 네트워크 학습 가능 | residual 개념 이해 필요 | gradient 문제와 skip connection |
| EfficientNet | compound scaling | 정확도와 효율 균형 | 설계 원리가 상대적으로 복잡 | 효율적 CNN 설계 |

네 모델은 CNN 발전의 서로 다른 단계를 보여준다. AlexNet은 CNN이 대규모 이미지 분류에서 강력하다는 사실을 증명했다. VGG는 작은 convolution filter를 반복적으로 사용해 깊이를 늘리는 단순한 원리를 제시했다. ResNet은 깊이가 증가할 때 발생하는 학습 문제를 skip connection으로 해결했다. EfficientNet은 정확도뿐 아니라 계산량과 모델 크기를 함께 고려하는 방향을 제시했다.

구조적 관점에서 보면 AlexNet과 VGG는 순차적으로 layer를 쌓는 방식에 가깝다. 특히 VGG는 동일한 convolution block을 반복하기 때문에 CNN 구조를 이해하기 쉽다. 반면 ResNet은 layer 사이에 우회 연결을 추가하여 정보와 gradient가 더 직접적으로 흐르도록 한다. EfficientNet은 단일 구조의 깊이만 늘리는 것이 아니라 depth, width, resolution을 균형 있게 확장한다.

학습 관점에서 보면 AlexNet은 ReLU와 dropout을 통해 깊은 CNN 학습의 가능성을 보였고, VGG는 더 깊은 구조가 좋은 표현을 만들 수 있음을 보여주었다. 그러나 깊이가 계속 증가하면 단순한 sequential 구조만으로는 학습이 어려워진다. ResNet은 이 문제를 residual block으로 해결했다. EfficientNet은 깊은 모델을 만드는 것 자체보다 제한된 계산 자원 안에서 좋은 성능을 내는 설계가 중요하다는 점을 강조한다.

활용성 관점에서는 VGG가 구조가 단순하여 교육용 설명과 feature extractor로 이해하기 쉽다. ResNet은 현재도 많은 비전 모델의 backbone으로 사용될 만큼 범용성이 높다. EfficientNet은 자원 효율이 중요한 환경에서 장점이 있다. 따라서 어떤 모델이 가장 좋다고 단정하기보다, 목적과 환경에 따라 적절한 모델을 선택해야 한다.

## 6. 본인 의견 및 향후 발전 방향

이번 비교를 통해 CNN의 발전은 단순히 layer를 많이 쌓는 방향으로만 진행된 것이 아니라는 점을 알 수 있다. AlexNet은 CNN의 가능성을 보여주었고, VGG는 깊이의 중요성을 보여주었다. 그러나 깊이가 증가하면서 학습이 어려워지는 문제가 나타났고, ResNet은 skip connection을 통해 이를 해결했다. 이후 EfficientNet은 정확도뿐 아니라 계산 효율까지 함께 고려하는 방향을 제시했다.

개인적으로 CNN을 이해하는 데 가장 중요한 모델은 ResNet이라고 생각한다. AlexNet과 VGG를 통해 CNN의 기본 구조와 깊이의 효과를 이해할 수 있지만, ResNet은 "깊은 네트워크가 왜 항상 잘 학습되지 않는가"라는 질문을 다루기 때문이다. 이는 단순한 구조 암기를 넘어 CNN 학습 원리를 이해하는 데 도움이 된다.

향후 컴퓨터 비전 모델은 정확도뿐 아니라 효율성, 해석 가능성, 다양한 환경에서의 적용 가능성을 함께 고려하는 방향으로 발전할 것이다. 최근에는 Vision Transformer 같은 구조가 널리 연구되고 있지만, CNN은 여전히 이미지의 지역적 구조를 효율적으로 활용한다는 강점을 가진다. 따라서 CNN은 앞으로도 단독 모델 또는 다른 구조와 결합된 형태로 계속 활용될 가능성이 높다.

## 7. 결론

본 보고서에서는 이미지 분류를 중심으로 AlexNet, VGG, ResNet, EfficientNet을 비교하였다. AlexNet은 CNN 기반 이미지 분류의 가능성을 보여주었고, VGG는 작은 convolution filter를 깊게 쌓는 설계의 효과를 제시했다. ResNet은 residual learning과 skip connection을 통해 매우 깊은 네트워크의 학습 문제를 해결했으며, EfficientNet은 depth, width, resolution을 균형 있게 확장하여 정확도와 효율을 함께 고려했다.

이 네 모델의 흐름을 통해 CNN의 핵심은 이미지의 지역적 특징을 계층적으로 학습하는 데 있으며, 모델 발전은 표현력, 학습 안정성, 계산 효율 사이의 균형을 찾아가는 과정임을 확인할 수 있다. 따라서 이미지 분류 모델의 비교는 CNN 구조와 학습 원리를 이해하는 데 효과적인 방법이다.

## 참고문헌

- Krizhevsky, A., Sutskever, I., and Hinton, G. E. "ImageNet Classification with Deep Convolutional Neural Networks." Advances in Neural Information Processing Systems, 2012.
- Simonyan, K., and Zisserman, A. "Very Deep Convolutional Networks for Large-Scale Image Recognition." arXiv:1409.1556, 2014.
- He, K., Zhang, X., Ren, S., and Sun, J. "Deep Residual Learning for Image Recognition." IEEE Conference on Computer Vision and Pattern Recognition, 2016.
- Tan, M., and Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." International Conference on Machine Learning, 2019.
- PyTorch Contributors. "Torchvision Models." PyTorch Documentation.
