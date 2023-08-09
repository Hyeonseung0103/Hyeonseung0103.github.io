---
layout: single
title: "GoogLeNet(Going deeper with convolutions)논문 요약"
toc: true
toc_sticky: true
category: CNN
---

# Abstract
본 논문은 Deep Convolutional Neural Networks의 아키텍처를 인셉션(Inception)이라는 이름으로 제안한다. ILSVRC14에서 분류와 탐지 부문 최고 수준을 달성했다. GoogLeNet이라 불리는 해당 아키텍처는 22개의 deep network로, 네트워크의 깊이와 너비는 키우면서 연산량을 일정하게 유지시켰다는 점에서 굉장히 의미 있는 연구였다. 아키텍처 품질을 최적화 시키기위해서는 Hebbian 원칙과 직관이 사용되었다.

<br><br>

# Details

## 도입부 & 관련 논문
- 최근 3년간 새로운 아이디어, 알고리즘, 네트워크 아키텍처 등의 발전으로 CNN 기반 딥러닝 모델이 크게 발전했음
- 2012년 개발된 AlexNet보다 12배나 적은 파라미터를 사용했음에도 좋은 정확도를 보임
- 본 논문에서는 정확도라는 숫자 뿐만 아니라 메모리 사용의 효율성도 고려
- 인셉션이라는 이름으로 DNN 아키텍처에 집중하여 새로운 형태의 구조를 소개하고, 네트워크의 깊이를 더 깊이함
- LeNet-5 부터 Convolutional layer와 Fully-conneted layer의 구조를 이루는 표준적인 CNN 아키텍처 활용
- Network in Network을 기반으로 1x1 필터와 ReLU를 사용해서 성능에 큰 영향을 주지 않으면서도 차원을 축소하여 네트워크의 크기를 줄이고, 비선형성을 추가하여 학습이 더 잘 이루어지게하는 방법론을 활용

<br><br>

## Motivation and High Level Considerations
DNN에서는 네트워크를 깊고 넓게 만들고 많은 양의 훈련 데이터를 사용한다면 성능이 좋을 것이다. 하지만, 좋은 데이터셋을 구축하기위해 발생하는 시간과 비용이 크고, 네트워크의 깊이와 넓이에 따라 많은 학습 시간이 소요된다.

이러한 문제를 해결하기위해 fully connected(dense) 된 convolution 계층을 sparsely connected로 변환하는 방법을 사용할 수 있다. 이 방법은 "잘 맞는 뉴런은 서로 연결 되어있다." 라는 Hebbian 원칙에 따라 
입력과 출력의 상관관계가 큰 뉴런을 통계적으로 분석(데이터셋의 확률 분포)하여 활성화한다. 모든 뉴런들이 서로 계산되는 것이 아니라 서로 잘 맞는 뉴런만을 활성화시켜 계산하는 방법이기때문에 연산량의 측면에서 훨씬 효율적이다.
아래 이미지는 dense와 sparse한 네트워크의 예시이다.

<br><br>

<div align="center">
  <p>
  <img width="560" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/d875d6eb-f78c-4d84-b835-9eca489b4ab6">
  </p>
</div>

<br><br>

그렇지만, AlexNet에서 병렬 GPU로 Full connection layer에 대한 학습이 가능하다는 것을 발견했고 많은 필터와 큰 배치 사이즈를 사용했을 때 더 좋은 성능을 낸다는 트렌드가 존재했다. 또한, 매층마다 다른 크기의 sparse network를 처리할 때 요구되는 리소스가 컸기때문에 오히려 dens한 network를 사용하는 것이 더 효율적이라는 의견이 많았다. 

따라서, 본 논문에서는 sparse와 dense network의 장점을 모두 사용하기 위해 sparse metrix를 클러스터링 하여 여러 개의 dense metrix로 만드는 방법을 활용했다. 여러 논문에 근거하여 network에 sparsity를 적용하면서도 
연산은 dense metrics를 활용하는 방법이다.  **Abstract**에 소개된 Inception 아키텍처가 이 방법을 실험하기 위해 구현되었고 learning rate와 hyper parameters를 좀 더 조정했을 때 localization과 
object detection 부분에서 좋은 성능을 냈다.

<br><br>

## 모델 구조
### 1. Architecture
GoogLeNet의 핵심 모듈인 인셉션 구조에서는 어떻게 최적의 local sparse structure를 찾고 이를 dense components에 가깝게 만들것이냐가 중요하다. 쉽게 말하자면, sparse matrix를 클러스터링 해서 dense matrix에 가까운
형태로 만드는 것이다. 이때 클러스터링 된 유닛은 다음 layer의 관련 유닛과 연결되어 활성화되는데 입력 이미지와 가까운 lower layer에서는 이미지의 낮은 수준의 특성을 학습하며 관련된 유닛이 특정 지역에만 집중되는 현상이 발생할 수 있다. 예를 들어, 얼굴을 인식하는 경우 눈, 코, 입과 같은 특징들이 연관되어 특정 지역에만 클러스터가 집중되는 것이다.

본 논문에서는 이를 방지하기 위해 1x1 필터를 사용했는데 1x1 필터는 각 픽셀에 대해 선형 조합을 수행하여 관련 유닛들이 비슷한 지역에만 집중되는 것을 해소하고, 넓은 영역에서 특징을 인식하도록 한다. 또한, 이미지의 정보를 더 많이 반영하기 위해서는 넓은 크기의 필터도 필요하기 때문에 3x3, 5x5 필터 또한 함께 사용했다.

하지만, 인셉션 모듈의 깊이가 깊어질수록 유닛이 특정 공간에만 몰려있는 공간 집중도가 해소됨으로써 5x5 크기의 필터가 더 많이 필요하게 될텐데 5x5 크기의 필터를 여러번 사용한다는 것은 연산량 측면에서 매우 비효율적인 방법이다. 따라서, 아래의 오른쪽 이미지(왼쪽은 처음에 고안된 기존 인셉션 모듈 이미지)처럼 3x3 혹은 5x5 필터 앞에 1x1 필터를 두어 출력값의 크기는 같으면서도 연산량을 크게 줄였다. 

<br><br>

<div align="center">
  <p>
  <img width="1000" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/2edf5f2e-3595-4c74-991e-573728f91747">
  </p>
</div>

<br><br>

1x1 필터는 중간에 병목 현상을 연상케 하기 때문에 병목층(bottleneck layer)라고도 불리며 1x1 필터를 추가한다는 것은 ReLU를 사용해 비선형성을 추가한다는 장점도 된다.

이 방법은 채널수를 크게 줄여서 성능이 떨어지지 않을까라는 걱정에 비해 성능에 큰 영향을 끼치지 않았고, 여러가지 크기의 필터를 통해 이미지의 다양한 특징을 추출할 수 있다는 좋은 방법이 되었다. 추가적으로, 저자는 메모리를 효율적으로 사용하기위해 lower layer 에는 기존의 CNN 기법을, higher layer 에는 인셉션 모듈을 적용했다고 언급한다.

<br><br>


### 2. GoogLeNet
아래의 표는 GoogLeNet의 구조와 파라미터 등을 간단하게 표현했다. 

<br><br>

<div align="center">
  <p>
  <img width="851" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/f59b058a-0cfb-4a84-8b0d-ff151f70f016">
  </p>
</div>

<br><br>

- 각 픽셀값에서 평균을 뺀 정규화 외에 별다른 전처리가 수행되지 않은 224x224 크기의 RGB 이미지를 입력으로 받음
- #3x3 reduce, #5x5 reduce는 각각 3x3, 5x5 필터 후 사용된 1x1 필터의 채널 수
- pool proj는 max pooling 뒤 사용된 1x1 필터의 채널 수
- 인셉션 모듈을 포함한 모든 합성곱층과 reduction, pool projection 층에서 ReLU 사용

<br><br>

위의 표를 그림으로 나타내면 다음과 같다.

<br><br>

<div align="center">
  <p>
  <img width="1000" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/0ff34eb7-f2c5-4cd2-800d-fbb77f56a137">
  </p>
</div>

<br><br>

그림을 보면 표에는 나타나지 않은 부분이 있는데 해당 부분은 빨간색 박스로 표시되어있는 인셉션 모듈 4(a)와 4(d)이다. GoogLeNet은 네트워크가 깊기때문에 역전파 과정에서 기울기 정보가 앞단까지 제대로 전달되지 않는 기울기 소실 문제가 발생할 수 있다. 따라서, 중간 network인 인셉션 모듈 4(a)와 4(d)에 분류기를 추가하여 기울기 소실 문제를 해결하고, 가중치가 추가적으로 정규화될 것을 기대했다. 해당 분류기에서 발생하는 손실 또한 훈련 손실에 더하여 사용했고(보조 분류기의 성능이 너무 많이 반영되지 않게 0.3의 비율만 적용), Inference 과정에서는 이 분류기를 제거했다.

또한, 최종 Classifier에서는 기존의 완전 결합층 대신 Global Average Pooling을 사용하여 가중치의 갯수를 크게 줄였다. 예를 들어, 이중분류에서 10x10x256의 합성곱 결과를 FC층에 통과시켜 예측을 수행한다고 하면 10x10x256의 입력을 1차원으로 Flatten 시킨 후 이것을 각 클래스에 해당하는 수만큼 다시 변환을 시켜야 하기때문에 총 10x10x256(뉴런 수)x2(클래스 수)의 가중치를 가진다. 

만약, GAP를 사용하면 10x10x256의 결과를 10x10 크기의 채널별로 평균을 내어 입력과 같은 256의 크기를 갖는 1차원 벡터를 만든 뒤 이것을 각 클래스의 속할 확률로 바로 분류하기 때문에 256 x 2의 가중치를 가진다.
GAP는 가중치의 갯수를 줄이는 것 뿐만 아니라 이미지의 공간정보를 유지하면서 과대적합을 해소할 수 있기때문에 유용하게 사용된다.

<br><br>

### 3. 훈련 및 성능
**훈련 방법**

- optimizer는 SGD, momentum은 0.9, learning rate는 8 epochs마다 4%씩 감소시키는 하이퍼파라미터를 사용
- 기존 이미지의 8~100% 사이의 크기와 가로 세로 비율을 3:4 혹은 4:3으로 랜덤하게 조정한 패치를 사용했을 때 좋은 성능을 보임
- 이미지의 명암, 밝기, 채도, 노이즈 등을 조절하는 photometric distortions를 사용하여 데이터 증강을 했을 때 과대적합이 어느 정도 해소
- 이후 시도한 random interpolation methods에 대해서는 효과를 알기 어려움

**성능**

- ILSVRC 2014 대회 분류 부문에서 7개의 모델을 앙상블 했을 때 top-5 에러율 6.67%로 1위를 차지했다.
    - 256, 288, 320, 252의 크기로 crop을 진행했다.
    - scale 마다 3장의 정사각형 이미지를 선택(왼쪽, 중앙, 오른쪽)했고 각 정사각형 이미지마다 4개의 모서리와 중앙 2장의 이미지를 crop 했다.
    - 추가로, 좌우반전에 똑같은 방법을 적용하면 하나의 이미지를 4x3x6x2 = 144개의 이미지로 crop 한다.
      
- ILSVRC 2014 대회 탐지 부문에서 6개의 모델을 앙상블 했을 때 mAP 43.9%를 달성하며 1위를 차지했다. Single model을 사용했을 때는 38.02%의 mAP를 기록했다.
    - 시간이 부족하여 Bounding box regression을 사용하지 않았는데도 높은 성능을 기록했다.
    - R-CNN과 유사한 방식을 사용했고, Region Proposal을 생성할 때 Inception 모듈을 분류기로 사용했다는 점에서 차이가 존재한다.
    - Selective search와 Multi-box predictions을 결합하여 성능을 높였다.
    - 앙상블 모델과 Single model의 성능 차이가 다른 모델들에 비해 크다.

<br><br>


## 결론
최적의 sparse structure를 dense structure와 비슷하게 변환시켜 연산을 수행하는 방법이 컴퓨터 비전에서 가능했고, 연산량이 조금 증가한 것에 비해 성능은 크게 향상되었다. 또한, 탐지 부문에서 context와 bounding box regression을 사용하지 않았는데도 성능이 좋았다는 점에서 의미있는 연구였다.

<br><br>

# 개인적인 생각
- GoogLeNet은 정확도 뿐만 아니라 연산량의 관점에서도 기존보다 훨씬 개선된 모델이라는 점에서 의미있는 연구였다. 합성곱층 중간중간에 1x1 필터를 넣어 연산량을 크게 줄임과 동시에 비선형성을 추가했고, AlexNet이나 VGG와는 달리 분류기에서 FC층이 아닌 GAP를 사용하여 연산량을 대폭 줄였다.
- VGG와 마찬가지로 1x1 필터가 핵심적인 역할로 사용이 되었는데 향후 논문에서는 어떤 방법을 통해 메모리를 효율적으로 사용하면서 성능을 높일 것인지 기대된다.
- 기울기 소실 문제를 해결하는 방법으로 인셉션 중간에 분류기를 배치해서 손실 정보가 앞단까지 잘 전달되게 한 점이 인상 깊었다. 향후 논문에서는 이러한 문제를 어떻게 해결할지 궁금증이 생겼다.
- GoogLeNet(22층)은 VGG보다 layer의 수(최대 19층)가 더 깊고 인셉션의 개념이 있어 복잡하다고 느껴지는데 이후 개발된 아키텍처에서는 VGG와 GoogLeNet 중 어떤 방법을 더 많이 사용할 것인지 궁금하다.

<br><br>

# 구현

# 이미지 출처
- [Dense and Sparse Network](https://www.baeldung.com/cs/neural-networks-dense-sparse)
- [Going deeper with convolutions Paper](https://arxiv.org/pdf/1409.4842v1.pdf)
