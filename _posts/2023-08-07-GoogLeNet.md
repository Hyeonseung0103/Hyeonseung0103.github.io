---
layout: single
title: "GoogLeNet(Going deeper with convolutions)논문 요약"
toc: true
toc_sticky: true
category: CNN
---

# Abstract
본 논문은 Deep Convolutional Neural Networks의 아키텍처를 인셉션(Inception)이라는 이름으로 제안한다. ILSVRC14에서 분류와 탐지 부문 최고 수준을 달성했다. GoogLeNet이라 불리는 해당 아키텍처는 22개의 deep network로,
네트워크의 깊이와 너비는 키우면서 연산량을 일정하게 유지시켰다는 점에서 굉장히 의미 있는 연구였다. 아키텍처 품질을 최적화 시키기위해서는 Hebbian 원칙과 직관이 사용되었다.

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

그렇지만, AlexNet에서 병렬 GPU로 Full connection layer에 대한 학습이 가능하다는 것을 발견했고 많은 필터와 큰 배치 사이즈를 사용했을 때 더 좋은 성능을 낸다는 트렌드가 존재했다. 또한, 매층마다 다른 크기의 sparse network를
처리할 때 요구되는 리소스가 컸기때문에 오히려 dens한 network를 사용하는 것이 더 효율적이라는 의견이 많았다. 

따라서, 본 논문에서는 sparse와 dense network의 장점을 모두 사용하기 위해 sparse metrix를 클러스터링 하여 여러 개의 dense metrix로 만드는 방법을 활용했다. 여러 논문에 근거하여 network에 sparsity를 적용하면서도 
연산은 dense metrics를 활용하는 방법이다.  **Abstract**에 소개된 Inception 아키텍처가 이 방법을 실험하기 위해 구현되었고 learning rate와 hyper parameters를 좀 더 조정했을 때 localization과 
object detection 부분에서 좋은 성능을 냈다.



<br><br>

## 모델 구조
### 1. Architecture


### 2. GoogLeNet


### 3. 훈련 방법

<br><br>

## 분류 및 결과


<br><br>


## 결론


<br><br>

# 개인적인 생각

<br><br>

# 구현

# 이미지 출처
- [Dense and Sparse Network](https://www.baeldung.com/cs/neural-networks-dense-sparse)
