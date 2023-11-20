---
layout: single
title: "EfficientNet, EfficientDet 정리"
toc: true
toc_sticky: true
category: Detection
---

Computer vision 분야의 발달로 네트워크의 층을 더해 정확도를 높이면서 연산량을 줄여 속도는 높이는 방법들이 고안되었다. 이 과정속에서 네트워크의 깊이와 높이, 이미지의 크기, 필터의 채널 수 등 다양한 파라미터들이
실험적으로 사용되었는데 EfficientNet과 EfficientDet에서는 compoung scaling을 사용해 해당 네트워크에 최적의 파라미터들을 찾아 성능을 개선시켰다. 이번 포스팅에서는 EfficientNet과 EfficientDet에 대해 간단히
정리해보자.

# EfficientNet
EfficientNet에서는 기본 backbone구조에서 네트워크의 넓이(필터 수), 깊이, 이미지 resolution을 독립적으로 scaling하거나 compound scaling을 수행하여 여러 네트워크에 대한 성능을 비교했다. 여기서 compound scaling이란,
각각의 파라미터를 scaling하고 이를 조합하여 최적의 파라미터 조합을 찾는 것으로 네트워크의 깊이, 필터 수, 이미지 resolution 크기가 파라미터로 사용된다.

<br> 
<div align="center">
  <p>
  <img width="600" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/6d9564f5-2c1d-4985-812d-3a2cc7fbf109">
  </p>
</div>

<br>

Figure 3.을 참고하면 각각의 파라미터들을 독립적으로 scaling을 수행할 때 resolution을 제외하고는 연산량이 증가해도 결국 정확도의 증가폭이 작아지는 구간이 생긴다.

<br> 
<div align="center">
  <p>
  <img width="600" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/fbc8b1b5-55a1-450e-97dc-75fdcbb287d0">
  </p>
</div>

<br>

이를 해결하기 위해 EfficientNet에서는 모든 파라미터를 scaling하는 compound scaling 기법을 사용했다. Depth, width, resolution 이 3가지 factor를 동시에 고려하는데 아래의 식처럼 최초에는
$\sigma$를 1로 고정하고 grid search를 기반으로 최적의 $\alpha, \beta, \gamma$ 값을 찾는다(EfficientNetB0는 1.2, 1.1, 1.15).

다음으로 $\alpha, \beta, \gamma$를 고정하고 $\sigma$ 값을 증가시켜가면서 EfficientB1 ~ B7까지 scale 조합을 구성한다. 특히, width와 resolution을 경우 factor가 2배가 되면 FLOPS가 4배가 되기대문에
너무 커지지않도록 제곱을 통해 scale을 조정했다. 이러한 방법으로 depth, width, resolution에 따른 FLOPS 변화를 기반으로 최적의 식을 도출한다.

<br> 
<div align="center">
  <p>
  <img width="400" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/94200458-8a65-469a-9354-1d17a4d139e4">
  </p>
</div>

<br>

이런 다양한 실험을 기반으로 EfficientNet은 ImageNet Classification에서 기존 SOTA 모델들에 비해 정확도와 속도 모두 향상된 performance를 기록했다.

<br> 
<div align="center">
  <p>
  <img width="600" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/8008b4da-84be-412c-8baf-4cfea84fa324">
  </p>
</div>

<br>

# EfficientDet
## Bi-FPN
RetinaNet이 FPN을 성공적으로 사용함으로써 FPN에 대한 연구가 매우 활발해졌다. EfficientDet은 EfficientNet에 Bi-FPN을 접목시켜 Detection task를 수행하는 모델이다.

FPN이 top-down 과정에서 bottom-up의 피처맵을 결합하여 예측했다면, Bi-FPN은 top-down뿐만 아니라 다시 bottom-up을 수행해 이 과정에서 이전 bottom-up의 피처맵과 top-down에서 결합된 피처맵을 모두 사용한다.
FPN의 아이디어는 높은 추상화 정보를 가지고 있는 피처맵에 낮은 추상화 정보를 가지고 있는 피처맵을 결합하여 공간 정보를 유지시키는 것이었다. Bi-FPN은 여기서 한발 더 나아가 낮은 추상화 정보의 피처맵도 높은 추상화 정보의 피처맵이
있다면 예측을 더 잘 수행할 것이라는 아이디어다. 즉, FPN보다 공간 정보를 훨씬 더 많이 사용하여 예측을 수행한다는 것이다.

<br> 
<div align="center">
  <p>
  <img width="600" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/104659c0-5305-47a2-9a34-ea843a1914d2">
  </p>
</div>

<br>

이 Bi-FPN block이 하나만 있는 것이 아니라 여러개가 존재하고, 각각의 피처맵들을 더할 때 가중치가 너무 커지지 않도록 Weighted Feature Fusion을 사용한다. Weighted Feature Fusion의 기법으로는
softmax, unbounded fusion을 사용해봤지만 Fast normalized fusion을 사용했을 때 연산량의 측면에서 가장 효율적이었다.

<br> 
<div align="center">
  <p>
  <img width="200" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/a8b35e94-1bda-407b-ada0-2e30cd784f02">
    <br>
  <img width="400" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/ebdfcd75-de5b-48e7-a46e-2e614b91ba72">
  </p>
</div>
<br>

## Compound Scaling
EfficientDet은 Backbone, Neck, Head에 모두 compound scaling을 적용했는데 backbone은 EfficientNet scaling을 그대로 적용했고, Bi-FPn network에는 $D_{bifpn} = 3 + \sigma$로
기본 반복 block을 3개로 적용하여 scaling했다.

Width(채널 수)는 $W_{bifpn} = 64 * (1.35 \sigma)$, Prediction network의 depth는 $D_{box} = D_{class} = 3 + [\sigma / 3]$, 이미지 resolution은 $R_{input} = 512 + \sigma * 128$
로 scaling했다.

여러 실험을 통해 최종적으로는 네트워크의 아키텍처에 따라 아래와 같이 compound scaling 되었다.

<br> 
<div align="center">
  <p>
  <img width="400" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/7e05ed67-7e79-44c3-9a90-f63b33abd945">
  </p>
</div>

<br>

EfficientDet은 EfficientNet을 기반으로 compound scaling을 사용했고, FPN을 개선한 Bi-FPN, activation으로 SiLU, loss로 Focal Loss, NMS를 조금더 보완한 Soft-NMS(기존 NMS가 근처의 detection된 박스까지 제거하는 것을
보완하여 많이 겹치는 박스에 가중치를 크게 부여해 confidence score를 낮춰서 겹치는 다른 박스도 어느 정도 유지시킬 수 있도록 하는 기법)기법을 사용하여 성능을 향상시켰다.

<br> 
<div align="center">
  <p>
  <img width="600" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/8674f097-ef44-4fb0-b53b-7cea7cbad70a">
  </p>
</div>

<br>

하지만 여전히, small object 탐지는 어렵다는 한계가 존재하긴한다.

# Reference
- [EfficientNet Paper](https://arxiv.org/pdf/1905.11946.pdf)
- [EfficientDet Paper](https://arxiv.org/pdf/1911.09070.pdf)
