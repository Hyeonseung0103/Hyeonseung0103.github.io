---
layout: single
title: "RetinaNet 정리"
toc: true
toc_sticky: true
category: Detection
---

RetinaNet은 one stage detector의 대표주자인 YOLO, SSD보다 높은 성능을 기록하면서 Faster R-CNN보다 빠른 수행시간을 기록한 모델이다. 특히, 작은 object에 대한 detection 능력도 뛰어난데 이번 포스팅에서는
이 RetinaNet에 대해 간단히 정리해보자.

# Focal Loss의 필요성
Classification에서는 cross entropy를 손실함수로 많이 사용한다. 하지만, class가 imbalance할 때는 cross entropy가 상대적으로 잘 맞추고있는 class임에도 단순히 데이터 수가 많아서 loss의 많은 부분을
차지할 수 있다는 문제가 발생한다. Two stage detector에서는 RPN을 통해 객체가 있을만한 높은 확률 순으로 필터링을 수행한 후 탐지를 할 수 있지만 one stage detector에서는 모든 region(e.g. anchor box)에 대해
탐지를 수행해야하기때문에 class imbalance 문제가 더 도드라진다.

예를 들어, detection이 쉬운 데이터를 easy examples, 어려운 데이터를 hard examples이라고 할 때 배경과 같이 흔한 easy examples이 10,000개 자전거와 같이 예측하고자 하는 hard examples이 50개 라고 하자.
만약, Easy examples이 평균적으로 loss가 0.1이고 hard example이 1이라면 에러의 총합은 easy examples이 1,000(0.1 * 10,000), hard examples이 50(1 * 50)으로 이미 잘 맞추고있는 easy examples의 에러가
더 크게 취급된다. 결국, 우리가 잘 예측해야하는 것은 hard example이기때문에 cross entropy를 사용하면 이와 같이 데이터의 분포는 고려되지 않은채 학습이 진행되어 학습이 불안정할 수 있다.

기존에는 augmentation이나 데이터셋 샘플링으로 이를 보완하려고 했지만 너무 많은 리소스가 필요하기때문에 RetinaNet에서는 Focal loss를 사용하여 이를 해결했다. 
Focal loss는 cross entropy($CE(p,y) = -\Sigma y_i ln p_i$) 공식에 가중치를 적용하는 방식이고 해당 클래스에 대한 확률이 높을수록(객체가 존재한다고 확신할수록) $\gamma$를 조절해 loss를 더 낮게하여 오히려 잘 예측하지 못한
클래스에 더 집중하도록 한다.

$$ FL(p_t) = -\Sigma y_i (1-p_t)^{\gamma} log(p_t) $$

<br> 
<div align="center">
  <p>
  <img width="600" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/088cde99-0ccb-4930-83db-e529162b5962">
  </p>
</div>

<br>

이 Focal loss를 활용해서 Cross entropy를 손실함수로 사용했을때보다 더 좋은 정확도를 기록했다.

# Feature Pyramid Network(FPN)
CNN에서는 층이 깊어질수록 추상적인 정보만 남아서 앞단의 세밀한 이미지 정보를 기억하기 어렵다는 문제가 있다. FPN은 이러한 문제를 해결하기 위한 기법으로 각 층의 피처맵을 예측에 사용할 피처맵과 결합하여 이미지 정보를
최대한 유지시키는 아이디어다.

<br> 
<div align="center">
  <p>
  <img width="500" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/862606a5-8611-40f8-b77f-cdd671cd5ddf">
  </p>
</div>
<br>

Backbone에서 bottom-up(사이즈는 줄이고, 채널은 늘림)으로 추출한 피처맵을 top-down(사이즈를 2배로 키우고, 채널은 그대로)으로 upsampling한 피처맵과 결합하여 이 결합한 피처맵을 예측에 사용하는 것이다.
해당 피처맵에서 계산된 손실을 모두 반영하여 loss를 계산한다. 이 방법은 여러 layer의 피처맵을 예측에 사용함으로써 단일 피처맵을 사용하는 것보다 다양한 이미지 정보를 사용할 수 있다는 장점이 있다. 
또한, 각 layer의 피처맵마다 grid에 9개의 anchor box가 할당되고 anchor는 k개의 클래스 확률값과 4개의 box regression 좌표를 가진다.

<br> 
<div align="center">
  <p>
  <img width="700" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/6c78e185-9eb2-403b-96e8-f7986da10975">
  </p>
</div>
<br>

Faster R-CNN에 FPN을 적용했을 때 성능이 향상했고, RetinaNet의 성능이 one stage detector뿐만 아니라 two satge detector인 Faster R-CNN과 비교해도 가장 높은 것을 알 수 있다.

# Reference
- [RetinaNet Paper](https://arxiv.org/pdf/1708.02002.pdf)
