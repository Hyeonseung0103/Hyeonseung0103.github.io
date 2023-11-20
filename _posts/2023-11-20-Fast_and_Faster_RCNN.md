---
layout: single
title: "Fast & Faster R-CNN 정리"
toc: true
toc_sticky: true
category: Detection
---

R-CNN이 다른 모델들에 비해 높은 mAP를 기록했고 이후 Fast-RCNN, Faster R-CNN, Mask-RCNN 등 여러 모델들이 R-CNN을 develop하여 object detection 분야에서 더 많은 발전을 이루어냈다. 이번 포스팅에서는
R-CNN 이후의 모델인 Fast R-CNN과 Faster R-CNN에 대해 간단하게 정리해보자.

# Spatial Pyramid Pooling Net(SPPNet)
Fast & Faster R-CNN을 더 잘 이해하기위해 먼저, SPPNet에 대해 알아보자.

R-CNN은 selective search와 CNN, SVM을 통해 object detection을 수행했는데 region proposals, feature extraction, detection이 모두 다른 네트워크에서 수행되어 학습 및 추론 시간이 느리다는 단점을 가지고있다.
또한, 2000개의 region proposals이 CNN에 입력되기 위해서는 모든 region이 고정된 크기의 벡터여야하는데 이를 위해 crop이나 warp를 사용해 사이즈를 조절했는데 crop/warp를 사용할 경우 실제 region과는 조금은 다른
이미지가 만들어지기때문에 성능에도 영향이 있다.

SPPNet은 이러한 문제점을 개선해서 2000개의 region proposals 이미지를 전부 CNN에 통과시키는 것이 아니라 CNN은 원본 이미지만 통과시켜 피처맵을 만들고 selective search로 나온 region을 이 피처맵과 맵핑시켜
학습하는 방법을 사용했다. 가장 중요한 점은 피처맵과 맵핑된 region을 고정된 크기의 벡터로 flatten시켜야 분류기에 넣어 detection을 수행할 수 있는데 모든 region의 크기가 제각각이기때문에 이를 고정된 크기의 벡터로
만드는 것이 어려웠다.

SPPNet에서는 이를 해결하기위해 SPP Layer를 만들어서 Flatten을 시키기전에 어떤 크기의 region이 들어와도 분류기에 입력하기 전에 고정된 크기의 벡터로 변환했다. 아래 그림처럼 다양한
크기의 피처맵이 들어오면 해당 피처맵을 여러개의 분면으로 쪼개고 이를 합쳐서 고정된 크기의 벡터를 만들어 분류기에 입력으로 사용하는 것이다.

<br> 
<div align="center">
  <p>
  <img width="500" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/52ba1d69-7923-400d-bcf8-624bd5cca7ba">
  </p>
</div>

<br>

이를 통해 성능 뿐만 아니라 detection 수행 시간도 크게 단축됐다.

# Fast R-CNN
Fast R-CNN은 SPPNet을 조금 더 보완시켜 SPP layer를 RoI pooling layer로 바꾼 네트워크라고 할 수 있다. 또한, 위에서 언급한 R-CNN의 문제를 여러가지 방법을 통해 해결했는데 먼저 SVM을 softmax 네트워크로 변환시켰고
classification과 regression을 혼합하여 사용하는 multi-task loss 함수를 통해 end-to-end network(RoI proposals 제외)를 구축했다.

SPPNet이 다양한 크기의 피처맵을 미리 정해놓은 여러개의 분면으로 쪼개 고정된 크기의 벡터로 만들었다면, Fast R-CNN은 RoI pooling을 통해 어떤 크기의 피처맵이 들어와도 모두 고정된 크기로 max pooling하는 기법을 사용한다.
보통 7 x 7 크기로 pooling 시키는데 만약 피처맵의 크기가 7의 배수가 아니라면 보간이나 이미지를 resizing하여 크기를 맞춘다.

손실함수로는 multi task loss를 사용하고 특히 regression loss에는 smooth $L_1$을 적용해서 R-CNN과 SPPNet에서 사용된 $L_2$ loss보다 outliers에 덜 민감하도록 하고, 
loss가 1보다 작으면 loss를 더 작게해서 큰 loss에 더 집중할 수 있도록 한다. $\lambda$는 classification과 box regression loss의 balance를 맞추는 용도로 사용된다.

<br> 
<div align="center">
  <p>
  <img width="500" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/a12312fb-2d42-4850-bf1a-078d715be168">
  <img width="500" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/ecf36288-8f1f-4017-89a8-b251ef7459c7">
  <img width="450" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/6b7d4f67-7234-4a08-9a28-a5fb794dc05a">

  </p>
</div>
<br>

Fast R-CNN의 학습 과정을 정리하면 다음과 같다.

1) 원본 이미지를 CNN에 통과시켜 feature extraction을 수행
   
2) selective search를 통해 나온 2000개의 region proposals을 원본 이미지의 피처맵과 맵핑

3) 맵핑된 다양한 크기의 region proposals 피처맵에 7x7 RoI pooling 적용

4) RoI pooling을 통해 나온 ouput으로 detection 수행(multi task loss)


<br> 
<div align="center">
  <p>
  <img width="700" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/19f94db4-a3da-4c4f-b07a-b4af21e67fa2">
  <br>
  <img width="300" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/95ca23fc-703a-4540-9348-901ee76d56e8">\
  </p>
</div>
<br>

결과적으로 Fast R-CNN은 속도나 정확도 측면 모두에서 기존 R-CNN보다 크게 개선된 mAP를 기록했다.

# Faster R-CNN
Fast R-CNN은 CNN, RoI pooling, 분류기를 결합해 특징 추출과 detection을 하나의 네트워크에서 수행하여 R-CNN보다 훨씬 빠른 속도로 detection이 가능했지만 여전히 region proposals에는 selective search 알고리즘을 사용하여 완벽한 end-to-end network를
구축하진 못했고 one stage model보다 속도 측면에서 좋지 않은 performance를 보였다. Faster R-CNN은 이러한 문제를 해결하기위해 Fast R-CNN에 RPN(Region Proposals Network)을 결합하여 selective search를 대체한
완벽한 end-to-end network를 구축했다.

End-to-End Network가 구축되면 classification, box regression 뿐만 아니라 region에 대한 back propagation도 가능해지기때문에 2000개로 한정된 selective search 알고리즘보다 더 효율적인 proposals이 가능하다.
그렇다면, 이미지에서 어떻게 selective search와 같이 region을 예상하여 제안할 수 있을까?

여기에서 사용된 개념이 anchor box이다. Anchor box는 한 픽셀당 고정된 여러 스케일의 boxes를 사용하여 region proposals을 수행한다. 아래 그림처럼 각각의 피처맵의 한 픽셀당 여러 ratio와 크기를 가진 k개의 anchor box가 있다고 할 때
anchor box마다 object의 존재여부 scores와($2k$ scores) 4개의 box 좌표($4k$ scores)가 output으로 도출된다. Anchor boxes가 ground truth box와 일치한다의 기준은 IoU가 가장 높은 anchor나 0.7이상인 anchor를
positive, 0.3이하를 negative로 분류하고 그 외의 애매한 box는 학습에서 제외시킨다.

<br> 
<div align="center">
  <p>
  <img width="700" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/a37bfcfd-7fb1-44a9-83da-afe1129efbab">
  </p>
</div>
<br>

손실 함수는 Fast R-CNN처럼 multi task loss를 사용하지만 anchor box와 관련된 계수들이 추가되었다. $p_i$는 anchor box내 객체가 object일 확률이고, $p_{i}^{\star}$는 해당 object가 ground trurh와 일치하면
positive로 1, 일치하지 않으면 negative 0으로 취급한다. Box regression은 anchor box가 클래스를 올바르게 예측한 즉, positive에 대해서만 수행하고 $t_i$는 anchor box와 모델이 예측한 bbox와의 차이,
$t_{i}^{\star}$는 anchor box와 ground truth간의 차이이다. Faster R-CNN의 손실함수에서 특이한 점은 예측 bounding box와 ground truth box의 차이를 anchor box와 각각의 box와의 차이를 사용하여 계산한다는 것이다.
이 방법은 anchor box를 참고해서 anchor를 기준으로 GT와 predicted box의 차이가 비슷할수록 두 박스의 거리가 가까울 것이라는 아이디어이다. 객체의 존재여부를 하나도 모르는 bbox를 생성하고 조정하는 것보다 positive anchor를 참고하여
bbox를 GT에 가깝게 조금씩 조정하는 것이 더 효율적인 방법이 된다. $N_{cls}$는 positive와 negative anchor의 비율을 동일하게 가져가기 위한 정규화 파라미터고 $N_{reg}$는 박스 갯수를 정규화한 값이다.

<br> 
<div align="center">
  <p>
  <img width="450" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/9911a43e-0cde-4a93-8700-cd1a62dacf80">
  <br>
  <img width="500" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/6db411c0-3104-4626-bcbc-d353de5b57e6">
  </p>
</div>
<br>

Faster R-CNN의 학습 과정을 정리하면 다음과 같다.
1) 원본 이미지를 CNN을 통과시켜 특징 추출

2) 각각의 피처맵에 대해 픽셀당 여러 스케일을 가진 anchor box를 그리고 각각의 boxes를 RoI pooling으로 고정된 크기의 벡터로 변환

3) Anchor boxes에 대해 object 존재여부와 ground truth와의 일치여부를 계산하고, 일치한다면 anchor box, predicted box, ground truth box를 통해 box regression 수행

<br> 
<div align="center">
  <p>
  <img width="700" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/129d93c3-c0d4-4c8b-a7f2-e78b70484aef">
  <br>
  <img width="700" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/bcc38a5d-40ec-4d16-9b0a-138a76217953">
  </p>
</div>
<br>

Faster R-CNN은 selective search 보다 anchor boxes를 활용한 RPN을 사용했을 때 성능이 크게 향상됐고, Fast R-CNN보다 개선된 모델임을 알 수 있다.

# Reference
- [SPPNet](https://arxiv.org/pdf/1406.4729v4.pdf)
- [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
- [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)
