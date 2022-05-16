---
layout: single
title: "Note 433 AutoEncoder"
toc: true
toc_sticky: true
category: Section4
---

AutoEncoder는 입력된 데이터를 저차원의 벡터로 압축시키고 다시 원본 크기의 데이터로 복원하는 신경망으로, 유용하게 쓰일 수 있다. 이 AutoEncoder를 활용하면 차원을 축소할 수 있고,
데이터의 어느 정도 형태는 유지하면서 데이터를 압축시킬 수 있기 때문에 저장면에서도 메모리 공간을 절약할 수 있다. 또한, 데이터를 압축시킬 때 노이즈들을 제거함으로써
필요한 데이터만 추출할 수 있고, 이 과정에서 이상치를 탐지할 수 있다는 장점을 가지고 있다. 이번 포스팅에서는 이렇게 유용하게 사용되는 AutoEncoder에 대해 알아보자.

### AutoEncoder의 구조

![image](https://user-images.githubusercontent.com/97672187/168503097-1ceabff9-c895-4878-86bf-fc0c31d8e25e.png){: .align-center}

이미지출처: https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798

AutoEncoder는 입력데이터를 저차원으로 축소시키는 인코더 부분과 축소된 데이터를 다시 원본 크기로 복구하는 디코더 부분으로 나눌 수 있다. 특히 위 그림에서 Code라고 표현된 가장
저차원의 벡터를 Latent(잠재)벡터라고 하는데 Latent 벡터는 원본 데이터보다 저차원이지만, 원본 데이터의 특징을 잘 보존하는 벡터이다. 결국 AutoEncoder의 학습목표는 원본 데이터의
특징을 가장 잘 보존하는 이 Latent 벡터를 잘 얻기 위함이다.  Latent 벡터는 원본 데이터를 압축했다가 복원하는 과정에서 원본 데이터의 특징일 최대한 보존할 수 있도록 학습된다.


### 매니폴드 학습(Manifold Learning)
Manifold란 고차원의 데이터가 이루는 저차원의 공간이고, Manifold 학습은 고차원 데이터를 데이터 공간에 뿌렸을 때 sample들을 가장 잘 표현하는 Manifold 공간을 찾는 것이다. 즉,
기존 sample들의 정보를 최대한 잘 반영한 고차원 데이터가 가지고 있는 저차원 공간을 찾는 것이다. 

차원의 저주는 데이터의 차원이 커질수록 데이터 분석이나 좋은 모델을 만들기 위해 필요한 샘플 데이터 수가 증가하는 것을 말한다. 데이터의 차원이 커질수록 데이터의 밀도가 낮아지는데
이 고차원의 데이터는 저차원의 매니폴드를 포함하고 있고, 이 저차원의 매니폴드를 벗어나는 순간 밀도는 급격히 낮아진다. 밀도가 낮으면(샘플들끼리 거리가 멀리 떨어져있으면) 유사한 데이터를
찾기 어렵고 데이터 간의 관계를 파악하기가 힘들 것이다. 따라서, Manifold 학습을 차원의 저주 관점에서 보면 고차원의 데이터를
잘 표현하는 manifold 공간을 찾으면서 샘플 데이터의 특징을 파악할 수 있고, 샘플 데이터의 특징을 잘 파악한다면 차원이 커져서 더 많은 샘플들이 필요로한 차원의 저주 문제를 해결할 수 있다.

차원의 크기는 Feature(특성)의 수로 표현되는데 결국 데이터를 잘 표현한 manifold를 찾는다는것은 중요한 특성을 찾고, 이를 기반으로 manifold 좌표를 조금씩 변경해가며 데이터를 유의미하게
변화시키는 것이다. 반대로, manifold를 잘 찾았다는 것은 중요한 특성이 유사하게 표현된 sample을 찾을 수 있다는 것이다. 

![image](https://user-images.githubusercontent.com/97672187/168505598-7cae8777-6279-4c9b-bc4f-85bfbdec9a96.png){: .align-center}

이미지출처: https://www.analyticsvidhya.com/blog/2021/02/a-quick-introduction-to-manifold-learning/

아래의 그림처럼, 고차원으로 복잡하게 표현되어 있는 데이터를 저차원 평면으로 바꿔서 차원은 줄이면서, 고차원에 표현된 데이터의 정보를 잘 반영한 저차원 공간을 찾을 수 있다. 
AutoEncoder의 Latent 벡터는 원본 데이터를 잘 표현한 저차원 데이터를 의미하는데 이 관점에서 매니폴드 학습이 잘 이루어지면 기존 샘플들의 정보를 잘 유지하면서 더 적은 차원으로 
데이터를 표현할 수 있기 때문에 Manifold는 AutoEncoder에서 핵심이 되는 개념이 된다.




