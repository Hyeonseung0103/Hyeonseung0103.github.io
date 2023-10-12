---
layout: single
title: "DeepLabv3+ : Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation 논문 요약"
toc: true
toc_sticky: true
category: Segmentation
---

# Atrous Convolution이란?
DeepLabv3+ 논문을 리뷰하기 앞서 DeepLab 모델들에서 사용된 핵심 기법인 Atrous Convolution에 대해 알아보자.

<br> 
<div align="center">
  <p>
  <img width="700" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/f677c6c9-c341-4081-bba7-cb7c8d6218f1">
  </p>
  <p>3x3 Atrous covolution, stride 1, padding = rate - 1 일 때 출력되는 피처맵</p>
</div>

<br><br>

Atrous convolution은 구멍이라는 말처럼 convolution 단계에서 필터 사이사이에 0을 입력하여 간격을 두는 기법으로 더 넓은 receptive field를 가질 수 있게 한다. 여기서 recpetive field란, 필터의 한 노드가 입력으로부터 수용할 수 있는 영역으로 segmentation과 같이 분류에 비해 정밀한 task가 요구된다면 주변의 공간 정보를 넓게 잘 활용해야하므로 넓은 receptive field를 가지는 것이 좋다.

넓은 receptive field를 위해 더 사이즈가 큰 필터를 사용하면 되지 않느냐는 생각이 들 수 있는데 필터의 사이즈가 커지면 파라미터의 갯수도 증가하기때문에 파라미터의 갯수는 그대로 유지시키면서 넓은 receptive field의 효과를 낼 수 있는 방법이 atrous(dilated) convolution이다.

위의 그림에서 빨간색 점을 필터의 크기, dilation rate을 $2^n$씩 증가시킨다고 할 때 (a)는 dilation rate이 1이기때문에 일반적인 convolution과 같다. (b)는 dilation rate이 2이고, padding은 1이기때문에 수용 영역이 7x7이 된다. 같은 방법으로 (c)는 dilation rate이 4이고, padding은 3이기때문에 15x15의 receptive field를 가진다. 결과적으로 3x3이라는 같은 필터의 크기를 가졌음에도 rate에 따라 필터사이에 0을 집어넣어 연산량은 일반적인 3x3 합성곱과 똑같이 가져가면서 padding까지 겹쳐져 훨씬 더 큰 receptive field를 가지는 것을 알 수 있다.

Receptive field를 구하는 공식은 다음과 같다.

<br> 

$$ReceptiveField = ((Filter Size - 1) * Dilated Rate) + 1 + (Filter Size  - 1) * (Stride - 1) + 2 * Padding$$

<br> 
추가적으로, receptive field는 layer가 깊어질수록 커진다. Receptive field란, 필터의 한 노드가 입력으로부터 수용할 수 있는 영역의 크기라고 했는데 아래 그림과 같이 층이 깊어질수록 한 노드가 갖는 수용영역의 크기는 넓어질 것이다. layer2 필터의 한 노드는 3x3의 receptive field를 가지고, layer3 필터의 한 노드는 layer 2의 3x3 -> 결국 layer 1의 5x5 모든 영역을 receptive field로 가지는 것이 된다. 층이 깊어질수록 receptive field가 커지고 이로 인해 저수준의 특징에서 점점 고수준의 특징을 학습할 수 있게된다.

<br> 
<div align="center">
  <p>
  <img width="500" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/59cfdef6-42f5-4d31-a34e-6e42700dce6e">
  </p>
</div>

<br>

아래 그림을 보면 똑같은 필터 크기를 사용하더라도 일반적인 convolution을 사용해 추출한 sparse한 features보다 atrous convolution 기법을 적용한 dense한 features의 object가 더 선명하게 보이는 것을 알 수 있다. 

<div align="center">
  <p>
  <img width="400" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/57d8afaa-2932-4f92-99a4-f80f01334fb3">
  </p>
</div>

<br><br>

Classification task에서는 한 이미지 내에서 object가 존재하는지의 여부가 중요했다면 semantic segmentation task에서는 단순히 object의 존재여부보다 object 간의 경계를 잘 파악하는 훨씬 중요하다. 
따라서, classification model은 receptive field가 상대적으로 작고 분류기에 fully connected layer를 통해 공간정보를 많이 손실하기 때문에 object의 경계간 detail한 정보를 얻기가 힘들어서 deeplab에서는 이를 해결하기 위해 atrous convolution을 사용하여 receptive field를 키우고 분류기에는 U-net처럼 fully convolutional layer를 사용해 공간정보를 최대한 활용하여 segmentation task를 잘 수행했다.



# Abstract
Spatial pyramid pooling module이나 encode-decoder 구조가 semantic segmentation 분야에서 잘 사용되고 있다. Spatial pyramid pooling 구조는 필터의 크기를 크게해서 넓은 영역의 context information을 encoding할 수 있고, encode-decoder 구조는 network의 공간정보를 점진적으로 복구해서 object 간의 경계를 더 선명하게 포착할 수 있게 한다. 본 논문에서는 이 두가지 방법을 모두 활용한 DeepLabv3+ 모델에 대해 설명한다. 추가적으로 Xception 모델과 ASPP(Atrous Spatial Pyyramid Pooling), decoder modules을 적용하여 더 빠르고 강한 encoder-decoder network를 구축했다. PASCAL VOC 2012와 Cityscapes 데이터셋에서 어떠한 post-processing 없이도 각각 89%, 82.1%의 정확도를 기록했다.
<br><br>

# Details
## 도입부 & 관련 연구
- FCN을 기반으로 한 Deep convolutional netrowrks는 benchmark tasks(BMT, 실제로 존재하는 하드웨어 혹은 소프트웨어의 성능을 비교 분석하는 방법)에서 수작업에 의존하던 시스템보다 훨씬 더 큰 개선을 이루어냈다.
- 본 연구에서는 spatial pyramid pooling 모듈이나 encoder-decoder 구조 라는 두 가지 종류의 networks를 사용하여 semantic segmentaion task를 수행했다.
- PSPNet이 다양한 grid scale을 사용하여 pooling 연산을 수행할 때, 본 연구의 이전 버전의 모델인 Deeplabv3에서는 다양한 크기의 context information을 캡처하기 위해 여러 rates을 가진 atrous convolution을 병렬로 적용한 ASPP 기법을 사용했다.
- layer의 마지막 피처 맵에 많은 semantic information이 잘 인코딩되더라도 pooling과 stride를 통해 객체 간의 경계와 관련된 세부 정보가 누락될 수 있다. 이러한 문제는 Atrous convolution을 적용하여 더 세밀하게 피처맵을 추출함으로써 완화시킬 수 있다.
- 하지만, 한정적인 메모리를 고려할 때 입력 이미지보다 8배, 4배 더 작은 피처맵을 추출하기에는 어려움이 있다. 더 큰 receptive field를 가지기위해(넓은 범위의 공간정보를 활용하기위해) 더 작은 피처맵을 추출한다는 것은 그만큼 더 많은 층을 거쳐야한다는 것이니까 층이 깊어질수록 훨씬 더 많은 연산량을 요구한다.
- 본 연구에서 사용한 encoder-decoder 모델에 encoder 경로에서는 일반적인 covolution에 비해 receptive field가 커져도 atrous convolution을 통해 피처수가 증가하지 않기 때문에 빠른 연산이 가능하고, decoder 경로에서는 객체간의 경계를 점진적으로 복구시켜 위의 문제를 해결할 수 있다. 쉽게 말해, DeepLabv3와 같이 컴퓨팅 자원에 따라 atrous convolution을 적용시켜 인코더 단계에서 풍부한 semantic segmentation 정보를 저장하고, 디코더 단계에서 해상도를 높이는 기법을 사용한다.
- 연산 속도와 정확도를 개선시킨 Xception model을 추가로 개발했고, ASPP와 encoder-decoder 구조를 사용하여 좋은 성능을 기록했다.


<br>

<div align="center">
  <p>
  <img width="700" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/65c1e4a4-e896-4ba9-90fc-868d5817e7e4">
  </p>
</div>

<br><br>

- DeepLabv3+ 모델은 다음과 같은 구조로 예측을 수행한다.
  
  1) Xception 혹은 ResNet을 backbone으로 사용하여 low level features(ex. ResNet에서 채널이 256인 conv2x 블록)과 GAP와 완전 결합층이 시작되기 이전의 피처(편의상 x)를 가져온다.
  2) Encoder 단계에서는 backbone의 x를 사용하는데 x를 가져올 때 encoder의 output stride에 맞춰서 크기가 유지되도록 하고, 논문에 근거해 특정 층(ex. ImageNet dataset이라고 가정하면 Resnet101에서 마지막 블록인conv5x의 ouput이 downsample을 진행하면 7x7이 되니까 downsample을 진행하지 않고, 오히려 atrous convolution을 사용하여 피처맵의 크기는 그대로 유지시키고 receptive field를 키움)에는 atrous convolution이 적용될 수 있다.
  3) Backbone에서 출력된 x에 ASPP(1x1, 3x3_rate6, 3x3_rate12, 3x3_rate18, GAP)를 수행하고 gap를 통해 작아진 output은 upsampling을 통해 크기를 다른 피처맵과 같이한 뒤 concatenate를 진행한다.
  4) Concatenate 후에는 1x1 합성곱을 통해 채널 수를 줄인 뒤 low_level_feature와 동일한 크기의 피처맵이 되도록 upsampling을 진행한다.
  5) Decoder 단계에서는 backbone에서 추출한 low level features에 1x1 합성곱을 적용해 채널은 48로 줄인뒤 이를 encoder의 출력과 다시 concatenate 한다.
  6) Encoder와 Decoder의 피처맵이 합쳐졌다면 3x3x256 크기를 가진 필터를 두번 거친 후 class 수만큼의 채널을 가진 1x1 합성곱을 통과한다.
  7) 마지막으로, 피처맵의 크기를 원본과 같은 크기로 upsampling하여 에측을 수행한다.

<br>

<div align="center">
  <p>
  <img width="700" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/3ae598ad-7207-4357-a45e-15c783bb9dac">
  </p>
</div>

<br><br>




## Methods
### 1. Encoder-Decoder with Atrous Convolution
**Atrous convolution**

Atrous convolution은 피처의 해상도를 컨트롤 하고, multi-scale의 정보를 캡쳐하기 위해 필터의 field-of-view를 조정한다.
<br>

<div align="center">
  <p>
  <img width="200" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/dd7a6934-b576-4d95-bfb5-130b80e99892">
  </p>
</div>

<br>

여기서 r은 atrous rate으로 input signal에 얼만큼의 stride을 적용할 것인지를 의미하고(일반적인 convolution은 r=1), i는 location, k는 kernel size, w는 필터, x는 input 피처맵, y는 output 피처맵을 나타낸다. 필터의 field-of-view(receptive field)는 rate에 따라 달라진다.

Ex 1) rate = 1일 때, k = 3(sigma의 k는 0부터 2까지) 일 때 x는 x[0 + 1 * 0] x[0 + 1 * 1]  x[0 + 1 * 2]로 증가하기떄문에

y[0] = x[0] * w[0] + x[1] * w[1] + x[2] * w[2]

Ex 2) rate = 2일 때, k = 3일 때 x는 x[0 + 2 * 0] x[0 + 2 * 1]  x[0 + 2 * 2]로 증가하기떄문에
y[0] = x[0] * w[0] + x[2] * w[1] + x[4] * w[2]

rate 2일때, receptive field가 더 커진 것을 알 수 있다.
<br>

**Depthwise separable convolution**
Depthwise separable convolution은 입력된 채널과 같은 채널을 갖는 필터로 각각의 필터는 자신이 담당한 입력 채널만을 상대로 연산을 수행하기때문에 필터의 모든 채널들이 입력 채널들에 대해 교차로 연산되는 일반적인 covolution보다 연산량이 훨씬 작다는 장점을 가지고 있다. 

아래 그림은 3x3 depthwise convolution이 각각의 채널마다(separable) 하나의 필터를 배치해 나온 특징들을 결합해 pointwise convolution을 만드는데 depthwise convolution 단계에서 atrous convolution을 적용해 rate = 2로 하는 (c)와 같은 방법론을 사용한다는 예시이다. Pointwise convolution 단계에서는 1x1 convolution을 통해 depthwise convolution의 결과를 선형결합하고 채널수를 줄일 수 있다.

<br>

<div align="center">
  <p>
  <img width="700" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/1fb8c1ff-886e-4630-9d86-3a3d614c8e7e">
  </p>
</div>

<br>

DeepLabv3+ 모델에서는 depthwise separable convolution을 사용하여 일반적인 covolution을 depthwise covolution과 pointwise covolution으로 변환하여 계산 복잡도를 크게 줄였다. 특히, depthwise convolution에 atrous convolution이 적용되어 atrous separable convolution이라고도 불린다. 이는 계산 복잡도를 크게 줄이면서 성능은 유지할 수 있도록 한 좋은 방법이었다.

**DeepLabv3 as encoder**

Output stride는 최종 output resolution(Global Average Pooling이나 Fully Connected 층 이전)에 대한 입력 이미지의 spatial resolution의 비율로 나타낸다. Classsification에서는 일반적으로 output의 resolution이 input보다 32배가
작기 때문에 output stride = 32가 된다(224x224 이미지가 FC층 출력으로 사용되기 직전에 7x7로 줄어들었을 때의 예시). Segmentation task에서는 좀 더 세밀한 특징을 추출하기 위해 마지막 한개 혹은 두개의 블록을 제거한 후, atrous convolution을 적용해
output stride = 16 혹은 8로 만들어서 resolution을 높인다.

또한, DeepLabv3에서처럼 다양한 비율의 atrous convolution을 적용한 atrous spatial pyramid pooling module을 사용하여 multi-scale의 특징을 추출했고, logits 이전의 마지막 피처맵을 인코더의 출력으로 사용했다. 인코더의 출력은 256개의 채널로 풍부한 semantic information을 담고있다. 컴퓨팅 자원에 따라 atrous rate을 조절하면 추출되는 피처맵의 해상도를 임의로 조정할 수도 있다.

**Proposed decoder**

DeepLabv3의 인코더는 일반적으로 output stride가 16이기떄문에 피처맵은 원본 이미지보다 16배 줄어들었고 디코더 과정에서 이를 복원하기 위해 피처가 16배 upsampling 된다. 하지만, 이러한 naive decoder module은 segment의 세부 정보를 잘 복구시키지 못할 수도 있기 때문에 DeepLabv3+에서는 다른 방법을 사용한다.

먼저, 인코더 피처를 4배 upsampling 하고, 동일한 spatial resolution을 갖는 네트워크 백본의 저수준 피처와 결합한다. 저수준 피처가 너무 많은 채널을 가지면 합치는 과정에서 연산량이 증가할 수 있기때문에 1x1 필터를 사용해 채널 수를 줄인다. 
연결 후에는 3x3 convolution을 몇번 적용하여 특징을 업데이트 시키고 upsampling을 통해 피처맵의 크기를 4배(원본과 같이) 키운다. 

또한, 실험을 통해 output stride가 16일 때 정확도와 연산 속도의 trade-off가 가장 좋았고, stride가 8일 때는 성능이 좀 더 좋았지만 그에 비해 연산 속도가 많이 느렸다.

### 2. Modified Aligned Xception
Xception 모델이 빠른 계산으로 ImageNet에서 좋은 분류 성능을 기록했고 MSRA팀이 Alined Xception 모델을 통해 객체 탐지에서 성능을 더욱 향상시켰다. 본 연구에서는 sementic segmentation 분야에서도 성능을 향상시키기 위해 다음과 같이 
MSRA 팀의 Alined Xception 모델보다 몇 가지 구조를 더 수정했다.

- 전체적인 구조는 크게 바꾸지 않고도 깊이를 깊게하여 연산 속도와 메모리 효율을 높였다.
- 모든 max pooling 연산을 atrous covonlution이 적용된 depthwise separable covolution으로 바꿨다.
- MobileNet의 구조처럼 3x3 convolution 연산 후에는 batch normalization과 ReLU를 추가했다.

<br><br>

## Experimental Evaluation
모델의 구조는 아래와 같고, 라벨은 20개의 foreground object와 1개의 backgound object 이루어져있다. 평가지표는 mIOU를 사용했고 initial lerning rate 0.007, crop size 513 x 513, output stride = 16,
학습에서는 random scale data augmentation을 사용했다. 

<br>

<div align="center">
  <p>
  <img width="700" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/73587e4a-0e4f-4a1c-8fec-bb5480820dc3">
  </p>
</div>

<br>

### 1. Decoder Design Choices
Decoder module에서 1x1 convolution의 효과를 평가하기 위해 3x3x256의 피처맵과 Resnet-101 네트워크의 Conv2 피처(conv2x residual block의 마지막 피처맵)를 사용한다. 
먼저, 인코더 모듈에서는 저수준 피처맵의 채널을 48이나 32로 줄이면 성능이 향상되기때문에 1x1x48의 필터를 적용하고, 그 후, 디코더 모듈에는 3x3 컨불루션을 배치했다. 그 결과, 디코더 모듈의 피처맵과 인코더의 피처맵을 합쳤을 때 단순히 여러개의 convolution을 사용하는 것보다 3x3x256 합성곱을 두번 사용한 것이 더 효과적이었다. 256에서 128로 필터의 채널을 바꾸거나 필터의 크기를 3x3을 1x1으로 바꾸면 성능이 하락했다.

이외에도 다양한 실험을 해봤지만 결국에는 위와 같이 가장 간단하면서도 효과적인 Decoder module을 선택했다. DeepLabv3의 피처맵과 1x1 필터가 적용되어 채널이 감소된 Conv2 피처맵은 두 개의 3x3x256 필터로 연결된다. DeepLabv3+ 모델의 output stride는 최대 4까지 추천한다.

### 2. ResNet-101 as Network Backbone
모델의 변화에 따른 정확도와 속도를 비교하기 위해 mIOU와 Multi-Adds를 사용했다. ResNet-101을 backbone으로 사용했을 때, Atrous convolution 덕분에 하나의 model 만을 가지고도 rate을 다르게 해서 다양한 resolution을 갖는 피처를 얻어 성능을 비교해볼 수 있었다.
- Baseline: output stride = 8로 하고 multi-scale input을 사용하면 성능이 향상된다. 좌우반전을 적용한 이미지를 추가하면 계산 복잡도는 두배가 되지만 그에 비해 성능 향상은 미비했다.
- Adding decoder: ouptut stride를 8로 하면 16으로 설정했을 때보다 성능이 더 좋지만 약 20억개에 달하는 오버헤드가 발생한다. Multi-scale과 좌우반전을 적용하면 성능이 향상된다.
- Coarser feature maps: atrous convolution을 사용하지 않고 빠른 연산을 위해 train output stride를 32, 디코더를 추가했을 때 성능이 2% 개선됐지만 train output stride = 16, eval output stride를 여러 값으로 사용했을 때보다
성능이 떨어졌다. 따라서, train, test output stride는 16이나 8을 쓸 것을 권장한다.

### 3. Xception as Network Backbone
- ImangeNet pretraining: 큰 하이퍼파라미터 조정 없이 ImageNet pretraining을 시켰을 때 배치 정규화나 ReLU를 추가하지 않은 Alined Xception에서 성능이 하락했다.
- Baseline: 디코더를 사용하고, output stride = 16으로 통일시켰을 때 Xception 모델은 ResNet-101보다 성능이 약 2% 높았다. Test output stride = 8 로 하고 좌우반전을 적용하면 추가적인 성능 향상을 얻을 수 있다.
- Adding decoder: 디코더를 출력하면 eval test = 16으로 사용했을 때 약 0.8% 성능이 올라간다.
- Using depthwise separable covolution: depthwise separable convolution을 ASPP와 디코더 모듈에 추가했을 때 연산 복잡도는 크게 감소했는데도 비슷한 mIOU score를 얻었다.
- Pretraining on COCO: MS-COCO dataset을 추가로 사전학습 하여 성능을 2% 개선시켰다.
- Pretraining on JFT: JFT-300M dataset을 추가로 사전학습 하여 성능을 1% 개선시켰다.
- Test set resluts: 계산 복잡도를 고려하지 않아서 output stride = 8, 배치 정규화를 사용하여 학습했을 때 JFT dataset의 사전학습과는 관계없이 mIOU 87.8%와 89%의 성능을 달성했다.
- Failure mode: DeepLabv3+ 모델은 소파와 의자, 많이 가려진 물체, 잘 보이지 않는 물체를 잘 세분화하지 못한다.

### 4. Improvement along Object Boundaries
<br>
<div align="center">
  <p>
  <img width="700" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/854ac372-d2fe-4440-a96b-a346164cc4e9">
  </p>
</div>

<br>

Object의 Boundaries 근처에서 proposed decoder를 활용하여 정확도를 높이기 위해 trimap experiment를 수행했다. Trimap은 배경, 전경, 배경과 전경 사이에 있는 불확실한 영역을 분류하는 작업을 수행하는 과정이라고 할 수 있다.
일반적으로 객체 경계 주변에서 발생하는 val set의 비어있는 label annotations(불확실한 영역)에 morphological dilation(trimap width를 조정해서 해당 영역을 더 부각시킴)을 적용했다. 

<br>


그 후, dilated band(확장된 trimap)에 대해 mIOU를 구했다. 단순히 naive decoder를 사용했을 때보다 proposed decoder를 사용했을 때 ResNet과 Xception 모델에서 4.8%, 5.4%의 mIOU 향상을 기록했고 dilated band가 좁을수록 성능은 더 잘 향상되었다.

### 5. Experimental Results on Cityscapes
Cityscapes dataset에서도 DeepLabv3+를 실험해봤다. Aligned Xception 모델을 backbone으로 사용하고 DeepLabv3의 ASPP를 사용했을 때 proposed decoder를 적용하면 mIOU가 78.79%로 naive decoder를 사용했을 때보다 약 1.46% 성능이 향상됐다.
Augmentation image를 제거했을 때 성능이 79.14%로 향상된 것을 보아 DeepLab 모델에서는 PASCAL VOC 2012 dataset에서 augmentation이 효과적이고 Cityscapes에서는 그렇지 않다는 것을 알 수 있다.

DeepLabv3+ 모델은 네트워크의 layers를 더 추가한 결과(X-71) val set에서 79.55%, test set에서는 82.1%의 최고 성능을 달성했다.

## Conclusion
- DeepLabv3+ 모델은 풍부한 cotext information을 위한 DeepLabv3의 인코딩과 Object Boundaries를 잘 복구하기 위한 디코더 모듈로 이루어진 encoder-decoder 구조를 사용한다.
- 컴퓨팅 자원에 따라 atrous convolution을 적용하여 인코더의 특징을 임의의 resolution으로 추출할 수도 있다.
- Xception 모델과 atrous separable convolution을 사용하여 더 빠르고 좋은 모델을 만들었다.
- PASCAL VOC 2012와 Cityscapes datasets에서 SOTA를 기록했다.

# 개인적인 생각
- DeepLabv1 이후 지속적으로 연구하여 architecture에 큰 변화없이 모델을 발전시켜서 SOTA를 기록해나갔다는 점이 인상깊었다.
- DeepLabv3+도 U-Net처럼 인코더-디코더 구조를 사용했는데, Segmentation 모델에 인코더-디코더 구조를 사용하지않으면서 성능을 좋게하는 방법은 없을지 궁금하다.
- ResNet101 모델을 backbone으로 사용할 때 마지막 residual block 뿐만 아니라 처음부터 atrous convolution을 사용했다면 성능이 어떻게 변화했을지 궁금하다.
- Pooling을 convolution 연산으로 대체하고, atrous convolution을 사용하고, fully connected layer를 fully convolutional layer로 대체하며 공간적인 정보를 최대한 보존하려고하는데 향후 논문에서는 이 공간 정보를 어떻게 더 잘 보존하는지 기대된다.

<br><br>

# 구현
DeepLabv3+를 pytorch로 구현해보자.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

```python
def conv3x3(in_channels, out_channels, stride = 1, dilation = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding = dilation, dilation=dilation, bias = False)

def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias = False)
```

```python
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride = 1, dilation = 1, downsample = None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride, dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        # 1x1 필터
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3 필터
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1 필터
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out
```

```python
class ResNet(nn.Module):
    def __init__(self, block, layers_num):
        super(ResNet, self).__init__()
        
        multi_grid = [1,2,4]
        '''
        이 multi_grid 비율에 맞게 rate이 곱해져서 atrous convolution
        이전 버전인 deeplabv3 논문에 근거. For example, when output stride = 16 and Multi Grid = (1, 2, 4), 
        the three convolutions will have rates = 2 · (1, 2, 4) = (2, 4, 8) in the block4, respectively.
        '''
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
        
        self.layer1 = self._make_layer(block, 64, layers_num[0])
        self.layer2 = self._make_layer(block, 128, layers_num[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers_num[2], stride = 2)
        self.layer4 = self._make_atrous_layer(block, 512, stride = 1, dilation = 2, multi_grid = multi_grid)
        # output stride = 16이면 최종 출력값이 14라서 downsampling이 되면 안 됨. 오히려, atrous convolution으로 receptive field를 넓혔음.

        # Deeplab에서는 residual block 까지만 사용.
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def _make_layer(self, block, out_channels, blocks_num, stride = 1):
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        
        # conv2,3,4x에서는 dilation이 항상 1이니까 굳이 block을 만들때 rate을 지정해줄 필요 없음.
        layers.append(block(self.in_channels, out_channels, stride, downsample))  
        self.in_channels = out_channels * block.expansion 
        for _ in range(1, blocks_num):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _make_atrous_layer(self, block, out_channels, stride = 1, dilation = 1, multi_grid = [1, 2, 4]):
        # resnet 가장 마지막 층에 사용하는 atrous_convolution에서는 output stride를 유지하기위해 downsampling이 일어나지않음.
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, dilation = multi_grid[0] * dilation)) 
        # 첫번째 residual block에 dilation 2 적용.

        self.in_channels = out_channels * block.expansion
        for i in range(1, len(multi_grid)):
            layers.append(block(self.in_channels, out_channels, dilation = multi_grid[i] * dilation))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_features = x

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x, low_level_features

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
```

```python
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding = 1, dilation = 1):
        super(ASPP, self).__init__()
        self.aspp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = 1, 
                      padding=padding, dilation = dilation, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    
    def forward(self, x):
        x = self.aspp(x)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
```

```python
# output stride가 16일 때
# rates = 1, 6, 12, 18
class DeepLabv3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabv3Plus, self).__init__()
        self.resnet = ResNet(Bottleneck, [3, 4, 23, 3])

        
        # deeplabv3 논문. atrous convolution -> "all with 256 filters and batch normalization"
        self.aspp1 = ASPP(2048, 256, kernel_size=1, padding=0, dilation = 1)
        self.aspp2 = ASPP(2048, 256, kernel_size=3, padding=6, dilation = 6)
        self.aspp3 = ASPP(2048, 256, kernel_size=3, padding=12, dilation = 12)
        self.aspp4 = ASPP(2048, 256, kernel_size=3, padding=18, dilation = 18)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.pool_conv = conv1x1(2048, 256) # 풀링 후 풀링 채널도 256으로 통일
        self.encoder_conv1x1 = conv1x1(256 * 5, 256)

        self.decoder_conv1x1 = conv1x1(256, 48)
        self.decoder_conv3x3_a = conv3x3(256 + 48, 256)
        self.decoder_conv3x3_b = conv3x3(256, 256)
        self.last_conv = conv1x1(256, num_classes)
    
    def forward(self, input):
        # 인코더
        x, low_level_features = self.resnet(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.avgpool(x)
        x5 = self.pool_conv(x5)
        x5 = F.upsample(x5, size = x4.size()[2:], mode='bilinear') # pooling으로 1,1로 줄어들었으니까 다시 다른 피처맵과 같은 크기로
        
        x = torch.cat([x1,x2,x3,x4,x5], 1)
        x = self.encoder_conv1x1(x)
        x = F.upsample(x, size = (input.size()[:-2] // 4, input.size()[:-1] // 4), mode = 'bilinear')
        # decoder와 합치기위해 low_level_feature와 같은 크기로 low level feature는 ResNet에서 input보다 4배 작은 크기로 넘어옴

        # 디코더
        xd = self.decoder_conv1x1(low_level_features)
        xd = torch.cat([x, xd], 1) 
        xd = self.decoder_conv3x3_a(xd)
        xd = self.decoder_conv3x3_b(xd)
        xd = self.last_conv(xd)
        output = F.upsample(xd, size = input.size()[2:], mode = 'bilinear')
        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
```

# 이미지 출처
- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation Paper](https://browse.arxiv.org/pdf/1802.02611.pdf)
- [Atrous convolution DeepLabv2 논문 참고](https://arxiv.org/pdf/1606.00915v2.pdf)
- [Atrous convolution 이미지 예시](https://medium.com/mini-distill/pps-deeplab-semantic-image-segmentation-with-deep-convolutional-nets-atrous-convolution-and-ae6463029410)
- [Receptive field 이미지](https://itrepo.tistory.com/32)
- [Trimap 예시](https://devocean.sk.com/blog/techBoardDetail.do?ID=163989)
