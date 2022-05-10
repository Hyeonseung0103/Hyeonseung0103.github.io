---
layout: single
title: "Note 424 Transformer"
toc: true
toc_sticky: true
---

RNN에서는 단어가 순서대로 들어오기 때문에 시퀸스가 길면 연산 시간이 길어진다는 단점이 있다. 단어를 time-step별로 처리하기 때문에 병렬처리를 하지 못하기 때문인데 트랜스포머(Transformer)는
모든 토큰을 동시에 입력받아 병렬 연산을 함으로써 이 문제를 해결한다. 트랜스포머는 NLP에서 나온 모델이지만, CV(Computer Vision)에서도 잘 사용되기도 하는 Mutlti modal model이다.

![image](https://user-images.githubusercontent.com/97672187/167523948-5f1751a7-7008-4f2b-808a-94f200cf790c.png){: .align-center}

이미지출처: https://blog.promedius.ai/transformer/

트랜스포머는 다음과 같은 구조를 지니고 있는데 이번 포스팅에서는 트랜스포머의 세부 구조들이 어떻게 동작하는지 정리해보자.(실제 구조는 왼쪽의 인코더 블록이 6개, 
오른쪽의 디코더 블록이 6개로 쌓여있다.)

### Positional Encoding(위치 인코딩)

![image](https://user-images.githubusercontent.com/97672187/167524104-618f964d-33b8-4ba8-a83a-83959e0d5706.png){:. align-center}

이미지출처: https://timodenk.com/blog/linear-relationships-in-the-transformers-positional-encoding/

트랜스포머는 병렬화를 위해 임베딩된 모든 단어 벡터를 동시에 입력받는데 컴퓨터는 동시에 들어온 단어들의 원래 위치를 알 수 없다. 따라서, 컴퓨터가 이해할 수 있도록 단어의 위치 정보를
제공하는 벡터를 따로 제공해줘야하고, 단어의 상대적인 위치 정보를 제공하는 벡터를 만드는 과정을 Positional Encoding이라고 한다. 의미 정보를 임베딩 결과에 더해주는 것.

### Self-Attention(셀프-어텐션)
Self-Attention은 위의 구조에서 Multi-Head Attetion부분에 해당한다. 번역을 정확하게 하기 위해서는 문장 내에서 'it'과 같은 지시 대명사가 어떠한 단어를 가리키는지 알아야한다. 이처럼
단어들의 관계를 잘 파악하기 위해서, 문장 자신에게 어텐션 메커니즘을 적용하는 것을 Self-Attention이라고 한다. Transformer에서는 RNN이 없기 때문에 Self-Attention으로 
다른 단어와의 관계를 파악한다.

지난 포스팅에서(Note 423) Attention에는 질문을 하는 Query, 답을 하는 Key, 답들의 의미가 담겨져 있는 Value가 존재한다고 했다. Self-Attention에서도 역시 q,k,v가 존재하는데
기존 Attention과의 차이는 이 q,k,v가 모두 가중치 벡터라는 점이다. q는 분석하고자 하는 단어에 대한 가중치 벡터, k는 각 단어가 쿼리에 해당하는 단어와 얼마나 연관있는지 비교하기
위한 가중치 벡터, v는 각 단어의 의미를 나타내는 가중치 벡터이다. 기존의 Attention에는 Query가 디코더로부터 등장했지만, Transformer의 인코더에서의 q,k,v가 모두 인코더에서 등장한다.
기존 Attention의 Query, Key, Vector와는 별개로 생각하자.

![image](https://user-images.githubusercontent.com/97672187/167525285-ab215251-ca38-44be-8085-253fd0d19501.png){: .align-center}

이미지출처: https://wdprogrammer.tistory.com/72

Self-Attention의 과정을 위의 그림을 보며 설명해보자.

1) 각 단어의 해당하는 input vector를 가중치 행렬인 $ W_{Q} $, $ W_{K} $, $ W_{V} $ 와 곱해서 3개의 q,k,v 벡터를 만든다.

2) q와 v를 내적해서 특정 위치의 단어가 다른 단어와 얼마나 연관되어 있는지 점수를 계산한다.

위에서는 질문인 q1가 주어졌을 때, 이 q1에 대해서 다른 단어들이 (k1,k2) 얼마나 연관되어 있는지 내적하여 계산한다.

3) 계산된 점수를 key벡터의 차원 수에 루트를 씌워서 나눈 뒤 Softmax 함수를 취한다.

차원 수에 루트를 씌워서 나눠준 이유는 scale을 차원이 클수록 socre값이 자체가 매우 커지고(내적은 곱하고 모든 원소를 합해서) 여기에 softmax를 취하면, 
total sum이 크기 때문에 softmax를 지난 후의 확률값이 매우 작아져서 기울기 소실이 발생할 수 있기 때문이다. 따라서 q,v의 내적결과를 차원수의 제곱근으로 나눠줌으로써 scale을 조정한다.

4) Key vector의 의미를 나타내는 Value vector에 softmax score를 곱해서 해당 단어에 대한 Self-Attention 출력값을 얻는다. 

결국, softmax score는 해당 value vector를 얼마나 반영할 것인지를 나타낸다.

위와 같은 방식의 Self-Attention을 병렬적으로 실행한 것을 Multi-Head Attention이라고 한다. Multi-Head Attention은 여러개의 모델을 사용하는 앙상블 처럼 
총 8개의 Head를 사용하고 8번의 Self-Attention을 실행해서 각각의 출력 행렬인 $ Z_{0} $ ~ $ Z_{7} $ 을 만들고 이 8개의 Self-Attention 행렬을 이어 붙여서 또 다른 가중치 행렬인 $ W^o $ 와 내적에서 최종행렬 $ Z $ 를 만들어낸다. 최종적으로
생성된 $ Z $ 는 토큰 벡터로 이루어진 input 행렬 $ X $ 와 동일한 크기를 가진다. 각 Head에서 일어나는 가중치 연산은 모두 다르다.

![image](https://user-images.githubusercontent.com/97672187/167526656-81887995-b23c-4955-b502-df880247d2ef.png){: .align-center}

이미지출처: https://wdprogrammer.tistory.com/72

### Layer Normalization & Skip Connection
Add & Norm이라고 표현된 sub layer에서 출력된 벡터는 Layer Normalization과 Skip connection을 거치게 된다. Layer Normalization은 Batch normalization처럼 학습이 빠르고,
잘 되로고 하기 위함이고 Skip connection(Residual connection)은 역전파 과정에서 정보가 소실되지 않도록 한다.

### Feed Forward Neural Network(FFNN)
첫번째 sub-layer를 거친 벡터는 FFNN으로 들어가서 은닉층의 차원을 늘렸다가 다시 원래 차원으로 줄어들게 만드는 2층 신경망이다. 활성화 함수로는 ReLU를 사용하고, 차원을 늘리고
다시 줄임으로써 학습이 더 잘 되도록 한다. 출력된 벡터는 다시 sub-layer로 들어가서 layer normalization과 skip connection을 수행한다. 

여기까지 input vector -> positioning encoding -> Multi head attention(8개의 Self attention) -> FFNN 의 인코더 학습이 이루어진다.

### Masked Multi-Head Attention
인코더도 RNN을 대체하는 Attention이 있듯이, 디코더도 RNN을 대체할 Attention이 필요하다

1. 입력된 문장의 관계 파악 attention
 
2. encoder-decoder를 연결할 attention

Masked Self-Attention은 디코더 블록에서 사용되는 Self-Attention이다. 즉, 입력된 문장의 관계를 파악하는 attention이다. 디코더는 왼쪽의 단어를 보고 오른쪽의 단어를 예측하게 되는데 타겟이 되는 왼쪽 단어 이후의 단어는
모르는 상태로 예측해야한다. 따라서 이 타겟 단어 뒤에 위치한 단어는 Self-Attention(RNN 대신 사용되어 단어간의 관계를 파악하기 위한 attention)에 영향을 주지 않도록 마스킹(Masking)해야한다.

![image](https://user-images.githubusercontent.com/97672187/167539993-37272701-7e43-4f0c-9717-02d3a1d1b5c8.png){: .align-center}

이미지출처: https://aimb.tistory.com/182

위의 그림처럼 인코더에서의 Self-Attention은 문장의 주어진 단어들을 모두 활용하여 단어간의 연관정도를 파악하지만, Masked Self-Attention에서는 타겟 단어 뒤에 오는 단어는
모른다고 가정하게 된다. Attention이기 때문에 디코더에도 여전히 Query, key, value가 존재하게 되는데 Masked Self-Attention 에서는 디코더의 self-attention 이기 때문에 q,k,v가
모두 디코더에서 등장한다. 타겟 단어를 중심으로 이루어지는 연관성 계산(q,k,v 사용)은 다음과 같다.

1) 인코더의 Self-Attention처럼 q와 k를 내적해서 연관정도를 계산한다.

![image](https://user-images.githubusercontent.com/97672187/167540720-a654be69-e2a9-49e1-bf1b-f05d0ad8efb7.png){: .align-center}

2) 타겟 단어를 제외하고는 모두 scores를 -Inf 로 Masked한다.

위의 예시처럼 "robot must obey orders" 라는 문장이 있다고 하면, 해당 문장의 q,k의 내적 행렬은 중심 단어가 He(1행), He is(2행), He is a(3행), He is a doctor(4행)가 된다.
중심단어 외의 단어는 고려되지 않아야하기 때문에 중심 단어 외의 단어는 softmax를 지난 확률값이 0이 되어야 하고, 이를 위해 scores의 값을 -Inf로 Masked 해준다.

![image](https://user-images.githubusercontent.com/97672187/167541256-2101e843-5f6a-4ee3-b737-acdcc8ce937e.png){: .align-center}

3) Masked 된 scores를 softmax함수를 통과시켜 최종 scores를 도출한다.

그 이후에는 인코더의 Self-Attetion처럼 Key에 알맞는 Value Vector를 곱해주고 Self-Attention 출력 벡터를 얻는다. 여기서 Masked된 단어들은 0이기 때문에 Value 계산에 영향을 미치지 못한다.

![image](https://user-images.githubusercontent.com/97672187/167541315-6fb8c1bc-f9c7-4633-a85a-8699882f5997.png){: .align-center}

이미지출처: https://jalammar.github.io/illustrated-gpt2/

결국, Masked Self-Attetion도 인코더에서의 Self-Attetion과 같은 메커니즘이지만, 중심단어 이후의 단어를 Masked한다는 차이가 존재한다.

### Encoder-Decoder Attetion
Masked Multi-Head Attention이 입력된 문장 내의 단어 관계를 파악하기 위한 self-attention이라면, Encoder-Decoder Attention은 인코더와 디코더를 연결시켜주는 인코더이다.
디코더에서의 Multi-Head Attention은 인코더의 Multi-Head Attention과는 다르다. 인코더의 Self-Attention과 같은 역할은 Masked Self-Attetion이 수행하고, 디코더에서 Multi-Head
Attention이라고 불리는 Encoder-Decoder Attention은 인코더와 디코더를 연결시켜준다.

![image](https://user-images.githubusercontent.com/97672187/167543436-7a207356-0a98-400d-a3ef-7898e861d81b.png){: .align-center}

이미지출처: https://www.researchgate.net/figure/The-original-Transformer-architecture-with-six-encoder-blocks-and-six-decoder-blocks_fig5_358604689

디코더에서 Masked Self-Attention 층을 지난 벡터는 Layout Normalization과 Skip connection을 거쳐 Encoder-Decoder Attention 층으로 들어간다. 번역의 성능을 높이기 위해서는
문장 내의 단어관계도 중요하지만, 번역할 문장과 번역된 문장과의 관계도 중요하다. 이 층에서는 Masked Self-Attention에서 출력된 벡터를 Q 벡터로 사용하고, K, V 벡터는 인코더의
최상위 블록인 6번째 블록의 K,V 벡터를 가져와서 사용한다. 계산과정은 Self-Attention처럼 Q와 K 벡터 내적 -> scores를 softmax -> V vector 곱하기 -> 최종 벡터 출력으로 이루어진다.

디코더에서의 Query는 디코더에서, Key와 Value는 인코더의 벡터를 참조한다.

### Linear & Softmax Layer
디코더의 Encoder-Decoder Attetion -> Layer normalization & Skip connection 층을 지난 최종 벡터는 Linear층을 지나고 Softmax 활성화 함수를 지나 예측할 단어의 확률을 구하게 된다.
이 확률이 가장 높은 단어들로 번역된 문장이 이루어진다.

위의 블록을 참고하면, 먼저 인코더의 과정을 6번 반복하고 이 마지막 인코더의 key,value 벡터를 활용해서 디코더의 과정이 6번 반복된다. 하지만 꼭 6개의 블록을 쌓아야하는 것은 아니다.



