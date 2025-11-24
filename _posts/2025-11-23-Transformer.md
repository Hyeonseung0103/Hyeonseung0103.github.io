---
layout: single
title: "Transformer와 MLP"
toc: true
toc_sticky: true
category: Interpretability
---

머신러닝과 비교하여 딥러닝 모델의 가장 큰 문제는 모델이 왜 이런 결과를 만들어냈는지 해석할 수 없다는 것이고 이를 'Black Box Problem'이라고 한다. 그리고 이 Black Box 문제를 해결해보고자 예전부터 XAI(eXplainable AI)에 관한 연구가 활발한데 XAI에서 사용하는 방법은 2가지로 나눌 수 있다.
- **해석가능성(Interpretability)**: 엔지니어의 관점에서 모델의 구조를 기반으로 알고리즘이 어떻게 변화하여 해당 결과를 도출해내는지 해석할 수 있는 능력
- **설명가능성(Explainability)**: 해석가능성에서 더 나아가 사회과학, HCI 등의 요소를 결합하여 일반적인 사용자들도 이해할 수 있도록 설명하는 능력

이번 포스팅에서는 딥러닝 모델에서 가장 많이 사용하는 Transformer를 기반으로 interpretability에 초점을 맞춰보자. 예전에도 [Transformer에 관한 포스팅](https://hyeonseung0103.github.io/section4/S4_S2_Transformer/)을 작성한적이 있지만 이번에는 Attention과 MLP 부분을 좀 더 자세하게 다루어보겠다.

# Embedding
Transformer에 문장이 입력되면 문장은 여러 토큰(무조건 토큰 하나가, 단어를 의미하는 것은 아님)으로 나뉘고 각 토큰은 해당 단어의 의미와 주변 단어들과의 관계, 위치 정보 등을 담은 고차원의 임베딩 벡터로 표현된다.

토큰이 임베딩 벡터로 표현된다는 것은 내적을 통해 단어사이의 유사도를 구할 수 있게 되는 것이고,
- 내적이 양수이면 단어가 유사한 것
- 내적이 0이면 전혀 관련이 없는 것
- 내적이 음수이면 단어가 반대의미를 가지는 것이다.

<br> 
<div align="center">
  <p>
  <img style="width: 50%;" alt="image" src="https://github.com/user-attachments/assets/db8323a8-8fbb-4815-92b9-6acc79db485a" />
  </p>
</div>

<br>

임베딩은 처음에는 랜덤한 값으로 이루어져있다가 여러 layer를 거치면서 data를 기반으로 아래와 같이 토큰의 정보를 가장 잘 담고 있는 상태로 업데이트된다.
이 embedding matrix의 크기는 모델이 미리 정의해놓은 차원이 행, token의 갯수가 열로 이루어진다.

<br> 
<div align="center">
  <p>
    <img style="width: 70%;" alt="image" src="https://github.com/user-attachments/assets/1eac480d-3019-430c-a5de-19b0a2cdbfe1" />
  </p>
</div>

<br>

이 embedding matrix는 모델이 어느 정도 단어까지를 문맥으로 볼 것인가에 대한 embedding_dimension x context_size의 크기로 변환되고

<br> 
<div align="center">
  <p>
    <img style="width: 70%;" alt="image" src="https://github.com/user-attachments/assets/1a1cec7a-b9d4-4851-a0bd-bd677518d7f8" />
  </p>
</div>

<br>

변환된 embedding matrix의 마지막열 벡터를 사용하여 아래와 같은 unembedding matrix와 곱해(여기까지 logits) 확률(softmax 후)을 계산한다.
Unembedding matrix는 토큰갯수 x embedding_dimension의 크기를 가지고 있어 내적 후의 logits의 크기는 토큰갯수x1이다.

- unembedding matrix: **number of token** x embedding dimension
- embedding matrix 마지막열 벡터: embedding dimension x **1(last token)**
- 둘의 내적: **number of token** x **1(last token)** -> 토큰 갯수만큼의 logit값

<br> 
<div align="center">
  <p>
    <img style="width: 70%;" alt="image" src="https://github.com/user-attachments/assets/4de1dbfd-5f9a-4cc7-964d-5f552b9c05c9" />
  </p>
</div>

<br>

softmax에서 temperature는 0에 가까울수록 가장 높은 logits값의 영향이 크도록(logit의 크기를 그대로 반영), 0과 멀어질수록 가장 높은 logits값의 영향이 작도록(다른 logits도 확률에 영향을 미치도록. 높은 logits값이 그대로 적용된다기 보다는 이 영향을 어느 정도 감쇠시켜서 다양한 토큰이 output으로 출력될 수 있음) 조절한다.


# Attention
Attention(여기서는 Self-Attention. Cross-Attention도 있는데 이건 나중에 다뤄보자)은 Query, Key, Value 를 통해 모델이 각 token들의 관계와 변화량을 계산하는 과정이다. 즉, 어떤 단어가 지금 나를 Attend(주목)하고 있는지 파악하는 과정.

- Query는 나(현재 token)와 관계가 있는 token이 무엇인지 질문을 하는 역할
- Key는 어떤 token이 현재 token과 연관이 있는지 Query에 대한 답을 하는 역할
- Value는 Key에 담긴 의미를 나타내어, 기존 token의 embedding에 어떤 의미가 얼만큼 추가되어야 하는지 업데이트 하는 역할 

Q와 K는 관계(연관성)를 계산하기 위한 것이고, V는 실제 정보를 전달하기 위한 것이다.

$W_Q$ 는 embedding space에 있는 token vector를 query space로 옮겨주는 역할을 한다. $W_K$, $W_V$  는 key space, value space로 옮겨주는 역할을 한다.

$$ E_{1} \times W_Q = Q_{1}, \quad E_{1} \times W_K = K_{1}, \quad E_{1} \times W_V = V_{1} $$

$Q_{1}$ 은 전체 토큰에 대해 만들어진 Query 행렬이며 각 행인 $i$번째 토큰의 Query 벡터는 $Q_{1}[i]$ 으로 표현할 수 있을 것(예시일뿐)이다. 즉, $Q_{1}$ 자체는 하나의 토큰을 나타내는 것이 아니라 모든 토큰에 대한 Query 벡터들의 집합이다. K, V도 마찬가지.

**예시**
- Q1: 각 토큰에 대한 모든 query를 담고 있는 행렬
- Q[i]: “나는 이런 특징과 관련 있는 token의 정보를 받아오고 싶어.”
- K[j]: “나는 이런 특징을 가진 token이야.”
- Q[i]·K[j]: “그럼 우리 둘이 얼마나 관련 있음?”
- Value(V[j]): “내가 가진 실제 정보는 이거야. 네가 준 가중치(Q[i]·K[j])만큼 반영해줄게!”

<br> 
<div align="center">
  <p>
    <img style="width: 70%;" alt="image" src="https://github.com/user-attachments/assets/4bbac9c7-99e5-42d0-b5f1-2dcad41b33ce" />
  </p>
</div>

<br>

token 사이의 관계는 Query와 Key의 내적으로 계산할 수 있고, 내적이 크면 두 token은 깊은 관계를 가지고 있는 것. 그리고 이 내적값은 softmax를 통해 확률값으로 변환할 수 있다.


<br> 
<div align="center">
  <p>
    <img style="width: 70%;" alt="image" src="https://github.com/user-attachments/assets/421dc06f-6f37-4e21-b35f-82057dafe0dc" />
  </p>
</div>

<br>

추가적으로, Transformer는 RNN과 달리 전체 sequence를 한 번에 병렬로 처리할 수 있다는 것이 큰 장점이다. 하지만 decoder에서 학습 시, 뒤에 나올 단어를 이미 알면 학습에 의미가 없으므로 future token을 가린 상태(masking)로 Attention을 수행한다. 이때 Q와 K의 내적값에서 뒤의 단어에 해당하는 부분에 -Inf를 넣어 softmax 확률이 0이 되도록 하여, 해당 단어가 사용되지 않도록 한다.

<br> 
<div align="center">
  <p>
    <img style="width: 70%;" alt="image" src="https://github.com/user-attachments/assets/d5fea525-8c8f-4063-9c76-81a83bc608ba" />
  </p>
</div>

<br>

Value vector는 문맥 정보를 잘 담은 최종 token을 만들기 위해 다른 token에서 더해져야 할 정보를 담고 있는 벡터로, 기존 embedding이 얼마나 변해야 하는지를 나타낸다. Q,K가 내적된 후 softmax를 통해 확률값으로 변하면 이 확률값은 Value를 얼만큼 적용할지를 결정하는 가중치 역할을 하게 되고,
이 Value가 기존 embedding vector를 얼만큼 바꿀지를 결정하게 되는것이다.

<br> 
<div align="center">
  <p>
    <img style="width: 70%;" alt="image" src="https://github.com/user-attachments/assets/ce1e22d4-c098-4cc2-940d-99845af0201b" />
  </p>
</div>

<br>

따라서, 이와 같이 모델에 의해 학습되는 $W_{Q1}$, $W_{K1}$, $W_{V1}$ -> $Q_{1}$, $K_{1}$, $V_{1}$ 을 한 세트로 한 가지 관점에서 token의 관계를 파악하는 것을 Single Head Attention이라고 하고, 각기 다른 $W_{Q}$, $W_{K}$ , $W_{V}$ 를 병렬로 배치하여
다양한 문맥과 관계를 학습할 수 있도록 하는 것을 Multi-Head Attention이라고 한다.

Sequence of token embeddings: $E = [e_1, e_2, \dots, e_n]$

Dimension: $d_{\text{model}}$


| Single | Multi |
|:--:|:--:|
| <img src="https://github.com/user-attachments/assets/cb9a42da-afe5-4d98-b526-a5452af6df17" width="70%"> | <img src="https://github.com/user-attachments/assets/fcac310c-d051-4e7e-91aa-bae81bf329e3" width="70%"> |

# MultiLayer Perceptron
Attention이 문맥 정보를 계산한다면, MLP는 문맥 정보를 기반으로 중요한 feature를 더 강화한다. MLP는 Attention의 output을 입력으로 받아 linear, R(G)eLU, linear 연산 후 입력을 다시 더해주는(skip connection) 방법을 사용한다.

예를 들어, Attention의 output에서 마이클 조던이라는 이름에 대한 정보를 담고 있는 임베딩 벡터하나를 MLP에 넣으면 여러 연산을 통해 농구와 관련된 임베딩 벡터가 더 강화되고 이걸 다시 마이클 조던이라는 input과 더해서 마이클 조던이 농구와 관련되었다는 정보를 더 강화하게 된다. 마이클 조던과 농구가 관련 있다는 것은 이미 Attention에서 문맥 정보를 파악하며 이루어졌고, MLP는 이렇게 중요한 정보를 비선형 연산을 통해 더 강화하는 역할을 한다. 이런 과정이 모든 토큰 벡터에 똑같이 적용된다.

<br> 
<div align="center">
  <p>
    <img style="width: 70%;" alt="image" src="https://github.com/user-attachments/assets/66ebf2d2-4c82-46c0-a843-a405189dffdc" />
  </p>
</div>

<br>

LLM은 이렇게 text를 입력으로 받아 Attention으로 문맥을 파악하고, MultiLayer Perceptrion에서 중요한 토큰은 더 강화하고, 중요하지 않은 토큰은 약화시키는 과정이 여러번 반복되어 다음 단어를 예측하는 구조로 동작한다.

특히, 요즘에는 MLP의 각 차원이 하나의 의미만 담당하는 것이 아니라(마이클 조던이 마이클 조던의 이름을 넘어 농구, 시카고 불스, 나이키, 23번, 은퇴 등 훨씬 더 많은 정보를 담고 있도록), 여러 의미가 희소하게 겹쳐져 인코딩되는 'Super Position'이라는 개념이 주목을 받아, LLM이 제한된 파라미터로 방대한 지식을 저장할 수 있도록 하고, 이 구조를 해석하려는 interpretability 연구가 활발히 이루어지고 있다.

Transformer와 MLP 학습에는 아래 자료를 참고했다.

- [트랜스포머, ChatGPT가 트랜스포머로 만들어졌죠. - DL5](https://www.youtube.com/watch?v=g38aoGttLhI)
- [그 이름도 유명한 어텐션, 이 영상만 보면 이해 완료! - DL6](https://www.youtube.com/watch?v=_Z3rXeJahMs)
- [수많은 정보는 LLM 모델 속 어디에 저장되어있는걸까? - DL 7](https://www.youtube.com/watch?v=zHQLPJ8-9Qc)
