---
layout: single
title: "Activation Steering"
toc: true
toc_sticky: true
category: Interpretability
---

기존에는 LLM의 행동을 제어하는 방법은 주로 프롬프팅이었다. 사용자가 원하는 스타일로 답변받으려면 프롬프트를 정교하게 작성해야 했고, 이를 따를지 말지는 온전히 모델의 선택이었다. 하지만 최근 주목받는 Activation Steering 기법은 이 패러다임을 바꾼다.

Steering(조종, 조작) Vector는 모델의 internal activation에 직접 개입하여 모델의 행동을 수학적으로 강제하는 기법이다. 프롬프팅처럼 '부탁하는' 방식이 아니라, 내부 표현 공간에서 특정 방향 벡터를 더함으로써 모델이 반드시 원하는 방식으로 동작하도록 만드는 것이다.

이 기법은 단순한 제어 방법을 넘어 **Interpretability(해석가능성)**의 관점에서도 큰 강점을 가진다. 모델의 내부 활성화를 직접 조작한다는 것은, 동시에 모델이 특정 개념을 어떻게 표현하고 처리하는지를 명확하게 관찰할 수 있다는 의미다. 예를 들어, Love﻿ 벡터에서 Hate﻿ 벡터를 뺀 결과인 steering vector는 모델 내부에서 '긍정성'이 어떻게 인코딩되어 있는지를 직접 보여준다. 이는 단순히 출력을 설명하는 XAI 방식과는 달리, 모델의 '생각 과정' 자체를 벡터 공간에서 들여다보는 것이다. 모델이 왜 특정 결과를 생성했는지 사후 분석하는 것이 아니라, 모델이 내부적으로 어떤 개념적 방향을 따라 이동하는지를 실시간으로 관찰하고 조작할 수 있게 되는 것이다.

### Steering Vector
model내 특정 layer의 activation의 출력에 steering vector를 더해주면 출력을 의도한대로 변형할 수 있다. 예를 들어, Love와 Hate사이의 steering vector를 구하고
이를 I dislike the cars에 더하면 I like cars에 가까운 문장으로 변하는 것이다.

신경망이 개발된 초기에는 각 뉴런이 하나의 특징에만 반응한다고 생각했다(e.g., 이미지에서 공이 있을 때만 활성화, 축구화가 있을 때만 활성화).
하지만, polysemanticity(다의성) 특징에 의해 한 뉴런이 여러 개의 완전히 다른 개념에 반응한다(e.g., 공이 있을때도, 축구화가 있을 때도, 골키퍼가 있을 때도 활성화)는 특징을 발견했고,
이는 신경망은 뉴런 갯수보다 훨씬 많은 특징을 담아낸다는 것을 발견했다.

공, 축구화, 긍정, 부정과 같은 모델이 표현하고자 하는 추상적인 것을 feature라고 한다면 이 복잡한 feature들을 지정된 activation(hidden layer를 통해 계산되고 학습되는 수치들)에 압축하여 저장하고 있는 것이다.
이 feature는 추상적인 개념으로 관련된 activation이 활성화되었을 때 나타내는 방향이라고 표현할 수 있고 따라서 steering vector를 activation에 더해주면서 출력을 원하는 방향으로 조종할 수 있다.


참고한 자료
- [Implementing activation steering](https://www.lesswrong.com/posts/ndyngghzFY388Dnew/implementing-activation-steering)
- [Steering vectors: tailor LLMs without training. Part I: Theory (Interpretability Series)](https://www.youtube.com/watch?v=cp-YSyc5aW8)
