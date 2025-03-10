---
layout: single
title: "Section1 Sprint3 Note131 벡터와 행렬"
categories: Section1
toc: true
toc_sticky: true
---
선형대수의 기본인 벡터와 행렬에 대해 공부했다. 선형대수를 구성하는 기본 단위는 스칼라와 벡터이다. 또한, 행렬은 여러 벡터로 이루어져있다.

### Scalar(스칼라)
특정 크기를 표현하는 숫자
- 하나의 숫자로 이루어진 데이터
- 실수, 정수 모두 가능

### Vector(벡터)
- 주로 1차원 array로 사용되며, DF에서는 각각의 행과 열로 사용됨.
- Matrix는 벡터의 모음이다.
- 벡터의 차원은 벡터의 요소(컴포넌트)의 모음이다. 하지만 이 컴포넌트가 스칼라를 의미하진 않는다.

![image](https://user-images.githubusercontent.com/97672187/152500364-0c341518-7ad9-4a6e-abc2-5416cf8e2367.png)

위 벡터들을 보면 원래 벡터가 1차원 array로 이루어져 있으니까 벡터가 다 1차원처럼 보이지만, 벡터의 차원은 컴포넌트의 갯수라고 했기 때문에 각각 2,3,1,4차원의 벡터가 된다.

### 벡터의 크기(Magnitude, Norm, Length)

![image](https://user-images.githubusercontent.com/97672187/152500930-1b5a79f3-32b6-400f-8b69-b3188c31cda4.png)

- 모든 원소를 제곱하고 더한 후 루트를 씌우면 벡터의 크기가 된다.
- 벡터의 크기는 단순히 길이(차원)가 아니다. 따라서 크기가 0이라는 것은 모든 원소가 0인 것.
- 벡터의 크기는 스칼라값으로 표현할 수 있다 = 하나의 값이기 때문에 이 크기들을 비교할 수 있게된다.

### 벡터의 내적(Dot product)
- 벡터의 내적은 각 구성요소를 곱한 뒤 합한 값과 같다.
- 벡터의 내적의 결과는 스칼라 값이다.(행렬의 내적 결과는 행렬이다.)
- 내적을 하는 이유: 벡터는 스칼라에 방향이 더해진 것인데, 방향을 무시하고 벡터의 크기를 구할 수 있기 때문에. 이 크기를 통해 데이터를 더 잘(유의미하게) 다룰 수 있게 된다. 결국 df도 벡터로 이루어져있으니까.

![image](https://user-images.githubusercontent.com/97672187/152501339-58757289-4843-4eb2-8135-8c0913ddaa69.png)

### 행렬(Matrix)
- 행과 열을 통해 배치되어 있는 숫자들
- 행과 열의 숫자를 차원(dimension)이라 표현한다.
- Transpose는 행과 열을 바꾸는 것을 의미한다. 대각선으로 바꾸는게 아니라 행과 열을 바꿈. 1행 2열이랑 2행 1열이랑 바꿈.

### Matrix Calculation
Matrix multiplication(행렬의 곱)
- 첫번째 행렬(1x 3)의 열의 수와 두번째 행렬(3 x 4)의 열의 수가 같아야 multiplication(곱)을 할 수 있음.(공식의 안쪽값) 
- 내적 결과는 (1 x 4) 첫번째 행렬의 행, 두번째 행렬의 열. (공식의 바깥쪽 값)
- 곱할 때는 첫번째 행렬에 행 고정시키고, 두번째 행렬에는 열 고정시키고 각 요소를 다 곱하면 된다.

![image](https://user-images.githubusercontent.com/97672187/152502189-921880bc-5651-46c2-ab34-9b10a1cba9dd.png)

### 정사각 매트릭스(square matrix)
- Diagonal(대각): 대각선 부분에만 값이 있고 나머지는 전부 0

![image](https://user-images.githubusercontent.com/97672187/152502600-f20f9741-07f1-46d7-8de5-b3c933938cbf.png)

- 상삼각, 하삼각도 있는데 별로 중요하진 않음.
- Identity(단위 매트릭스): Diagonal 행렬 중에서, 모든 값이 1인 경우

![image](https://user-images.githubusercontent.com/97672187/152502741-6555f508-2484-4ff3-a24f-054aae35d2db.png)
- Symmetric(대칭): 대각선을 기준으로 위 아래의 값이 대칭인 경우

![image](https://user-images.githubusercontent.com/97672187/152502899-bc7785fa-0dc3-497d-8303-68603429e3b9.png)

### 행렬식(Determinant)
- 행렬식은 모든 정사각 매트릭스가 갖는 속성으로, det(A) 혹은 |A|로 표기된다.

![image](https://user-images.githubusercontent.com/97672187/152503248-cdc955d0-97b1-4e5a-b3c0-aca07fe99c96.png)

- 2x2 matrix에서는 ad-bc로 행렬식을 구할 수 있다.
- 3x3도 있는데 어차피 손으로 계산할 일 없으니까..

```python
import numpy as np
np.linalg.det(A) #A 행렬의 행렬식 구하는 방법
```

### 역행렬(Inverse matrix)
- 행렬에 역행렬을 곱하면 단위 행렬을 된다.
- 어떤 행렬을 단위 행렬로 만들어주는 행렬이다.
- 2차원에서의 역행렬

![image](https://user-images.githubusercontent.com/97672187/152503880-f5c06bde-b6db-4620-bdf6-f0ca52437bc0.png)

- 모든 행렬이 역행렬이 존재하는 것은 아니다.
- 위 식을 보면 알 수 있듯이 행렬식이 0이 되면 역행렬이 존재할 수 없다.
- 행렬식이 0이되는 행렬을 특이(singular) 행렬이라고 하고, 이들은 2개의 행 혹은 열이 선형 관계를 이루고 있을 때 발생한다.
- 선형의 관계란 서로 연관이 있는 관계.

![image](https://user-images.githubusercontent.com/97672187/152504063-7b8eca80-b467-4d93-b90c-61c8f6e54be9.png)

위 행렬을 보면 1열은 3열의 5배고, 2열은 각각 *2, *3를 하게됨.

```python
#역행렬 구하기
np.linalg.inv(m)
```

### Numpy를 이용한 선형대수
- List와 Array는 연산을 처리하는 방식이 다름.
- List일 때

```python
a = [1, 2, 3]
b = [4, 5, 6]

a + b # => [1,2,3,4,5,6]
3*a # => [1,2,3,1,2,3,1,2,3]
```

- Array일때

```python
a = np.array(a)
b = np.array(b)
a + b # => [5,7,9]
a * b # => [4,10,18]
```

- 따라서 Array를 쓰는것이 더 좋다.
- 내적할 때

```python
np.dot(a,b)
```

- 행과 열을 구분하는 어레이를 만들고 싶으면 2차원으로 만들면 됨. [안에 또 []]

### L1 norm, L2 norm
- 벡터의 크기를 나타내는 norm에는 두 가지 종류가 있다. 이 두 가지가 어떻게 다르게 쓰일 것인가는 나중에 다시.
- L1 norm

![image](https://user-images.githubusercontent.com/97672187/152505846-e0b34205-c77d-4bcc-835c-bcc5cbcad5ee.png)

- L2 norm

![image](https://user-images.githubusercontent.com/97672187/152505889-e38f0a60-5726-4997-ae51-2566c0504da3.png)

### MSE와 MAE
- MSE(Mean Squared Error): 평균제곱오차, 예측값과 실제값의 차이(잔차) 제곱의 평균

![image](https://user-images.githubusercontent.com/97672187/152506117-1998f097-24d2-4066-8188-5be8ec6cd00d.png)

- MAE(Mean Absolute Error): 평균절대오차, 예측값과 실제값의 차이(잔차) 절댓값의 평균

![image](https://user-images.githubusercontent.com/97672187/152506710-38ccca6f-1018-4f1c-a861-f7e73fc61b97.png)

- 두 지표모두 회귀 모델의 성능을 평가하는데 사용.
- MSE는 손실함수로 쓰여서 모델의 파라미터를 최적화 하는데 사용되고, MAE는 모델을 훈련시키는 동안 예측값의 성능을 확인하는 회귀지표로 사용됨.

### 데이터 과학과 선형대수의 관계
- 우리가 다루는 데이터는 보통 행과 열로 구성된 matrix로 이루어져있다.
- DF도 결국 행렬과 유사한 형태
- 이 매트릭스는 벡터로 이루어져 있고, 벡터는 스칼라로 이루어져있다고 할 수 있다.
- 따라서 선형대수학을 배우는 이유는 데이터를 더 잘 다루고 이해하기 위해서이다.

### Discussion
벡터화를 진행하는 이유는 무엇일까요?

답변:

프로그래밍이라는 것은 결국 컴퓨터가 인식할 수 있는 언어를 사용해 매우 큰 데이터나 복잡한 연산을 처리하는 것이라고 생각한다. 하지만, 컴퓨터는 이미지를 이미지대로, 텍스트를 텍스트대로 인식하는 것이 아니라 주어진 데이터를 숫자로 변환해야지만 인식할 수 있다. 예를 들어, 이미지는 RGB값으로 각각 빨강, 초록, 파랑색을 조합해서 여러가지 색을 표현하는데 이 색상은 숫자로 표현된다. 우리가 AI를 사용하는 가장 큰 이유는 주어진 데이터를 통해 미래를 예측할 수 있는 모델을 개발하는 것이다. 위에서 이미지는 숫자로 표현된 여러가지 색들의 조합이라고 설명했는데, 이해하기 쉽게 컴퓨터가 하나의 파란색 자동차 사진을 벡터가 아니라 이미지로 기억하고 있다고 가정해보자. 이와 똑같은 자동차이지만 다른 각도와 다른 배경에서 사진을 찍었다면, 자동차의 크기와 배경이 다르기 때문에 두 자동차는 같은 자동차임에도 다른 이미지로 예측을 할 것이다. 하지만, 이미지를 벡터화 시킨다면 해당 자동차의 크기와 배경에 상관없이 기존 자동차의 벡터와 새로운 사진의 자동차의 벡터가 유사하게 나올 것이고 우린 이를 통해 두 자동차는 같은 자동차라는 것을 알 수가 있다.

두번째 예시를 보면, 텍스트를 one-hot encoding으로 벡터화해서 표현했다. 'the'라는 단어를 벡터화한다면 새로운 문장이 들어왔을 때 이 벡터화를 기반으로 겹치는 단어가 있는지 여부를 알 수 있다. 아주 간단한 방법으로 '비교'가 가능해진다. 만약에 새로운 문장에도 'the'의 value가 1이라면 그 문장도 'the'를 포함하고 있다는 것을 알 수 있기 때문에. 하지만, One-hot encoding은 단어(categories)의 갯수만큼 차원이 증가하기 때문에 차원이 커질수록 연산 속도가 매우 느려진다는 단점이 있다. 이를 보완하기 위해 임베딩이라는 것이 존재한다고 한다.

결론: 벡터화를 통해 컴퓨터가 인식할 수 있게 함으로써 데이터를 비교할 수 있게 되고, 더 효율적인 분석이 가능해진다.



참조링크:
https://ko.wikipedia.org/wiki/%ED%96%89%EB%A0%AC_%EA%B3%B1%EC%85%88
https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=heygun&logNo=221516529668
