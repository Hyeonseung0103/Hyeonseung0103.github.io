---
layout: single
title: "Section2 Sprint1 Note 211 Simple Linear Regression"
categories: Section2
toc: true
toc_sticky: true
---

이번 포스팅에서는 단순 선형 회귀에 대해 배웠다. 기존에 알고 있던 개념이지만, 나의 언어로 다시 적어보자.

### 단순 선형 회귀(Simple Linear Regression)
회귀란, 독립변수에 따른 종속변수의 변화를 정량적으로 파악하는 것을 말한다. 독립변수와 종속변수가 서로 어떠한 관계를 나타내는 가를 회귀식을 통해 도출해 낼 수 있는데 단순 선형 회귀는
종속변수(반응 변수, target, label, 정답, 예측하고자 하는 값)가 1개, 독립변수(예측변수, 특성, 설명) 가 1개인 회귀를 말한다. 다중 선형 회귀는 종속변수가 1개, 독립변수가 2개 이상인 회귀를 말한다.

선형(선선선) 회귀는 주어진 데이터를 가장 잘 설명할 수 있는 직선을 그리는 것이라고 할 수 있다. 이 직선은 독립변수와 종속변수의 관계를 요약해준다.

데이터를 가장 잘 설명한다는 것은 데이터와 가장 비슷하다는 것을 의미하고, 데이터와 가장 비슷하기 위해서는 오차가 최소화 되어야 한다.
이 오차를 계산하는 데에는 여러가지 방법이 있겠지만, 주로 최소제곱법이라는 기법을 사용해 오차가 최소화 되는 직선을 찾는다.

TMI: 잔차는 표본에서 실제값과 예측값의 차이를 말하고, 오차는 모집단에서 실제값과 예측값의 차이를 말함. 의미는 비슷하나 표본이냐 모집단이냐의 차이.

선형 회귀를 통해 보간과 외삽을 할 수 있다. 보간은 주어진 데이터 내의 범위에서 새로운 관측치가 생겼을 때 그 관측치의 타겟값을 회귀식을 통해 예측하는 것을 말 하고, 외삽은 주어진 데이터 
범위 밖의 관측치가 새로 생겼을 때 그 관측치의 타겟값을 예측하는 것을 말한다. 데이터 범위 안이든 밖이든 새로운 값이 주어져도 우린 회귀식을 통해 알고자 하는 값을 예측할 수 있게 된다.

또한, 기울기를 통해 데이터의 방향이 다른 데이터에 따라 어떻게 변하는지 증감 방향도 확인 할 수 있다.

하지만 외삽의 경우, 범위 밖의 데이터의 분포가 어떻게 이루어져있는지 모르는데 새로운 타겟을 기존 데이터에 의존한(기존 회귀식) 결과로만 예측할 수 밖에 없다는 문제점이 존재한다.

### 최소제곱법 (Ordinary least squares, OLS, Least Squared Method)
위에서 데이터를 가장 잘 설명하는 직선을 찾기 위해 최소제곱법을 사용한다고 했다. 최소제곱법은 오차의 제곱합을 최소로 하는 직선을 찾는 기법이다. 
오차의 절댓값을 사용할 수도 있는데 오차의 제곱합을 사용하는 이유는 오차가 큰 값에 더 큰 가중치를 부여하기 위함이다(제곱을 하면 값이 커질수록 scale이 더 커지니까).

![image](https://user-images.githubusercontent.com/97672187/155123309-a43624d9-59f4-4107-a203-dea3db77f314.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/155123346-3f725dab-818c-4737-bf1b-3b594f91fd9d.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/155123365-49020b96-2a98-44ba-87a1-ddea4cea8a4c.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/155123394-cb1c71c4-b0ac-4be2-9b5e-cbeef482a387.png){: .align-center}

이미지 출처: https://terms.naver.com/entry.naver?cid=58944&docId=3569970&categoryId=58970

우리가 직선을 그릴 때는 직선의 기울기(coefficient)와 y절편(intercept)이 필요하다. 따라서 단순선형회귀에서 최소제곱법은 오차의 제곱합이 최소가 되는 회귀 계수 2개(기울기, y 절편)를 
찾는 기법이라고 할 수 있다. 위의 식을 보면 오차를 최소화 시키려면 편미분 값이 0이 되어야 한다(경사하강법에서 사용하는 손실함수를 생각하면 기울기가 0이 되는 부분(볼록한 부분의 꼭짓점)
이 오차가 최소가 되는 지점이다). 여러 유도과정을 거쳐 마지막 식처럼 오차 제곱합을 최소화 시키는 기울기  **_a_** 와  y절편 **_b_** 를 구할 수 있다.

최소제곱법은 위와 같은 방법으로 오차 제곱합이 최소가 되는 직선을 찾아서 데이터를 가장 잘 설명하는 회귀식을 도출해낸다.

### 기준모델(Baseline Model)
가장 간단하면서, 최소한의 성능을 나타내는 기준이 되는 모델. 이것보다는 잘 만들어야 한다는 지표가 된다.
선형 회귀에서는 주로 평균을 기준 모델로 하는데, 꼭 정해져 있는 것이 아니다.
다른 회귀 모델에서는 선형 회귀 자체를 기준 모델로 정하기도 한다.

보통 기준모델은 다음과 같이 설정한다.

회귀: 타겟의 평균값

분류: 타겟의 최빈값

시계열회귀: 이전 타임스탬프의 값

이 기준 모델의 성능을 측정해, 내가 새로 만든 모델과 성능을 비교한다. 회귀에서는 mse나 mae, 분류에서는 accuracy가 되겠지.

### Scikit-learn 패키지를 사용한 선형 회귀
```python
import pandas as pd
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression
# 경고메세지 끄기
warnings.filterwarnings(action='ignore')

df = pd.read_csv('https://ds-lecture-data.s3.ap-northeast-2.amazonaws.com/kc_house_data/kc_house_data.csv')
model = LinearRegression()
#다중 회귀하면 X_train에 data frame이 들어가는데 이건 단순회귀라 1차원 변수만 들어가니까
#[]를 하나 더해서 2차원으로 만들어줌.
X_train = df[['sqft_living']]
y_train = df['price']
model.fit(X_train, y_train)

X_test = [[15000]] # 15000이라는 새로운 값 예측
y_pred = model.predict(X_test) #모델에서 도출한 회귀식으로 새로운 y값 예측
print(round(y_pred[0],0)) #예측값

print(model.coef_[0]) #회귀식의 기울기. 데이터가 1 증가할 때 얼만큼 변화하는가
print(model.intercept_) #회귀식의 y절편
```

### 비용함수(Cost function)
실제값과 예측값의 차이(오차)를 계산하는 함수. 이 비용함수를 통해 오차가 가장 적은 모델을 찾게 된다.
최소제곱법에서 사용하는 RSS(Residual Sum of Squares,오차의 제곱합)도 비용함수.

학습 목표

- 선형회귀모델을 이해한다. -> 독립변수와 종속변수의 관계를 가장 잘 나타내는(오차의 제곱합이 최소가 되는) 회귀식을 도출해내는 모델. 이 모델을 통해 보간, 외삽을 할 수 있다. 

- 지도학습(Supervised Learning)을 이해한다. -> 데이터에게 정답을 알려주며 학습 시키는 것. 관측치에 정답을 매칭시키며 훈련시켜서 새로운 관측치가 들어오더라도 기존 학습을 바탕으로 
연속형 또는 범주형의 타겟을 예측할 수 있다.

- 회귀모델에 기준모델을 설정할 수 있다. -> 회귀모델의 기준은 타겟의 평균, 분류는 타겟의 최빈값, 시계열 회귀는, 이전 타임스탬프의 값.


### 1. 단순선형회귀모델을 만들기 위해 전제되어야 하는 조건

조건1. 두 변수는 선형관계에 있어야 한다. = 종속변수와 독립변수는 연관이 있어야 한다.

조건2. 값이 다른 데이터가 2개 이상이어야한다. = 직선을 그리기 위해서는 최소 2개의 점이 있어야 한다.

조건3. 표본은 랜덤하게 추출되어야 한다. = 규칙을 찾아서 추출하면 직선 위에 아예 핏하게 되는 것이고, 오차가 0이 된다. 그럼 예측을 하는 의미가 없어짐. 과적합이 발생할 수도 있고.

조건4. 오차의 평균은 0이 되어야 한다. = 오차들이 회귀선을 중심으로 대칭적으로 분포되어있어야 한다.

조건5. 오차들은 관측치에 대해서 등분산을 가져야한다. = 각 독립변수 X의 값에 있어서 종속변수 Y가 평균으로부터 떨어져있는 분포가 같아야한다.

조건 6. 오차끼리는 독립이어야 한다. = 오차끼리 연관이 있으면 안 된다. = 오차끼리의 공분산이 0이어야 한다.

조건 7. 오차들의 분포는 정규 분포를 이뤄야한다.

### 2. OLS란?

OLS는 Ordinary Least Square, Least Square Method라고 불리며 최소제곱법이라고 한다. 이는 오차의 제곱합이 최소가 되는 회귀식을 찾는 것이다. 왜냐하면 데이터가 랜덤하게 분포해 있으면 이 관측치들을 모두 지나는 직선을 그리기 힘들기 때문에 직선이 실제 데이터들의 분포를 가장 잘 설명할 수 있는, 즉, 오차가 최소가 되는 직선을 찾아야한다. 오차가 최소가 되는 직선을 찾으려면 관측치와 회귀선의 차이의 합이 최소가 되는 직선의 식을 구해야 하므로 우린 결국 오차의 제곱합이 최소가 되는 기울기와 y절편을 찾아야한다.

기울기는 여러 증명과정을 거쳐 두 변수(독립,종속)들의 평균의 차이를 곱한 것의 합을 분자, 독립변수의 평균과 실제값의 차이의 제곱합을 분모로 한다. y 절편은 종속변수의 값과 방금 구한 기울기와 독립변수의 값을 곱한 값의 차이로 구할 수 있다.

한 줄 요약. OLS는 오차의 제곱합이 최소가 되는 회귀식을 찾는 기법.
