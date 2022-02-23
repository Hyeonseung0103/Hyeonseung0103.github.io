---
layout: single
title: "Section2 Sprint1 Note 212 Multiple Linear Regression"
category: Section2
---

다중 선형 회귀, 편향, 분산, 다중 공선성, 여러가지 회귀지표 등에 대해 학습했다.

## Note 212
### 다중 선형 회귀(Multiple Linear Regression)
2개 이상의 독립변수의 변화에 따른 종속변수의 변화를 정량적으로 파악한 것. 

2개 이상의 독립변수를 사용하여 종속변수를 가장 잘 예측하는(데이터를 가장 잘 설명할 수 있는) 회귀식을 도출하는 것.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X_train = df_train[['bathrooms', 'sqft_living']]
y_train = df_train['price']
X_test = df_test[['bathrooms', 'sqft_living']]
y_test = df_test['price']

lm = LinearRegression()
lm.fit(X_train,y_train)

y_pred = lm.predict(X_test)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print(mse,mae,rmse,r2)
```

단순 회귀와의 차이는 독립변수가 많아 진다는 것인데 변수가 많아지면 타겟을 예측하기 위한 "힌트,정보"가 많아지고 정보가 많아지면 모델의 설명력이 높아진다는 장점이 있다.
하지만, 동시에 모델의 복잡도가 증가해서 과적합이 생길 수도 있다.

### 다중공선성(Multicollinearity)
선형회귀분석은 선형성(독립변수와 종속변수), 독립성(독립변수끼리), 등분산성(오차항끼리), 정규성(오차항끼리)의 기본 가정이 필요하다.

그 중 다중공선성은 독립변수들끼리 큰 상관관계에 있는 것으로 독립성을 만족해야한다는 가정에 위배된다. 보통 상관분석에서 독립변수들끼리 상관계수가 0.9이상이면 다중공선성 문제가 있다고 한다.

다중공선성은 상관분석을 통해서는 상관계수가 0.9이상 이거나, 

회귀분석을 통해서는 공차한계(Tolerance)가 0.1보다 작거나, 혹은 분산팽창요인(VIF, Variance Inflation Factor)가 10 이상일 때 문제가 된다고 판단한다.

### train, test data
전체 데이터를 train data와 test data로 분리하는 이유는 모델의 성능을 확인하기 위함이다. 학습 데이터를 통해 학습된 모델을 테스트 데이터에서 성능을 평가함으로써 모델이 과적합인지, 과소적합인지
등을 확인할 수 있다. 만약 원하는 성능에 미치지 못한다면 여러가지 방법으로 모델을 재학습시켜서 성능을 올릴 수 있다. 

일반화는 모델이 한 번도 보지 못한 데이터에서도 기존 데이터에서와 비슷한 성능을 낼 수 있는 능력을 말한다. 즉, 어느 데이터를 가져다줘도 성능이 큰 차이가 없는 것이다. 모델을 최적화함으로 일반화
시킬 수 있다. 최적화된 모델이라는 것은 오차를 줄인 모델일 것이기 때문에 다른 데이터에서도 비슷한 성능을 낼 것으로 기대할 수 있다.

일반화를 하는 방법
- 규제(Regularization)을 통해 모델의 복잡도를 줄인다.
- 데이터의 양을 늘린다. 데이터를 늘릴 수록 모수와 가까워지니까(큰 수의 법칙) 과적합을 해소할 수 있다.
- 편중되어 있는 데이터를 균형있게 일반화 시킨다. 정규화 같은 방법 사용. 이상치의 영향 최소화.

### 편향과 분산(Bias and Variance)

![image](https://user-images.githubusercontent.com/97672187/155313170-1ecf8346-7310-4b32-94d9-0f4a7b7a56d3.png)

출처: http://scott.fortmann-roe.com

편향은 정답에 치우쳐져 있는 정도, 분산은 정답에 모여 있는 정도이다. 편향이 크다는 것은 정답으로부터 멀리 치우쳐져 있다는 뜻이므로, 모델의 학습자체가 잘못되었다고 할 수 있다. 분산이 큰 것은
예측값들이 여기저기 흩어져 있는 것이다. 따라서, 편향이 작고 분산이 작은 모델이 가장 좋은 모델이지만 편향과 분산이 모두 작은 모델을 만드는 것은 힘들다.
편향을 줄이면 분산이 커지고, 분산을 줄이면 편향이 커지기 때문이다. 이를 편향과 분산의 trade-off 라고 한다. 

학습 데이터는 잘 예측하는데(편향이 작음) 테스트 데이터에서는 예측을 잘 하지 못하는(분산이 큼) 경우를 과적합,
학습과 테스트 데이터 모두 비슷한 성능을 내긴 하지만(분산이 작음), 성능이 좋지 않은 것. 즉, 학습 데이터 자체를 잘 예측하지 못한 경우(편향이 큼)를 과소적합이라고 한다.

과적합 = 분산이 큼
과소적합 = 편향이 큼

에러를 나타내는 식은 편향과 분산의 합으로 표현 될 수 있는데 결국 편향과 분산을 줄이는 것이 오차를 최소화 하는 것이 되는 것이고, 오차가 최소화 된 모델이 최적화 된 모델, 즉 일반화가 잘 된 모델이라고 할 수 있다.

### 회귀지표(MSE, RMSE, MAE, R2)
MSE(Mean Squared Error): 평균제곱오차 = 오차 제곱의 합의 평균, 오차를 제곱하기 때문에 오차가 큰 값은 기존단위보다 더 큰 scale을 가지게 된다. 오차가 큰 값에 더 가중치를 부여한 느낌.

RMSE(Root Mean Squared Error): MSE에 루트를 씌운 것. MSE는 제곱을 통해 기존 데이터에 비해 단위가 커졌기때문에 이 단위를 원래 scale로 돌리기 위해 root를 씌움.

MAE(Mean Absolute Error): 평균절대오차 = 오차 절댓값의 평균, MSE와는 달리 제곱을 하지 않고 절댓값을 씌웠기 때문에 기존 데이터와 똑같은 단위로 비교가 가능하다.

R2(R squared): 1-(오차 제곱의 합 / 실제값과 평균의 차이 제곱의 합) = 데이터를 설명할 수 있는 능력. 
0에 가까울 수록 오차의 제곱합이 실제값과 평균의 차이보다 크다는 것이니까 데이터를 잘 설명하지 못하는 것. 1에 가까울 수록 데이터를 잘 설명하는 것.

### OLS함수를 사용한 다중 선형회귀

OLS 함수를 사용하면 model을 summary한 결과를 볼 수 있다. sklearn패키지와 statsmodel 패키지 중 더 편한 것 사용하면 될듯.
결과는 똑같았다.

```python
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

model = sm.OLS.from_formula("price ~ sqft_living + grade + sqft_living_rank + grade_rank", data = df_train)
X_test3 = df_test[['sqft_living', 'grade', 'sqft_living_rank', 'grade_rank']]
result = model.fit()
print(result.summary())
y_pred = result.predict(X_test3)

print(mean_squared_error(df_test['price'], y_pred) ** 0.5)
print(r2_score(df_test['price'], y_pred))
```

### 한줄요약

머신러닝모델을 만들 때 학습과 테스트 데이터를 분리 해야 하는 이유: 한 번도 보지 못한 데이터에서 모델의 성능을 평가하기 위해

다중선형회귀: 2개 이상의 독립변수를 통해 종속변수를 가장 잘 예측하는 회귀식을 도출하는 기법.

과적합/과소적합을 일반화 관점에서 설명: 과적합은 학습 데이터에서는 예측을 잘 하지만, 테스트 데이터에서 예측을 잘 하지 못한 경우를 말하는데 어느 데이터에서든지 모델이 좋은 성능을 유지하는 능력인
일반화가 잘 되지 않았다고 할 수 있다. 과소적합은 학습 데이터에서도, 테스트 데이터에서도 성능이 좋지 않는 것인데 일반화는 어떤 데이터든 좋은 성능을 유지해야 하니까 과소적합 또한 일반화가 잘 되지
않은 상태이다.

편향/분산의 트레이트오프 개념을 이해하고 일반화 관점에서 설명: 편향이 크면 데이터가 정답에서 멀리 떨어져있다는 것이기 때문에 좋은 성능을 낼 수가 없다. 또한 분산이 크면 데이터가 많이 흩어져
있는 것이기 때문에 데이터가 많이 흩어져있는 것이기 때문에 이것 역시 좋은 성능을 낼 수 없다. 따라서 편향과 분산이 모두 작은 모델이 좋은 모델이지만, 편향과 분산은 한쪽을 줄이면 한쪽이 커지는
trade-off 관계에 있다. trade off 관계는 학습 데이터에서 편향을 줄이면 테스트 데이터에서는 분산이 커져있고, 분산을 줄이면 편향이 커지는 관계를 말한다. 모델을 최적화 시킨다는 것은 결국 편향과
분산을 줄여서 오차를 최소화 시키는 것이기 때문에 일반화가 잘 되도록 하는 과정이라고도 해석할 수 있다. 데이터가 정답에 가까이 있고, 잘 모여있을 수록(편향과 분산이 모두 작은 것) 일반화가 잘 된
모델이다.
