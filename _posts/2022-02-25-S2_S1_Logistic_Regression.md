---
layout: single
title: "Section2 Sprint1 Note 214 Logistic Regression, Validation set, Scaler"
category: Section2
---

분류모델 중 하나인 Logistic Regression에 대해 배웠다. 또한, 검증 데이터의 중요성을 알 수 있었고, 다양한 Scaler에 대해 공부했다.

## Note 214
### Train, Test and Validation Set
학습데이터는 모델을 학습시키기 위한 데이터, 테스트 데이터는 학습된 모델의 성능을 평가하기 위한 데이터이다. 우리는 테스트 데이터의 타겟 정보를 모르는 상태여야 하는데 테스트 데이터에서 성능을 계속 측정하며 모델을 수정한다면, 올바르게 일반화 능력을 측정할 수 없다. 모델이 학습 데이터 뿐만 아니라 테스트 데이터에도 핏하도록 수정이 되어버렸기 때문에. 그럼 새로운 데이터가 주어졌을 때도 모델의 일반화 능력이 좋을 것이라는 보장이 없다. 이처럼 테스트 데이터에서 성능을 여러번 측정하는 것을 방지하기 위해 학습 데이터를 나누어서 검증 데이터로 사용할 수 있다. 데이터가 충분히 많다면 학습 데이터에서 검증 데이터를 떼어내면 되지만, 그렇지 않다면 학습 데이터 내에서 K개의 그룹을 만들어서 학습 그룹과 검증 그룹을 교차하며 성능을 평가하는 방법인 K Fold CV를 사용할 수도 있다. 검증 데이터는 학습에는 사용되지 않고, 테스트 데이터 대신 모델의 성능을 평가하는 지표로만 사용된다.

```python
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(df,train_size = 0.8, test_size = 0.2, random_state = 2)
X_train, X_val = train_test_split(X_train, train_size = 0.8, test_size = 0.2, random_state = 2)
y_train, y_val, y_test = X_train.pop('종속변수'), X_val.pop('종속변수'), X_test.pop('종속변수')
```

### Logistic Regression
로지스틱 회귀는 회귀를 사용하여 데이터가 어떤 범주에 속할 확률을 0에서 1사이의 값으로 예측하고, 그 확률값이 정해진 기준보다 크면 종속 변수를 1 아니면 0으로 분류해주는(이진분류) 지도학습 알고리즘이다.
일반 회귀모델을 사용하면 타겟변수값이 음수에서 양수까지 나타나므로 이 값으로는 범주를 선택하기 어렵다. 따라서 회귀식을 확률값으로 변화하여 범주를 분류하게 된다. 

![image](https://user-images.githubusercontent.com/97672187/155836321-f8543892-8762-4f23-9803-cb936895cf7c.png)

이미지 출처: http://faculty.cas.usf.edu/mbrannick/regression/Logistic.html

위와 같이 p를 구해서 범주를 분류할 수 있지만, 로지스틱 회귀의 계수는 비선형 함수 내에 있기 때문에 직관적으로 해석하기 어렵다. 따라서 실패확률에 대한 성공확률에 비율인 Odds라는 개념을 사용하고
분류문제에서 Odds는 타겟이 클래스 1 확률에 대한 클래스 0확률의 비라고 해석할 수 있다. 

![image](https://user-images.githubusercontent.com/97672187/155836444-333f3c6e-f0ca-4ad6-8979-a1650c846a71.png)

이미지 출처: http://faculty.cas.usf.edu/mbrannick/regression/Logistic.html

p는 어떤 사건이 일어날 확률, 1-p는 일어나지 않을 확률이다. 즉, p가 1일 확률이면, 1-p는 0일 확률이 된다. p가 1이면 odds 가 무한대이고, p가 0이면 odds가 0이 된다. 
Odds가 3이라는 것은 어떤 사건이 일어날 확률이 일어나지 확률의 3배라는 뜻이다. ex) 1일 확률이 0인 확률의 3배


![image](https://user-images.githubusercontent.com/97672187/155836582-a3b05d4f-2c62-456d-8293-1c3be3774854.png)

이미지 출처: https://medium.com/codex/logistic-regression-and-maximum-likelihood-estimation-function-5d8d998245f9

이 비선형형태의 회귀계수를 선형형태로 바꾸기 위해서는 Odds에 로그를 취하면 되고, 이를 로짓변환(Logit transformation)이라고 한다. 
로짓변환을 통해 다음과 같이 선형형태의 회귀식을 도출해낼 수 있고, 특성 X의 증가에 따라 로짓(ln(odds))이 얼마나 증감 했는가를 해석할 수 있다.
하지만, 이는 회귀분석처럼 타겟값의 수치가 커지고, 작아지는 것이 아니라 특정 범주에 속할 확률을 높이고 낮추는 것이로 해석해야한다. 확률 자체가 아니라 확률을 높이고 낮추는 수치가 된다.

타이타닉 데이터를 예제로 들면 타겟이 생존여부인데 로짓변환을 통해 변환시킨 회귀계수가 양수이면 생존의 가능성을 높이고, 음수이면 생존의 가능성을 낮춘다고 해석할 수 있다.



### Sklearn 패키지를 활용한 Logistic Model

범주형 데이터 인코딩, 데이터 정규화가 진행되면 좋다. 하지만, 무조건 성능이 오르는 것은 아니다.

```python
from category_encoders import OneHotEncoder
from sklearn.preprocessing import StandardScaler

encoder = OneHotEncoder(cols = ['gender'])
X_train_encoded = encoder.fit_transform(X_train)
X_val_encoded = encoder.transform(X_val)
X_test_encoded = encoder.transform(X_test)

s = StandardScaler()
X_train_scaled = s.fit_transform(X_train_encoded)
X_val_scaled = s.transform(X_val_encoded)

lg = LogisticRegression(max_iter = 1000)
lg.fit(X_train_scaled, y_train)

#score함수 쓰면 predict를 하지 않아도 성능을 측정할 수 있음.
#predict하고나서 성능 측정하고 싶으면 accuracy_score() 함수 쓰면 됨.
print(lg.score(X_train_scaled, y_train))
print(lg.score(X_val_scaled, y_val))
```

threshold를 따로 정하지 않으면 로지스틱회귀 모델은 default값인 0.5를 사용한다. threshold를 수정하고 싶으면 predict_proba()함수를 사용해서 확률값을 추출한 뒤
내가 정한 threhsold를 사용하면 된다.

```python
from sklearn.metrics import accuracy_score

lg = LogisticRegression(max_iter = 1000)
lg.fit(X_train_scaled, y_train)
threshold = np.arange(0.4,0.6,0.01)
train_score = [(0,0)]
val_score = [(0,0)]
for t in threshold:
  preds = np.where(lg.predict_proba(X_train_scaled)[:,1] > t,1,0)
  if train_score[0][1] < accuracy_score(y_train, preds):
    del train_score[0]
    train_score.append((t,accuracy_score(y_train, preds)))

  preds_val = np.where(lg.predict_proba(X_val_scaled)[:,1] > t,1,0)
  accuracy_score(y_val, preds_val)

  if val_score[0][1] < accuracy_score(y_val, preds_val):
    del val_score[0]
    val_score.append((t,accuracy_score(y_val, preds_val)))

print(train_score)
print(val_score)
```

### Outlier 제거
데이터에 이상치가 존재하면 모델이 이상치의 영향을 많이 받아서 성능이 떨어질 수도 있다. 이상치는 데이터들의 일반적인 분포 밖에 있는 값들을 의미하는데 사분위수의 상위 75%, 하위 25%의 값 차이인
IQR(Interquartile range, Q3 - Q1)을 사용하여 판단할 수 있다. 이 이상치가 오타일 수도 있고, 실제값일 수도 있어서 연구자의 주관에 따라 이상치의 기준을 정하고, 제거하거나 제거하지 않을 수 있다. 제거한다고 무조건 성능이 오르는 것은 아니다.

이상치를 제거하는 방법은 여러가지가 있지만 Z-score 패키지를 사용하면 쉽게 제거할 수 있다. Z-score는 해당 데이터가 평균에서 얼마나 벗어나 있는지 알 수 있는 지표이다.
보통 Z-score가 +- 3 범위 밖에 있으면 이상치로 확인한다.

![image](https://user-images.githubusercontent.com/97672187/155837223-93ad389d-3908-4272-8937-f36e822e1d31.png)

이미지 출처: https://medium.com/@gulcanogundur/normal-da%C4%9F%C4%B1l%C4%B1m-z-score-ve-standardizasyon-782963bc123e

```python
print(df.shape[0])
df = df.drop(df.loc[abs(ss.zscore(df[['height','weight','ap_hi','ap_lo']])) >= 3].index) #outlier가 2개이상인 행도 있다. 그래서 각 변수의 outlier들의 갯수합이랑 사라지는 값이랑 다름.
print(df.shape[0])
```

### Scale
1) 데이터의 scale 범위가 크면 노이즈가 생성되기 쉽고, 과적합이 일어나기 쉽다. 
2) 데이터의 scale 범위를 줄여줌으로써 학습을 더 빨리 시킬 수 있고, 최적화 과정에서 local optimum에 빠질 가능성을 줄여준다. 
3) Scale이 너무 크면 값의 분포가 넓기 때문에 값을 예측하기 어려워진다.

데이터마다 적합할 여러가지 정규화 방법이 있는데 크게 4가지의 정규화에 대해 다뤄보자.

1) StandardScaler: 각 열의 평균을 0, 표준편차 1을 기준으로 정규화. 각 데이터가 평균에서 몇 표준 편차만큼 떨어져있는가. 데이터의 특징을 모르는 경우 무난한 정규화 방법.

2) MinMaxScaler: 각 열의 최솟값과 최댓값을 기준으로 0 ~ 1구간 내에 균등하게 값을 배정. 이상치에 민감하지만, 각 열의 범위를 모두 0 ~ 1로 동등하게 분포를 바꿀 수 있다.

3) RobustScaler: 각 열의 median(Q2)에 해당하는 데이터를 0으로 잡고, Q1, Q3 사분위수와의 IQR차이 만큼을 기준으로 정규화. 이상치가 많은 데이터를 다루는 경우 유용하다.

4) Normalizer: 열이 아니라 행을 사용. 한 행의 모든 변수의 거리가 1이 되도록 스케일링. 일반적인 데이터 전처리보다는 딥러닝의 학습 벡터에 사용됨. 특히 피처들이 (키,나이,몸무게)와 같이 단위가 다르면 사용하지 않아야함.

```python
s = StandardScaler()
# s = RobustScaler()
X_train_scaled = s.fit_transform(X_train)
X_val_scaled = s.transform(X_val)
```

### 한줄요약

훈련/검증/테스트(train/validate/test) 데이터를 분리하는 이유는? 테스트 데이터를 여러번 사용하므로 발생하는 Data leak을 방지하고, 모델의 일반화 능력을 올바르게 측정하기 위해서

분류(classification) 문제와 회귀문제의 차이점은? 분류는 타겟이 범주형이기 때문에 타겟이 특정 범주에 속할 확률을 기반으로 타겟값을 예측한다. 회귀문제는 타겟이 연속형이기 때문에 연속적인 수치형 값으로 타겟을 예측한다.

로지스틱회귀(Logistic regression)란? 회귀를 사용하여 독립변수를 통해 타겟이 특정 범주에 속할 확률을 계산하고, 확률이 임계치보다 크면 1, 작으면 0으로 분류하는 지도학습 알고리즘이다.
