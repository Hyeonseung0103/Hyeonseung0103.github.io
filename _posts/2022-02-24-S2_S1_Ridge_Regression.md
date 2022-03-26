---
layout: single
title: "Section2 Sprint1 Note 213 Ridge Regression, Select Best Number of K Features"
category: Section2
toc: true
toc_sticky: true
---

다중 선형회귀에서 과적합이 발생하는 경우 회귀계수에 패널티를 부과해서 과적합을 해소하는 Ridge Regression에 대해 배웠다. 또한, 가장 적절한 변수의 갯수를 선택하는 방법도 알게 되었다.

### OHE(One Hot Encoding)
범주형 데이터는 대소관계가 있는 순서형과, 대소관계가 없는 명목형으로 나눌 수 있다. 머신러닝에서 변수가 텍스트 형식으로 되어있으면 컴퓨터가 인식할 수 없기 때문에 모델을 학습 할 수 없다.
따라서, 이를 숫자로 인코딩 해줘야지만 컴퓨터가 인식하고 학습할 수 있는데 여러가지 인코딩 방법 중 원핫인코딩은 범주형 변수를 범주의 수만큼 차원을 생성하여 해당하는 범주의 변수에는 1 
나머지 범주의 변수는 0으로 표시하는 희소표현 방법이다. 

1) get_dummies를 사용한 OHE

```python
import pandas as pd
pd.get_dummies(df, prefix=['변수']) #option에 drop_first = True하면 범주의 첫번째열은 제거 됨. 다중공선성 문제 해소
```

2) category_encoders를 사용한 OHE

```python
encoder = OneHotEncoder(use_cat_names = True) #범주 이름들 사용
X_train = encoder.fit_transform(X_train) #fit_transform은 학습 데이터에서 사용
X_test = encoder.transform(X_test) #transform은 학습 데이터에서 사용된 기준을 test 데이터에 적용하게 함. test 데이터에서 새로운 기준을 생성하여 인코딩 하는 Data leak의 문제를 막음.
```

원핫 인코딩은 여러 패키지들을 사용하여 범주형 변수를 간단하게 인코딩 할 수 있다는 장점이 있지만, 값이 0인 변수들을 통해 값이 1인 변수를 예측할 수 
있기 때문에 선형 관계가 생기고 독립변수들 간에 상관관계가 높다는 다중공선성의 문제를 야기한다. 다중 공선성 문제를 해결하기 위해서 데이터에서 해당 범주에 관련된 변수 하나를 삭제하는 방법을 사용할
수 있다. 변수 하나를 삭제해도 나머지 변수들의 0,1값에 의해서 없어진 변수의 값이 원래 무엇인지 파악할 수 있기 때문이다. 없어진 변수로 인해 때문에 독립 변수들간의 상관성을 낮출 수 있다.

또한, 범주의 수만큼 차원이 커진다는 문제가 있는데 차원이 커진다는 것은 모델이 복잡해진다는 것이고, 모델이 복잡해지면 결국 과적합의 문제가 발생한다.
범주의 차원이 증가하는 문제를 해결하기 위해서는 label encoding, count encoding, target encoding과 같은 방법을 쓸 수 있다. label encoding은 범주를 순서형으로 표현하는 방법, count encoding은 각 범주의 count를 세서 표현하는 방법, target encoding은 각 범주별 타겟의 평균으로 인코딩하는 방법이다.

하지만, target encoding 같은 경우 타겟의 정보를 예측에 사용하게 되는 것이기 때문에 Data leak의 문제가 생긴다. 따라서, 무조건 학습 데이터의 타겟 정보만 사용해야 하고, 학습 데이터의 정보만 사용해서 일어날 수 있는 과적합 문제는 smoothing이나 CV 를 통해 해소한다고 한다.

어느 데이터에서나 항상 좋은 인코딩 방법은 없고, 데이터에 따라 더 좋은 인코딩이 무엇인지 알고 쓰는게 중요할 것 같다.

### 특성선택(Feature selection)
특성선택은 데이터에서 필요한 변수만 선택함으로써 차원은 줄이고, 유의미한 성능은 유지시킬 수 있게 한다.

SelectionKBest 함수는 변수 갯수 K에 따라 점수를 매겨서 가장 중요한 변수를 선택할 수 있게 된다. 
이를 이용하면 K에 따라 모델의 성능을 비교해서 최적의 K는 몇 개 인지 알 수 있게된다. 하지만 이게 무조건 최종모델에서도 성능이 높을 거라는 보장은 없다.

SelectionKBest함수는 단순히 타겟과 독립변수의 상관 계수가 높은 변수를 택하는 것이 아니라 f-stat, p-value도 사용해서 결정한다.

회귀분석에서의 좋은특성은 독립변수들끼리 상관성이 낮고, 종속변수와는 상관성이 높은 특성

```python
from sklearn.linear_model import LinearRegression

training = dict()
testing = dict()

#여기서는 mae를 기준으로 K의 성능을 평가함.

for i in range(1,len(X_train.columns) + 1):
  selector = SelectKBest(score_func = f_regression, k = i)
  X_train_selected = selector.fit_transform(X_train,y_train)
  X_test_selected = selector.transform(X_test)

  #선택된 변수들
  #selected_col = X_train.columns[selector.get_support()]
  #선택되지 않은 변수들
  #만약에 선택된 변수들로 데이터 보고 싶으면 X_train.loc[:, selected_col] 이런식으로 하면 될 듯.
  
  lm = LinearRegression()
  lm.fit(X_train_selected,y_train)
  y_pred = lm.predict(X_train_selected)
  mae = mean_absolute_error(y_train,y_pred)
  training[i] = mae

  y_pred = lm.predict(X_test_selected)
  mae = mean_absolute_error(y_test,y_pred)
  testing[i] = mae
  ```
 
특성 선택에는 여러가지 방법이 있겠지만, 조사한 결과 4가지 정도로 압축할 수 있을 것 같다.

첫째로는 분산에 따른 선택이다. 분산이 기준치보다 낮은 특성은 제거하는 방법이다. 하지만, 분산에 의한 선택이 반드시 상관관계와 일치한다는 보장은 없다.

둘째로는 단일 변수 선택법이다. 변수를 하나만 사용했을 때 예측모델의 성능을 평가해서 정확도나 상관관계가 가장 좋은 특성만 선택하는 방법이다. 하지만, 좋은 단일 변수를 모았더라도 성능이 개선된다는 보장은 없다.

셋째는 모델 기반 변수 선택이다. 전체 데이터에 대해 트리 모델을 학습시켜 특성 중요도를 확인하고, 중요도가 높은 변수를 선택한다. 하지만, 이것 역시 최종 모델에서도 꼭 좋은 성능을 내는 것은 아니다.

넷째는 반복적 특성 선택이다. 모든 특성의 조합을 다 시도해보고 가장 좋은 조합을 찾는 방법이다. 보통 모든 특성을 사용하는 것부터 출발해서 지정한 특성 개수에 도달할 때가지 특성을 제거해나가며 최상의 조합을 찾는다.

어떤 방법을 사용하냐에 따라 성능이 다르겠지만, 언제나 성능이 좋은 방법은 정해져있지 않다.

### Ridge Regression
Ridge 회귀는 회귀식에서 회귀계수에 패널티를 줌으로써 특정 변수에 의해 큰 영향을 받아 모델이 과적합되는 것을 해소하기 위해 사용한다. 과적합이 일어나는 이유는 편향은 작지만 분산이 크기 때문이다.
편향과 분산은 trade off 관계에 있기 때문에 Ridge 회귀는 편향을 조금 키우는 동시에 분산을 줄이는 방법으로 정규화(Regularization, 과적합을 완화해 일반화 성능을 높여주는 기법)을 수행한다.
이 정규화의 강도는 패널티값 람다로 조절할 수 있다.

![image](https://user-images.githubusercontent.com/97672187/155529085-fc0945fd-14c3-4221-a448-fcb1e73ddc12.png)

이미지출처: https://sanghyu.tistory.com/13

위 수식을 보면 최소제곱법을 사용하는 선형회귀와 회귀계수를 구하는 공식이 유사하다. Ridge는 최소제곱법에 람다 * 기울기의 제곱합이 추가된 것.

따라서 람다가 0에 가까워질수록 선형회귀와 Ridge회귀는 같아지고, 람다가 커질수록 기울기는 0에 가까워진다. 

람다 증가 -> 다른 회귀계수들 감소 -> 회귀계수가 곧 기울기니까 기울기가 감소

-> 람다가 커지면 다른 계수들이 줄어든다. = 오차를 0으로 만드는게 좋은데 람다를 키우면 이 오차를 작게 하기 위해서는 다른 회귀계수가 줄어들어야 한다. 따라서 이 람다라는게 커질 수록 다른 계수들에 더 큰 패널티를 부여하게 되는 것.

-> 패널티가 크다. = 기울기가 작다. 독립변수의 변화에 종속변수가 덜 영향을 받는다.

-> 패널티가 큼 -> 변수의 영향력이 작아짐 -> 영향력이 0에 가까워지면 해당 변수는 거의 영향을 못 미치니까 모델이 단순화 됨. -> 과적합 해소

alpha = 람다

1) Ridge 패키지 사용

적당한 alpha, 즉 람다를 연구자가 결정함.

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=alpha, normalize=True)
ridge.fit(X_train, y_train)

#학습 데이터 성능
y_pred = model.predict(X_train)
mae = mean_absolute_error(y_train, y_pred)

#테스트 데이터 성능
y_pred2 = model.predict(X_test)
mae2 = mean_absolute_error(y_test, y_pred2)

print(mae,mae2)

```

2) RidgeCV 패키지 사용

CV를 통해 내가 정한 범위 내에서 성능이 가장 좋은 alpha를 알려줌.

여기서는 적절한 K가 무엇인지도 검증해보았다.
```python
train_mae = dict()
test_mae = dict()
train_r2 = dict()
test_r2 = dict()
for i in range(52,55):
  selector = SelectKBest(score_func = f_regression, k = i)
  X_train_selected = selector.fit_transform(X_train,y_train)
  X_test_selected = selector.transform(X_test)

  alphas = [0, 0.001, 0.01, 0.1, 1] 
  # alphas = np.arange(0,1,0.01) 

  ridge = RidgeCV(alphas=alphas, normalize=True, cv=5)
  ridge.fit(X_train_selected, y_train)
  print("alpha: ", ridge.alpha_)
  print("best score: ", ridge.best_score_)

  y_pred = ridge.predict(X_train_selected)
  mae = mean_absolute_error(y_train,y_pred)
  r2 = r2_score(y_train,y_pred)
  train_mae[i] = mae
  train_r2[i] = r2

  y_pred = ridge.predict(X_test_selected)
  mae = mean_absolute_error(y_test,y_pred)
  r2 = r2_score(y_test,y_pred)
  test_mae[i] = mae
  test_r2[i] = r2

print('\n')
print(f'학습 mae는 {sorted(train_mae.items(), key=lambda x: x[1])}')
print(f'테스트 mae는 {sorted(test_mae.items(), key=lambda x: x[1])}')
print(f'학습 r2는 {sorted(train_r2.items(), key=lambda x: -x[1])}')
print(f'테스트 r2는{sorted(test_r2.items(), key=lambda x: -x[1])}')
```

### Ridge(L2_norm) vs Lasso(L1_norm)
패널티를 부여하는 방식이 다르다.

Ridge: 연구자가 직접 판단하여 특성을 확인하고 제거한다. 중요하지 않은 독립변수들의 기울기를 0에 가까이 수렴하게 한다.

Lasso: 자동으로 특성을 선별해서 불필요한 특성을 제거한다. 중요하지 않은 변수를 아예 영향을 끼치지 못하도록 0에 수렴시켜 버린다. 

특성이 많은데 그 중 일부분만 중요하다면 Lasso, 중요도가 전체적으로 비슷하면 Ridge를 쓰는게 좋다고 한다.

### 한줄요약
OHE: 범주형 데이터를 학습시키기 위해 해당 변수를 범주의 갯수만큼 차원을 증가시켜 0,1을 사용한 희소표현으로 데이터를 인코딩한 기법.

Ridge 회귀를 통한 특성선택(Feature selection): Ridge 회귀에서 몇개의 변수를 사용해야 가장 좋은 성능을 내는지 SelectionKBest 함수를 통해 비교할 수 있다.

정규화(regularization)을 위한 Ridge 회귀모델: Ridge 회귀는 과적합을 해소하고, 일반화 능력을 증가시키기 위해 회귀계수에 패널티를 부과하여 모델이 특정 변수에 지나치게 민감하게 반응하지 않도록 한다.
