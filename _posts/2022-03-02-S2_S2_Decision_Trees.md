---
layout: single
title: "Section2 Sprint2 Note221 Decision Trees, Imputers"
category: Section2
---

데이터들의 분포가 비선형일 때도, 분류와 회귀 문제에서 모두 사용할 수 있는 모델 중 하나인 Decision Trees에 대해 배웠다. 또한, 결측치를 처리하는 다양한 imputer에 대해서도 알 수 있었다.

## Note221
### Pipelines
파이프라인은 코드를 단순화하여 반복되는 작업을 간단하게 처리할 수 있다. 또한, 똑같은 코드를 잘못사용해서 발생할 수 있는 데이터 누수도 방지할 수 있다는 장점이 있다.(ex) fit_transform, transform 등)

데이터 전처리와 모델링을 한 번에 연결시킬 수 있다. 

ex) OHE + Imputer + Scailing + Modeling

```python
from sklearn.pipeline import make_pipeline
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(
    OneHotEncoder(), 
    SimpleImputer(), 
    StandardScaler(), 
    LogisticRegression()
)
pipe.fit(X_train, y_train)

print(pipe.score(X_val, y_val))
y_pred = pipe.predict(X_test)
```

딕셔너리 형태로 구성되어 있는 names_steps 속성을 사용해서 학습 데이터 뿐만 아니라, 검증, 테스트 데이터만 따로도 처리가 가능해진다.

```python
import matplotlib.pyplot as plt

#검증 데이터 회귀계수 그래프 그리기.
model_lr = pipe.named_steps['logisticregression']
enc = pipe.named_steps['onehotencoder']
encoded_columns = enc.transform(X_val).columns
pd.Series(model_lr.coef_[0], encoded_columns).sort_values().plot.barh();
```

### Decision Trees(결정트리)
- 결정트리는 각 독립변수마다 조건을 부여해서 이 조건에 대해 boolean으로 대답하며 조건이 끝날 때까지 학습하여 적절한 타겟값을 의사결정하는 모델이다. 
- 조건은 독립변수의 수치나 범주의 기준으로 이루어져 있고, 이 조건이 노드(nodes)가 된다. 또 각 노드를 연결시켜주는 선을 엣지(edge)라고 한다.
- 각 노드는 최상위에 있는 root node, 중간에 있는 internal nodes, 가장 끝에 있는 external(leaf, termnal) nodes로 나누어진다.
- 분류와 회귀문제 모두 적용 가능하다.
- 독립변수의 type이 연속형이든 범주형이든 모두 결정트리 모델로 학습할 수 있지만, sklearn 패키지에서는 범주형을 수치형으로 인코딩 해줘야 학습할 수 있다.

장점

- 비선형 데이터 학습에 좋다.
- 스케일링을 사용할 필요가 없다.
- 해석에 용이하다.


단점

- 과적합이 잘 발생한다. (max_depth, min_samples_split과 같은 하이퍼 파라미터를 조정하여 해소한다.)
- 조건에 부합하지 않은 데이터는 누락되고, 시간이 오래 걸린다.
- 탐욕 알고리즘의 결과가 된다. = 최적의 결과는 아니고, 적절한 결과를 도출한다.
- 외삽을 하지 못한다. = 학습에 사용한 데이터 범위 밖의 값을 잘 예측하지 못한다.

**탐욕 알고리즘** : 여러 경우 중 하나를 결정해야 할 때마다 그 순간에 최적이라고 생각되는 것을 선택해 나가는 방식으로 최종적인 해답에 도달 하는 것. 
순간의 최선이 꼭 전체의 최선이 되진 않기 떄문에 이 해답이 최적의 결과라는 보장이 없다. 지역적 최선 but, 전역적 최선 X.

### 결정트리의 손실함수
결정트리를 학습할 때 가장 중요한 것은 노드를 어떠한 기준으로 분류하고, 얼마나 깊이 분류할 것이냐이다. 결정트리에서의 손실함수는 **지니불순도(Gini Impurity of Gini Index)** 나 
**엔트로피(Entropy)** 를 사용한다. 불순도란 데이터가 섞여있는 정도이다. 불순도가 클수록 데이터가 더 섞여 있는 것이기 때문에 분류가 어려워진다.

지니 불순도: 데이터가 반반 섞여있을 때가 가장 불순도가 큰 것이니까 범위는 0 ~ 0.5까지이다.

![image](https://user-images.githubusercontent.com/97672187/156348364-c074a760-ddff-425a-b0a1-d77aa59d8332.png){: .align-center}

엔트로피 지수: 지니 불순도 보다 연산이 조금 더 복잡하고, 0과 1사이 값을 반환한다.

![image](https://user-images.githubusercontent.com/97672187/156348399-8e1c8464-f954-420c-915a-284830d3d5ac.png){: .align-center}

이미지 출처: https://wooono.tistory.com/104

![image](https://user-images.githubusercontent.com/97672187/156352194-c1d5d3a8-a0c5-435b-998a-53c58742bafe.png){: .align-center}

이미지 출처: https://bigdaheta.tistory.com/26

- 분할에 사용할 특성이나 분할값은 타겟변수를 가장 잘 구별해 주는(불순도의 감소가 최대가 되는 = 정보획득이 가장 큰) 것을 선택한다.
- 정보획득(Information Gain)은 노드를 분할했을 때 엔트로피의 감소량이다. 엔트로피 감소했다는 것은 불순도가 감소했다는 것이고, 이 감소량이 크다는 것은 데이터를 더 명확하게 분류했다는 것이기
때문에 정보획득이 크다고 할 수 있다.

### sklearn에서 Decision Trees 구현

```python

from sklearn.tree import DecisionTreeClassifier

pipe = make_pipeline(
    OneHotEncoder(use_cat_names=True),  
    SimpleImputer(), 
    DecisionTreeClassifier(random_state=42, criterion='entropy')
)

pipe.fit(X_train, y_train)
print(pipe.score(X_train, y_train)) #학습 데이터 정확도
print(pipe.score(X_val, y_val)) #검증 데이터 정확도

```

### 과적합 해소를 위한 하이퍼 파라미터 튜닝

결정나무는 과적합이 잘 일어나는데 이를 해소하기 위해 min_samples_split, min_samples_leaf, max_depth와 같은 하이퍼 파라미터를 튜닝할 수 있다.

- min_samples_split: 분류할 때 샘플이 너무 적으면 분류하지 못하도록, 샘플 갯수 제한

- min_samples_leaf: leaf node에 존재하는 샘플의 최소 갯수 제한. 너무 세분화해서 학습하지 못하도록. leaf node에 이만큼은 있어야 한다.

- max_depth: tree의 최대 깊이를 제한해서, 너무 깊이 학습되지 않게함.

Grid search를 사용해서 하이퍼 파라미터를 여러개로 조합해보고 성능이 최대가 되는 하이퍼 파라미터를 찾아보자.(자세한 개념은 나중에 다시)

```python

from sklearn.model_selection import GridSearchCV

dt = DecisionTreeClassifier(random_state=42)
parameters = {'max_depth': [3, 5, 7, 9],
              'min_samples_split': [3, 5],
              'min_samples_split': [x for x in range(3, 15,2)],
              'min_samples_leaf': [x for x in range(1, 15,2)],
              'splitter': ['best', 'random']}

grid_dt = GridSearchCV(dt, # estimator 객체,
                      param_grid = parameters, cv = 5)
grid_dt.fit(X_train_encoded, y_train)

result = pd.DataFrame(grid_dt.cv_results_['params'])
result['mean_test_score'] = grid_dt.cv_results_['mean_test_score']
result = result.sort_values(by='mean_test_score', ascending=False)
result
```

```python

max_depth = result.iloc[0]['max_depth']
min_leaf = result.iloc[0]['min_samples_leaf']
min_split = result.iloc[0]['min_samples_split']
splitter = result.iloc[0]['splitter']

dt_best = DecisionTreeClassifier(random_state=42, max_depth = max_depth, min_samples_leaf= min_leaf, min_samples_split= min_split, splitter = splitter)
dt_best.fit(X_train_encoded,y_train)
y_pred = dt_best.predict(X_train_encoded)
print(f1_score(y_train, y_pred))

y_pred = dt_best.predict(X_val_encoded)
print(f1_score(y_val, y_pred))

```

```python
# 트리 모델은 특성 중요도도 그릴 수 있다.
# 중요도는 불순도가 최소화 되는 값, 빈도, 특성이 얼마나 일찍, 자주 분기에 사용되냐에 따라 결정.
# 항상 양수이다.
# 하지만 이 중요도가 꼭 최적의 모델을 만든다고는 할 수 없음.
import matplotlib.pyplot as plt
importances = pd.Series(dt_best.feature_importances_, X_train_encoded.columns)
plt.figure(figsize=(10,30))
importances.sort_values().plot.barh();

```

### 단조(Monotonic), 비단조(Non-monotonic), 특성상호작용
단조는 데이터의 관계에 규칙이 존재하는 것(선형) 비단조는 규칙이 존재하지 않는 것(비선형)을 말한다. (독립과 종속변수의 관계가)

특성상호작용은 특성끼리 상관관계가 높거나 독립성을 위반한 경우를 말한다. 선형회귀에서는 독립변수들끼리 상관관계가 높으면 학습이 올바르게 되지 않을 수 있는데
트리모델은 특성상호작용을 자동으로 걸러낸다.

따라서 트리 모델은 선형모델과는 달리 비선형, 비단조, 특성상호작용 특징을 가지고 있는 데이터 분석에 용의하다.

특성상호작용에 문제가 없는 이유: 한 번에 하나의 특성만 가지고 분할하기 때문에 다른 특성과의 관계가 고려되지 않는다. 하지만, 데이터간 관계가 고려되지 않아 외삽에 대한 신뢰가 떨어진다.

### Imputers(결측치 처리 방법)
**1. Simple Imputer**

simple imputer는 평균, 최빈값 등 사용자가 결측치를 각 변수의 어떤 값으로 대체 할 것인지 정할 수 있다. 수치형 변수에는 평균, 범주형 변수에는 최빈값을 사용한다. 장점은 쉽고 빠르고, 비교적 적은 숫자형 데이터에 적합하다. 단점은 설명변수 간의 상관관계는 고려하지 않고, 해당 칼럼에서만 결측치 처리를 적용한다. 정확성이 높지 않고, 설명력이 떨어진다. 불확실함. 특히 범주형 변수 같은 경우 안 그래도 많이 등장한 최빈값이 더 많이 생기기 때문에 데이터가 치우쳐진다.

```python
# 1. simple imputer
print(X_train.isnull().sum().sum())
print(X_test.isnull().sum().sum())
print(X_val.isnull().sum().sum())

num_cols = X_train.select_dtypes(include = 'number').columns.tolist()

# 수치형이랄게 딱히 없어서 최빈값으로 na를 처리. 수치형이지만 순서형으로 해석하는 게 맞다고 판단.
# 하지만 최빈값으로 바꾸면 타입이 object가 되므로 미리 이전 타입 저장해놨다가 다시 바꾸기
s = SimpleImputer(strategy = 'most_frequent')
X_train = pd.DataFrame(s.fit_transform(X_train), columns = X_train.columns)
X_val = pd.DataFrame(s.transform(X_val), columns = X_val.columns)
X_test = pd.DataFrame(s.transform(X_test), columns = X_test.columns)

X_train[num_cols] = X_train[num_cols].astype('float')
X_test[num_cols] = X_test[num_cols].astype('float')
X_val[num_cols] = X_val[num_cols].astype('float')

print(X_train.dtypes)

print(X_train.isnull().sum().sum())
print(X_test.isnull().sum().sum())
print(X_val.isnull().sum().sum())

```

**2. Multivariate Imputation**

회귀분석, KNN 등과 같은 분석 process를 거쳐서 가장 적절한 결측치를 예측한다. 따라서 Simple Imputer보다 훨씬 정확한 값으로 결측치를 대체 할 수 있다는 장점이 있다. 하지만, 데이터에 noise가 더 생기고, 분석에 사용되는 데이터가 random하게 이루어지기 때문에 각 데이터의 추정값이 계산할 때마다 약간 다를 수 있고, 완벽히 동일한 답을 내지는 않는다.

하지만 언제나 좋은 성능을 내는 imputer는 없기 때문에 데이터에 따라 적절한 imputer를 정해야 할 것 같다.

```python

#2. Iteratieve imputer
#각 피처의 결측치를 종속변수로 하고 회귀분석을 통해 결측치를 예측하는 방법.
#회귀분석을 여러번 시행해서 가장 적절한 결측치를 찾는다.
#회귀분석을 위해서 OHE가 들어가야함.
ohe2 = OneHotEncoder(use_cat_names = True)
it = IterativeImputer(max_iter = 10, random_state = 42)

X_train_imp = ohe2.fit_transform(X_train)
X_val_imp = ohe2.transform(X_val)
X_test_imp = ohe2.transform(X_test)

X_train_imp = pd.DataFrame(it.fit_transform(X_train_imp), columns = X_train_imp.columns)
X_val_imp = pd.DataFrame(it.transform(X_val_imp), columns = X_val_imp.columns)
X_test_imp = pd.DataFrame(it.transform(X_test_imp), columns = X_test_imp.columns)
X_train_imp.head()

print(X_train_imp.isnull().sum().sum())
print(X_test_imp.isnull().sum().sum())
print(X_val_imp.isnull().sum().sum())

```

