---
layout: single
title: "Section2 Sprint2 Note222 Random Forest, Various Encoder"
category: Section2
---

Decision Tree는 비단조 데이터, 특성상호작용과 관계 없이 좋은 성능을 낸다는 장점이 있다. 한 개의 트리만을 사용하기 때문에 한 노드에서 생긴 에러가 하부 노드에 계속 영향을 줄 수 있고,
Tree를 너무 깊게 학습하여 과적합이 일어날 수 있다. 이를 해소하기 위해 최대 깊이를 제한하는 등의 하이퍼 파라미터 튜닝을 할 수 있지만 이것만 가지고는 과적합을 충분히 해소하기 힘들다.

이러한 문제는 Random Forest라는 앙상블 모델을 활용하여 더 쉽게 해소할 수 있다.

## Note 222
### Random Forest
앙상블은 한 종류의 데이터로 여러 학습모델을 만들어 그 모델들의 예측 결과를 분류 문제에선 다수결, 회귀 문제에선 평균을 내어 예측하는 방법을 말한다.

랜덤 포레스트는 결정트리 모델을 기본 모델로 사용하는 앙상블 모델이다.

랜덤 포레스트에서는 배깅(Bagging) 이라는 기법을 사용한다. 배깅은 복원 추출을 의미하는 Bootstrap과, 결과를 합치는 것을 의미하는 aggregating을 합친 뜻으로, 학습 데이터를 랜덤하게 복원 추출하면서
여러개의 결정트리를 만들고 각 트리에서 나온 결과를 다수결 혹은 평균으로 합쳐서 최종 타겟값을 결정하는 것을 의미한다.

![image](https://user-images.githubusercontent.com/97672187/156551747-70fd4816-ab53-4b65-83ae-a0b9065212d1.png){: .align-center}

이미지 출처: https://psystat.tistory.com/92

복원추출이기 때문에 같은 샘플이 반복되어 추출될 수 있고, 위의 공식처럼 복원 추출에 선택되지 않은 데이터들은 전체 샘플에서 약 36.8%에 해당한다. 여기서 선택되지 않은 36.8% 데이터를
Out of bags(OOB)라고 하고, 이 OOB는 추출된 샘플들로 만들어진 결정트리의 성능을 평가하는 지표로 사용된다. 

ex) 만약 5개의 기본 모델이 있으면 5개의 모델마다 복원 추출을 하게 될텐데, 그럼 5개의 모델마다 OOB가 생길 것이다. 그럼 각각의 기본 모델이 OOB 데이터에서 성능을 평가하는데 이 기본 모델이
OOB 데이터를 예측한 값과 OOB의 실제값의 차이를 통해 OOB error가 생기고. 전체 기본 모델의 OOB error의 평균을 내면 이 random forest 모델의 OOB score를 계산할 수 있다. 이후 검증 결과를 바탕으로
하이퍼 파라미터를 조절하여 OOB error를 최소화 시킬 수 있다. (OOB 말고, accuracy나 f1 score 확인해서 하이퍼 파라미터 튜닝 할 수도 있음.)

또한, 랜덤포레스트는 기본 모델 트리를 만들 때 무작위로 특성을 선택하여 사용한다. 일반 결정트리는 전부 or 사용자가 지정한 만큼 변수를 사용하지만, 랜덤 포레스트는 전체 특성 중
k 개의 특성을 사용하는데 이 k개에서 최적의 특성을 찾아내서 노드를 분할한다. k개는 주로 ![image](https://user-images.githubusercontent.com/97672187/156554439-f12522b9-8e02-4e1c-ac27-4390b215a06b.png)
을 사용한다.

결정트리의 과적합 문제를 해결할 수 있다는 장점이 있지만(절대적인 건 아님), 데이터를 직관적으로 해석하기 어려워진다. 또한, 결정트리보다 학습 시간이 매우 길어진다.

### Ordinal Encoding(순서형 인코딩)
범주형 변수를 encoding 하는 방법으로 OHE를 사용할 수 있다. 하지만 OHE는 범주의 갯수만큼 차원이 증가하기 때문에 모델이 복잡해지고 과적합이 생길 수 있다.
또한, 트리모델에서는 범주의 차원이 클수록 한 범주가 갖는 중요도가 낮아져 해당 범주는 상위 노드에 선택될 기회가 적어진다. OHE를 사용하면 범주의 차원이 증가하기 때문에 이로인해 성능저하가 발생할 수 있다.
하지만, 모든 변수에 OHE를 사용해야 하는 것은 아니다. 범주형 변수가 명목형이 아니라 순서형이라면 대소 관계를 가지므로 OHE보다 Ordinal Encoding을 사용하는게 더 좋은 선택일 수 있다.(무조건은 아님)
데이터가 순서형 변수일 때 Ordinal encoding을 사용하면 차원이 늘어나지 않으면서 범주의 대소관계를 표현할 수 있어서 모델을 학습할 때 더 도움이 될 수 있다.
만약, 명목형에 순서형 인코딩을 사용하면, 순서가 없어야 할 데이터에 순서가 생기므로 모델의 성능이 떨어질 수 있기 때문에 주의해야한다.

선형 회귀에서는 범주형 변수를 순서형, 명목형에 따라 각각 다르게 인코딩을 해주어야 한다. ex) 김씨1, 이씨2, 박씨3 -> 명목형 변수에 이렇게 순서를 부여하면 박씨(3)가 김씨(1)의 세배가 된다는 대소관계가 생겨
회귀식에 영향을 주기 때문에.

트리 모델에서는 순서형, 명목형에 구애받지 않고 범주형 변수를 인코딩 할 수 있다.(하지만, 패키지에 따라 다를지도. 적어도 sklearn에서는) 

트리에서는 1,2,3,4,5의 범주가 이 범주들 외에도 다른 다양한 조건들로 인해 모두 다 분할되어 흩어지게 된다. 따라서 범주 1과 다른 조건들에 대해서 독립적인 타겟값, 범주 2와 다른 조건들에 대해서 독립적인 타겟값.....이 형성되기 때문에 데이터가 분할되기 전에 가졌던 순서가 의미가 없어진다.

또한, 트리모델에서 범주형 데이터는 중요하다고 생각하는 범주를(지니 불순도를 낮추고, 정보획득을 높여줄 범주) 우선적으로 사용한다. 데이터가 순서를 가지고 있더라도, 해당 순서가 크다고 중요한 범주가 아니라 지니불순도를 낮추는게 중요한 범주로 사용되기 때문에 순서에 영향을 받지 않는다. 따라서 순서형으로 인코딩해도 대소관계를 가지지 않는다.

하지만 범주가 많다면, 해당 범주를 통해 잘 나눌 수 있는(지니 불순도를 낮추는) 경우의 수가 많아지기 때문에, 범주가 많은 변수를 모델이 계속 중요하다고 판단해서 학습이 잘 되지 않을 수 있다.

범주의 순서를 정확히 알고 있다면, 순서를 직접 입력시켜서 Ordinal Encoding 할 수도 있다.

```python
from category_encoders import OrdinalEncoder
ord = OrdinalEncoder()
X_train_encoded = ord.fit_transform(X_train)
X_val_encoded = ord.transform(X_val)
X_test_encoded = ord.transform(X_test)
```

### Target Encoding(타겟 인코딩, Mean encoding)
target encoding: 타겟이 범주형일 경우, 범주형 독립변수가 주어졌을 때 타겟의 범주가 나올 사후 확률과, 모든 훈련 데이터에 대한 타겟의 사전 확률의 혼합으로 범주를 encoding 한다.

타겟이 연속형이라면, 해당 범주에서의 타겟 값의 평균을 범주에 encoding 한다.

target encoding을 사용하면 독립변수와 종속변수의 관계가 사용되어 설명력이 높아지고, 차원의 증가 없이 빠르게 학습할 수 있다는 장점이 있다. 
하지만, 종속변수의 정보를 사용하기 때문에 data leakage가 일어날 수 있고, 이를 줄이기 위해 학습데이터의 종속변수만 사용하는데 이로 인해 과적합이 일어날 수 있다. 
과적합은 encoding 된 값에 smoothing이라는 규제를 줌으로써 범주의 encoding 값이 특정 평균이나 확률에 치우치지 않게 함으로 줄일 수 있다.

```python
#1. Target encoding
from category_encoders import TargetEncoder
te = TargetEncoder() #smoothing이 1이 default 값으로 되어있다. 학습 데이터에 치우쳐진 평균을 전체 평균에 가깝도록 규제를 가한다. 1에 가까울 수록 더 큰 규제를 가하는 것.
X_train_te = te.fit_transform(X_train, y_train)
X_val_te = te.transform(X_val)
X_test_te = te.transform(X_test)
display(X_train_te.head())
display(X_val_te.head())
```

### Catboost Encoding
catboost encoding: boosting이란, 무작위로 선택하는 것보다는 약간 나은 규칙성을 띈 weak learner들을 결합시켜 strong leanrer를 만들어 더 정확성이 높은 모델을 만든다. 
여러 모델을 병렬로 처리하고 결과를 합치는 bagging과는 달리, 이전 weak learner의 에러를 활용해서 에러가 큰 weak learner에 더 큰 가중치를 부여하며 모델을 순차적으로 학습하여 편향을 줄여나간다. 하지만, 편향을 줄인만큼 분산이 커져서 과적합의 우려가 있다.

catboost는 데이터의 일부의 잔차만 활용하고, 그 뒤의 데이터의 잔차는 이전 모델이 예측한 값을 활용하는 ordered boosting 방식을 사용한다. 
catboost에서 활용하는 encoding을 Ordered Target Encoding이라고 할 수 있는데 이는 특정 범주에 인코딩해야 할 값을 이전 데이터에서 인코딩 된 값으로 사용함으로 data leakage와 오버피팅을 
줄여준다. 또한 target encoding 처럼 다양한 수치로 범주를 표현할 수 있고, 기존 부스팅의 문제점인 긴 학습 시간도 단축했다는 장점도 있다.

하지만 역시 학습 데이터에서 사용한 encoding 값을 테스트 데이터에서 사용하기 때문에 과적합을 완전히 해소하진 못한다.

```python

# 2. catboost encoding
from category_encoders import CatBoostEncoder
cat = CatBoostEncoder()
X_train_cat = cat.fit_transform(X_train, y_train)
X_val_cat = cat.transform(X_val)
X_test_cat = cat.transform(X_test)

display(X_train_cat.head())
display(X_val_cat.head())

```

### sklearn을 활용한 Random Forest
n_estimators = 기본모델 수(기본트리 수). 하지만 많을수록 학습 시간 증가

```python
rf_origin = RandomForestClassifier(n_estimators = 200,random_state=42, max_depth = 9, min_samples_leaf= 10, min_samples_split= 3)
rf_origin.fit(X_train_sim, y_train)
y_pred = rf_origin.predict(X_train_sim)
print(f1_score(y_train, y_pred))

y_pred = rf_origin.predict(X_val_sim)
print(f1_score(y_val, y_pred))
```

```python
#Grid search 사용
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(random_state=42)
parameters = {'n_estimators' : [50, 100], 
              'max_depth': [3, 5, 6, 8],
              'min_samples_split': [x for x in range(3, 10,2)],
              'min_samples_leaf': [x for x in range(3, 10,2)]}

grid_rf = GridSearchCV(rf, # estimator 객체,
                      param_grid = parameters, cv = 5, n_jobs = -1, verbose = 1, scoring = 'f1') #n_jobs = -1은 사용 가능한 컴퓨터 프로세서 모두 사용한다는 뜻.
grid_rf.fit(X_train_cat, y_train)

result = pd.DataFrame(grid_rf.cv_results_['params'])
result['mean_test_score'] = grid_rf.cv_results_['mean_test_score']
result = result.sort_values(by='mean_test_score', ascending=False)

from sklearn.metrics import f1_score

max_depth = int(result.iloc[0]['max_depth'])
min_leaf = int(result.iloc[0]['min_samples_leaf'])
min_split = int(result.iloc[0]['min_samples_split'])
n = int(result.iloc[0]['n_estimators'])

rf_best = RandomForestClassifier(n_estimators = n, random_state=42, max_depth = max_depth, min_samples_leaf= min_leaf, min_samples_split= min_split)
rf_best.fit(X_train_cat,y_train)
y_pred = rf_best.predict(X_train_cat)
print(f1_score(y_train, y_pred))

y_pred = rf_best.predict(X_val_cat) #catboost encoding 하고 하이퍼 파라미터 조금씩 수정해도 과적합 발생. 
print(f1_score(y_val, y_pred))

```

```python
#tree graph
# graphviz 설치방법: conda install -c conda-forge python-graphviz
import graphviz
from sklearn.tree import export_graphviz

estimator = rf_origin.estimators_[5]
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = X_train_sim.columns,
                class_names = ['no', 'yes'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')
```


