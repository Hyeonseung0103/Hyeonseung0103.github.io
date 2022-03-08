---
layout: single
title: "Section2 Sprint2 Note224 Cross Validation 과 Hyper Parameter Tuning"
category: Section2
---

데이터셋이 충분히 크다면 hold-out 교차 검증을 할 수 있다. 하지만, hold-out 교차 검증을 하면 검증 데이터에서 한 번씩 밖에 학습된 모델의 성능을 측정할 수 있기 때문에 이 일반화 능력을 100퍼센트
신뢰하기 어렵다. 검증 데이터를 여러번 쓸 수 있다면 일반화 능력을 측정하기 더 쉬울 것이다. 이를 위해 K Fold CV를 사용할 수 있다. 또한, 사용자가 수치의 범위를 지정하고 성능이 가장 좋은 조합을
찾아내는 하이퍼 파라미터 튜닝에 대해서도 배웠다.

## Note 224
### 교차검증(Cross Validation)
모든 문제에 적합한 모델과 하이퍼 파라미터는 없다. 하지만, 교차 검증을 사용함으로써 어떤 모델이, 어떤 하이퍼 파라미터가 가장 좋은 성능을 내는지 알 수 있다.

Hold-out CV는 데이터를 학습, 검증, 훈련 데이터로 나누어서 학습 된 모델을 테스트 데이터에서 성능을 평가하기 전, 미리 만들어 놓은 검증 데이터에서 성능을 평가하는 방법이다. 만약 학습 된 모델이
검증 데이터에서 성능이 잘 나왔다고 해도, 이 검증이 딱 한 번만 수행했다면 신뢰하기가 힘들 것이다. 또한, 주어진 데이터가 작으면 이를 학습 데이터와 검증 데이터셋으로 분리하기가 쉽지 않다.
데이터를 분리한다는 것은 결국, 학습에 사용될 데이터가 적어진다는 뜻이니까 학습이 충분히 이루어지지 못할 수 있다.

이러한 문제를 해결하기 위해 K-Fold CV를 사용할 수 있다. K-Fold CV는 학습 데이터를 K개의 그룹으로 나누어서 학습에는 K-1의 그룹을 검증에는 나머지 데이터 그룹을 사용하여 성능을 검증하는 방법이다.
K번의 교차검증을 할 수 있기 때문에 학습 데이터가 적어도 사용할 수 있고, 모델의 일반화 능력을 더 신뢰할 수 있게 된다.

하나의 모델 뿐만 아니라 여러 가지 모델에 교차 검증을 수행하며 가장 좋은 성능을 내는 모델과 하이퍼 파라미터를 찾을 수 있다.

```python
from sklearn.model_selection import cross_val_score
#학습 데이터를 교차 검증시키고, 간단하게 모델의 랜덤하게 생성된 검증 데이터의 점수를 알고 싶을 때
k = 3
scores = cross_val_score(pipe, X_train, y_train, cv = k, scoring = 'f1')
print(scores)
```

```python
#위에서 미리 어느 정도 점수를 확인하고, 적절한 K 값을 찾아서 이걸 돌리면 될 듯.
#해당 모델의 값들을 평균을 내서 최종 타겟값을 예측하고 싶을 때, 더 복잡하고 정확하게 써야함.
from sklearn.model_selection import StratifiedKFold , KFold

is_holdout = False
n_splits = 5

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) #타겟이 0이나 1, 이 한 가지 데이터만 가지고 학습하지 않게, 
# 0과1 모두를 포함한 데이터 사용. 기존 K Fold의 단점 보완(무작위로 타겟 범주의 수 고려하지 않고 뽑음)

from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder

models = []

#위에서 사용했던 target encoding + Randomizedsearch CV 결과 중 가장 과적합이 낮고 성능이 좋은 파라미터 사용
rank = pd.DataFrame(clf_rf.cv_results_).sort_values(by='rank_test_score') #Randomized searcv 한 값
max_depth = rank.loc[rank['param_randomforestclassifier__max_depth'] <= 7].iloc[0]['param_randomforestclassifier__max_depth']
max_feat = rank.loc[rank['param_randomforestclassifier__max_depth'] <= 7].iloc[0]['param_randomforestclassifier__max_features']
n = rank.loc[rank['param_randomforestclassifier__max_depth'] <= 7].iloc[0]['param_randomforestclassifier__n_estimators']

for tr, val in cv.split(X_train,y_train):
  print("="*50)
  preds = []
  pipe_target = Pipeline([('encoder', TargetEncoder(min_samples_leaf=2, smoothing=1000)),
                         ('imputer', SimpleImputer()),
                         ('model',   RandomForestClassifier(random_state = 42, max_depth = max_depth, max_features = max_feat, n_estimators = n, min_samples_leaf= 10))], #class_weight 안 쓸때가 더 성능 높음
                        verbose = 100
  )

  pipe_target.fit(X_train.iloc[tr], y_train.iloc[tr])
  models.append(pipe_target)
  if is_holdout:
    break
 ```
 
 ```python
 #위에 처럼 학습 후 저장한 모델을 가지고 점수 계산
 from sklearn.metrics import f1_score

threshold = 0.32
pred_list = []
tr_scores2 = []
val_scores2 = []

for i,(tri, vai) in enumerate( cv.split(X_train, y_train) ):
    y_pred_prob_tr = models[i].predict_proba(X_train.iloc[tr])[:, 1]
    pred_tr = np.where(y_pred_prob_tr >= threshold , 1, 0)
    score = f1_score(y_train.iloc[tr],pred_tr)
    tr_scores2.append(score)

    y_pred_proba = models[i].predict_proba(X_train.iloc[vai])[:, 1]
    pred_val = np.where(y_pred_proba >= threshold , 1, 0)
    score = f1_score(y_train.iloc[vai],pred_val)
    val_scores2.append(score)
    pred = models[i].predict_proba(X_test)[:, 1]
    pred_list.append(pred)
print(tr_scores2)
print(np.mean(tr_scores2))
print(val_scores2)
print(np.mean(val_scores2))
```

```python
y_pred_proba_test = np.mean(pred_list, axis = 0) # 테스트 데이터 확률들을 각 행마다 평균
y_pred_test = np.where(y_pred_proba_test >= threshold, 1, 0) #threshold에 따른 최종값.
```

### 하이퍼 파라미터 튜닝
최적화 vs 일반화

최적화: 학습 데이터로 더 좋은 성능을 얻기 위해 모델을 조정하는 것. 파라미터(모델) + 하이퍼 파라미터(연구자) 조정 -> 모델의 성능을 높이는 것.

일반화: 학습된 모델이 처음 본 데이터에서도 학습 데이터와 비슷한 성능을 내는 것.

이상적인 모델은 과소적합과 과적합 사이에 존재한다.

하이퍼 파라미터는 모델 훈련 중에 학습 되지 않는 파라미터이다. 따라서 사용자가 직접 정해줘야 한다. 하지만, 이를 손으로 직접 정해주면서 성능을 높이기엔 어렵기 때문에
하이퍼 파라미터 튜닝 툴을 사용한다.

1) GridSearchCV

검증하고 싶은 하이퍼파라미터들의 수치를 정해주고 이 수치들의 모든 조합을 검증하는 방법. 모든 조합을 보고 주어진 수치 내의 최적의 조합을 찾아낼 수 있다.

장점: 사용자가 정한 수치를 모두 보기 때문에 정확하다.

단점: 모든 조합을 검증하기 때문에 범위를 넓게 잡을수록 시간이 오래 걸린다. 

```python
%%time

from sklearn.model_selection import GridSearchCV

dists = {
    'simpleimputer__strategy': ['mean', 'median', 'most_frequent'], 
    'randomforestclassifier__n_estimators': [10,50,100,300], 
    'randomforestclassifier__max_depth': np.arange(7,11,2), 
    'randomforestclassifier__max_features': np.arange(0.01, 1, 0.25) # 소수로 입력하면 이 max_features * n_features가 고려된다. 0일 때는 에러남.
}

grid= GridSearchCV(
    pipe,
    param_grid=dists,
    cv = 3,        
    scoring='f1',  
    verbose=1,     
  )

grid.fit(X_train, y_train)

print('최적 하이퍼파라미터: ', grid.best_params_) #하지만 무조건 best로 하기에는 과적합이 일어날 수 있음. 확인해보고 수동으로 조정할 것.
print('F1: ', grid.best_score_)
```

```python
best = grid.best_estimator_ # 최적의 결과
pred = best.predict_proba(X_val)[:,1]

#만약 최적의 결과 말고, 과적합을 해소하는 다른 조합 보려고 할 때는
#이렇게 DF 만들어서 다른 조합 직접 찾으면 될듯
rank = pd.DataFrame(grid.cv_results_).sort_values(by='rank_test_score')

#이후에 threshold 여러개로 조정해보면서 스코어를 보면 된다.
```

2) Randomized Search CV

하이퍼 파라미터들의 범위를 정해지고, 범위 내에서 파라미터 수치를 랜덤하게 추출하여 조합한다.

장점: 주어진 모든 범위를 보진 않기 때문에 Grid Search보다 빠르고, 어느 정도 정확성을 보유한다.

단점: 범위를 넓게 잘을수록 최적의 하이퍼 파라미터 수치를 놓친다. Local minimum에 빠질 수 있다.

CV vs Search CV

Cross validation은 데이터의 일반화 능력을 향상시키기 위함이다.

Search CV는 CV의 개념을 활용하긴 했지만, 하이퍼 파라미터에 국한 되어서 여러가지 조합을 통해 모델의 성능을 가장 높이는 하이퍼 파라미터 수치를 찾기 위함이다.
데이터의 일반화 능력보다는 주어진 모델에서 가장 높은 성능을 내는 하이퍼 파라미터를 찾는 것이 목적이다.

```python
%%time
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

pipe_target = make_pipeline(
    TargetEncoder(min_samples_leaf=2, smoothing=1000),
    SimpleImputer(),
    RandomForestClassifier(random_state = 42))

dists = {
    'randomforestclassifier__n_estimators': randint(50, 500), 
    'randomforestclassifier__max_depth': np.arange(6,9,1), 
    'randomforestclassifier__max_features': uniform(0, 1) # 소수로 입력하면 이 max_features * n_features가 고려된다.
}

clf_rf = RandomizedSearchCV(
    pipe_target,
    param_distributions = dists,
    n_iter = 50, #이 수만큼 random하게 튜닝 파라미터를 조합하여 반복.random 조합 시도 횟수
    cv = 5,
    scoring = 'f1',
    verbose = 1,
    n_jobs = -1
)
clf_rf.fit(X_train,y_train)
```
grid search와 마찬가지로 최적의 모델은 best_estimator_ 를 활용 or refit 함수 활용. 과적합 발생시 grid search 처럼 DF 만들어서 수동으로 확인(다시 돌리기엔 시간이 너무 오래 걸려서).

시계열 데이터에서는 일반적인 CV가 아니라 다른 CV를 사용해야한다. 랜덤하게 검증 데이터를 추출하는 CV를 사용하면 미래의 데이터로 과거를 예측하는 Data leakage 같은 경우가 생길 수 있기 때문.
