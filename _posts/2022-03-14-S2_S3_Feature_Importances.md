---
layout: single
title: "Section2 Sprint3 Note233 Bagging, Boosting, Feature Importances"
category: Section2
toc: true
toc_sticky: true
---

RandomForest 모델은 기본 모델인 Decision Tree 모델을 각각 독립적으로 학습하여 모든 기본모델의 학습 결과를 반영해서 최종 타겟값을 예측하는 Bagging 모델이다. Bagging 외에도 
다양한 앙상블 기법 중 Boosting과 Stacking이 있는데 이번 시간에는 Boosting에 대해 정리해보자. 또한, Tree Model에서 Feature importnace 함수를 쓰면 변수의 중요도를 파악할 수 있었는데,
이외에도 다른 방법의 feature importance를 활용해보자.

### Feature Importance(Basic)
트리모델에서 기본적으로 사용되는 feature importance는 지니 불순도, 엔트로피를 가장 많이 감소시킨 변수가 높게 나온다. 간단하고 빠르게 중요도를 파악할 수 있긴 하지만, 범주의 차원이 많으면 별로 중요하지 않은 변수라도 중요하게
판단할 수 있어서 이 중요도를 무조건 신뢰하긴 어렵다. 왜냐하면, 차원이 큰 범주를 중요하다고 오인하여 학습 시킨다면 과적합의 우려가 있기 때문이다.

### Drop-Columnm Importance
위의 기본적인 feature importance의 문제를 해결하기 위해 Drop-Column importance를 활용할 수 있다. Drop-Column importance는 변수를 하나씩 빼보면서 이 변수를 뺏을 때 모델의 성능을 비교하여
만약 성능이 크게 떨어졌다면 이 변수를 중요하다고 판단하는 것이다. Feature importance에 비해 훨씬 직관적이고 정확한 방법이긴 하지만, 변수가 아주 많은데 이 변수를 하나하나 빼가면서 성능을
비교한다면 시간이 매우 오래 걸릴 것이다. 특성이 N개 존재하면, N + 1 번만큼의 학습이 필요하기 때문이다. 긴 시간 때문에 현업에서 많이 사용하는 방법은 아니라고 한다.

Ex) 독립변수가 2개라면, 처음에 모든 변수를 넣고 한 번, 특정 변수 빼고 한 번, 다른 변수 빼고 한 번 이렇게 총 3번의 학습이 필요한다.

### Permutation Importance(순열 중요도)
Permutation Importance는 기본적인 feature importance보다는 정확하고, Drop-Column Importance 보다는 시간을 훨씬 단축시키는 중간 단계의 중요도다. Drop-Column 보다는 정확하지 않을 수 있지만,
특정 변수의 값의 순서를 무작위로 섞고 나서 모델의 성능을 재측정하기 때문에 모델을 재학습 시킬 필요가 없다. 모델의 재학습 없이 만들어진 모델에서 변수의 값만 섞은 다음 성능을 측정하는 것이다.
다른 변수는 그대로 있고, 한 변수의 값만 무작위로 섞고 성능을 측정하고, 그 다음 변수를 무작위로 섞고 성능을 측정하는 과정을 계속 반복한다. 여기서 중요한 점은 변수를 섞을 때 랜덤하게 값을 부여하는 것이 아니라 원래 있던 값을 랜덤하게 순서만 섞는 것이다. 새롭게 noise를 추가하는 게 아님. 

![image](https://user-images.githubusercontent.com/97672187/158338700-8b8ec2b3-de5e-47b0-8826-fbee2d916f61.png)


최종적으로 나온 중요도는 양수 값이 클수록 그 변수를
무작위로 섞었을 때 모델의 성능이 떨어졌다는 것이고, 이것은 결국 해당 변수가 중요하다고 해석할 수 있다. 값이 0에 가까우면 별로 중요하지 않은 변수, 음수 값이 나오면 오히려 변수의 값을 섞었을 때
모델의 성능이 나왔다는 뜻으므로 더 중요하지 않다고 해석할 수 있다.

Permutaion importance는 이렇게 모델의 재학습이 없어서 시간을 단축시키고, 모델의 성능과 직접적인 연관이 있기 때문에 feature importance보다 더 정확하다는 장점이 있다.
하지만, 각 변수들이 독립적이라는 가정을 하고 진행하기 때문에 독립 변수끼리의 상관성을 고려하지 않는다. 예를 들어, 나이와 연령대의 변수가 있다고 했을 때, 두 변수는 상관성이 높다. 그러나
순열 중요도에서는 나이 변수의 값을 랜덤하게 섞어서 연령대 변수의 값이 나이와 전혀 상관없이 매칭이 되어있다면 두 변수는 상호작용을 갖지 못하게 되고, 만약 모델에서 각각의 변수들의 중요도를
낮게 평가한다면 두 변수는 중요한 변수에서 제외되어 버릴 수 있다. 실제로 나이와 연령대가 잘 매칭되어 쓰였을 때 아주 중요한 변수였는데 따로따로 쓰여서 중요하지 않게 판단하는 경우가 생길 수 있다.
따라서, 상황에 따라서 변수들 간의 관계를 잘 파악하고 적절한 중요도 방법을 사용해야 한다.

```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import eli5
from eli5.sklearn import PermutationImportance

# permuter 정의
permuter = PermutationImportance(
    pipe.named_steps['randomforestclassification'],
    scoring='accuracy', # 회귀에서는 r2나 negative_mean_.... 등등
    n_iter=5, # 다른 random 한 방법으로 5번 반복
    random_state=42
)

# permuter 계산은 전처리된 X_val을 사용해야함.
# 학습 데이터의 중요도를 파악하는 건 의미 없음. 우리는 성능을 평가하고 싶은 것이니까
X_val_transformed = pipe.named_steps['전처리 시키는 단계의 파이프'].transform(X_val)

# 스코어를 다시 계산하기 위한 fit. 재학습은 아니다.
permuter.fit(X_val_transformed, y_val);

eli5.show_weights(
    permuter, 
    top=None, # 상위 몇개의 변수를 볼 것인가. None은 전체
    feature_names=X_val.columns # 변수 이름들
)

#이 중요도를 보고 중요하지 않은 변수는 삭제할 수도 있음. 예를 들어 중요도가 0.001 보다 큰 변수만 사용.

```

### Boosting
Bagging은 기본 모델인 weak learner들의 결과를 각각 독립적으로 반영해서 타겟을 예측한다. 하지만 Boosting은 bagging과는 달리 각 weak learner들은 이전 weak learner들의 오차를 반영해서 
이 오차를 줄여나가는 방법으로 모델을 순차적으로 학습하게 된다. 오차를 줄이기 때문에 편향이 줄어든다는 장점이 있지만, 그만큼 분산이 커져서 과적합의 우려가 있다.

RandomForest의 장점 중 하나는 하이퍼파라미터에 상대적으로 덜 민김한 것인데, 그래디언트 부스팅의 경우 하이퍼파라미터에 민감하지만 잘 셋팅하면 RandomForest보다 더 좋은 성능을 낼 수 있다.

Bagging vs Boosting: weak learner가 독립적이냐 vs weak learner가 이전 weak learner에 영향을 받냐

- AdaBoost(회귀, 분류 모두 적용 가능, 트리의 기본 단위인 stump 사용)
AdaBoost는 weak learner가 잘못 예측한 관측치에 더 큰 가중치를 주어 이 관측치들이 다음 샘플링에서 더 많이 뽑히게 함으로써 다음에는 해당 관측치를 더 잘 예측하도록 하는 방법을 사용한다.

1) 모든 관측치에 대해 가중치를 똑같이 설정

2) 관측치를 복원추출하여 weak learner를 학습

3) 잘못 예측한 관측치에 더 큰 가중치를 부여해서, 샘플링을 할 때 더 자주 뽑히도록 함.(가중치는 계속 달라짐)

4) 위의 과정을 계속 반복

5) 이 여러가지 분류기들을 결합하여 최종 예측.

최종 예측은 각 weak learner들의 결과를 가중합으로 결정. weak learner에 가중치가 클수록(아까 샘플링할 때 관측치의 가중치랑 다른 것임.) 더 좋은 성능을 냈다는 뜻이므로 더 큰 voting을 할 수 있음.

- Gradient Boosting
Adaboost와 유사하지만 비용함수를 최소화하면서 모델을 최적화 시킨다. Adaboost는 예측을 잘못한 관측치를 더 많이 샘플링 되게 하여 최적화 시키지만 Gradient Boosting은 샘플링이 많이 되게 하진않고,
타겟의 예측값이 아니라 잔차를 학습하도록 한다. 가중치를 조정하는게 아니라 잔차가 큰 데이터를 더 학습하도록 하는 것. 

1) 각 관측치의 초기 예측값은 타겟의 평균을 사용한다.

2) 각 관측치마다 이 예측값과 실제값의 차이인 잔차를 구하고, 이 잔차를 첫번째 weak learner로 학습시킨다.

3) 학습 된 결과에 따라 잔차의 예측값의 평균을 구한다.

4) 잔차의 예측값 * 학습률을 이전 예측값에 대해 예측값을 새로 업데이트 시킨다.

5) 이 과정을 반복해서 주어진 모든 학습이 끝나면 각 weak leaner들의 결과를 가중합으로 최종 타겟값을 도출한다.

GBM에서 예측값을 도출한 다음 잔차를 구하지 않고, 잔차를 굳이 예측하는 이유는 잔차는 원래 실제값이 존재해야 여기서 예측값을 빼서 구할 수 있게 되는데 즉, 실제값이 존재하는 데이터 포인트가 있어야만 구할 수 있다. 하지만, 이 잔차를 예측하게 되면 데이터 포인트가 없더라도 모든 실수 범위에서 값을 구할 수 있기 때문에 x가 어떻든 예측된 잔차값을 알 수 있어서 일반화가 가능해진다.
이 잔차를 집중적으로 학습하면 결국 정답을 잘 맞추기 때문에 편향이 줄어들게 되는데 편향이 줄어들수록 분산이 증가해 과적합의 우려가 있다.
이 편향을 줄어나갈 때 사용하는 것이 learning rate(학습률)인데 이 learning rate(0~1)가 클수록 학습 데이터에 실제 타겟값과 가깝게, 또 빨리 예측하게 된다. 하지만, 이 learning rate을 1로줘서
바로 타겟값을 예측하게 한다면 모델이 모든 잔차를 타겟과 일치하게 바로 예측해버리고, 이렇게 되면 과적합이 생긴다. 반대로 learning rate가 너무 작으면 잔차를 줄여나가는데 시간이 오래 걸리고,
잔차를 많이 못 줄인 상태에서 학습이 끝나 과소적합이 발생할 수도 있다. 따라서 적절한 learning rate를 적용해야한다.

learning rate는 트리의 갯수은 n_estimators와 trade off 관계이다. 왜냐면 learning rate가 크면 바로 타겟값을 예측해서 성능을 높일 수 있기 때문에 많은 트리가 필요하지 않고,
learning rate가 작으면 성능을 높이기 위해 여러개의 트리를 사용해서 잔차를 줄여나가야 한다. learning_rate와 n_estimators를 잘 튜닝하는 것이 중요하다. 다른 하이퍼 파라미터들도 마찬가지.

- Early Stopping
Randomized Search CV나 Grid Search CV를 활용하면 n_estimators에서 성능의 저하되는 부분을 찾아서 중단시킬 수가 없기 때문에 하이퍼 파라미터 조합의 수가 많을 수록 시간이 오래걸린다.
많다고 꼭 좋은게 아닌데, 주어진 조합 안에서 수행함으로 많은 값으로도 계속 학습시켜 버릴수도 있기 때문에.
하지만, Early Stopping methods를 쓰면 성능이 안 좋은 부분에서 저절로 n_estimators를 끊을 수가 있기 때문에 시간을 훨씬 더 단축시킬 수 있다.

- XGBoost

기존에 Gradient Boosting은 모델이 학습 데이터에 대해서 잔차를 너무 줄여서 과적합의 우려가 있었다. 이를 해결하기 위해 XGboost는 기존 Gradient Boosting 방법에서 규제항을 추가해 트리가 복잡할수록 패널티를 부여한다. 이를 통해 과적합을 어느 정도 해소할 수 있다.


```python
#XGBoost 구현(회귀도 가능)
from xgboost import XGBClassifier
model = XGBClassifier(
    n_estimators=1000,  # ES에 따라 조기 종료 될 수도 있음.
    max_depth=7,
    learning_rate=0.2, #학습률. 1에 가까울수록 적은 n_estimator가 필요하고 잔차를 더 빨리 줄임.
#     scale_pos_weight=ratio, # RF의 class_weight 데이터가 불균형일 때 비율 적용
    n_jobs=-1
)

eval_set = [(X_train_encoded, y_train), 
            (X_val_encoded, y_val)]

model.fit(X_train_encoded, y_train, 
          eval_set=eval_set,
          eval_metric='error', # logloss도 있고, 회귀에서는 r2나 negative_mean_... 사용할 수 있음.
          early_stopping_rounds=50
         ) # 50 회동안 성능의 개선이 없으면 강제 종료
         
results = model.evals_result() # 이 result로 어떻게 개선되었는지 확인 가능

```






