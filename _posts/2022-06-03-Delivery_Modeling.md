---
layout: single
title: "배달 데이터 프로젝트 Modeling"
toc: true
toc_sticky: true
category: Delivery
---

지난 포스팅까지 모델링을 위한 모든 전처리 과정을 맞췄다. 이번 포스팅에서는 여러가지 모델을 사용한 모델링의 과정과 성능에 대해
정리해보자.

## Delivery Project 모델링

### 데이터 불러오기

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
plt.rc('font', family='NanumBarunGothic') 
from category_encoders import OneHotEncoder,OrdinalEncoder, TargetEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import shap
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from catboost import Pool,CatBoostClassifier
```

```python
DATA_PATH = '.........'
df_final = pd.read_csv(f'{DATA_PATH}df_final.csv')

#예측에 필요없는 변수 제거
df_final.drop(columns = ['날짜', 'time', '시도'],inplace = True)
df_final.head()         
```

![image](https://user-images.githubusercontent.com/97672187/171805327-124f854a-1374-4b81-922c-088a403beaef.png){: .align-center}

<br>


<br>


학습, 검증, 테스트 데이터로 나누기

```python
X_train, X_test = train_test_split(df_final, test_size = 0.2, random_state = 42)
X_train, X_val = train_test_split(X_train, test_size = 0.2, random_state = 42)
y_train, y_val, y_test = X_train.pop('주문정도'), X_val.pop('주문정도'), X_test.pop('주문정도')
X_train.shape, X_val.shape, X_test.shape
```

![image](https://user-images.githubusercontent.com/97672187/171805505-183f57c3-e3da-4e6c-975c-ebf698266a74.png){: .align-center}

<br>


<br>

### 1. RandomForest
성능이 올랐지만, 과적합이 있다.

```python
#랜덤포레스트
pipe_rf = make_pipeline(TargetEncoder(min_samples_leaf = 2, smoothing = 1000),
                       RandomForestClassifier(n_estimators = 100, random_state = 42, n_jobs = -1))  

pipe_rf.fit(X_train, y_train)

print(pipe_rf.score(X_train, y_train))
print(pipe_rf.score(X_val, y_val))

y_prob = pipe_rf.predict_proba(X_train) 
print('학습 AUC',roc_auc_score(y_train, y_prob, multi_class="ovr", average="weighted"))

y_prob = pipe_rf.predict_proba(X_val)
print(roc_auc_score(y_val, y_prob, multi_class="ovr", average="weighted"))
```

![image](https://user-images.githubusercontent.com/97672187/171805713-6ffff02a-d1a5-41d9-afa8-56aa30019315.png){: .align-center}

<br>


<br>

### 2. XGBoost
AUC가 검증데이터에서 과적합이 발생한 RandomForest에 비해 훨씬 증가했다.

0.8359 -> 0.8602

```python
#xgboost
pipe_xgb = make_pipeline(TargetEncoder(min_samples_leaf = 2, smoothing = 1000),
                       XGBClassifier(n_estimators = 100, random_state = 42, n_jobs = -1))  

pipe_xgb.fit(X_train, y_train)

print(pipe_xgb.score(X_train, y_train))
print(pipe_xgb.score(X_val, y_val)) #정확도도 증가

y_prob = pipe_xgb.predict_proba(X_train) 
print('학습 AUC',roc_auc_score(y_train, y_prob, multi_class="ovr", average="weighted"))

y_prob = pipe_xgb.predict_proba(X_val) 
print('검증 AUC', roc_auc_score(y_val, y_prob, multi_class="ovr", average="weighted"))
```

![image](https://user-images.githubusercontent.com/97672187/171806031-d498cb86-ca51-440b-890b-4bd960b23b35.png){: .align-center}

<br>


<br>

### 3. LightGBM

XGBoost보단 아니지만, 꽤 준수한 성능을 보인다.

```python
#LGBM
pipe_lgbm = make_pipeline(TargetEncoder(min_samples_leaf = 2, smoothing = 1000),
                       LGBMClassifier(n_estimators = 100, random_state = 42, n_jobs = -1))  

pipe_lgbm.fit(X_train, y_train)

print(pipe_lgbm.score(X_train, y_train))
print(pipe_lgbm.score(X_val, y_val)) #정확도도 증가

y_prob = pipe_lgbm.predict_proba(X_train) 
print('학습 AUC',roc_auc_score(y_train, y_prob, multi_class="ovr", average="weighted"))

y_prob = pipe_lgbm.predict_proba(X_val) 
print(roc_auc_score(y_val, y_prob, multi_class="ovr", average="weighted"))
```

![image](https://user-images.githubusercontent.com/97672187/171806473-ea84956d-795e-4071-bb53-350a39d20147.png){: .align-center}

<br>


<br>

### 4. XGBoost와 LGBM으로 하이퍼 파라미터 튜닝
#### 1) XGBoost
Logistic Regression이나 RandomForest에 비해 준수한 성능을 냈던 두 모델로 Randomized Search CV을 진행해보자.

```python
#파이프를 넣으면 정확도와 AUC를 계산하는 함수
def accuracy_and_auc(pipe):
    print('학습 정확도', pipe.score(X_train, y_train))
    print('검증 정확도', pipe.score(X_val, y_val))
    y_prob = pipe.predict_proba(X_train)
    train_roc = roc_auc_score(y_train, y_prob, multi_class="ovr", average="weighted")
    print('학습 AUC',train_roc)

    y_prob = pipe.predict_proba(X_val)
    val_roc = roc_auc_score(y_val, y_prob, multi_class="ovr", average="weighted")
    print('검증 AUC', val_roc)
    print('학습과 검증 AUC 차이', train_roc - val_roc )
    

#모델을 넣으면 정확도와 AUC를 계산하는 함수
def accuracy_and_auc2(xgb):
    print('학습 정확도', xgb.score(X_train_encoded, y_train))
    print('검증 정확도', xgb.score(X_val_encoded, y_val))
    y_prob = xgb.predict_proba(X_train_encoded)
    train_roc = roc_auc_score(y_train, y_prob, multi_class="ovr", average="weighted")
    print('학습 AUC',roc_auc_score(y_train, y_prob, multi_class="ovr", average="weighted"))

    y_prob = xgb.predict_proba(X_val_encoded)
    val_roc = roc_auc_score(y_val, y_prob, multi_class="ovr", average="weighted")

    print('검증 AUC', roc_auc_score(y_val, y_prob, multi_class="ovr", average="weighted"))
    print('학습과 검증 AUC 차이', train_roc - val_roc )
```

XGBoost로 Randmoized Search CV, Target Encoding

```python
%%time
encoder = TargetEncoder(min_samples_leaf = 2, smoothing = 1000)
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_val_encoded = encoder.transform(X_val)
xgb =  XGBClassifier(n_estimators = 1000, random_state = 42, n_jobs = -1, tree_method = 'gpu_hist', predictor = 'gpu_predictor')

dists = {
    'xgbclassifier__max_depth': np.arange(5,9,1),
    'xgbclassifier__learning_rate' : np.arange(0.025, 0.05, 0.005),
    'xgbclassifier__max_features': uniform(0,1)
}

clf_xgb = RandomizedSearchCV(
    xgb,
    param_distributions = dists,
    n_iter = 100,
    cv = 5,
    scoring = 'roc_auc_ovr_weighted',
    verbose = 1,
    n_jobs = -1
)
eval_set = [(X_train_encoded, y_train),
           (X_val_encoded, y_val)]
clf_xgb.fit(X_train_encoded, y_train, eval_set = eval_set, eval_metric = 'auc', early_stopping_rounds = 10)

#최적의 파라미터
print("Best Parameter: {}".format(clf_xgb.best_params_))
# Best Parameter: {'xgbclassifier__learning_rate': 0.030, 
                   'xgbclassifier__max_depth': 5, 'xgbclassifier__max_features': 0.7698}
```

최적의 파라미터로 재학습

```python
xgb_best = clf_xgb.best_estimator_
xgb_best.fit(X_train_encoded, y_train, eval_set = eval_set, eval_metric = 'auc', early_stopping_rounds = 10)
```

```python
#과적합이 좀 있다.
accuracy_and_auc2(xgb_best)
```

![image](https://user-images.githubusercontent.com/97672187/171810178-e4a57e9f-b6ef-4efe-bea3-a9a85fc8b1a7.png){: .align-center}

<br>


<br>

Shap 중요도 그래프

```python
#과적합을 줄이기 위해 필요없는 변수를 없애고 다시 돌려보자.
#permutation importance는 다중분류에서 적용되지 않는다.
explainer = shap.TreeExplainer(xgb_best)
shap_values = explainer.shap_values(X_val_encoded.iloc[:500])

shap.initjs()

#다중 분류라 해당 변수가 어떤 클래스에 영향을 미쳤는지 표시됨.
shap.summary_plot(shap_values, X_val_encoded.iloc[:500], plot_type = 'bar')
```

![image](https://user-images.githubusercontent.com/97672187/171810581-ddc59c64-5a98-4de7-9468-07f1fbb97789.png){: .align-center}

<br>


<br>

Feature importance 그래프

```python
fig = plt.figure(figsize = (10,8))
pd.Series(xgb_best.feature_importances_, index = X_val.columns).sort_values().plot.barh()
```

![image](https://user-images.githubusercontent.com/97672187/171810701-06776209-1458-408c-92c8-8e27440139c2.png){: .align-center}

<br>


<br>


중요도가 낮은 변수 제거

```python
#두 가지 중요도 그래프에서 모두 순위가 낮은 변수는 제거해보자.
# 주말, 초미세, 풍속, 미세,축구
print(df_final.shape)
df_final2 = df_final.drop(columns = ['주말', '초미세', '풍속', '미세', '축구'])
print(df_final2.shape)

X_train, X_test = train_test_split(df_final2, test_size = 0.2, random_state = 42)
X_train, X_val = train_test_split(X_train, test_size = 0.2, random_state = 42)
y_train, y_val, y_test = X_train.pop('주문정도'), X_val.pop('주문정도'), X_test.pop('주문정도')

encoder = TargetEncoder(min_samples_leaf = 2, smoothing = 1000)
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_val_encoded = encoder.transform(X_val)
```

![image](https://user-images.githubusercontent.com/97672187/171810910-62c50d7f-9454-4f16-9d6c-e03661afa39c.png){: .align-center}

<br>


<br>


필요없는 변수 줄이고 재학습

```python
#필요없는 변수를 좀 줄이고, n_estimators가 학습이 좀 덜 되게 함으로써 과적합을 줄인다.
xgb2 =  XGBClassifier(n_estimators = 120, random_state = 42, n_jobs = -1, tree_method = 'gpu_hist', predictor = 'gpu_predictor',
                     max_depth = 6, learning_rate = 0.25)
eval_set = [(X_train_encoded, y_train),
           (X_val_encoded, y_val)]
xgb2.fit(X_train_encoded, y_train, eval_set = eval_set, eval_metric = 'auc', early_stopping_rounds = 5)
```

하이퍼 파라미터 튜닝결과 XGBoost는 0.8602에서 0.8610으로 성능이 아주 조금 개선됐다.

```python
#학습과 검증 데이터의 AUC 차이가 0.01 보다 작을 때까지 수동으로 돌린 결과 다음과 같은 결과가 젤 좋았다.
accuracy_and_auc2(xgb2)
```

![image](https://user-images.githubusercontent.com/97672187/171811059-d3979962-e955-4fcf-bccd-8d113cee5f66.png){: .align-center}

<br>


<br>

#### 2) LGBM
LGBM으로 Randomized Search CV 진행.

```python
#LGBM
lgbm = LGBMClassifier(random_state = 42, n_jobs = -1, objective = 'multiclass', num_class = 3)
dists = {
    'lightgbmclassifier__max_depth': np.arange(5,8,1),
    'lightgbmclassifier__learning_rate' : np.arange(0.1, 0.35, 0.01),
    'lightgbmclassifier__max_features': uniform(0,1),
    'lightgbmclassifier__n_estimators': np.arange(150,1000,50)
}

clf_lgbm = RandomizedSearchCV(
    lgbm,
    param_distributions = dists,
    n_iter = 100,
    cv = 5,
    scoring = 'roc_auc_ovr_weighted',
    verbose = 1,
    n_jobs = -1
)
eval_set = [(X_train_encoded, y_train),
           (X_val_encoded, y_val)]
clf_lgbm.fit(X_train_encoded, y_train, eval_set = eval_set, eval_metric = 'multi_logloss', early_stopping_rounds = 5)
```


```python
#Randomized search CV가 수동으로 만질 때보다 성능이 더 떨어짐.
print("Best Parameter: {}".format(clf_lgbm.best_params_))
#Best Parameter: {'lightgbmclassifier__learning_rate': 0.21999999999999995, 'lightgbmclassifier__max_depth': 6, 
'lightgbmclassifier__max_features': 0.668879150010527, 'lightgbmclassifier__n_estimators': 200}

lgbm_best = clf_lgbm.best_estimator_
lgbm_best.fit(X_train_encoded, y_train, eval_set = eval_set, eval_metric = 'multi_logloss', early_stopping_rounds = 10)

accuracy_and_auc2(lgbm_best)
```

![image](https://user-images.githubusercontent.com/97672187/171811854-22ae5110-cd71-4bde-85a0-5614ff2cc84f.png){: .align-center}

<br>


<br>

Randomized Search CV가 성능이 오히려 더 떨어져서 수동으로 하이퍼 파라미터를 튜닝해보았다.
그 결과 원래 검증 데이터 성능이 0.8451이었던 LGBM모델의 성능이 0.8679로 증가했고, 이는 XGBoost보다 높은 성능을 갖는다.

```python
#xgboost 보다 성능이 더 좋다.
lgbm2 =  LGBMClassifier(n_estimators = 200, random_state = 42, n_jobs = -1,learning_rate = 0.17)
eval_set = [(X_train_encoded, y_train),
           (X_val_encoded, y_val)]
lgbm2.fit(X_train_encoded, y_train, eval_set = eval_set, eval_metric = 'multi_logloss', early_stopping_rounds = 5)

#검증 데이터 성능이 xgboost보다 0.006 올랐다.
accuracy_and_auc2(lgbm2)
```

![image](https://user-images.githubusercontent.com/97672187/171812278-9c697f95-d3f2-4476-9ba7-bc38f9a759c1.png){: .align-center}

<br>


<br>


여러가지 변수 조합을 통해 불필요한 변수를 제거하고, 성능을 더 높여보자. 최종 모델은 성능이 가장 좋은
LGBM을 활용하자.

Shap 중요도 그래프

```python
explainer = shap.TreeExplainer(lgbm2)
shap_values = explainer.shap_values(X_val_encoded.iloc[:500])

shap.initjs()
shap.summary_plot(shap_values, X_val_encoded.iloc[:500], plot_type = 'bar')
```

![image](https://user-images.githubusercontent.com/97672187/171813065-7eac9313-95b5-4d51-9454-f61b980518d1.png){: .align-center}

<br>


<br>

Feature importances 그래프

```python
fig = plt.figure(figsize = (10,8))
pd.Series(lgbm2.feature_importances_, index = X_val.columns).sort_values().plot.barh()
```

![image](https://user-images.githubusercontent.com/97672187/171813375-ef85f549-6f5b-4207-8317-b194f490429f.png){: .align-center}

<br>


<br>

**여러가지 조합 결과 '업종', '시간', '월', '요일', '구', '50대를 제외한 나이대 모두, '공휴일' 을 변수로 사용하는게 가장 좋았다.**
열심히 추가했던 날씨, 미세먼지, 축구 변수는 아쉽지만 성능을 올리는데 크게 영향을 못 미쳤다.

```python
#날씨관련해서 모두 제거하고 돌려보자.
#df_final2 = df_final.drop(columns = ['축구', '풍속','상대습도', '초미세', '미세', '일강수량', '적설', '체감온도'])
df_final2 = df_final.drop(columns = ['풍속','상대습도', '초미세', '미세', '일강수량', '적설', '체감온도'])
print(df_final2.shape)
df_final2.drop_duplicates(inplace = True)
print(df_final2.shape)
display(df_final2.head())
X_train, X_test = train_test_split(df_final2, test_size = 0.2, random_state = 42)
X_train, X_val = train_test_split(X_train, test_size = 0.2, random_state = 42)
y_train, y_val, y_test = X_train.pop('주문정도'), X_val.pop('주문정도'), X_test.pop('주문정도')

#encoder = TargetEncoder(min_samples_leaf = 2, smoothing = 1000)
#encoder = OrdinalEncoder()
# OHE가 성능이 가장 좋았다.
encoder = OneHotEncoder(use_cat_names = True)
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_val_encoded = encoder.transform(X_val)
```

![image](https://user-images.githubusercontent.com/97672187/171814276-3ed54952-787e-4477-8991-f020c21329b7.png){: .align-center}

<br>


<br>


```python
lgbm3 =  LGBMClassifier(n_estimators = 330, random_state = 42, n_jobs = -1,max_depth = 5,learning_rate = 0.22)
eval_set = [(X_train_encoded, y_train),
           (X_val_encoded, y_val)]
lgbm3.fit(X_train_encoded, y_train, eval_set = eval_set, eval_metric = 'multi_logloss', early_stopping_rounds = 5)

#OHE가 성능이 더 좋다.
accuracy_and_auc2(lgbm3)
```

![image](https://user-images.githubusercontent.com/97672187/171814418-de06f611-3bfc-4f78-8369-c06e253e1ff4.png){: .align-center}

<br>


<br>

**최종 테스트 데이터 성능**

학습, 검증 데이터와 모두 비슷한 성능을 내는 LGBM 모델을 만들 수 있었다. 최종 score는 **정확도가 72.62%**, **AUC가 0.8723**으로
초기에 만들었던 Logistic Regression 모델의 성능인 **정확도 58.66%**, **AUC 0.7116**보다 훨씬 높다는 것을 확인할 수 있다.

```python
X_test_encoded = encoder.transform(X_test)
y_prob = lgbm3.predict_proba(X_test_encoded)
test_roc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

print('테스트 정확도', lgbm3.score(X_test_encoded,y_test))
print('테스트 AUC', roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"))
```

![image](https://user-images.githubusercontent.com/97672187/171814531-4dd7f43f-88d1-4548-a002-84115ac75c1d.png){: .align-center}

<br>


<br>

이로써 모델링까지의 과정을 모두 마쳤고, baseline 모델보다 더 좋은 성능을 내는 LGBM 모델을 만들 수 있었다. 다음 포스팅에서는
분석을 진행하며 도출했던 인사이트를 정리하고, 분석 전에 생각했던 가정과 분석 결과가 어떻게 같고, 차이가 있는지 결론 파트를
작성해보자.


