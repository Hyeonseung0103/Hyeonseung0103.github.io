---
layout: single
title: "배달 데이터 프로젝트 전처리 및 EDA Part.1"
toc: true
toc_sticky: true
category: Delivery
---

KT 빅데이터 플랫폼에서 제공하는 배달 데이터를 활용하여 서울 특별시의 시간대별, 구별로 주문의 정도(많음, 보통, 적음)를 예측하는 모델을 개발했다. 이번 포스팅에서는 어떤 데이터를 추가로 활용하여
모델을 만들고, 성능을 높였는지 코드를 정리해보자.

## Delivery Project 전처리 및 EDA Part.1

### 1. 데이터 탐색

```python
#사용할 라이브러리
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
import json
from urllib.request import urlopen
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
```

```python
#데이터 불러오기
DATA_PATH = '......'

df_19 = pd.read_csv(f'{DATA_PATH}delivery_2019.csv', names = ['날짜', '시간', '업종', '시도', '구', '주문건수'])
df_20 = pd.read_csv(f'{DATA_PATH}delivery_2020.csv', names = ['날짜', '시간', '업종', '시도', '구', '주문건수'])
df_21 = pd.read_csv(f'{DATA_PATH}delivery_2021.csv')
df_21 = df_21.rename(columns = {'date' : '날짜', 'hour' : '시간', 'category' : '업종', 'sido': '시도', 'gu' : '구', 'order' : '주문건수' })
```

데이터는 밑의 이미지처럼 날짜, 시간, 업종, 시간, 구, 주문건수 변수로 이루어져 있다.

```python
display(df_19.head())
```

![image](https://user-images.githubusercontent.com/97672187/171601618-35d01613-c93f-481e-9f25-fc1a1b228945.png){: .align-center}

<br>


<br>

도단위로 주문건수를 시각화한 그래프. 경기도가 주문량이 젤 많고, 서울특별시는 도 단위가 아닌데도 주문량이 두번째로 많다.

```python
df_all = pd.concat([df_19,df_20,df_21], axis = 0)
df_pie = (df_all.groupby(['시도'])['주문건수'].sum().sort_values(ascending = False).to_frame() / sum(df_all['주문건수'])) * 100
fig = plt.figure(figsize = (8,15))
plt.pie(df_pie['주문건수'].iloc[:10], labels = df_pie.index[:10], autopct = '%.2f%%',textprops={'fontsize': 14})
plt.show()
```

![image](https://user-images.githubusercontent.com/97672187/171602367-8715e052-63f1-4088-94a5-1905fa258a23.png){: .align-center}

<br>

<br>

시 단위로 주문건수를 시각화한 그래프. 서울특별시가 의정부시 다음으로 주문량이 많다. 

```python
df_gu = df_all.groupby(['구'])['주문건수'].sum()
df_seoul = df_all.loc[df_all['시도'] == '서울특별시'].groupby('시도')['주문건수'].sum()
df_pie2 = pd.concat([df_gu, df_seoul], axis = 0).sort_values(ascending = False)
fig = plt.figure(figsize = (8,15))
plt.pie(df_pie2.iloc[:10], labels = df_pie2.index[:10], autopct = '%.2f%%',textprops={'fontsize': 14})
plt.show()
```

![image](https://user-images.githubusercontent.com/97672187/171602403-3e46f307-bbf2-4ffd-8f24-a401a862d2a8.png){: .align-center}

<br>


<br>

가장 배달이 활발한 의정부시를 사용하면 좋겠지만, 의정부시는 주어진 데이터에서 동 단위로 세분화 되어 있지 않기 때문에 더 정확하고 세분화된 인사이트 도출을 위해
데이터에 구별로 세분화 되어 있는 서울 특별시를 타겟도시로 선정하자.

```python
#결측치와 중복 데이터가 없는 것을 확인
df_all = df_all.loc[df_all['시도'] == '서울특별시']
display(df_all.head())

print(df_all.isnull().sum())
print(df_all.duplicated().sum())
```

![image](https://user-images.githubusercontent.com/97672187/171603831-ca2936f3-c2ce-4e20-9e44-3c36733e3403.png){: .align-center}

<br>


<br>

가장 많이 팔리는 배달음식 종류 확인. 치킨이 제일 많다. 도시락과 심부름은 주문 건수가 1퍼센트 미만이므로 중요하지 않다고 판단해서 제거하자.

```python
df_pie = pd.DataFrame((df_all.groupby('업종')['주문건수'].sum().sort_values(ascending = False) / sum(df_all['주문건수'])) * 100)
df_pie
fig = plt.figure(figsize = (8,15))
plt.pie(df_pie['주문건수'], labels = df_pie.index, autopct = '%.2f%%', textprops={'fontsize': 14})
plt.show()

df_all = df_all.loc[~df_all['업종'].isin(['도시락', '심부름'])]
```

![image](https://user-images.githubusercontent.com/97672187/171604277-472fb28b-4ab3-4ac8-8553-78090625ef4b.png){: .align-center}

<br>


<br>

### 2. Target 변수 선정

타겟도시를 서울 특별시로 선정했고 원래 시간대별 구별 주문건수를 예측하려고 했다. 하지만, 구별 주문건수를 예측하는 것보다 주문이 얼마나 될 것인지의 정도(많음, 보통, 적음)를 예측하는 것이 더 좋은 분석일 것
같다는 생각이 들었다. 만약에 데이터가 지점단위로 세분화 되어있다면 주문의 많고 적음을 나타내는 정도를 예측하는 것보다 그 식당에서 몇건의 주문이 발생할 것이라고 예측하는 것이 더
좋은 분석이 될테지만, 시간대별 구단위로 예측을 하는 것이기 때문에 주문의 건수가 각 지점의 입장에서 크게 와닿지 않을 것 같아서 주문 건수를 기반으로 주문 정도 변수를 생성하고
구별로 주문의 많고 적음을 예측하는 것이 더 유용한 분석이라고 판단했다.

Ex) 내일 마포구에서 **100건** 정도의 주문이 발생할거야! -> 100건이 많은건가???우리 식당에서는 그럼 몇 건 정도 발생하는데??

내일 마포구에서는 주문이 **많이** 들어올거야! -> 내일은 많이 팔리는구나! 평소보다 더 준비를 해야겠다.

```python
#2건까지 주문 적음(1), 4건까지 주문 보통(2), 5건부터 주문 많음으로 하자(3)
df_all['주문건수'].quantile([0.35,0.65,0.85,0.95]) 
```

![image](https://user-images.githubusercontent.com/97672187/171607380-61c151ea-5d06-453c-b213-ff5a2d6c28c5.png){: .align-center}

<br>


<br>


```python
df_all.loc[df_all['주문건수'] <= 2, '주문정도'] = 1
df_all.loc[(df_all['주문건수'] > 2) & (df_all['주문건수'] <= 4), '주문정도'] = 2
df_all.loc[df_all['주문건수'] > 4, '주문정도'] = 3
display(df_all.head())
```

![image](https://user-images.githubusercontent.com/97672187/171607466-a4a2e0c0-fe1e-4ef9-bb99-1f8cf04a4979.png){: .align-center}

<br>


<br>


```python
df_all['주문정도'] = df_all['주문정도'].astype(int)
df_all['주문정도'].value_counts(normalize = True) #class가 엄청 불균형 하지는 않은 것 같다.
```

![image](https://user-images.githubusercontent.com/97672187/171607524-f39e80fb-ccc3-4e46-b9f6-6c0765771ab5.png){: .align-center}

<br>


<br>


```python
#data leak을 피하기 위해 주문 건수는 제거
df_all.drop(columns = ['주문건수'],inplace = True)
```

<br>

<br>

### 3. Baseline 모델(Logistic Regression)
Logistic Regression 모델을 활용하여 baseline 모델을 만들어보자. 이 모델의 성능보다는 더 좋은 성능을 내는 모델을 만드는 것이 이번 프로젝트의 목표가 될 것이다.
주문의 정도인 많음(3), 보통(2), 적음(1)을 예측하고 분류해야 할 대상이 3개이상이므로 다중 분류 모델이 될 것이다.

학습, 검증, 테스트 데이터로 분리

```python
X_train, X_test = train_test_split(df_all, test_size = 0.2, random_state = 42)
X_train, X_val = train_test_split(X_train, test_size = 0.2, random_state = 42)
y_train, y_val, y_test = X_train.pop('주문정도'), X_val.pop('주문정도'), X_test.pop('주문정도')
display(X_train.head())
display(X_val.head())
display(y_train.head())
```

![image](https://user-images.githubusercontent.com/97672187/171608449-56fa8493-6beb-49b9-81d8-552bbde22357.png){: .align-center}

<br>


<br>


```python
#파이프를 넣으면 정확도와 AUC를 계산하는 함수
def accuracy_and_auc(pipe):
    print('학습 정확도', pipe.score(X_train, y_train))
    print('검증 정확도', pipe.score(X_val, y_val))
    y_prob = pipe.predict_proba(X_train)
    print('학습 AUC',roc_auc_score(y_train, y_prob, multi_class="ovr", average="weighted"))

    y_prob = pipe.predict_proba(X_val)
    print('검증 AUC', roc_auc_score(y_val, y_prob, multi_class="ovr", average="weighted"))
    
```

Logistic Regression 모델

```python
pipe_glm = make_pipeline(OneHotEncoder(use_cat_names = True),
                       LogisticRegression(solver='liblinear', multi_class='auto', C=100.0, random_state=42))  
# C는 규제. RIdge와 다르게 작을수록 규제가 큰 것. 과적합 방지

pipe_glm.fit(X_train, y_train)

#과적합은 없지만 정확도는 별로 높지 않다.
accuracy_and_auc(pipe_glm)
```

![image](https://user-images.githubusercontent.com/97672187/171608617-410a16d4-6584-427b-91e0-52352ed03c98.png){: .align-center}

<br>


<br>

추가적인 변수를 사용하지 않고, 다중분류 모델을 만들었을 때 학습 데이터와 검증데이터에서는 정확도가 58%, AUC가 약 0.71 정도가 나왔다.
이 모델보다 더 좋은 성능을 내는 모델을 만들기 위해 다음 포스팅에서는 여러 변수들을 추가한 내용과 전처리 작업을 정리해보겠다.


