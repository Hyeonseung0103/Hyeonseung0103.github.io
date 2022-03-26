---
layout: single
title: "Section2 Sprint3 Note231,232 ML Problems and Data Wrangling"
category: Section2
toc: true
toc_sticky: true
---

지난 이틀동안 ML에서 일어날 수 있는 문제들과, Data Wrangling에 대해 배웠다. 또한, 불균형한 데이터를 처리하는 방법에 대해서도 알 수 있었다.

### 불균형 데이터 처리
회귀에서든 분류에서든 타겟의 분포를 확인하는 것이 중요하다. 타겟이 정규분포를 따른다면 큰 문제가 없지만, 값이 한 쪽으로 치우쳐져 있으면 이상치의 영향을 많이 받는다는 뜻이므로
성능이 저하될 수 있다.


```python
#데이터 분포 확인(회귀, 분류 상관 X)
sns.displot(target)

```

분류에서는 범주별 가중치를 계산하여 모델 학습에 비율이 작은 범주에 더 큰 가중치를 부여하고, 비율이 큰 범주에 더 작은 가중치를 부여해 불균형을 해소할 수 있다.

**class weight = n_samples / (n_classes * np.bincount(y))**

ex) RandomForest의 class_weight 파라미터

```python
cw = len(y_train) / (2*np.bincount(y_train))
RandomForestClassifier(class_weight={False:cw[0],True:cw[1]}, random_state=42)

```

![image](https://user-images.githubusercontent.com/97672187/157814245-d4cb18bd-5b46-4c63-abcf-86ce8a01c4d9.png){: .align-center}

회귀에서는 이상치를 제거하는 방법도 있지만, 타겟값의 분포가 Positively skewed(right skewed, 오른쪽이 꼬리가 나와있음)한 경우 Log를 씌워 정규분포 형태로 변환 할 수 있다.

Standard Scaler vs 분포변환(ex. Log)

Standard Scaler 사용할 때: 회귀분석에서 규제항이 있을 때(ex. Ridge), 독립변수의 중요도를 파악할 때.
결국, 규제항을 주는 것은 중요한 변수에 더 큰 규제를 주게 되는데 단위가 다르면 같은 데이터라도 다른 중요도를 가지게 되니까 scaling 필요. 독립변수 중요도도 비슷한 개념.

분포 변환(Log 사용): 데이터가 불균형할 때, 불균형한 데이터를 정규분포 형태로 바꿔주기 위해서. 데이터가 치우쳐져 있으면 이상치에 영향을 많이 받아 예측값이 왜곡될 수 있기 때문에 분포 변환이 필요하다. 데이터가 평균에 모여있는 분포로. Log를 씌우면 데이터간 편차가 줄어들기 때문에 데이터가 한쪽으로 치우쳐진 정도인 왜도와 데이터가 뾰족한 정도인 첨도를 줄일 수 있기 때문에 정규성이 높아져서 더 정확한 분석이 가능해진다. 왜도가 높으면 데이터가 한 쪽으로 치우처진 정도가 크고, 첨도가 높으면 데이터의 분포가 뾰족해서 데이터의 꼬리 부분이 더 무겁고 이는 곧 이상치가 많은 것이 된다.

![image](https://user-images.githubusercontent.com/97672187/160058216-f5d41747-1bfd-4adc-bfe2-6db6f1f110ce.png){: .align-center}


또한, 자연로그를 취하면 독립변수와 종속변수의 관계가 기존에 비선형인 데이터를 선형 관계로 만들 수 있었기 때문에 회귀분석에서도 유용하게 사용된다.

```python
#로그변환
np.log1p(target)

#로그를 원래대로
np.expm1(np.log1p(target))

#Transformed Target Regressor
t = TransformedTargetRegressor(regressor = lm,
                                func=np.log1p, inverse_func=np.expm1)
#이렇게 하면 로그로 타겟을 변환시켜서 학습함.

```

## Note 232

### Data Wrangling
분석을 하거나 모델을 만들기 전에 데이터를 사용하기 쉽게 변형, 맵핑하는 과정(전처리). 모델이 돌아가기만 한다면 정해진 데이터 형식은 없고, 해당 모델에 최적의 데이터형태를 만드는 것이 중요하다.

### 최빈값 함수

```python
df['변수'].mode()
```

### get_group 함수

```python
#원하는 그룹만 가져와서 계산할 수 있음.
#이때 선택하는 그룹은 내가 그룹으로 지정한 수와 같아야한다.
#2개로 group을 묶었으니까, 내가 선택하는 group도 각각 1개씩 해야함.
df.groupby(['Age','Gender']).get_group((20,'Male'))['Price'].sum()

```

### lambda
변수를 입력으로 사용하기 위하여 간단하게 lambda 함수 사용 가능.

```python
df['변수'].apply(lambda x: x.split())

```



