---
layout: single
title: "Section1 Sprint2 Note 123 Confidence Interval"
categories:
  - Section1
toc: true
toc_sticky: true
---

Note123 에서는 ANOVA, 신뢰 구간, 큰 수의 법칙, 중심 극한 정리 등의 개념을 배웠다.

## ANOVA

지난 posting 에서는 모집단이 정규 분포를 따른다는 가정하에 표본의 평균을 이용하여 가설을 검정할 수 있는 T-test, 
모집단이 정규 분포를 따르지 않을 때 표본의 분포를 이용하여 가설을 검정하는 기법인 Chi-square test 대해 공부했다.
두 가지 기법 모두 최대로 비교할 수 있는 데이터가 2개라는 한계가 존재했는데, 그렇다면 3개의 표본 그룹을 비교하려면 어떻게 해야할까?

ANOVA Test는 3개 이상의 표본 그룹의 평균을 비교하여 그룹들의 평균의 차이가 유의미한지 통계적으로 검증하는 기법이다. ANOVA는 T-test와 같이
모집단이 정규분포를 따라야하며 각 표본은 독립적이어야 한다는 가정이 있어야 한다.

귀무가설: 표본 그룹의 평균은 모두 동일할 것이다.( p-value > 0.05, 그룹들 간 유의미한 차이가 없을 것이다.)

대립가설: 표본 그룹 중 적어도 한 그룹의 평균은 다른 그룹의 평균들과 다를 것이다.

- 여러 그룹간에 유의미한 차이가 있다는 것은?

각 그룹끼리는 모여있고, 다른 그룹과는 명확하게 떨어져 있어야한다. (각 그룹끼리의 분산은 작아야 하고, 다른 그룹과의 분산은 커야한다.)
이것을 수식으로 표현하면,

![image](https://user-images.githubusercontent.com/97672187/151537943-18895462-8d60-498c-add3-1dd6ae180ab6.png)

따라서 F 값이 크다는 것은 분자가 크고(다른 그룹끼리의 분산이 크고), 분모는 작은 것(그룹 내의 분산은 작고)이다.

-> 그룹이 명확히 나눠져 있다는 것 = 그룹간의 차이가 유의미하게 나타난다. (p-value < 0.05, 귀무가설 기각)

* Anova를 사용하는 이유

Multiple Comparison: 3개 이상의 그룹을 비교하기 위해 2개씩 여러번 비교하는 것.

만약 A,B,C 그룹을 비교하기 위해 T-test를 사용해서 A vs B, A vs C, B vs C로 비교하고 각각 에러가 날 확률을 5% 라고 하자.

그렇다면 세개의 그룹에서 15% 에러가 발생하고 그룹이 10개면 50%가 될 것이다. 따라서 그룹을 기존의 방법으로 여러번 비교하는 것은 좋은 방법이 아니다.
Anova는 그룹을 한 번에 비교하기 때문에 Multiple Comparison 보다 에러가 더 적게 발생한다.

```python
from scipy.stats import f_oneway
f_oneway(g1, g2, g3)
```

pvalue > 0.05, 세 개의 그룹의 평균이 유의미한 차이가 없다.

pvalue < 0.05, 세 개의 그룹의 평균이 유의미한 차이가 있다.

## 큰 수의 법칙(Law of large numbers)

Sample 데이터의 수가 커질수록, sample의 통계치는 점점 모집단의 모수(모집단의 특성: 평균, 표준편차 등)와 같아진다.

어떻게 보면 당연한 말이다. 모집단이 100개라고 하면, 10개의 샘플보다 70개의 샘플의 통계치가 더 모집단의 모수와 같아지겠지.

## 중심극한정리(Central Limit Theorem, CLT)

Sample의 수가 커질수록(샘플을 여러번 반복해서 추출할수록), 각 샘플 그룹의 평균들의 분포는 정규분포를 따른다.

Sample의 수라는게 한 샘플 안에 있는 갯수가 아니라, 샘플을 여러번 반복해서 추출하는 것을 말함. 샘플을 여러번 반복해서 추출하고 각 샘플들의 평균을 기록하고,

이 샘플들의 평균 데이터를 분포로 표현하면 정규분포가 나온다.

CLT가 중요한 이유!

모집단이 어떤 분포를 가지고 있던지 간에 표본의 크기가 충분히 크다면, 표본평균들의 분포가 모집단의 모수를 기반으로 한 정규분포를 이룬다는 점을 이용하여, 특정 사건이 일어날 확률값을 계산할 수 있게 된다. 수집한 표본의 통계량을 이용하여 모집단의 모수를 추정하는 수학적 근거가 마련된다.

## Point estimate vs Interval estimate

값을 예측할 때 하나의 값으로 예측하면 Point, 구간으로 하면 Interval

ex) 메시의 올 시즌 골 수는 40골일거야(point), 메시의 올 시즌 골 수는 30~40골일거야(interval)

## 신뢰도(Confidence level)

표본을 통해 계산한 신뢰 구간 안에 모집단의 평균이 얼마나 잘 포함되었는지 신뢰할 수 있는 정도.

## 신뢰구간(Confidence Interval)

모집단의 평균과 표준 편차를 모르니까 표본으로부터 평균->구간을 구하고 이 구간 내에 모평균이 얼만큼의 확률로 포함되어 있는지를 믿을 수 있는 구간.

ex) 95% 신뢰구간을 사용한다면, 표본를 통해 하나의 신뢰구간을 구하고 이 구간내에 모수가 있을 확률이 95%라는 것.

100개의 표본 데이터를 통해 하나의 신뢰 구간 뽑고 이 구간내에 모평균이 포함되어 있을 것이다 라는 가정.

```python
from scipy.stats import t

# 표본의 크기
n = len(sample)
# 자유도
dof = n-1
# 표본의 평균
mean = np.mean(sample)
# 표본의 표준편차, ddof는 편차를 계산할 때, n-1로 나누라는 뜻. 보통 1을 많이씀
sample_std = np.std(sample, ddof = 1)
# 표준 오차
std_err = sample_std / n ** 0.5 # sample_std / sqrt(n)

#신뢰구간(신뢰도, 자유도, 평균, scale(표준오차))
t.interval(.95, dof, loc = mean, scale = std_err)
```
## 신뢰구간 시각화

1) Using plotlib

```python
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure() #그래프의 크기를 조절해준다
ax = fig.add_axes([0,0,1,1])

pop_mean = df['오존(ppm)'].mean() #모집단의 평균, 신뢰 구간과 비교하기 위해서.
print(pop_mean)

plt.bar('s1', s1_mean, yerr = CI1[0]-s1_mean,capsize = 10) # yerr은 오차막대, capsize는 오차막대의 넓이
#신뢰구간 low와 평균과의 차이가 오차가 됨.
plt.bar('s2', s2_mean,yerr = CI2[0]-s2_mean,capsize = 10)
plt.axhline(pop_mean, linestyle='--', color='#4000c7')
plt.axhline(s1_mean, color = 'black', xmin = 0.15, xmax = 0.35) #axhline은 가로선. 막대 끝에 검은색 선을 그리기 위함, 세로선은 axvline
plt.axhline(s2_mean, color = 'black', xmin = 0.65, xmax = 0.85)

#s2가 표본의 크기가 더 크기 때문에 신뢰구간이 더 좁음.(똑같이 alpha값이 0.95인 상황에서는 신뢰구간이 좁을수록 더 정확함)
#메시의 골수가 50~60골 사이일 거야가 95퍼센트의 확률로 맞는 것보다 메시의 골수가 50~55골 사이일 거야가 95퍼센트 확률로 맞는게 더 정확하고 좋은 분석.
plt.show()
```

![image](https://user-images.githubusercontent.com/97672187/151541853-0fb0ce5f-2a18-4dc8-8fc9-69051866e615.png)


2) Using seaborn (seaborn 한 그래프에 두 가지 데이터를 추가 못함. 그래서 두 가지 막대 표현하려면 DF에서 변수 하나 만들어서 데이터를 넣어줘야함)

```python

import seaborn as sns
#seaborn bar graph는 기본적으로 오차 막대를 그려준다.

df_graph = pd.DataFrame({'sample' : ['s1','s2'], 'mean' : [s1_mean,s2_mean]})
fig = plt.figure() #그래프의 크기를 조절해준다
ax = fig.add_axes([0,0,1,1])

s1['sample'] = 's1'
s2['sample'] = 's2'
df_sample = pd.concat([s1,s2], axis = 0)

ax = sns.barplot(x = 'sample',y= '오존(ppm)',data=df_sample,capsize = 0.1)
plt.axhline(pop_mean, linestyle='--', color='#4000c7')
plt.axhline(s1_mean, color = 'black', xmin = 0.15, xmax = 0.35)
plt.axhline(s2_mean, color = 'black', xmin = 0.65, xmax = 0.85)
plt.show()

```



## Discussion

Confidence interval을 (statistics에 대한 기초 지식이 없는) 초등학생이 이해할 수 있도록 예시를 들어 설명해주세요.

답변:

신뢰 구간이란 내가 모르는 친구들의 수학점수의 평균이 내가 예상한 친구들의 수학 점수 범위 안에 있다고 믿을 수 있는 구간이라고 할 수 있다. 한 학년 전체가 100명이라고 하면 내가 우리반인 25명의 친구들의 수학 점수만 가지고 구간을 예상(아마 70 ~ 80점 사이일 거야!)하고, 이 구간 안에 95퍼센트의 확률로 100명의 수학평균이 포함(95퍼센트의 확률로 100명의 수학평균이 70~80점 사이일 것이다.)된다고 예측하는 것이다. 그럼 이 신뢰구간을 95퍼센트의 확률로 신뢰할 수 있다고 얘기한다. 보통 신뢰구간을 95퍼센트로 많이 사용하지만, 90퍼센트, 99퍼센트도 있다.

그렇다면 신뢰구간이 99퍼센트면 더 정확하니까 좋은게 아닌가? 정확하긴 하지만 더 좋은 예측이라고 하긴 힘들다. 예를 들어 위에서 95퍼센트 신뢰구간을 위해 점수를 70 ~ 80점으로 사용했었는데, 더 정확한 예측을 위해 '친구들의 수학점수는 60 ~ 80 사이일 거야!'라고 하면, 어지간하면 다 이 사이의 점수를 형성할테니까 좋은 예측이라고 말하기는 힘들다. 이말은 실제 100명의 친구들의 수학 점수 평균이 진짜 얼마일것인지 예상하는게 힘들다는 뜻이다. 신뢰구간이 넓다 = 실제로 예상해야 할 범위가 넓다.

또 반대로, 신뢰구간이 90퍼센트면(예를 들어, 75~80점 사이일 거야!) 실제 100명의 친구들의 수학점수 평균을 예상하는 데에는 도움이 되지만, 그 구간을 신뢰하기가 어렵다. 구간이 좁을수록 신뢰하기가 힘들어진다.

따라서 적절한 신뢰구간이 정해져 있기보다는 상황에 따라 각각 다른 신뢰 구간을 사용해서 실제 우리가 원하고자 하는 값을 예상하는데 도움이 되도록 하는것이 중요하다.

