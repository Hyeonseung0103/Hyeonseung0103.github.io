---
layout: single
title: "Section1 Sprint3 Note134 Clustering"
categories: Section1
---

비지도 학습의 예시 중 하나인 Clustering에 대해 배웠다. 그리고 많은 Clustering 방식이 있지만, 이번 Posting에서는 K-means와 Hierarchical clustering 알고리즘에 대해 정리해보자.

## Note 134
### Scree Plots
Scree Plot은 PCA 분석 후 주성분(PC, 고유벡터) 수를 선정하기 위해 주성분의 분산 변화를 보는 그래프다. 고유값 변화율이 급격히 완만해지는 부분 전까지가 필요한 주성분의 수이다.

예를 들어, 그래프에서 PC3에서 그래프가 급격히 완만해지면(분산이 다음 분산과 별로 차이가 없을 때 = PC3나 PC4나 분산의 변화가 차이가 없을 때) PC2까지 쓰는게 좋다.

![image](https://user-images.githubusercontent.com/97672187/153151428-e1d4c5ad-c436-4c16-a564-b94498737ea5.png)

```python
import matplotlib.pyplot as plt

scaler = StandardScaler()
Z = scaler.fit_transform(df.loc[:, df.columns != "species"])
pca = PCA(4)
pca.fit(Z)

def scree_plot(pca):
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_ 
    
    ax = plt.subplot()
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals, color = ['#00da75', '#f1c40f',  '#ff6f15', '#3498db']) # Bar plot
    ax.plot(ind, cumvals, color = '#c0392b') # 누적분포 Line plot 
    
    for i in range(num_components): #막대 위에 annotation 추가
        ax.annotate(r"%s" % ((str(vals[i]*100)[:3]+'%')), (ind[i], vals[i]), va = "bottom", ha = "center", fontsize = 13)
     
    ax.set_xlabel("PC")
    ax.set_ylabel("Variance")
    plt.title('Scree plot')
    
scree_plot(pca)
# PC를 2개 고르면 약 87%, 3개 고르면 약 96% 까지 데이터를 설명할 수 있다. 
# 많이 고를수록 설명력은 높아지지만, 차원 축소한 메리트가 사라지게 됨.
# q보통 누적 기여율이 85% 이상이면 해당 지점까지 주성분의 수로 결정하게 됨.
# 90% 설명을 하기 위해서는 PC를 3개 선택해야 하지만, 85% 기준을 충족시킨 2개를 선택하는 것이 더 나을 것 같다.
```

### 강화학습(Reinforcement Learning)
머신러닝의 한 형태로, 모델이 좋은 성능을 내면 해당 파라미터에 보상, 그렇지 않으면 패널티를 줌으로써 학습해 나가는 형태.

### Clustering
비지도 학습의 예시 중 하나로, 정답이 없이 주어진 데이터로부터 패턴을 파악하여 유사한 패턴을 가진 그룹끼리 군집화 시키는 것.
정답이 없기 때문에 예측을 위한 모델링 보다는, 지도학습 전에 EDA를 위해 많이 쓰인다.

**Clustering의 종류**

**Hirarchical clustering(계층 군집화)**

- Agglomerative clustering(병합 군집화): 개별 포인트에서 시작후 점점 크게 합쳐감. 군집에 데이터를 추가 하면서 군집을 형성

- Divisive clustering(분리 군집화): 한개의 큰 군집에서 점점 작은 군집으로 나누어감.

Agglomeriative clustering 코드 구현

- ward: 군집 내에서의 분산을 최소화 하는 접근법, 각 군집의 관측치 중 분산을 가장 작게 증가시키는 두 클러스터를 합침

- average: 군집 쌍의 모든 관측치 간 거리의 평균을 최소화. 각 군집의 관측치 중 평균거리가 가장 짧은 두 클러스터를 합침

- complete 군집쌍의 관측치 사이의 최대 거리 최소화. 각 군집의 관측치 중 최대거리가 가장 짧은 두 클러스터를 합침

- single: 군집 쌍의 가장 가까운 관측치 사이 거리 최소화. 각 군집의 관측치 중 최소거리가 가장 짧은 두 클러스터를 합침

```python
#1) sklearn패키지로 ward, complete, single, average를 사용하여 각각 다른 clustering 해보기

from matplotlib import pyplot as plt
from time import time
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

#features는 이미 그 전에 표준화 작업을 다 해놓은 DF
#표준화가 안 되어있으면 표준화 시키고 군집화를 해야함.

#ARI는 군집화에서 얼마나 잘 맞췄는지 지표
clustering_ari = []
linkage_settings = ["ward", "average", "complete", "single"]
for linkage in linkage_settings:
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=2)
    clustering.fit(features)
    label3 =  np.where(clustering.labels_ == 0,'M','B')
    clustering_ari.append(adjusted_rand_score(label, clustering.labels_))
    plt.scatter(features.iloc[:,2], features.iloc[:,5],s = 20, c = label3, cmap = 'rainbow') # 무작위로 변수 뽑아서 군집 분포비교
    plt.title(linkage, size = 17)
    plt.show()
d = {'linkage' : linkage_settings, 'ARI': clustering_ari}
pd.DataFrame(d)
#ARI 결과 ward방식이 이 데이터에서는 젤 낫다는 것을 알 수 있음
```

```python
#2) Scipy패키지로 계층 군집화 구현. k가 없어도 군집화 해줌.
#Dendogram으로 군집을 표현할 수 있음
# 여기서는 합당한 cluster 갯수를 2라고 함
# 파란색 선 밑에가 합당한 클러스터 갯수
# scipy 라이브러리로 적절한 클러스터 갯수를 파악할 수 있을까?
# 계층 군집화하려면
# 이 dendogram을 먼저 그린 후에 agglomerative clustering을 하면 도움 되지 않을까

##보통 k-means 를 돌리기전에 hirearchical clustering을 먼저하고(or Elbow methods)를 통해 최적의 k를 찾은 뒤 
## 이 k를 참고해서 k-means를 돌릴 수 있다.

import scipy.cluster.hierarchy as shc
plt.title("Diagnosis Dendograms using method")
pd.DataFrame(shc.linkage(features, metric = 'euclidean',method='ward'))
dend = shc.dendrogram(shc.linkage(features, method='ward'))
```

Point Assignment Clustering
- 시작시에 cluster의 수를 정하고, 데이터들을 하나씩 cluster에 할당시킴(ex. K-means)

이 두가지와 다른 기준으로 clustering을 나누면 Hard vs Soft로도 나눌 수 있다.

- Hard clustering: 데이터는 하나의 군집에만 할당.
- Soft clustering: 데이터가 여러 군집에 확률을 가지고 할당.

일반적으로 군집화는 Hard clustering을 말한다.

### Similarity(유사도)
군집화를 할 때 데이터가 해당 군집 즉, 해당 데이터들의 특성과 얼마나 유사하냐를 판단하는 기준이다. Euclidean, Cosine, Jaccard 등 여러가지가 있으나 보통
Eucledian 거리를 많이 활용한다.

![image](https://user-images.githubusercontent.com/97672187/153155122-0b6a2238-c2da-46c0-bdc7-627c5c6b637c.png)

이미지 출처: https://en.wikipedia.org/wiki/Euclidean_distance

### K-means Clustering
과정

1) 관찰자가 k개의 cluster 갯수를 설정하고, 이 k에 따라 데이터 내에 랜덤하게 centroid가 설정된다

2) 해당 centroid 와 가장 유사한(거리가 가장 가까운) 데이터를 같은 cluster로 할당한다.

3) 군집에 변화가 생기면 각 군집 내의 새로운 centroid를 계산한다.

4) 2,3 과정을 유의미한 차이가 없을 때까지 계속 반복한다.

코드구현

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 
from google.colab import files
import io

uploaded = files.upload()
df = pd.read_csv(io.StringIO(uploaded['data.csv'].decode('cp949')))
df = df.drop(['Unnamed: 32'],axis = 1)

s = StandardScaler()
features = df.loc[:, 'radius_mean':'fractal_dimension_worst'] #numeric 변수만 추출
label = df['diagnosis'] # 나중에 결과를 확인할 정답
features = pd.DataFrame(s.fit_transform(features),columns = features.columns)

#Kmeans Algorhithm
kmeans = KMeans(n_clusters = 2,random_state = 42) #여기서는 나와야하는 label을 B,M으로 이미 알고 있으니까 2
kmeans.fit(features)
label2 = kmeans.labels_ #나온 군집
np.unique(label2, return_counts = True)

import sklearn.metrics as metrics

label2 = np.where(label2 == 0,'M','B') #군집이 1,0으로 나와서 M,B로 바꿔줌
np.unique(label2, return_counts = True)

np.mean(np.equal(label,label2)) #accuracy 계산
```

K-means 단점: 적절한 k의 갯수를 관찰자가 정해야 한다. 랜덤하게 부여된 centroid에 따라 성능의 차이가 있다. convex하지 않은 데이터(데이터의 분포가 원형을 띄거나 나선을 띄는경우)에 대해서는 
좋은 성능을 내지 못한다.

### Elbow methods
kmeans는 결과적으로 관찰자가 정한 k개의 군집이 형성된다. 그렇다면 정답이 없을때 적절한 k는 어떻게 구할 수 있을까?

Elbow methods는 군집내의 관측치들의 거리합을 나타내는 itertia의 감소율이 급격히 작아지면 유의미한 군집화가 더 이상 이루어지지 않는다고 판단한다. 따라서 itertia의 기울기가 급격하게 완만해지기 전까지의 
군집의 갯수를 최적의 k라고 판단하는 방법이다. 밑의 예시에서는 3부터 기울기가 급격하게 낮아졌기 때문에 k를 3으로 선택한다.

```python
sum_of_squared_distances = []
K = range(1, 15)
for k in K:
    km = KMeans(n_clusters = k)
    km = km.fit(points)
    sum_of_squared_distances.append(km.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
```
Elbow methods는 날카로운 팔꿈치가 존재하지 않을 수가 있고, 급격히 완만해지는 구간은 주관적이라는 단점 또한 있다.

Elbow 외에도 다음과 같은 방법들을 활용할 수 있음.

https://steadiness-193.tistory.com/285

### Discussion
1. ML에서 Supervised , Unsupervised Learning의 차이는?

답변:

Supervised learning은 모델링을 할 때 모델에게 정답을 알려주면서 학습시키는 것을 말한다. 관측치가 주어지면 이 관측치가 정답을 잘 예측할 수 있도록 매칭시킨다. label의 형태에 따라 분류와 회귀로 나눌 수 있고, 분류는 사용자의 영상 시청여부 파악, 회귀는 학생의 점수 예측 등이 예시가 된다.
Unsupervised learning은 모델에게 정답을 알려주지 않고 주어진 데이터의 패턴으로 데이터를 분류하는 것을 말한다. 정답을 모르기 때문에 오로지 데이터의 특성에 따라 군집화나 차원을 축소 할 수 있고, 예시로는 PCA와 Clustering이 있다. 관찰자가 눈으로는 파악하기 힘든 특성을 비지도 학습을 통해 파악할 수 있다.

2.K-means Clustering을 활용해 아래처럼 데이터를 분류해 보았습니다.
아래에서 볼 수 있듯이, Circles/Moons 형태 의 데이터는 분류가 잘 되지 않습니다. 왜 그럴까요?
그리고 두 형태에는 어떤 Clustering 방법을 사용하는 것이 좋을까요?

답변:

K-means는 군집의 형태가 convex(볼록)하지 않거나(동그랗거나, U 형태의 군집) 군집들의 크기가 매우 다른 군집이 존재하면 성능이 떨어진다. Circles와 Moons 같은 경우 군집이 U형태를 띄고 있는 것 같다. 따라서 k-means로 이를 예측하면 군집을 잘 예측하기가 힘들다.
k-means의 단점은 관찰자의 주관으로 k를 정해야 한다는 것과, 초기에 랜덤하게 설정된 centroid에 따라 성능이 크게 좌우 될 수 있다는 것이다.

Circles와 Moons의 데이터와 같은 경우 DBSCAN이라는 알고리즘을 사용하면 좋을 것 같다. DBSCAN은 임의의 점 p를 설정하고 p를 중심으로 주어진 클러스터의 반경 안에 포함 된 관측치의 갯수를 센다. 만일 해당원에 내가 지정한 갯수 이상의 점이 포함 되어 있으면 해당 점 p를 핵심 포인트로 간주하고 원에 포함된 점들을 하나의 클러스터로 묶는다. 지정한 갯수 미만이면 pass한다. 이 과정을 계속 반복하는데 만약 새로운 점 p'가 기존의 클러스터에 이미 포함된 점이라면 두 클러스터는 연결되어 있다고 판단하여 하나의 클러스터로 묶는다. 모든 점에 대하여 클러스터링이 끝났는데 아직 군집화에 속하지 못한 점이 있으면 noise로 간주한다.

DBSCAN은 K-means와 다르게 군집이 합쳐질 수 있다. 따라서 k-means에 비해 초기에 랜덤하게 설정 된 centroid의 영향을 훨씬 작게 받을 수 있을 것이다. 따라서 원하는 반경과 최소 갯수만 잘 지정한다면 k-means보다 더 좋은 성능을 가진 모델을 만들 수 있다. 하지만, 데이터의 수가 많아질수록 알고리즘 수행시간이 급격히 상승하고, k(클러스터 수)를 지정할 필요는 없지만 데이터 분포에 맞게 최소 갯수와 반경을 지정해주어야한다는 단점이 있다.

