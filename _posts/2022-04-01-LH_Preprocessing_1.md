---
layout: single
title: "전처리 Part.1"
toc: true
toc_sticky: true
category: LH
---

"대전광역시 교통사고 핫스팟 지역 100개 도출"을 주제로 한 LH 주관 공모전에 참가했다. 전국단위의 공모전이라 많은 참가팀들이 있었지만, 약 한달이라는 시간동안 열심히 준비한 결과
장려상을 수상했다.

지역은 사각형으로 표현되는 격자를 의미하고, 데이터분석을 통해 교통사고가 가장 잘 일어날 것 같은 격자(사각형) 100개를 도출하는 것이 최종 목표이다.
5명이 한 팀으로 참가했고 언어는 R과 파이썬을 사용했다. 코드가 매우 길지만 전처리부터 최종결과 도출까지 최대한 쉽게 정리해봐야겠다.

## 전처리 Part.1(by using Python)

```python
# 필요한 라이브러리 import

import platform
import folium
import math
from tqdm import tqdm
from shapely import wkt
from shapely.geometry import Point, Polygon, LineString

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

if platform.system() == 'Darwin': # 맥
        plt.rc('font', family='AppleGothic') 
elif platform.system() == 'Windows': # 윈도우
        plt.rc('font', family='Malgun Gothic') 
elif platform.system() == 'Linux': # 리눅스 (구글 Colab)
        plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings(action = 'ignore')
```

#### 1) 도로가 지나가는 격자만 추출

대전시의 격자는 약 54,000개가 있다. 하지만 이중에는 도로가 지나가지 않는 산지 등의 격자도 있기 때문에 실제로 교통사고가 일어날 확률이 높은
**도로가 지나다니는 격자만 추출**하도록 한다.

```python
accident_grid = gpd.read_file('2.대전광역시_교통사고격자(2017~2019).geojson')
road_detail = gpd.read_file('19.대전광역시_상세도로망(2018).geojson')

print(np.shape(accident_grid))
print(np.shape(road_detail))
```

![image](https://user-images.githubusercontent.com/97672187/161378123-083fcda6-d8a9-4bc2-a6b1-d08be80d263a.png){: .align-center}

Crosses 함수를 사용하면 교통사고격자에 상세 도로망이 지나가는지를 알 수 있다. 도로가 지나가는 격자만 추출해서 한 격자에 어떤 도로가 지나가는지의 정보를 추가한다.

```python
# 교통사고격자와 상세도로망 데이터에서 서로 교차하는게 있는 부분만 추출하기
new  = []
accident_gid = pd.DataFrame()
for i in tqdm(range(len(accident_grid))): 
    for j in range(len(road_detail)): 
        if accident_grid.loc[i, 'geometry'].crosses(road_detail.loc[j, 'geometry']) : # 교통사고격자에 도로가 지나간다면
            accident_gid = pd.concat([accident_gid, accident_grid.loc[[i, ]]])
            new += list(road_detail.loc[j, 'geometry'])
    accident_gid['geometry_2'] = new
accident_gid.to_csv('accident_gid.csv')

accident_gid = pd.read_csv('accident_gid.csv', index_col = 0)
accident_gid['geometry'] = [wkt.loads(line) for line in list(accident_gid['geometry'])] # geometry type으로 변환
accident_gid = accident_gid[['gid', 'acci_cnt', 'geometry']]
print(np.shape(accident_gid))
accident_gid.head()
print(len(accident_gid['gid'].unique()))
```

![image](https://user-images.githubusercontent.com/97672187/161378237-cc19d7d5-dd33-43b3-85fd-e6d7807d777e.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/161378321-fb627e0d-e179-4afb-a8e2-78b9b3879efe.png){: .align-center}

약 54,000가지의 격자 중 실제 도로가 지나가는 격자는 25,000개 정도였다. 그리고 추려진 25,000개 격자 중 unique한 격자는 11,870개였다.

folium 함수를 사용해서 지도 위에 격자들을 표시했다. 줌을 확대하면 사각형 형태의 격자가 보이는 것을 확인할 수가 있다.

```python
center = [36.348315, 127.390594]

mks = folium.Map(location=center, zoom_start=10)
for i in list(accident_gid.index) : 
    folium.GeoJson(data=accident_gid.loc[i,'geometry']).add_to(mks)
mks
```

![image](https://user-images.githubusercontent.com/97672187/161378367-4478f48c-b0a9-4383-9365-c060b4cc56b4.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/161378413-a9e6810b-bc20-478a-bed7-e50eb931cde3.png){: .align-center}

#### 2) 격자별 사고 건수, 사상자 수 계산 후, 위에서 추려진 accident_gid 데이터와 병합

교통사고 내역 데이터를 불러온 후 격자별로 사고건수, 사상자 수 등을 계산했다. 사고건수는 격자별로 빈도수를 사용했고, 사상자 수는 사망자, 중상자, 경상자수를 합쳐서 만들었다.

```python
accident_list = pd.read_csv('1.대전광역시_교통사고내역(2017~2019).csv')

acci_count = accident_list.groupby('gid').count()
acci_count.reset_index(inplace=True)
acci_count = acci_count[['gid', '사고일']]
print(np.shape(acci_count))
```

![image](https://user-images.githubusercontent.com/97672187/161378513-b21ddaa3-34ae-454c-b772-92a9f67c673c.png){: .align-center}


```python
accident_list['사상자수'] = accident_list['사망자수'] + accident_list['중상자수'] + accident_list['경상자수']

acci_injury = accident_list.groupby('gid').sum('사상자수')
acci_injury = acci_injury[['사망자수', '중상자수', '경상자수', '사상자수']]
acci_injury.reset_index(inplace=True)
acci_injury = acci_injury[['gid', '사상자수']]

acci_count['사상자수'] = acci_injury['사상자수']
acci_count.columns = ['gid', '사고건수', '사상자수']
print(np.shape(acci_count))
acci_count.head() # 격자별로 사고건수와 사상자수(사망자수, 중상자수, 경상자수의 합)를 계산
```

![image](https://user-images.githubusercontent.com/97672187/161378558-c7fbf350-d8f2-450d-bc31-ac7f61cd5a80.png){: .align-center}


geometry 변수는 다각형인 Polygon 형태로 되어있기 때문에 centroid 함수를 사용해서 위도와 경도의 가운데 지점을 찾아 해당 격자의 위도, 경도를 구했다.

```python
# 사고건수 및 사상자수 데이터를 첫번째 과정에서 구했던 도로가 지나는 격자 데이터와 merge
acci_count = accident_gid.merge(acci_count, on='gid', how='left')
acci_count['사고건수'] = acci_count['사고건수'].fillna(0) # 사고건수가 없어서 NaN이 된 지역을 0으로 변경
acci_count['사상자수'] = acci_count['사상자수'].fillna(0)

acci_count['x'] = [acci_count.loc[c,'geometry'].centroid.x for c in list(acci_count.index)] # 위도, 경도 추가
acci_count['y'] = [acci_count.loc[c,'geometry'].centroid.y for c in list(acci_count.index)]

acci_count = acci_count.loc[list(acci_count['gid'].drop_duplicates().index), ].reset_index(drop = True) # 겹치는 데이터 제외
acci_count.to_csv('accident_count.csv')
print(np.shape(acci_count))
acci_count.head()
```

![image](https://user-images.githubusercontent.com/97672187/161378774-bbf7b3e2-5ae4-41b3-9514-7d5c96a237d6.png){: .align-center}

밑의 그래프를 보면 사고가 한 번도 발생하지 않은 격자가 6900개 정도로 **이 데이터는 불균형한 상태**라는 것을 알 수 있다.

```python
sum(acci_count['사고건수'] == 0)
```

![image](https://user-images.githubusercontent.com/97672187/161378806-f45d5752-6eaa-4cf0-98e3-42f22c1d4913.png){: .align-center}


```python
plt.bar(acci_count['사고건수'].value_counts().index, acci_count['사고건수'].value_counts())
plt.show()
```

![image](https://user-images.githubusercontent.com/97672187/161378819-83255c8e-f526-440a-907e-70bd984eaa82.png){: .align-center}

#### 3) 교통사고격자 데이터에 위도와 경도 추가

```python
accident_grid['x'] = [accident_grid.loc[c,'geometry'].centroid.x for c in list(accident_grid.index)]
accident_grid['y'] = [accident_grid.loc[c,'geometry'].centroid.y for c in list(accident_grid.index)]
accident_grid[['gid', 'acci_cnt', 'x', 'y']].to_csv('accident_grid.csv')
print(np.shape(accident_grid))
accident_grid.head()
```

![image](https://user-images.githubusercontent.com/97672187/161378905-2d12765b-3a12-4b33-85db-331c9950640e.png){: .align-center}
