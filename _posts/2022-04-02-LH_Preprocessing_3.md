---
layout: single
title: "전처리 Part.3"
category: LH
toc: true
toc_sticky: true
---

전처리 Part.3에서는 격자별로 신호등이나 CCTV와 같은 교통시설물들에 대한 정보와 일별, 시간별 혼잡도 데이터를 추가했다.

## 전처리 Part.3(by using Python)

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

```python
#데이터 불러오기
daejeon_signal_walk = gpd.read_file('3.대전광역시_신호등(보행등).geojson')
daejeon_signal_car = gpd.read_file('4.대전광역시_신호등(차량등).geojson')
cctv = gpd.read_file('10.대전광역시_교통CCTV.geojson')
weather = pd.read_csv('16.대전광역시_기상데이터(2017~2019).csv')
road_detail = gpd.read_file('19.대전광역시_상세도로망(2018).geojson')
day_time_accident = pd.read_csv('20.대전광역시_평일_일별_시간대별_추정교통량(2018).csv')
day_time_confuse = pd.read_csv('21.대전광역시_평일_일별_혼잡빈도강도(2018).csv')
day_time_times = pd.read_csv('22.대전광역시_평일_일별_혼잡시간강도(2018).csv')
```

<br>



<br>


### 5. accident_count_filter 데이터 불러오기(유의미한 격자만 추려낸 데이터)

도로가 지나다니는 격자인 accident_gid 데이터와 모란지수를 활용해서 유의미한 격자만 추려낸 accident_count_filter 데이터를 불러온다.

```python
accident_gid = pd.read_csv('accident_gid.csv', index_col = 0)
print(np.shape(accident_gid))
accident_gid.head(3)
```

![image](https://user-images.githubusercontent.com/97672187/161382072-f32d10bc-0dce-49f3-b3a4-9099934902b0.png){: .align-center}


<br><br>


```python
accident_count_filter = pd.read_csv('accident_count_filter.csv')
accident_count_filter['geometry'] = [wkt.loads(line) for line in list(accident_count_filter['geometry'])]
print(np.shape(accident_count_filter))
accident_count_filter.head(3)
```

![image](https://user-images.githubusercontent.com/97672187/161382097-b3c495eb-48ef-465f-a4f0-aa47a24f08b5.png){: .align-center}


<br>



<br>


### 6. 교통안전시설물 - 보행자 신호등의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합

get_gid함수는 within 함수를 사용해서 신호등 좌표와 격자좌표가 교차하면 그 신호등에 해당격자를 맵핑해주는 함수이다.

```python
def get_gid(criteria, df) : # within 함수 사용
    gids = pd.DataFrame()
    gidss = []
    for i in tqdm(list(df.index)) :
        for j in list(criteria.index) : 
            if df.loc[i, 'geometry'].within(criteria.loc[j, 'geometry']) : # criteria와 df의 geometry가 교차하는지 확인
                gids = pd.concat([gids, df.loc[[i,]]])
                gidss.append(criteria.loc[j, 'gid'])
                break
    gids['gid'] = gidss
    return gids
```

<br>



<br>



get_gid 함수를 통해 **criteria**와 **df**의 **geometry** 값에 대하여 두 데이터가 **서로 교차(within)** 하는지 확인하여 그에 맞는 **gid**를 **Labeling**한다.
또한, 이 함수를 통해 도출된 gid와 유의미한 격자 데이터만 추려낸 gid를 사용하여 데이터를 병합했다.

```python
## 보행자 신호등에 격자를 맵핑
signal_walk_samp = get_gid(accident_count_filter, daejeon_signal_walk)
signal_walk_samp.to_csv('final_signal_walk.csv')
final_signal_walk = pd.read_csv('final_signal_walk.csv', index_col = 0)
print(np.shape(final_signal_walk))
final_signal_walk.head()
```

![image](https://user-images.githubusercontent.com/97672187/161382422-432190c4-9719-430f-b29d-69c86ef0e20a.png){: .align-center}

<br>

신호등이 없어서 NA로 표시된 격자는 신호등 수를 0으로 대체.

```python
signal_walk_count = final_signal_walk.groupby('gid').count()[['geometry']]
signal_walk_count.reset_index(inplace=True)
signal_walk_count.columns = ['gid', '신호등_보행자수']

accident_count_filter = accident_count_filter.merge(signal_walk_count, on = 'gid', how = 'left')
accident_count_filter['신호등_보행자수'] = accident_count_filter['신호등_보행자수'].fillna(0)
print(np.shape(accident_count_filter))
accident_count_filter.head(3)
```

![image](https://user-images.githubusercontent.com/97672187/161382512-9aa2a225-b0bd-4732-975a-d8e3b69fa8f5.png){: .align-center}


<br>



<br>



### 7. 교통안전시설물 - 차량 신호등의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합
보행자 신호등과 똑같은 방법으로 진행


```python
## 차량 신호등
signal_car_samp = get_gid(accident_count_filter, daejeon_signal_car)
signal_car_samp.to_csv('final_signal_car.csv')
final_signal_car = pd.read_csv('final_signal_car.csv', index_col = 0)
print(np.shape(final_signal_car))
final_signal_car.head()
```

![image](https://user-images.githubusercontent.com/97672187/161382575-ce8d92b0-940b-4bfc-a4a4-c5778a1b2f5f.png){: .align-center}


<br>



<br>


```python
signal_car_count = final_signal_car.groupby('gid').count()[['geometry']]
signal_car_count.reset_index(inplace=True)
signal_car_count.columns = ['gid', '신호등_차량등수']

accident_count_filter = accident_count_filter.merge(signal_car_count, on = 'gid', how = 'left')
accident_count_filter['신호등_차량등수'] = accident_count_filter['신호등_차량등수'].fillna(0)
print(np.shape(accident_count_filter))
accident_count_filter.head(3)
```

![image](https://user-images.githubusercontent.com/97672187/161382591-51ae8cc8-0fce-465a-a23a-e4c8f449d794.png){: .align-center}


<br>



<br>


### 8. 교통안전시설물 - CCTV의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합

```python
## CCTV
cctv_samp = get_gid(accident_count_filter, cctv)
cctv_samp.to_csv('final_cctv.csv')
final_cctv = pd.read_csv('final_cctv.csv', index_col = 0)
print(np.shape(final_cctv))
final_cctv.head()
```

![image](https://user-images.githubusercontent.com/97672187/161382618-d8061bb8-581d-4a46-bce6-961198245cae.png){: .align-center}


<br>



<br>




```python
cctv_count = final_cctv.groupby('gid').count()[['geometry']]
cctv_count.reset_index(inplace=True)
cctv_count.columns = ['gid', 'cctv수']

accident_count_filter = accident_count_filter.merge(cctv_count, on = 'gid', how = 'left')
accident_count_filter['cctv수'] = accident_count_filter['cctv수'].fillna(0)
print(np.shape(accident_count_filter))
accident_count_filter.head(3)
```

![image](https://user-images.githubusercontent.com/97672187/161382633-d02db5bb-b27b-4d54-992a-3ffe6fea6175.png){: .align-center}

<br>



<br>




```python
accident_count_filter.to_csv('accident_count_filter_1.csv') # 신호등(보행등), 신호등(차량등), cctv수까지 merge한 상태.
```


<br>



<br>


### 9. 교통혼잡빈도 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합


```python
accident_count_filter_1 = pd.read_csv('accident_count_filter_1.csv', index_col = 0)
accident_count_filter_1['geometry'] = [wkt.loads(line) for line in list(accident_count_filter_1['geometry'])] # str 형태에서 geometry 형태로 형 변환
print(np.shape(accident_count_filter_1))
accident_count_filter_1.head(3)
```

![image](https://user-images.githubusercontent.com/97672187/161382703-e4af4248-ca24-416c-91cf-f2b327e528d1.png){: .align-center}

<br>



<br>



상세도로망 데이터에 geometry 정보가 있기 때문에 혼잡빈도 데이터와 상세도로망 데이터를 LinkID를 기준으로 합친다면 get_gid2 함수를 통해 격자별로 혼잡빈도강도를 적용할 수 있게 된다.

```python
day_time_confuse['link_id'] = [int(id/100) for id in list(day_time_confuse['상세도로망_LinkID'])] # 상세도로망_LinkID에서 link_id 추출 
road_detail = road_detail.astype({'link_id' : 'int64'})
day_time_confused = day_time_confuse.merge(road_detail, on='link_id', how='left') # 위에서 만든 link_id를 geometry가 있는 road_detail 데이터와 병합
print(np.shape(day_time_confused))
day_time_confused.head(2)
```

![image](https://user-images.githubusercontent.com/97672187/161382953-c7f6cf14-1e2c-4bd6-8525-375b7dbe6858.png){: .align-center}


<br>



<br>



get_gid2 함수는 위에서 만든 get_gid와 원리는 똑같지만 도로 형태는 일반 좌표정보가 아닌 Polygon 형태이기 때문에 within이 아닌 crosses 사용했다. 
geometry가 서로 교차하면 해당 LinkID에 격자를 맵핑해 줄 것이다.

```python
def get_gid2(criteria, df) :
    gids = pd.DataFrame()
    gidss = []
    for i in tqdm(list(df.index)) :
        for j in list(criteria.index) : 
            if df.loc[i, 'geometry'].crosses(criteria.loc[j, 'geometry']) : 
                gids = pd.concat([gids, df.loc[[i,]]])
                gidss.append(criteria.loc[j, 'gid'])
                break
    gids['gid'] = gidss
    return gids
```

get_gid2함수를 사용하여 도로링크ID별 gid를 얻어낸다.

```python
## 교통혼잡도
accident_count_filter_2 = get_gid2(accident_count_filter_1, day_time_confused)
accident_count_filter_2.to_csv('accident_count_filter_2.csv')
accident_count_filter_2 = pd.read_csv('accident_count_filter_2.csv', index_col = 0)
accident_count_filter_2['geometry'] = [wkt.loads(line) for line in list(accident_count_filter_2['geometry'])] #str을 geometry 타입으로 변환
print(np.shape(accident_count_filter_2))
accident_count_filter_2.head(3)
```

![image](https://user-images.githubusercontent.com/97672187/161383010-0e27f390-21bc-4fb0-8856-16d52cc5e4f7.png){: .align-center}


<br>



<br>




격자별로 또, 일별로 혼잡빈도강도가 다를 것이기 때문에 한 격자라도 여러 일에 걸쳐서 혼잡빈도강도를 가지고 있을 것이다. 따라서 격자별로 이 혼잡빈도강도를 평균내서 새로운 변수를
추가하자.


```python
confused = accident_count_filter_2.groupby('gid').mean()[['혼잡빈도강도']] # 격자별로 혼잡빈도강도의 평균 값을 사용
confused.reset_index(inplace=True)

accident_count_filter_1 = accident_count_filter_1.merge(confused, on = 'gid', how = 'left')
accident_count_filter_1['혼잡빈도강도'] = accident_count_filter_1['혼잡빈도강도'].fillna(0) #격자별 혼잡빈도강도가 없는 격자는 0으로 대체
print(np.shape(accident_count_filter_1))
accident_count_filter_1.head(3)
```

![image](https://user-images.githubusercontent.com/97672187/161383272-d5b91cd3-a487-4035-8e14-047f55c0630c.png){: .align-center}


<br><br>

### 10. 교통혼잡시간 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합
위에서 다뤘던 교통 혼잡빈도강도 데이터와 똑같은 방식으로 교통 혼잡시간강도 데이터를 처리했다. 

```python
#링크ID 추출
day_time_times['link_id'] = [int(id/100) for id in list(day_time_confuse['상세도로망_LinkID']) ]
day_time_timed = day_time_times.merge(road_detail, on='link_id', how='left')
print(np.shape(day_time_timed))
day_time_times.head(2)
```

![image](https://user-images.githubusercontent.com/97672187/161383414-c7ced44a-922c-4cc6-8f13-1f4c5084d706.png)



<br>



<br>



```python
# 교통혼잡시간
# 혼잡시간강도 데이터에도 gid 맵핑
accident_count_filter_4 = get_gid2(accident_count_filter_3, day_time_times)
accident_count_filter_4.to_csv('accident_count_filter_4.csv')
accident_count_filter_4 = pd.read_csv('accident_count_filter_4.csv', index_col = 0)
accident_count_filter_4 = accident_count_filter_4.reset_index(drop = True)
accident_count_filter_4['geometry'] = [wkt.loads(line) for line in list(accident_count_filter_4['geometry'])]
print(np.shape(accident_count_filter_4))
accident_count_filter_4.head(3)
```

![image](https://user-images.githubusercontent.com/97672187/161383448-9c59d805-7dd5-4ebc-8852-1a788ae7d141.png){: .align-center}



<br>



<br>




```python
timed = accident_count_filter_4.groupby('gid').mean()[['혼잡시간강도']] # 격자별로 혼잡시간강도의 평균 값을 사용
timed.reset_index(inplace=True)

accident_count_filter_5 = accident_count_filter_3.merge(timed, on = 'gid', how = 'left')
accident_count_filter_5['혼잡시간강도'] = accident_count_filter_5['혼잡시간강도'].fillna(0)
print(np.shape(accident_count_filter_5))
accident_count_filter_5.head(3)
```


![image](https://user-images.githubusercontent.com/97672187/161383520-a31dc692-9d65-45b8-b648-aae420acd7dd.png){: .align-center}


<br>



<br>



### 11. 시간대별 추정교통량 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합

accident_count_filter_2 데이터에 이미 LinkID별로 gid가 있기 때문에 굳이 get_gid2함수를 사용하지 않아도 
LinkID별로 일별, 시간별, 전체 추정 교통량을 평균내어서 격자별 전체 추정 교통량 변수를 추가할 수 있다.

```python
day_time_accident['link_id'] = [int(id/100) for id in list(day_time_accident['상세도로망_LinkID']) ]
day_time_accidentd = day_time_accident.merge(road_detail, on='link_id', how='left')
day_time_accidentd.groupby('link_id').mean('전체_추정교통량')
day_time_accidentd.reset_index(inplace=True)
```


```python
day_time_accidentd = day_time_accidentd.merge(accident_count_filter_2[['gid', 'link_id']], on='link_id', how='left')
day_time_sample = day_time_accidentd.loc[:, ['전체_추정교통량', 'gid']]
day_time_sample = day_time_sample.groupby('gid').mean()
day_time_sample.reset_index(inplace = True)
```


```python
accident_count_filter_5 = accident_count_filter_5.merge(day_time_sample, on = 'gid', how = 'left')
accident_count_filter_5 = accident_count_filter_5.fillna(0)
print(np.shape(accident_count_filter_5))
accident_count_filter_5.head(5)
```

![image](https://user-images.githubusercontent.com/97672187/161384298-83055900-129e-4e76-93e6-42a8e5402af4.png){: .align-center}


<br>



<br>



```python
accident_count_filter_5.to_csv('accident_count_filter_5.csv') # 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량까지 merge한 상태.
```


### Weather 변수를 연도별로 나눠주기
```python
weather_2017 = weather[weather['일시'] < '2018-01-01']
weather_2018 = weather[(weather['일시'] >= '2018-01-01') & (weather['일시'] <= '2018-12-31')].reset_index(drop = True)
weather_2019 = weather[weather['일시'] >= '2019-01-01'].reset_index(drop = True)
weather_2017.to_csv('weather_2017.csv')
weather_2018.to_csv('weather_2018.csv')
weather_2019.to_csv('weather_2019.csv')
```
