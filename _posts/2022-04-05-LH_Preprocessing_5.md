---
layout: single
title: "전처리 Part.5"
toc: true
toc_sticky: true
category: LH
---

전처리 Part.5 에서는 R로 작업했던 날씨 데이터를 불러와서 기존 데이터와 병합한 뒤, 교통 시설물 데이터를 추가했다.

## 전처리 Part.5(by using Python)

```python
import platform
import folium
import math
from tqdm import tqdm

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings(action = 'ignore')

from geoband.API import *
from shapely import wkt
from shapely.geometry import Point, Polygon, LineString

if platform.system() == 'Darwin': # 맥
        plt.rc('font', family='AppleGothic') 
elif platform.system() == 'Windows': # 윈도우
        plt.rc('font', family='Malgun Gothic') 
elif platform.system() == 'Linux': # 리눅스 (구글 Colab)
        plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
```

### 전처리한 weather 데이터 불러오기
```python
accident_list = pd.read_csv('1.대전광역시_교통사고내역(2017~2019).csv')
weather_arrange = pd.read_csv('weather_arrange.csv')
weather_arrange.columns = ['사고일', '평균온도', '최저온도', '최고온도', '최대풍속' ,'평균풍속', '평균습도', '평균지면온도', '강수량' ,'적설량', '안개시간']

print(np.shape(weather_arrange))
weather_arrange.head(3)
```

![image](https://user-images.githubusercontent.com/97672187/161971892-2029ede1-4681-40d0-83f7-0093ca132121.png){: .align-center}

<br>



<br>

격자별로 날씨변수들 계산

```python
acci = accident_list.merge(weather_arrange, on='사고일', how='left')
acci_sum = acci.groupby('gid').sum(['평균온도', '최고온도', '최저온도', '최대풍속', '평균풍속', '평균습도','평균지면온도', '강수량', '적설량', '안개시간']) 
acci_sum.reset_index(inplace=True)
acci_sum = acci_sum[['gid', '평균온도', '최저온도', '최고온도', '최대풍속', '평균풍속', '평균습도', '평균지면온도', '강수량', '적설량', '안개시간']]
print(np.shape(acci_sum))
acci_sum.head(3)
```

![image](https://user-images.githubusercontent.com/97672187/161972100-27721c50-1b65-47f1-9b4f-e3b1340f621c.png){: .align-center}

<br>



<br>

### 12. 전처리한 weather 데이터와 현재까지 구축한 데이터와 병합

```python
accident_count_filter_5 = pd.read_csv('accident_count_filter_5.csv', index_col = 0) # 기존데이터
accident_count_filter_6 = accident_count_filter_5.merge(acci_sum, on='gid', how='left') #날씨데이터 병합
accident_count_filter_6 = accident_count_filter_6.fillna(0)
print(np.shape(accident_count_filter_6))
accident_count_filter_6.head(2)
accident_count_filter_6.to_csv('accident_count_filter_6.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨 데이터까지 merge한 상태.
```

![image](https://user-images.githubusercontent.com/97672187/161972345-5b2e8e29-1fba-4420-8ee8-03ed31551f29.png){: .align-center}


기존에 전처리를 진행할 때 POLYGON 타입의 좌표 정보가 주어지면 within함수를 사용해서 좌표가 격자 내에 있는지 확인한 후 좌표가 격자 내에 있으면 해당 격자를 리턴해주었다. 
여기서도 역시 get_gid 함수를 사용해서 교통 시설물 좌표가 주어지면 좌표가 어느 격자 내에 있는지 파악한다.

```python
def get_gid(criteria, df) : 
    gids = pd.DataFrame()
    gidss = []
    for i in tqdm(list(df.index)) :
        for j in list(criteria.index) : 
            if df.loc[i, 'point'].within(criteria.loc[j, 'geometry']) : 
                gids = pd.concat([gids, df.loc[[i,]]])
                gidss.append(criteria.loc[j, 'gid'])
                break
    gids['gid'] = gidss
    return gids
```


### 13. 안전지대 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합

격자당 안전지대 시설물이 몇개 있는가.

```python
accident_count_filter_6 = pd.read_csv("accident_count_filter_6.csv", index_col = 0) # 지금까지 전처리 된 데이터
accident_count_filter_6['geometry'] = [wkt.loads(line) for line in list(accident_count_filter_6['geometry'])] # str을 geometry 타입으로 변환
print(np.shape(accident_count_filter_6))
accident_count_filter_6.head(3)
```

![image](https://user-images.githubusercontent.com/97672187/161973302-ff7a95f2-b5f3-45c9-97cf-24717dfb846b.png){: .align-center}

<br>


<br>


centroid 함수를 사용하고 geometry 타입의 좌표 정보 중 중앙값을 구해서 위도, 경도 좌표로 활용한다.

```python
safe = gpd.read_file("5.대전광역시_안전지대.geojson")
safe["longitude"] = safe.centroid.x
safe["latitude"] = safe.centroid.y
safe['point'] = safe.apply(lambda row : Point([row['longitude'], row['latitude']]), axis=1) # get_gid 함수를 사용하기 위해 Point 타입으로 위경도를 묶음.
```


```python
accident_count_filter_7 = get_gid(accident_count_filter_6, safe[['point']]) # POINT 좌표 정보와 일치하는 gid 정보를 추가
accident_count_filter_7 = accident_count_filter_7.reset_index(drop = True)
accident_count_filter_7.to_csv('accident_count_filter_7.csv')
```


```python
accident_count_filter_7 = pd.read_csv('accident_count_filter_7.csv', index_col = 0)
safe_count = accident_count_filter_7.groupby('gid').count() # 격자별로 안전지대가 얼마나 있는지 파악.
safe_count.reset_index(inplace = True)
safe_count.columns = ['gid', '안전지대수']

accident_count_filter_8 = accident_count_filter_6.merge(safe_count, on = 'gid', how = 'left') # 기존 데이터에 격자별 안전지대수 데이터 병합
accident_count_filter_8['안전지대수'] = accident_count_filter_8['안전지대수'].fillna(0)
print(np.shape(accident_count_filter_8))
accident_count_filter_8.head(3)
accident_count_filter_8.to_csv('accident_count_filter_8.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수 데이터까지 merge한 상태.
```

![image](https://user-images.githubusercontent.com/97672187/161975278-905c5b96-8024-4f31-93c0-131fa12a82b8.png){: .align-center}

<br>



<br>

### 14. 중앙분리대수 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합
안전지대수를 추가했던 것과 똑같은 방식으로 진행했지만, 중앙 분리대수는 전체데이터가 16개 밖에 되지 않아서 기존 데이터의 격자 내에 좌표가 없을 수도 있으므로
교통사고격자 전체 데이터를 사용해서 전체 격자 중 어떤 격자에 좌표가 포함되어 있는지 알아냈다. 즉, 전체 교통사고격자 데이터를 기준으로 get_gid함수를 사용했다.


격자당 중앙분리대가 몇개 있는가.

```python
center = gpd.read_file("31.대전시_중앙분리대.geojson")
center["longitude"] = center.centroid.x
center["latitude"] = center.centroid.y
center['point'] = center.apply(lambda row : Point([row['longitude'], row['latitude']]), axis=1)

accident_grid = gpd.read_file('2.대전광역시_교통사고격자(2017~2019).geojson')
center_filter = get_gid(accident_grid, center)
center_filter.to_csv('center_filter.csv')
```


```python
accident_count_filter_8 = pd.read_csv('accident_count_filter_8.csv', index_col = 0)
center_filter = pd.read_csv('center_filter.csv', index_col = 0)
center_filter = center_filter.groupby('gid').count()[['point']]
center_filter.reset_index(inplace = True)
center_filter.columns = ['gid', '중앙분리대수']

accident_count_filter_9 = accident_count_filter_8.merge(center_filter, on = 'gid', how = 'left')
accident_count_filter_9['중앙분리대수'] = accident_count_filter_9['중앙분리대수'].fillna(0)
print(np.shape(accident_count_filter_9))
accident_count_filter_9.head(3)
accident_count_filter_9.to_csv('accident_count_filter_9.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수 데이터까지 merge한 상태.
```

![image](https://user-images.githubusercontent.com/97672187/161982049-9475c85b-d8a7-4e23-893e-837e9369b5f3.png){: .align-center}


<br>



<br>


### 15. 정차금지지대 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합

격자당 정차금지지대가 몇개 있는가.


```python
no_stop = gpd.read_file("8.대전광역시_정차금지지대.geojson")
no_stop["longitude"] = no_stop.centroid.x
no_stop["latitude"] = no_stop.centroid.y
no_stop['point'] = no_stop.apply(lambda row : Point([row['longitude'], row['latitude']]), axis=1)
```


```python
accident_count_filter_10 = get_gid(accident_count_filter_6, no_stop[['point']])
accident_count_filter_10 = accident_count_filter_10.reset_index(drop = True)
accident_count_filter_10.to_csv('accident_count_filter_10.csv')
accident_count_filter_10 = pd.read_csv('accident_count_filter_10.csv', index_col = 0)
no_stop_zone = accident_count_filter_10.groupby('gid').count()
no_stop_zone.reset_index(inplace = True)
no_stop_zone.columns = ['gid', '정차금지지대수']

accident_count_filter_11 = accident_count_filter_9.merge(no_stop_zone, on = 'gid', how = 'left')
accident_count_filter_11['정차금지지대수'] = accident_count_filter_11['정차금지지대수'].fillna(0)
print(np.shape(accident_count_filter_11))
accident_count_filter_11.head(3)
accident_count_filter_11.to_csv('accident_count_filter_11.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수, 정차금지지대수 데이터까지 merge한 상태.
```

![image](https://user-images.githubusercontent.com/97672187/161982569-fa6f1c00-c352-44e6-9ab8-de1664558641.png){: .align-center}

<br>



<br>


### 16. 도로속도표시 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합

격자당 도로속도표시 시설물이 몇개 있는가.

```python
speed = gpd.read_file("7.대전광역시_도로속도표시.geojson")
speed["longitude"] = speed.centroid.x
speed["latitude"] = speed.centroid.y
speed['point'] = speed.apply(lambda row : Point([row['longitude'], row['latitude']]), axis=1)
```


```python
accident_count_filter_11['geometry'] = [wkt.loads(line) for line in list(accident_count_filter_11['geometry'])] # str to geometry 타입변환
accident_count_filter_12 = get_gid(accident_count_filter_11, speed[['point']]) # 좌표를 사용해서 격자 정보 가져오기
accident_count_filter_12 = accident_count_filter_12.reset_index(drop = True)
accident_count_filter_12.to_csv('accident_count_filter_12.csv')
accident_count_filter_12 = pd.read_csv('accident_count_filter_12.csv', index_col = 0)

speed_count = accident_count_filter_12.groupby('gid').count()
speed_count.reset_index(inplace = True)
speed_count.columns = ['gid', '도로속도표시수']

accident_count_filter_13 = accident_count_filter_11.merge(speed_count, on = 'gid', how = 'left')
accident_count_filter_13['도로속도표시수'] = accident_count_filter_13['도로속도표시수'].fillna(0)
print(np.shape(accident_count_filter_13))
accident_count_filter_13.head(3)
accident_count_filter_13.to_csv('accident_count_filter_13.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수,
# 정차금지지대수, 도로속도표시수 데이터까지 merge한 상태.
```

![image](https://user-images.githubusercontent.com/97672187/161982934-533ce00f-7f98-49a9-beee-9afd48a3a12f.png){: .align-center}


<br>



<br>

### 17. 교통안전표지 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합

격자당 교통안전표지가 몇개 있는가.

```python
safe_signal = gpd.read_file("9.대전광역시_교통안전표지.geojson")
safe_signal["longitude"] = safe_signal.centroid.x
safe_signal["latitude"] = safe_signal.centroid.y
safe_signal['point'] = safe_signal.apply(lambda row : Point([row['longitude'], row['latitude']]), axis=1)
```


```python
accident_count_filter_14 = get_gid(accident_count_filter_13, safe_signal[['point']])
accident_count_filter_14 = accident_count_filter_14.reset_index(drop = True)
accident_count_filter_14.to_csv('accident_count_filter_14.csv')

accident_count_filter_14 = pd.read_csv('accident_count_filter_14.csv', index_col = 0)
safe_signal_count = accident_count_filter_14.groupby('gid').count()
safe_signal_count.reset_index(inplace = True)
safe_signal_count.columns = ['gid', '교통안전표지수']

accident_count_filter_15 = accident_count_filter_13.merge(safe_signal_count, on = 'gid', how = 'left')
accident_count_filter_15['교통안전표지수'] = accident_count_filter_15['교통안전표지수'].fillna(0)
print(np.shape(accident_count_filter_15))
accident_count_filter_15.head(3)
accident_count_filter_15.to_csv('accident_count_filter_15.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수,
# 정차금지지대수, 도로속도표시수, 교통안전표지수 데이터까지 merge한 상태.
```

![image](https://user-images.githubusercontent.com/97672187/161984800-26a261d1-54e4-4cc6-9ed0-202ba8ba7d15.png){: .align-center}


<br>




<br>


### 18. 교통노드 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합

격자당 몇개의 노드가 포함이 되어 있는가.

```python
traffic_node = gpd.read_file('18.대전광역시_교통노드(2018).geojson')
traffic_node["longitude"] = traffic_node.centroid.x
traffic_node["latitude"] = traffic_node.centroid.y
traffic_node['point'] = traffic_node.apply(lambda row : Point([row['longitude'], row['latitude']]), axis=1)
```


```python
accident_count_filter_16 = get_gid(accident_count_filter_15, traffic_node[['point']])
accident_count_filter_16 = accident_count_filter_16.reset_index(drop = True)
accident_count_filter_16.to_csv('accident_count_filter_16.csv')
print(np.shape(accident_count_filter_16))

accident_count_filter_16 = pd.read_csv('accident_count_filter_16.csv', index_col = 0)
node_count = accident_count_filter_16.groupby('gid').count()
node_count.reset_index(inplace = True)
node_count.columns = ['gid', '노드개수']

accident_count_filter_17 = accident_count_filter_15.merge(node_count, on = 'gid', how = 'left')
accident_count_filter_17['노드개수'] = accident_count_filter_17['노드개수'].fillna(0)
print(np.shape(accident_count_filter_17))
accident_count_filter_17.head(3)
accident_count_filter_17.to_csv('accident_count_filter_17.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수,
# 정차금지지대수, 도로속도표시수, 교통안전표지수, 노드개수 데이터까지 merge한 상태.
```


![image](https://user-images.githubusercontent.com/97672187/161985241-8a4e7fd7-4394-4a72-8a58-a52264249284.png){: .align-center}

<br>




<br>

### 19. 횡단보도 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합
격자당 횡단보도가 몇개 있는가.


```python
crosswalk = gpd.read_file('6.대전광역시_횡단보도.geojson')
crosswalk["longitude"] = crosswalk.centroid.x
crosswalk["latitude"] = crosswalk.centroid.y
crosswalk['point'] = crosswalk.apply(lambda row : Point([row['longitude'], row['latitude']]), axis=1)
```


```python
accident_count_filter_18 = get_gid(accident_count_filter_17, crosswalk[['point']])
accident_count_filter_18 = accident_count_filter_18.reset_index(drop = True)
accident_count_filter_18.to_csv('accident_count_filter_18.csv')

accident_count_filter_18 = pd.read_csv('accident_count_filter_18.csv', index_col = 0)
crosswalk_count = accident_count_filter_18.groupby('gid').count()
crosswalk_count.reset_index(inplace = True)
crosswalk_count.columns = ['gid', '횡단보도수']

accident_count_filter_19 = accident_count_filter_17.merge(crosswalk_count, on = 'gid', how = 'left')
accident_count_filter_19['횡단보도수'] = accident_count_filter_19['횡단보도수'].fillna(0)
print(np.shape(accident_count_filter_19))
accident_count_filter_19.head(3)
accident_count_filter_19.to_csv('accident_count_filter_19.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수,
# 정차금지지대수, 도로속도표시수, 교통안전표지수, 노드개수, 횡단보도수 데이터까지 merge한 상태.
```

![image](https://user-images.githubusercontent.com/97672187/161985428-6ae0d6c0-2f41-465c-a362-a57444801a88.png){: .align-center}

<br>



<br>

### 20. 건물 면적 데이터와 현재까지 구축된 데이터를 병합
격자에 포함되어 있는 건물들의 면적이 얼마나 되는가. 건물면적에 있는 gid 변수를 활용하여 기존 데이터와 병합한다.

```python
build_area_grid = gpd.read_file('24.대전광역시_건물연면적_격자.geojson')
build_area_grid.columns = ['gid', '건물면적', 'geometry']
build_area_grid = build_area_grid.iloc[:, :2]
```


```python
accident_count_filter_20 = accident_count_filter_19.merge(build_area_grid, on = 'gid', how = 'left')
accident_count_filter_20['건물면적'] = accident_count_filter_20['건물면적'].fillna(0)
print(np.shape(accident_count_filter_20))
accident_count_filter_20.head(3)
accident_count_filter_20.to_csv('accident_count_filter_20.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수,
# 정차금지지대수, 도로속도표시수, 교통안전표지수, 노드개수, 횡단보도수, 건물면적 데이터까지 merge한 상태.
```

![image](https://user-images.githubusercontent.com/97672187/161985907-533fe199-890d-4258-9450-9ac2ec2dff82.png){: .align-center}

<br>


<br>


### 21. 차량등록현황 데이터와 현재까지 구축된 데이터를 병합
격자당 등록된 차량들이 몇대인지 병합.

```python
car_list = gpd.read_file('30.대전광역시_차량등록현황_격자.geojson')
car_list.columns = ['gid', '자동차대수', 'geometry']
car_list = car_list.iloc[:, :2]
```



```python
accident_count_filter_21 = accident_count_filter_20.merge(car_list, on = 'gid', how = 'left')
accident_count_filter_21['자동차대수'] = accident_count_filter_21['자동차대수']
print(np.shape(accident_count_filter_21))
accident_count_filter_21.head(3)
accident_count_filter_21.to_csv('accident_count_filter_21.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수,
# 정차금지지대수, 도로속도표시수, 교통안전표지수, 노드개수, 횡단보도수, 건물면적, 자동차대수 데이터까지 merge한 상태.
```

![image](https://user-images.githubusercontent.com/97672187/161986111-9c282cd7-1d94-431b-967c-6aa64919f39a.png){: .align-center}


<br>


<br>

### 22. 총인구 데이터와 현재까지 구축된 데이터를 병합

격자당 몇명의 사람이 살고있는지 파악. 기존 데이터와 병합

```python
total_popu = gpd.read_file('12.대전광역시_인구정보(총인구).geojson')
total_popu.columns = ['gid', '총거주인구수', 'geometry']
total_popu = total_popu.iloc[:, :2]
```



```python
accident_count_filter_22 = accident_count_filter_21.merge(total_popu, on = 'gid', how = 'left')
accident_count_filter_22['총거주인구수'] = accident_count_filter_22['총거주인구수'].fillna(0)
print(np.shape(accident_count_filter_22))
accident_count_filter_22.head(3)
accident_count_filter_22.to_csv('accident_count_filter_22.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수,
# 정차금지지대수, 도로속도표시수, 교통안전표지수, 노드개수, 횡단보도수, 건물면적, 자동차대수, 총거주인구수 데이터까지 merge한 상태.
```

![image](https://user-images.githubusercontent.com/97672187/161986303-b3958f3a-18d8-41a4-8d01-f16bef2cbf42.png){: .align-center}


<br>


<br>


### weather 관련 열 이름 바꾸기

```python
col_name_lst = ['gid', 'acci_cnt', 'geometry', '사고건수', '사상자수', 'x', 'y', '신호등_보행자수', '신호등_차량등수', 'cctv수', '혼잡빈도강도',
                '혼잡시간강도', '전체_추정교통량', '이상평균기온동반사고건수', '이상최저온도동반사고건수', '이상최고온도동반사고건수', '이상최대풍속동반사고건수', '이상평균풍속동반사고건수',
                '이상평균습도동반사고건수', '이상평균지면온도동반사고건수', '이상강수량동반사고건수', '이상적설량동반사고건수', '이상안개시간동반사고건수', '안전지대수',
                '중앙분리대수', '정차금지지대수', '도로속도표시수', '교통안전표지수', '노드개수', '횡단보도수', '건물면적', '자동차대수', '총거주인구수']
accident_count_filter_22.columns = col_name_lst
print(np.shape(accident_count_filter_22))
accident_count_filter_22.head(3)
accident_count_filter_22.to_csv('accident_count_filter_23.csv')
```

![image](https://user-images.githubusercontent.com/97672187/161986486-9c924a75-35af-48ce-9562-261314ba99db.png){: .align-center}
