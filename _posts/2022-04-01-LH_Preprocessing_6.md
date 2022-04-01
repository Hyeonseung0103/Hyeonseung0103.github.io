---
layout: single
title:  "전처리 Part.6"
categories: coding
tag: [python, blog, jekyll]
toc: true
toc_sticky: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>



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

<pre>
/opt/app-root/lib/python3.6/site-packages/geopandas/_compat.py:91: UserWarning: The Shapely GEOS version (3.8.0-CAPI-1.13.1 ) is incompatible with the GEOS version PyGEOS was compiled with (3.9.0-CAPI-1.16.2). Conversions between both will be slow.
  shapely_geos_version, geos_capi_version_string
</pre>

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

## 13. 안전지대 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합



```python
accident_count_filter_6 = pd.read_csv("accident_count_filter_6.csv", index_col = 0)
accident_count_filter_6['geometry'] = [wkt.loads(line) for line in list(accident_count_filter_6['geometry'])]
print(np.shape(accident_count_filter_6))
accident_count_filter_6.head(3)
```

<pre>
(5556, 23)
</pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gid</th>
      <th>acci_cnt</th>
      <th>geometry</th>
      <th>사고건수</th>
      <th>사상자수</th>
      <th>x</th>
      <th>y</th>
      <th>신호등_보행자수</th>
      <th>신호등_차량등수</th>
      <th>cctv수</th>
      <th>...</th>
      <th>평균온도</th>
      <th>최저온도</th>
      <th>최고온도</th>
      <th>최대풍속</th>
      <th>평균풍속</th>
      <th>평균습도</th>
      <th>평균지면온도</th>
      <th>강수량</th>
      <th>적설량</th>
      <th>안개시간</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>다바931203</td>
      <td>2</td>
      <td>(POLYGON ((127.4230710131166 36.38013455218083...</td>
      <td>2</td>
      <td>2</td>
      <td>127.423628</td>
      <td>36.380586</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>다바861174</td>
      <td>0</td>
      <td>(POLYGON ((127.3450791441312 36.35391426501025...</td>
      <td>0</td>
      <td>0</td>
      <td>127.345636</td>
      <td>36.354366</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>다바900127</td>
      <td>1</td>
      <td>(POLYGON ((127.3886063974092 36.31159021601022...</td>
      <td>1</td>
      <td>1</td>
      <td>127.389163</td>
      <td>36.312042</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 23 columns</p>
</div>



```python
safe = gpd.read_file("5.대전광역시_안전지대.geojson")
safe["longitude"] = safe.centroid.x
safe["latitude"] = safe.centroid.y
safe['point'] = safe.apply(lambda row : Point([row['longitude'], row['latitude']]), axis=1)
```


```python
accident_count_filter_7 = get_gid(accident_count_filter_6, safe[['point']])
accident_count_filter_7 = accident_count_filter_7.reset_index(drop = True)
accident_count_filter_7.to_csv('accident_count_filter_7.csv')
print(np.shape(accident_count_filter_7))
accident_count_filter_7.head(3)
```


```python
accident_count_filter_7 = pd.read_csv('accident_count_filter_7.csv', index_col = 0)
safe_count = accident_count_filter_7.groupby('gid').count()
safe_count.reset_index(inplace = True)
safe_count.columns = ['gid', '안전지대수']

accident_count_filter_8 = accident_count_filter_6.merge(safe_count, on = 'gid', how = 'left')
accident_count_filter_8['안전지대수'] = accident_count_filter_8['안전지대수'].fillna(0)
print(np.shape(accident_count_filter_8))
accident_count_filter_8.head(3)
```

<pre>
(5556, 24)
</pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gid</th>
      <th>acci_cnt</th>
      <th>geometry</th>
      <th>사고건수</th>
      <th>사상자수</th>
      <th>x</th>
      <th>y</th>
      <th>신호등_보행자수</th>
      <th>신호등_차량등수</th>
      <th>cctv수</th>
      <th>...</th>
      <th>최저온도</th>
      <th>최고온도</th>
      <th>최대풍속</th>
      <th>평균풍속</th>
      <th>평균습도</th>
      <th>평균지면온도</th>
      <th>강수량</th>
      <th>적설량</th>
      <th>안개시간</th>
      <th>안전지대수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>다바931203</td>
      <td>2</td>
      <td>(POLYGON ((127.4230710131166 36.38013455218083...</td>
      <td>2</td>
      <td>2</td>
      <td>127.423628</td>
      <td>36.380586</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>다바861174</td>
      <td>0</td>
      <td>(POLYGON ((127.3450791441312 36.35391426501025...</td>
      <td>0</td>
      <td>0</td>
      <td>127.345636</td>
      <td>36.354366</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>다바900127</td>
      <td>1</td>
      <td>(POLYGON ((127.3886063974092 36.31159021601022...</td>
      <td>1</td>
      <td>1</td>
      <td>127.389163</td>
      <td>36.312042</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 24 columns</p>
</div>



```python
accident_count_filter_8.to_csv('accident_count_filter_8.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수 데이터까지 merge한 상태.
```

## 14. 중앙분리대수 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합



```python
center = gpd.read_file("31.대전시_중앙분리대.geojson")
center["longitude"] = center.centroid.x
center["latitude"] = center.centroid.y
center['point'] = center.apply(lambda row : Point([row['longitude'], row['latitude']]), axis=1)

accident_grid = gpd.read_file('2.대전광역시_교통사고격자(2017~2019).geojson')
center_filter = get_gid(accident_grid, center)
center_filter.to_csv('center_filter.csv')
```

<pre>
100%|██████████| 16/16 [00:49<00:00,  3.12s/it]
</pre>

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
```

<pre>
(5556, 25)
</pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gid</th>
      <th>acci_cnt</th>
      <th>geometry</th>
      <th>사고건수</th>
      <th>사상자수</th>
      <th>x</th>
      <th>y</th>
      <th>신호등_보행자수</th>
      <th>신호등_차량등수</th>
      <th>cctv수</th>
      <th>...</th>
      <th>최고온도</th>
      <th>최대풍속</th>
      <th>평균풍속</th>
      <th>평균습도</th>
      <th>평균지면온도</th>
      <th>강수량</th>
      <th>적설량</th>
      <th>안개시간</th>
      <th>안전지대수</th>
      <th>중앙분리대수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>다바931203</td>
      <td>2</td>
      <td>MULTIPOLYGON (((127.4230710131166 36.380134552...</td>
      <td>2</td>
      <td>2</td>
      <td>127.423628</td>
      <td>36.380586</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>다바861174</td>
      <td>0</td>
      <td>MULTIPOLYGON (((127.3450791441312 36.353914265...</td>
      <td>0</td>
      <td>0</td>
      <td>127.345636</td>
      <td>36.354366</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>다바900127</td>
      <td>1</td>
      <td>MULTIPOLYGON (((127.3886063974092 36.311590216...</td>
      <td>1</td>
      <td>1</td>
      <td>127.389163</td>
      <td>36.312042</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 25 columns</p>
</div>



```python
accident_count_filter_9.to_csv('accident_count_filter_9.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수 데이터까지 merge한 상태.
```

## 15. 정차금지지대 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합



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
print(np.shape(accident_count_filter_10))
accident_count_filter_10.head(3)
```


```python
accident_count_filter_10 = pd.read_csv('accident_count_filter_10.csv', index_col = 0)
no_stop_zone = accident_count_filter_10.groupby('gid').count()
no_stop_zone.reset_index(inplace = True)
no_stop_zone.columns = ['gid', '정차금지지대수']

accident_count_filter_11 = accident_count_filter_9.merge(no_stop_zone, on = 'gid', how = 'left')
accident_count_filter_11['정차금지지대수'] = accident_count_filter_11['정차금지지대수'].fillna(0)
print(np.shape(accident_count_filter_11))
accident_count_filter_11.head(3)
```

<pre>
(5556, 26)
</pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gid</th>
      <th>acci_cnt</th>
      <th>geometry</th>
      <th>사고건수</th>
      <th>사상자수</th>
      <th>x</th>
      <th>y</th>
      <th>신호등_보행자수</th>
      <th>신호등_차량등수</th>
      <th>cctv수</th>
      <th>...</th>
      <th>최대풍속</th>
      <th>평균풍속</th>
      <th>평균습도</th>
      <th>평균지면온도</th>
      <th>강수량</th>
      <th>적설량</th>
      <th>안개시간</th>
      <th>안전지대수</th>
      <th>중앙분리대수</th>
      <th>정차금지지대수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>다바931203</td>
      <td>2</td>
      <td>MULTIPOLYGON (((127.4230710131166 36.380134552...</td>
      <td>2</td>
      <td>2</td>
      <td>127.423628</td>
      <td>36.380586</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>다바861174</td>
      <td>0</td>
      <td>MULTIPOLYGON (((127.3450791441312 36.353914265...</td>
      <td>0</td>
      <td>0</td>
      <td>127.345636</td>
      <td>36.354366</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>다바900127</td>
      <td>1</td>
      <td>MULTIPOLYGON (((127.3886063974092 36.311590216...</td>
      <td>1</td>
      <td>1</td>
      <td>127.389163</td>
      <td>36.312042</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 26 columns</p>
</div>



```python
accident_count_filter_11.to_csv('accident_count_filter_11.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수, 정차금지지대수 데이터까지 merge한 상태.
```

## 16. 도로속도표시 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합



```python
speed = gpd.read_file("7.대전광역시_도로속도표시.geojson")
speed["longitude"] = speed.centroid.x
speed["latitude"] = speed.centroid.y
speed['point'] = speed.apply(lambda row : Point([row['longitude'], row['latitude']]), axis=1)
```


```python
accident_count_filter_11['geometry'] = [wkt.loads(line) for line in list(accident_count_filter_11['geometry'])]
accident_count_filter_12 = get_gid(accident_count_filter_11, speed[['point']])
accident_count_filter_12 = accident_count_filter_12.reset_index(drop = True)
accident_count_filter_12.to_csv('accident_count_filter_12.csv')
print(np.shape(accident_count_filter_12))
accident_count_filter_12.head(3)
```


```python
accident_count_filter_12 = pd.read_csv('accident_count_filter_12.csv', index_col = 0)
speed_count = accident_count_filter_12.groupby('gid').count()
speed_count.reset_index(inplace = True)
speed_count.columns = ['gid', '도로속도표시수']

accident_count_filter_13 = accident_count_filter_11.merge(speed_count, on = 'gid', how = 'left')
accident_count_filter_13['도로속도표시수'] = accident_count_filter_13['도로속도표시수'].fillna(0)
print(np.shape(accident_count_filter_13))
accident_count_filter_13.head(3)
```

<pre>
(5556, 27)
</pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gid</th>
      <th>acci_cnt</th>
      <th>geometry</th>
      <th>사고건수</th>
      <th>사상자수</th>
      <th>x</th>
      <th>y</th>
      <th>신호등_보행자수</th>
      <th>신호등_차량등수</th>
      <th>cctv수</th>
      <th>...</th>
      <th>평균풍속</th>
      <th>평균습도</th>
      <th>평균지면온도</th>
      <th>강수량</th>
      <th>적설량</th>
      <th>안개시간</th>
      <th>안전지대수</th>
      <th>중앙분리대수</th>
      <th>정차금지지대수</th>
      <th>도로속도표시수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>다바931203</td>
      <td>2</td>
      <td>MULTIPOLYGON (((127.4230710131166 36.380134552...</td>
      <td>2</td>
      <td>2</td>
      <td>127.423628</td>
      <td>36.380586</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>다바861174</td>
      <td>0</td>
      <td>MULTIPOLYGON (((127.3450791441312 36.353914265...</td>
      <td>0</td>
      <td>0</td>
      <td>127.345636</td>
      <td>36.354366</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>다바900127</td>
      <td>1</td>
      <td>MULTIPOLYGON (((127.3886063974092 36.311590216...</td>
      <td>1</td>
      <td>1</td>
      <td>127.389163</td>
      <td>36.312042</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 27 columns</p>
</div>



```python
accident_count_filter_13.to_csv('accident_count_filter_13.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수,
# 정차금지지대수, 도로속도표시수 데이터까지 merge한 상태.
```

## 17. 교통안전표지 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합



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
print(np.shape(accident_count_filter_14))
accident_count_filter_14.head(3)
```


```python
accident_count_filter_14 = pd.read_csv('accident_count_filter_14.csv', index_col = 0)
safe_signal_count = accident_count_filter_14.groupby('gid').count()
safe_signal_count.reset_index(inplace = True)
safe_signal_count.columns = ['gid', '교통안전표지수']

accident_count_filter_15 = accident_count_filter_13.merge(safe_signal_count, on = 'gid', how = 'left')
accident_count_filter_15['교통안전표지수'] = accident_count_filter_15['교통안전표지수'].fillna(0)
print(np.shape(accident_count_filter_15))
accident_count_filter_15.head(3)
```

<pre>
(5556, 28)
</pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gid</th>
      <th>acci_cnt</th>
      <th>geometry</th>
      <th>사고건수</th>
      <th>사상자수</th>
      <th>x</th>
      <th>y</th>
      <th>신호등_보행자수</th>
      <th>신호등_차량등수</th>
      <th>cctv수</th>
      <th>...</th>
      <th>평균습도</th>
      <th>평균지면온도</th>
      <th>강수량</th>
      <th>적설량</th>
      <th>안개시간</th>
      <th>안전지대수</th>
      <th>중앙분리대수</th>
      <th>정차금지지대수</th>
      <th>도로속도표시수</th>
      <th>교통안전표지수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>다바931203</td>
      <td>2</td>
      <td>MULTIPOLYGON (((127.4230710131166 36.380134552...</td>
      <td>2</td>
      <td>2</td>
      <td>127.423628</td>
      <td>36.380586</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>다바861174</td>
      <td>0</td>
      <td>MULTIPOLYGON (((127.3450791441312 36.353914265...</td>
      <td>0</td>
      <td>0</td>
      <td>127.345636</td>
      <td>36.354366</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>다바900127</td>
      <td>1</td>
      <td>MULTIPOLYGON (((127.3886063974092 36.311590216...</td>
      <td>1</td>
      <td>1</td>
      <td>127.389163</td>
      <td>36.312042</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 28 columns</p>
</div>



```python
accident_count_filter_15.to_csv('accident_count_filter_15.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수,
# 정차금지지대수, 도로속도표시수, 교통안전표지수 데이터까지 merge한 상태.
```

## 18. 교통노드 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합



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
accident_count_filter_16.head(3)
```


```python
accident_count_filter_16 = pd.read_csv('accident_count_filter_16.csv', index_col = 0)
node_count = accident_count_filter_16.groupby('gid').count()
node_count.reset_index(inplace = True)
node_count.columns = ['gid', '노드개수']

accident_count_filter_17 = accident_count_filter_15.merge(node_count, on = 'gid', how = 'left')
accident_count_filter_17['노드개수'] = accident_count_filter_17['노드개수'].fillna(0)
print(np.shape(accident_count_filter_17))
accident_count_filter_17.head(3)
```

<pre>
(5556, 29)
</pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gid</th>
      <th>acci_cnt</th>
      <th>geometry</th>
      <th>사고건수</th>
      <th>사상자수</th>
      <th>x</th>
      <th>y</th>
      <th>신호등_보행자수</th>
      <th>신호등_차량등수</th>
      <th>cctv수</th>
      <th>...</th>
      <th>평균지면온도</th>
      <th>강수량</th>
      <th>적설량</th>
      <th>안개시간</th>
      <th>안전지대수</th>
      <th>중앙분리대수</th>
      <th>정차금지지대수</th>
      <th>도로속도표시수</th>
      <th>교통안전표지수</th>
      <th>노드개수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>다바931203</td>
      <td>2</td>
      <td>MULTIPOLYGON (((127.4230710131166 36.380134552...</td>
      <td>2</td>
      <td>2</td>
      <td>127.423628</td>
      <td>36.380586</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>다바861174</td>
      <td>0</td>
      <td>MULTIPOLYGON (((127.3450791441312 36.353914265...</td>
      <td>0</td>
      <td>0</td>
      <td>127.345636</td>
      <td>36.354366</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>다바900127</td>
      <td>1</td>
      <td>MULTIPOLYGON (((127.3886063974092 36.311590216...</td>
      <td>1</td>
      <td>1</td>
      <td>127.389163</td>
      <td>36.312042</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 29 columns</p>
</div>



```python
accident_count_filter_17.to_csv('accident_count_filter_17.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수,
# 정차금지지대수, 도로속도표시수, 교통안전표지수, 노드개수 데이터까지 merge한 상태.
```

## 19. 횡단보도 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합



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
print(np.shape(accident_count_filter_18))
accident_count_filter_18.head(3)
```


```python
accident_count_filter_18 = pd.read_csv('accident_count_filter_18.csv', index_col = 0)
crosswalk_count = accident_count_filter_18.groupby('gid').count()
crosswalk_count.reset_index(inplace = True)
crosswalk_count.columns = ['gid', '횡단보도수']

accident_count_filter_19 = accident_count_filter_17.merge(crosswalk_count, on = 'gid', how = 'left')
accident_count_filter_19['횡단보도수'] = accident_count_filter_19['횡단보도수'].fillna(0)
print(np.shape(accident_count_filter_19))
accident_count_filter_19.head(3)
```

<pre>
(5556, 30)
</pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gid</th>
      <th>acci_cnt</th>
      <th>geometry</th>
      <th>사고건수</th>
      <th>사상자수</th>
      <th>x</th>
      <th>y</th>
      <th>신호등_보행자수</th>
      <th>신호등_차량등수</th>
      <th>cctv수</th>
      <th>...</th>
      <th>강수량</th>
      <th>적설량</th>
      <th>안개시간</th>
      <th>안전지대수</th>
      <th>중앙분리대수</th>
      <th>정차금지지대수</th>
      <th>도로속도표시수</th>
      <th>교통안전표지수</th>
      <th>노드개수</th>
      <th>횡단보도수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>다바931203</td>
      <td>2</td>
      <td>MULTIPOLYGON (((127.4230710131166 36.380134552...</td>
      <td>2</td>
      <td>2</td>
      <td>127.423628</td>
      <td>36.380586</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>다바861174</td>
      <td>0</td>
      <td>MULTIPOLYGON (((127.3450791441312 36.353914265...</td>
      <td>0</td>
      <td>0</td>
      <td>127.345636</td>
      <td>36.354366</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>다바900127</td>
      <td>1</td>
      <td>MULTIPOLYGON (((127.3886063974092 36.311590216...</td>
      <td>1</td>
      <td>1</td>
      <td>127.389163</td>
      <td>36.312042</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 30 columns</p>
</div>



```python
accident_count_filter_19.to_csv('accident_count_filter_19.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수,
# 정차금지지대수, 도로속도표시수, 교통안전표지수, 노드개수, 횡단보도수 데이터까지 merge한 상태.
```

## 20. 건물 면적 데이터와 현재까지 구축된 데이터를 병합



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
```

<pre>
(5556, 31)
</pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gid</th>
      <th>acci_cnt</th>
      <th>geometry</th>
      <th>사고건수</th>
      <th>사상자수</th>
      <th>x</th>
      <th>y</th>
      <th>신호등_보행자수</th>
      <th>신호등_차량등수</th>
      <th>cctv수</th>
      <th>...</th>
      <th>적설량</th>
      <th>안개시간</th>
      <th>안전지대수</th>
      <th>중앙분리대수</th>
      <th>정차금지지대수</th>
      <th>도로속도표시수</th>
      <th>교통안전표지수</th>
      <th>노드개수</th>
      <th>횡단보도수</th>
      <th>건물면적</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>다바931203</td>
      <td>2</td>
      <td>MULTIPOLYGON (((127.4230710131166 36.380134552...</td>
      <td>2</td>
      <td>2</td>
      <td>127.423628</td>
      <td>36.380586</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1291.19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>다바861174</td>
      <td>0</td>
      <td>MULTIPOLYGON (((127.3450791441312 36.353914265...</td>
      <td>0</td>
      <td>0</td>
      <td>127.345636</td>
      <td>36.354366</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1159.69</td>
    </tr>
    <tr>
      <th>2</th>
      <td>다바900127</td>
      <td>1</td>
      <td>MULTIPOLYGON (((127.3886063974092 36.311590216...</td>
      <td>1</td>
      <td>1</td>
      <td>127.389163</td>
      <td>36.312042</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>212.75</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 31 columns</p>
</div>



```python
accident_count_filter_20.to_csv('accident_count_filter_20.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수,
# 정차금지지대수, 도로속도표시수, 교통안전표지수, 노드개수, 횡단보도수, 건물면적 데이터까지 merge한 상태.
```

## 21. 차량등록현황 데이터와 현재까지 구축된 데이터를 병합



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
```

<pre>
(5556, 32)
</pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gid</th>
      <th>acci_cnt</th>
      <th>geometry</th>
      <th>사고건수</th>
      <th>사상자수</th>
      <th>x</th>
      <th>y</th>
      <th>신호등_보행자수</th>
      <th>신호등_차량등수</th>
      <th>cctv수</th>
      <th>...</th>
      <th>안개시간</th>
      <th>안전지대수</th>
      <th>중앙분리대수</th>
      <th>정차금지지대수</th>
      <th>도로속도표시수</th>
      <th>교통안전표지수</th>
      <th>노드개수</th>
      <th>횡단보도수</th>
      <th>건물면적</th>
      <th>자동차대수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>다바931203</td>
      <td>2</td>
      <td>MULTIPOLYGON (((127.4230710131166 36.380134552...</td>
      <td>2</td>
      <td>2</td>
      <td>127.423628</td>
      <td>36.380586</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1291.19</td>
      <td>409</td>
    </tr>
    <tr>
      <th>1</th>
      <td>다바861174</td>
      <td>0</td>
      <td>MULTIPOLYGON (((127.3450791441312 36.353914265...</td>
      <td>0</td>
      <td>0</td>
      <td>127.345636</td>
      <td>36.354366</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1159.69</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>다바900127</td>
      <td>1</td>
      <td>MULTIPOLYGON (((127.3886063974092 36.311590216...</td>
      <td>1</td>
      <td>1</td>
      <td>127.389163</td>
      <td>36.312042</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>212.75</td>
      <td>41</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 32 columns</p>
</div>



```python
accident_count_filter_21.to_csv('accident_count_filter_21.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수,
# 정차금지지대수, 도로속도표시수, 교통안전표지수, 노드개수, 횡단보도수, 건물면적, 자동차대수 데이터까지 merge한 상태.
```

## 22. 총인구 데이터와 현재까지 구축된 데이터를 병합



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
```

<pre>
(5556, 33)
</pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gid</th>
      <th>acci_cnt</th>
      <th>geometry</th>
      <th>사고건수</th>
      <th>사상자수</th>
      <th>x</th>
      <th>y</th>
      <th>신호등_보행자수</th>
      <th>신호등_차량등수</th>
      <th>cctv수</th>
      <th>...</th>
      <th>안전지대수</th>
      <th>중앙분리대수</th>
      <th>정차금지지대수</th>
      <th>도로속도표시수</th>
      <th>교통안전표지수</th>
      <th>노드개수</th>
      <th>횡단보도수</th>
      <th>건물면적</th>
      <th>자동차대수</th>
      <th>총거주인구수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>다바931203</td>
      <td>2</td>
      <td>MULTIPOLYGON (((127.4230710131166 36.380134552...</td>
      <td>2</td>
      <td>2</td>
      <td>127.423628</td>
      <td>36.380586</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1291.19</td>
      <td>409</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>다바861174</td>
      <td>0</td>
      <td>MULTIPOLYGON (((127.3450791441312 36.353914265...</td>
      <td>0</td>
      <td>0</td>
      <td>127.345636</td>
      <td>36.354366</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1159.69</td>
      <td>16</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>다바900127</td>
      <td>1</td>
      <td>MULTIPOLYGON (((127.3886063974092 36.311590216...</td>
      <td>1</td>
      <td>1</td>
      <td>127.389163</td>
      <td>36.312042</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>212.75</td>
      <td>41</td>
      <td>82.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 33 columns</p>
</div>



```python
accident_count_filter_22.to_csv('accident_count_filter_22.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수,
# 정차금지지대수, 도로속도표시수, 교통안전표지수, 노드개수, 횡단보도수, 건물면적, 자동차대수, 총거주인구수 데이터까지 merge한 상태.
```

## weather 관련 열 이름 바꾸기



```python
col_name_lst = ['gid', 'acci_cnt', 'geometry', '사고건수', '사상자수', 'x', 'y', '신호등_보행자수', '신호등_차량등수', 'cctv수', '혼잡빈도강도',
                '혼잡시간강도', '전체_추정교통량', '이상평균기온동반사고건수', '이상최저온도동반사고건수', '이상최고온도동반사고건수', '이상최대풍속동반사고건수', '이상평균풍속동반사고건수',
                '이상평균습도동반사고건수', '이상평균지면온도동반사고건수', '이상강수량동반사고건수', '이상적설량동반사고건수', '이상안개시간동반사고건수', '안전지대수',
                '중앙분리대수', '정차금지지대수', '도로속도표시수', '교통안전표지수', '노드개수', '횡단보도수', '건물면적', '자동차대수', '총거주인구수']
accident_count_filter_22.columns = col_name_lst
print(np.shape(accident_count_filter_22))
accident_count_filter_22.head(3)
```

<pre>
(5556, 33)
</pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gid</th>
      <th>acci_cnt</th>
      <th>geometry</th>
      <th>사고건수</th>
      <th>사상자수</th>
      <th>x</th>
      <th>y</th>
      <th>신호등_보행자수</th>
      <th>신호등_차량등수</th>
      <th>cctv수</th>
      <th>...</th>
      <th>안전지대수</th>
      <th>중앙분리대수</th>
      <th>정차금지지대수</th>
      <th>도로속도표시수</th>
      <th>교통안전표지수</th>
      <th>노드개수</th>
      <th>횡단보도수</th>
      <th>건물면적</th>
      <th>자동차대수</th>
      <th>총거주인구수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>다바931203</td>
      <td>2</td>
      <td>MULTIPOLYGON (((127.4230710131166 36.380134552...</td>
      <td>2</td>
      <td>2</td>
      <td>127.423628</td>
      <td>36.380586</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1291.19</td>
      <td>409</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>다바861174</td>
      <td>0</td>
      <td>MULTIPOLYGON (((127.3450791441312 36.353914265...</td>
      <td>0</td>
      <td>0</td>
      <td>127.345636</td>
      <td>36.354366</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1159.69</td>
      <td>16</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>다바900127</td>
      <td>1</td>
      <td>MULTIPOLYGON (((127.3886063974092 36.311590216...</td>
      <td>1</td>
      <td>1</td>
      <td>127.389163</td>
      <td>36.312042</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>212.75</td>
      <td>41</td>
      <td>82.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 33 columns</p>
</div>



```python
accident_count_filter_22.to_csv('accident_count_filter_23.csv')
```
