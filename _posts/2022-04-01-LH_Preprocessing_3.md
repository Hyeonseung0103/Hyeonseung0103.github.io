---
layout: single
title:  "전처리 Part.3"
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

<pre>
/opt/app-root/lib/python3.6/site-packages/geopandas/_compat.py:91: UserWarning: The Shapely GEOS version (3.8.0-CAPI-1.13.1 ) is incompatible with the GEOS version PyGEOS was compiled with (3.9.0-CAPI-1.16.2). Conversions between both will be slow.
  shapely_geos_version, geos_capi_version_string
</pre>

```python
# 데이터셋 다운로드
from geoband.API import *

GetCompasData('SBJ_2102_003', '3', '3.대전광역시_신호등(보행등).geojson')
GetCompasData('SBJ_2102_003', '4', '4.대전광역시_신호등(차량등).geojson')
GetCompasData('SBJ_2102_003', '10', '10.대전광역시_교통CCTV.geojson')
GetCompasData('SBJ_2102_003', '16', '16.대전광역시_기상데이터(2017~2019).csv')
GetCompasData('SBJ_2102_003', '19', '19.대전광역시_상세도로망(2018).geojson')
GetCompasData('SBJ_2102_003', '20', '20.대전광역시_평일_일별_시간대별_추정교통량(2018).csv')
GetCompasData('SBJ_2102_003', '21', '21.대전광역시_평일_일별_혼잡빈도강도(2018).csv')
GetCompasData('SBJ_2102_003', '22', '22.대전광역시_평일_일별_혼잡시간강도(2018).csv')

daejeon_signal_walk = gpd.read_file('3.대전광역시_신호등(보행등).geojson')
daejeon_signal_car = gpd.read_file('4.대전광역시_신호등(차량등).geojson')
cctv = gpd.read_file('10.대전광역시_교통CCTV.geojson')
weather = pd.read_csv('16.대전광역시_기상데이터(2017~2019).csv')
road_detail = gpd.read_file('19.대전광역시_상세도로망(2018).geojson')
day_time_accident = pd.read_csv('20.대전광역시_평일_일별_시간대별_추정교통량(2018).csv')
day_time_confuse = pd.read_csv('21.대전광역시_평일_일별_혼잡빈도강도(2018).csv')
day_time_times = pd.read_csv('22.대전광역시_평일_일별_혼잡시간강도(2018).csv')
```

## 5. accident_count_filter 데이터 불러오기(의미가 떨어지는 사고건수가 0인 데이터 Filter)



```python
accident_gid = pd.read_csv('accident_gid.csv', index_col = 0)
print(np.shape(accident_gid))
accident_gid.head(3)
```

<pre>
(25097, 4)
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
      <th>geometry_2</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>다바866110</td>
      <td>0.0</td>
      <td>LINESTRING (127.3486909175812 36.2983602119806...</td>
      <td>MULTIPOLYGON (((127.3507618774412 36.296222652...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>다바823157</td>
      <td>0.0</td>
      <td>LINESTRING (127.3039387031694 36.3388912702199...</td>
      <td>MULTIPOLYGON (((127.3027655376963 36.338525724...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>다바823157</td>
      <td>0.0</td>
      <td>LINESTRING (127.3039387031694 36.3388912702199...</td>
      <td>MULTIPOLYGON (((127.3027655376963 36.338525724...</td>
    </tr>
  </tbody>
</table>
</div>



```python
accident_count_filter = pd.read_csv('accident_count_filter.csv')
accident_count_filter['geometry'] = [wkt.loads(line) for line in list(accident_count_filter['geometry'])]
print(np.shape(accident_count_filter))
accident_count_filter.head(3)
```

<pre>
(5556, 7)
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
    </tr>
  </tbody>
</table>
</div>


## 6. 교통안전시설물 - 보행자 신호등의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합



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

get_gid 함수를 통해 **criteria와 df의 geometry 값에 대하여 두 데이터가 서로 교차(within)하는지 확인하여 그에 맞는 gid를 Labeling**한다.<br/>

이 함수를 통하여 gid 격자가 없는 데이터들을 Labeling 해서 1번 과정에서 만들었던 5,556개 격자 데이터와 병합하는 작업을 수행하도록 한다.



```python
## 보행자 신호등
signal_walk_samp = get_gid(accident_count_filter, daejeon_signal_walk)
signal_walk_samp.to_csv('final_signal_walk.csv')
signal_walk_samp.head()
```


```python
final_signal_walk = pd.read_csv('final_signal_walk.csv', index_col = 0)
print(np.shape(final_signal_walk))
final_signal_walk.head()
```

<pre>
(5552, 8)
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
      <th>gu</th>
      <th>dong</th>
      <th>jibun</th>
      <th>loc_cd</th>
      <th>sgnl_drn_cd</th>
      <th>sgnl_knd_cd</th>
      <th>geometry</th>
      <th>gid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>서구</td>
      <td>관저동</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>POINT (127.334619467568 36.29501431648465)</td>
      <td>다바851108</td>
    </tr>
    <tr>
      <th>1</th>
      <td>유성구</td>
      <td>노은동</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>POINT (127.3235205239782 36.3670329930303)</td>
      <td>다바841188</td>
    </tr>
    <tr>
      <th>2</th>
      <td>동구</td>
      <td>판암동</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>POINT (127.4613672442361 36.32029083293295)</td>
      <td>다바965136</td>
    </tr>
    <tr>
      <th>7</th>
      <td>동구</td>
      <td>천동</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>POINT (127.4445737985495 36.31713656664272)</td>
      <td>다바950133</td>
    </tr>
    <tr>
      <th>8</th>
      <td>동구</td>
      <td>천동</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>POINT (127.4444815869447 36.31659667794215)</td>
      <td>다바950132</td>
    </tr>
  </tbody>
</table>
</div>



```python
signal_walk_count = final_signal_walk.groupby('gid').count()[['geometry']]
signal_walk_count.reset_index(inplace=True)
signal_walk_count.columns = ['gid', '신호등_보행자수']

accident_count_filter = accident_count_filter.merge(signal_walk_count, on = 'gid', how = 'left')
accident_count_filter['신호등_보행자수'] = accident_count_filter['신호등_보행자수'].fillna(0)
print(np.shape(accident_count_filter))
accident_count_filter.head(3)
```

<pre>
(5556, 8)
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
    </tr>
  </tbody>
</table>
</div>


## 7. 교통안전시설물 - 차량 신호등의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합



```python
## 차량 신호등
signal_car_samp = get_gid(accident_count_filter, daejeon_signal_car)
signal_car_samp.to_csv('final_signal_car.csv')
signal_car_samp.head()
```


```python
final_signal_car = pd.read_csv('final_signal_car.csv', index_col = 0)
print(np.shape(final_signal_car))
final_signal_car.head()
```

<pre>
(8794, 8)
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
      <th>gu</th>
      <th>dong</th>
      <th>jibun</th>
      <th>loc_cd</th>
      <th>sgnl_drn_cd</th>
      <th>sgnl_knd_cd</th>
      <th>geometry</th>
      <th>gid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>유성구</td>
      <td>구암동</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>POINT (127.329215606638 36.34866636469033)</td>
      <td>다바846168</td>
    </tr>
    <tr>
      <th>1</th>
      <td>유성구</td>
      <td>구암동</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>POINT (127.3292166841326 36.34861926175856)</td>
      <td>다바846168</td>
    </tr>
    <tr>
      <th>2</th>
      <td>유성구</td>
      <td>구암동</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>POINT (127.3293927765914 36.34835870534383)</td>
      <td>다바846167</td>
    </tr>
    <tr>
      <th>3</th>
      <td>동구</td>
      <td>삼정동</td>
      <td>11-17전</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>POINT (127.4776798809445 36.33255601008285)</td>
      <td>다바979150</td>
    </tr>
    <tr>
      <th>4</th>
      <td>유성구</td>
      <td>구암동</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>POINT (127.3295454232338 36.34861674022679)</td>
      <td>다바847168</td>
    </tr>
  </tbody>
</table>
</div>



```python
signal_car_count = final_signal_car.groupby('gid').count()[['geometry']]
signal_car_count.reset_index(inplace=True)
signal_car_count.columns = ['gid', '신호등_차량등수']

accident_count_filter = accident_count_filter.merge(signal_car_count, on = 'gid', how = 'left')
accident_count_filter['신호등_차량등수'] = accident_count_filter['신호등_차량등수'].fillna(0)
print(np.shape(accident_count_filter))
accident_count_filter.head(3)
```

<pre>
(5556, 9)
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
    </tr>
  </tbody>
</table>
</div>


## 8. 교통안전시설물 - CCTV의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합



```python
## CCTV
cctv_samp = get_gid(accident_count_filter, cctv)
cctv_samp.to_csv('final_cctv.csv')
cctv_samp
```


```python
final_cctv = pd.read_csv('final_cctv.csv', index_col = 0)
print(np.shape(final_cctv))
final_cctv.head()
```

<pre>
(118, 5)
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
      <th>gu</th>
      <th>dong</th>
      <th>jibun</th>
      <th>geometry</th>
      <th>gid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>대덕구</td>
      <td>오정동</td>
      <td>NaN</td>
      <td>POINT (127.4152345319938 36.35875724574742)</td>
      <td>다바923179</td>
    </tr>
    <tr>
      <th>1</th>
      <td>대덕구</td>
      <td>오정동</td>
      <td>NaN</td>
      <td>POINT (127.4152392844907 36.35876507827574)</td>
      <td>다바923179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>서구</td>
      <td>탄방동</td>
      <td>NaN</td>
      <td>POINT (127.401849404806 36.34503114747884)</td>
      <td>다바911164</td>
    </tr>
    <tr>
      <th>3</th>
      <td>동구</td>
      <td>가양동</td>
      <td>NaN</td>
      <td>POINT (127.4423500116377 36.35097538486666)</td>
      <td>다바948170</td>
    </tr>
    <tr>
      <th>4</th>
      <td>서구</td>
      <td>용문동</td>
      <td>NaN</td>
      <td>POINT (127.393257603758 36.33811979940576)</td>
      <td>다바904156</td>
    </tr>
  </tbody>
</table>
</div>



```python
cctv_count = final_cctv.groupby('gid').count()[['geometry']]
cctv_count.reset_index(inplace=True)
cctv_count.columns = ['gid', 'cctv수']

accident_count_filter = accident_count_filter.merge(cctv_count, on = 'gid', how = 'left')
accident_count_filter['cctv수'] = accident_count_filter['cctv수'].fillna(0)
print(np.shape(accident_count_filter))
accident_count_filter.head(3)
```

<pre>
(5556, 10)
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
    </tr>
  </tbody>
</table>
</div>



```python
accident_count_filter.to_csv('accident_count_filter_1.csv') # 신호등(보행등), 신호등(차량등), cctv수까지 merge한 상태.
```

## 9. 교통혼잡빈도 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합



```python
accident_count_filter_1 = pd.read_csv('accident_count_filter_1.csv', index_col = 0)
accident_count_filter_1['geometry'] = [wkt.loads(line) for line in list(accident_count_filter_1['geometry'])] # str 형태에서 geometry 형태로 형 변환
print(np.shape(accident_count_filter_1))
accident_count_filter_1.head(3)
```

<pre>
(5556, 10)
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
    </tr>
  </tbody>
</table>
</div>



```python
day_time_confuse['link_id'] = [int(id/100) for id in list(day_time_confuse['상세도로망_LinkID'])] # 상세도로망_LinkID로 link_id 추출 
road_detail = road_detail.astype({'link_id' : 'int64'})
day_time_confused = day_time_confuse.merge(road_detail, on='link_id', how='left') # 위에서 만든 link_id를 gid가 있는 road_detail 데이터와 병합
print(np.shape(day_time_confused))
day_time_confused.head(2)
```

<pre>
(18044, 27)
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
      <th>상세도로망_LinkID</th>
      <th>도로등급</th>
      <th>링크길이</th>
      <th>도로명</th>
      <th>시도명</th>
      <th>시군구명</th>
      <th>읍면동명</th>
      <th>혼잡빈도강도</th>
      <th>link_id</th>
      <th>max_speed</th>
      <th>...</th>
      <th>dw_lanes</th>
      <th>oneway</th>
      <th>length</th>
      <th>width</th>
      <th>car_lane</th>
      <th>num_cross</th>
      <th>barrier</th>
      <th>up_its_id</th>
      <th>dw_its_id</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56421933101</td>
      <td>101</td>
      <td>0.121</td>
      <td>경부고속도로</td>
      <td>대전광역시</td>
      <td>대덕구</td>
      <td>덕암동</td>
      <td>28.24</td>
      <td>564219331</td>
      <td>100</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0.121</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1870196300</td>
      <td>0</td>
      <td>MULTILINESTRING ((127.41780 36.43107, 127.4181...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56421763601</td>
      <td>101</td>
      <td>0.244</td>
      <td>경부고속도로</td>
      <td>대전광역시</td>
      <td>대덕구</td>
      <td>덕암동</td>
      <td>32.06</td>
      <td>564217636</td>
      <td>100</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0.244</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1870196000</td>
      <td>0</td>
      <td>MULTILINESTRING ((127.41688 36.42635, 127.4168...</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 27 columns</p>
</div>



```python
def get_gid2(criteria, df) : # 도로 형태의 Polygon 형태. 따라서 within이 아닌 crosses 사용
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


```python
## 교통혼잡도
accident_count_filter_2 = get_gid2(accident_count_filter_1, day_time_confused)
accident_count_filter_2.to_csv('accident_count_filter_2.csv')
accident_count_filter_2
```


```python
accident_count_filter_2 = pd.read_csv('accident_count_filter_2.csv', index_col = 0)
accident_count_filter_2['geometry'] = [wkt.loads(line) for line in list(accident_count_filter_2['geometry'])]
print(np.shape(accident_count_filter_2))
accident_count_filter_2.head(3)
```

<pre>
(10028, 28)
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
      <th>상세도로망_LinkID</th>
      <th>도로등급</th>
      <th>링크길이</th>
      <th>도로명</th>
      <th>시도명</th>
      <th>시군구명</th>
      <th>읍면동명</th>
      <th>혼잡빈도강도</th>
      <th>link_id</th>
      <th>max_speed</th>
      <th>...</th>
      <th>oneway</th>
      <th>length</th>
      <th>width</th>
      <th>car_lane</th>
      <th>num_cross</th>
      <th>barrier</th>
      <th>up_its_id</th>
      <th>dw_its_id</th>
      <th>geometry</th>
      <th>gid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>56421763601</td>
      <td>101</td>
      <td>0.244</td>
      <td>경부고속도로</td>
      <td>대전광역시</td>
      <td>대덕구</td>
      <td>덕암동</td>
      <td>32.06</td>
      <td>564217636</td>
      <td>100</td>
      <td>...</td>
      <td>1</td>
      <td>0.244</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1870196000</td>
      <td>0</td>
      <td>(LINESTRING (127.4168847073474 36.426348006591...</td>
      <td>다바925253</td>
    </tr>
    <tr>
      <th>7</th>
      <td>56420113501</td>
      <td>101</td>
      <td>1.151</td>
      <td>경부고속도로</td>
      <td>대전광역시</td>
      <td>대덕구</td>
      <td>덕암동</td>
      <td>36.20</td>
      <td>564201135</td>
      <td>100</td>
      <td>...</td>
      <td>1</td>
      <td>1.151</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1870198000</td>
      <td>0</td>
      <td>(LINESTRING (127.4172644168076 36.453428662051...</td>
      <td>다바927278</td>
    </tr>
    <tr>
      <th>8</th>
      <td>56420113201</td>
      <td>101</td>
      <td>1.151</td>
      <td>경부고속도로</td>
      <td>대전광역시</td>
      <td>대덕구</td>
      <td>덕암동</td>
      <td>42.36</td>
      <td>564201132</td>
      <td>100</td>
      <td>...</td>
      <td>1</td>
      <td>1.151</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1870198100</td>
      <td>0</td>
      <td>(LINESTRING (127.4197828255578 36.443253549970...</td>
      <td>다바927278</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 28 columns</p>
</div>



```python
confused = accident_count_filter_2.groupby('gid').mean()[['혼잡빈도강도']] # 격자별로 혼잡빈도강도의 평균 값을 사용
confused.reset_index(inplace=True)

accident_count_filter_1 = accident_count_filter_1.merge(confused, on = 'gid', how = 'left')
accident_count_filter_1['혼잡빈도강도'] = accident_count_filter_1['혼잡빈도강도'].fillna(0)
print(np.shape(accident_count_filter_1))
accident_count_filter_1.head(3)
```

<pre>
(5556, 11)
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
      <th>혼잡빈도강도</th>
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
      <td>97.36</td>
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
      <td>62.85</td>
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
      <td>94.56</td>
    </tr>
  </tbody>
</table>
</div>



```python
accident_count_filter_1.to_csv('accident_count_filter_3.csv') # 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도까지 merge한 상태.
```

## 10. 교통혼잡시간 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합



```python
accident_count_filter_3 = pd.read_csv('accident_count_filter_3.csv', index_col = 0)
accident_count_filter_3['geometry'] = [wkt.loads(line) for line in list(accident_count_filter_3['geometry'])]
print(np.shape(accident_count_filter_3))
accident_count_filter_3.head(3)
```

<pre>
(5556, 11)
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
      <th>혼잡빈도강도</th>
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
      <td>97.36</td>
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
      <td>62.85</td>
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
      <td>94.56</td>
    </tr>
  </tbody>
</table>
</div>



```python
day_time_times['link_id'] = [int(id/100) for id in list(day_time_confuse['상세도로망_LinkID']) ]
day_time_timed = day_time_times.merge(road_detail, on='link_id', how='left')
print(np.shape(day_time_timed))
day_time_timed.head(2)
```

<pre>
(18044, 27)
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
      <th>상세도로망_LinkID</th>
      <th>도로등급</th>
      <th>링크길이</th>
      <th>도로명</th>
      <th>시도명</th>
      <th>시군구명</th>
      <th>읍면동명</th>
      <th>혼잡시간강도</th>
      <th>link_id</th>
      <th>max_speed</th>
      <th>...</th>
      <th>dw_lanes</th>
      <th>oneway</th>
      <th>length</th>
      <th>width</th>
      <th>car_lane</th>
      <th>num_cross</th>
      <th>barrier</th>
      <th>up_its_id</th>
      <th>dw_its_id</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56421933101</td>
      <td>101</td>
      <td>0.121</td>
      <td>경부고속도로</td>
      <td>대전광역시</td>
      <td>대덕구</td>
      <td>덕암동</td>
      <td>35.50</td>
      <td>564219331</td>
      <td>100</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0.121</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1870196300</td>
      <td>0</td>
      <td>MULTILINESTRING ((127.41780 36.43107, 127.4181...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56421763601</td>
      <td>101</td>
      <td>0.244</td>
      <td>경부고속도로</td>
      <td>대전광역시</td>
      <td>대덕구</td>
      <td>덕암동</td>
      <td>47.61</td>
      <td>564217636</td>
      <td>100</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0.244</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1870196000</td>
      <td>0</td>
      <td>MULTILINESTRING ((127.41688 36.42635, 127.4168...</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 27 columns</p>
</div>



```python
## 교통혼잡시간
accident_count_filter_4 = get_gid2(accident_count_filter_3, day_time_timed)
accident_count_filter_4.to_csv('accident_count_filter_4.csv')
accident_count_filter_4
```


```python
accident_count_filter_4 = pd.read_csv('accident_count_filter_4.csv', index_col = 0)
accident_count_filter_4 = accident_count_filter_4.reset_index(drop = True)
accident_count_filter_4['geometry'] = [wkt.loads(line) for line in list(accident_count_filter_4['geometry'])]
print(np.shape(accident_count_filter_4))
accident_count_filter_4.head(3)
```

<pre>
(10028, 28)
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
      <th>상세도로망_LinkID</th>
      <th>도로등급</th>
      <th>링크길이</th>
      <th>도로명</th>
      <th>시도명</th>
      <th>시군구명</th>
      <th>읍면동명</th>
      <th>혼잡시간강도</th>
      <th>link_id</th>
      <th>max_speed</th>
      <th>...</th>
      <th>oneway</th>
      <th>length</th>
      <th>width</th>
      <th>car_lane</th>
      <th>num_cross</th>
      <th>barrier</th>
      <th>up_its_id</th>
      <th>dw_its_id</th>
      <th>geometry</th>
      <th>gid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56421763601</td>
      <td>101</td>
      <td>0.244</td>
      <td>경부고속도로</td>
      <td>대전광역시</td>
      <td>대덕구</td>
      <td>덕암동</td>
      <td>47.61</td>
      <td>564217636</td>
      <td>100</td>
      <td>...</td>
      <td>1</td>
      <td>0.244</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1870196000</td>
      <td>0</td>
      <td>(LINESTRING (127.4168847073474 36.426348006591...</td>
      <td>다바925253</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56420113501</td>
      <td>101</td>
      <td>1.151</td>
      <td>경부고속도로</td>
      <td>대전광역시</td>
      <td>대덕구</td>
      <td>덕암동</td>
      <td>43.17</td>
      <td>564201135</td>
      <td>100</td>
      <td>...</td>
      <td>1</td>
      <td>1.151</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1870198000</td>
      <td>0</td>
      <td>(LINESTRING (127.4172644168076 36.453428662051...</td>
      <td>다바927278</td>
    </tr>
    <tr>
      <th>2</th>
      <td>56420113201</td>
      <td>101</td>
      <td>1.151</td>
      <td>경부고속도로</td>
      <td>대전광역시</td>
      <td>대덕구</td>
      <td>덕암동</td>
      <td>47.59</td>
      <td>564201132</td>
      <td>100</td>
      <td>...</td>
      <td>1</td>
      <td>1.151</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1870198100</td>
      <td>0</td>
      <td>(LINESTRING (127.4197828255578 36.443253549970...</td>
      <td>다바927278</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 28 columns</p>
</div>



```python
timed = accident_count_filter_4.groupby('gid').mean()[['혼잡시간강도']] # 격자별로 혼잡시간강도의 평균 값을 사용
timed.reset_index(inplace=True)

accident_count_filter_5 = accident_count_filter_3.merge(timed, on = 'gid', how = 'left')
accident_count_filter_5['혼잡시간강도'] = accident_count_filter_5['혼잡시간강도'].fillna(0)
print(np.shape(accident_count_filter_5))
accident_count_filter_5.head(3)
```

<pre>
(5556, 12)
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
      <th>혼잡빈도강도</th>
      <th>혼잡시간강도</th>
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
      <td>97.36</td>
      <td>98.5000</td>
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
      <td>62.85</td>
      <td>86.1475</td>
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
      <td>94.56</td>
      <td>17.4900</td>
    </tr>
  </tbody>
</table>
</div>


## 11. 시간대별 추정교통량 데이터의 geometry에 따라 그에 일치하는 gid(격자) Labeling 및 병합



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

<pre>
(5556, 13)
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
      <th>혼잡빈도강도</th>
      <th>혼잡시간강도</th>
      <th>전체_추정교통량</th>
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
      <td>97.360000</td>
      <td>98.500000</td>
      <td>927.740000</td>
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
      <td>62.850000</td>
      <td>86.147500</td>
      <td>280.790000</td>
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
      <td>94.560000</td>
      <td>17.490000</td>
      <td>568.760000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>다바885170</td>
      <td>15</td>
      <td>(POLYGON ((127.3718339133395 36.35033979440872...</td>
      <td>15</td>
      <td>15</td>
      <td>127.372390</td>
      <td>36.350791</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38.888333</td>
      <td>54.821667</td>
      <td>1710.090000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>다바957151</td>
      <td>4</td>
      <td>(POLYGON ((127.452087475365 36.33326960850717,...</td>
      <td>4</td>
      <td>4</td>
      <td>127.452644</td>
      <td>36.333721</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>62.542222</td>
      <td>75.783333</td>
      <td>185.816471</td>
    </tr>
  </tbody>
</table>
</div>



```python
accident_count_filter_5.to_csv('accident_count_filter_5.csv') # 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량까지 merge한 상태.
```

## weather 변수를 연도별로 나눠주기



```python
weather_2017 = weather[weather['일시'] < '2018-01-01']
weather_2018 = weather[(weather['일시'] >= '2018-01-01') & (weather['일시'] <= '2018-12-31')].reset_index(drop = True)
weather_2019 = weather[weather['일시'] >= '2019-01-01'].reset_index(drop = True)
```


```python
weather_2017.to_csv('weather_2017.csv')
weather_2018.to_csv('weather_2018.csv')
weather_2019.to_csv('weather_2019.csv')
```
