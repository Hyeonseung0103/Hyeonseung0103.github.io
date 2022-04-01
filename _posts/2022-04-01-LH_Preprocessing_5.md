---
layout: single
title:  "전처리 Part.5"
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
## 전처리한 weather 데이터 불러오기



```python
accident_list = pd.read_csv('1.대전광역시_교통사고내역(2017~2019).csv')
```


```python
weather_arrange = pd.read_csv('weather_arrange.csv')
weather_arrange.columns = ['사고일', '평균온도', '최저온도', '최고온도', '최대풍속' ,'평균풍속', '평균습도', '평균지면온도', '강수량' ,'적설량', '안개시간']

print(np.shape(weather_arrange))
weather_arrange.head(3)
```

<pre>
(1095, 11)
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
      <th>사고일</th>
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
      <td>2017-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-01-02</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-01-03</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
acci = accident_list.merge(weather_arrange, on='사고일', how='left')
acci_sum = acci.groupby('gid').sum(['평균온도', '최고온도', '최저온도', '최대풍속', '평균풍속', '평균습도','평균지면온도', '강수량', '적설량', '안개시간']) 
acci_sum.reset_index(inplace=True)
acci_sum = acci_sum[['gid', '평균온도', '최저온도', '최고온도', '최대풍속', '평균풍속', '평균습도', '평균지면온도', '강수량', '적설량', '안개시간']]
print(np.shape(acci_sum))
acci_sum.head(3)
```

<pre>
(6068, 11)
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
      <td>다바780093</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>다바781090</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>다바781091</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


## 12. 전처리한 weather 데이터와 현재까지 구축한 데이터와 병합



```python
accident_count_filter_5 = pd.read_csv('accident_count_filter_5.csv', index_col = 0)
accident_count_filter_6 = accident_count_filter_5.merge(acci_sum, on='gid', how='left')
accident_count_filter_6 = accident_count_filter_6.fillna(0)
print(np.shape(accident_count_filter_6))
accident_count_filter_6.head(2)
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
  </tbody>
</table>
<p>2 rows × 23 columns</p>
</div>



```python
accident_count_filter_6.to_csv('accident_count_filter_6.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨 데이터까지 merge한 상태.
```
