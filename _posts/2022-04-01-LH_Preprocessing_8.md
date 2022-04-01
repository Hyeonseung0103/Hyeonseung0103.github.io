---
layout: single
title:  "전처리 Part.8"
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
## 현재까지 구축했던 데이터 불러오기



```python
accident_count_filter_24 = pd.read_csv('accident_count_filter_24.csv', index_col = 0)
accident_count_filter_24.head()
```

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
      <th>X</th>
      <th>gid</th>
      <th>acci_cnt</th>
      <th>geometry</th>
      <th>사고건수</th>
      <th>사상자수</th>
      <th>x</th>
      <th>y</th>
      <th>신호등_보행자수</th>
      <th>신호등_차량등수</th>
      <th>...</th>
      <th>차대사람 -30대</th>
      <th>차대사람 -40대</th>
      <th>차대사람 -50대</th>
      <th>차대사람 -60대 이상</th>
      <th>차대차 -20대</th>
      <th>차대차 -20대 미만</th>
      <th>차대차 -30대</th>
      <th>차대차 -40대</th>
      <th>차대차 -50대</th>
      <th>차대차 -60대 이상</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>다바931203</td>
      <td>2</td>
      <td>MULTIPOLYGON (((127.4230710131166 36.380134552...</td>
      <td>2</td>
      <td>2</td>
      <td>127.423628</td>
      <td>36.380586</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>다바861174</td>
      <td>0</td>
      <td>MULTIPOLYGON (((127.3450791441312 36.353914265...</td>
      <td>0</td>
      <td>0</td>
      <td>127.345636</td>
      <td>36.354366</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>다바900127</td>
      <td>1</td>
      <td>MULTIPOLYGON (((127.3886063974092 36.311590216...</td>
      <td>1</td>
      <td>1</td>
      <td>127.389163</td>
      <td>36.312042</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>다바885170</td>
      <td>15</td>
      <td>MULTIPOLYGON (((127.3718339133395 36.350339794...</td>
      <td>15</td>
      <td>15</td>
      <td>127.372390</td>
      <td>36.350791</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>다바957151</td>
      <td>4</td>
      <td>MULTIPOLYGON (((127.452087475365 36.3332696085...</td>
      <td>4</td>
      <td>4</td>
      <td>127.452644</td>
      <td>36.333721</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>


## 24. 교통사고내역 데이터로 사망자, 중상자, 경상자의 수를 gid 격자에 따라 현재까지 구축한 데이터와 병합



```python
accident_list = pd.read_csv('1.대전광역시_교통사고내역(2017~2019).csv')
count = accident_list.groupby(['gid', '사망자수', '중상자수', '경상자수']).count()
count.reset_index(inplace=True)
count = count[['gid', '사망자수', '중상자수', '경상자수']]
count.head()
```

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
      <th>사망자수</th>
      <th>중상자수</th>
      <th>경상자수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>다바780093</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>다바781090</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>다바781091</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>다바781091</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>다바787132</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
count_death = count[['gid','사망자수']].groupby('gid').sum('사망자수')
count_inj = count[['gid','중상자수']].groupby('gid').sum('중상자수')
count_weak = count[['gid','경상자수']].groupby('gid').sum('경상자수')
```


```python
count_all = count_death.merge(count_weak, on='gid', how='left').merge(count_inj, on='gid', how='left')
count_all.reset_index(inplace=True)

count_all 
```

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
      <th>사망자수</th>
      <th>경상자수</th>
      <th>중상자수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>다바780093</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>다바781090</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>다바781091</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>다바787132</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>다바787134</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6063</th>
      <td>라바010169</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6064</th>
      <td>라바016173</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6065</th>
      <td>라바024244</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6066</th>
      <td>라바033232</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6067</th>
      <td>라바033242</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>6068 rows × 4 columns</p>
</div>



```python
accident_count_filter_25 = accident_count_filter_24.merge(count_all, on='gid', how='left')
accident_count_filter_25.head()
```

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
      <th>X</th>
      <th>gid</th>
      <th>acci_cnt</th>
      <th>geometry</th>
      <th>사고건수</th>
      <th>사상자수</th>
      <th>x</th>
      <th>y</th>
      <th>신호등_보행자수</th>
      <th>신호등_차량등수</th>
      <th>...</th>
      <th>차대사람 -60대 이상</th>
      <th>차대차 -20대</th>
      <th>차대차 -20대 미만</th>
      <th>차대차 -30대</th>
      <th>차대차 -40대</th>
      <th>차대차 -50대</th>
      <th>차대차 -60대 이상</th>
      <th>사망자수</th>
      <th>경상자수</th>
      <th>중상자수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>다바931203</td>
      <td>2</td>
      <td>MULTIPOLYGON (((127.4230710131166 36.380134552...</td>
      <td>2</td>
      <td>2</td>
      <td>127.423628</td>
      <td>36.380586</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>다바861174</td>
      <td>0</td>
      <td>MULTIPOLYGON (((127.3450791441312 36.353914265...</td>
      <td>0</td>
      <td>0</td>
      <td>127.345636</td>
      <td>36.354366</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>다바900127</td>
      <td>1</td>
      <td>MULTIPOLYGON (((127.3886063974092 36.311590216...</td>
      <td>1</td>
      <td>1</td>
      <td>127.389163</td>
      <td>36.312042</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>다바885170</td>
      <td>15</td>
      <td>MULTIPOLYGON (((127.3718339133395 36.350339794...</td>
      <td>15</td>
      <td>15</td>
      <td>127.372390</td>
      <td>36.350791</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>다바957151</td>
      <td>4</td>
      <td>MULTIPOLYGON (((127.452087475365 36.3332696085...</td>
      <td>4</td>
      <td>4</td>
      <td>127.452644</td>
      <td>36.333721</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 49 columns</p>
</div>



```python
accident_count_filter_25 = accident_count_filter_25.fillna(0)
accident_count_filter_25 = accident_count_filter_25.iloc[:, 1:]
print(np.shape(accident_count_filter_25))
accident_count_filter_25.head(3)
```

<pre>
(5556, 48)
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
      <th>차대사람 -60대 이상</th>
      <th>차대차 -20대</th>
      <th>차대차 -20대 미만</th>
      <th>차대차 -30대</th>
      <th>차대차 -40대</th>
      <th>차대차 -50대</th>
      <th>차대차 -60대 이상</th>
      <th>사망자수</th>
      <th>경상자수</th>
      <th>중상자수</th>
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
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 48 columns</p>
</div>



```python
accident_count_filter_25.to_csv('accident_count_filter_25.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수, 정차금지지대수, 도로속도표시수
# 교통안전표지수, 노드개수, 횡단보도수, 건물면적, 자동차대수, 총거주인구수, 연령 및 사고유형, 사망&경상&중상자수에 따라 나눈 데이터까지 merge한 상태.
```
