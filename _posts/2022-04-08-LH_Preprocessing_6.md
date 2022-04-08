---
layout: single
title: "전처리 Part.6"
toc: true
toc_sticky: true
category: LH
---

R을 사용하여 교통사고내역 데이터의 사고유형, 연령대 변수를 현재까지 구축한 데이터와 병합한다.
그 후, python으로 사망자, 중상자 ,경상자 수를 gid 격자별로 count하고 병합한다.

## 전처리 Part.6(by usung R and Python)

```R
library(dplyr)
library(tidyr)
```
### R로 전처리
### 23. 교통사고내역 데이터를 활용하여 사고유형과 연령대를 현재까지 구축한 데이터와 병합

현재까지 구축했던 데이터 불러오기

```R
accident_count_filter_23 <- read.csv('accident_count_filter_23.csv')
dim(accident_count_filter_23)
accident_count_filter_23 %>% head(2)
```

![image](https://user-images.githubusercontent.com/97672187/162198854-8d2572d9-0b18-49bd-98ec-fead54c353c8.png){: .align-center}

<br>



<br>

사고유형(차대차, 차대사람)과 연령대 변수(20대 미만, 20대, 30대, 40대, 50대, 60대 이상)를 모두 고려하기 위하여
각각의 데이터를 병합하여 총 12개의 새로운 그룹을 만들고, 전체 데이터와 병합하도록 한다.

그룹은 사고유형 그룹 2개 * 연령대 그룹 6개로 총 12개이다.

```R
accident_list <- read.csv('1.대전광역시_교통사고내역(2017~2019).csv')
accident_list_sep <- accident_list %>% separate('사고유형', sep='-', into=c('사고유형메인', '사고유형디테일')) # 사고유형 변수를 종류와 디테일로 나누기
dim(accident_list_sep)
accident_list_sep %>% head(2)
```

![image](https://user-images.githubusercontent.com/97672187/162199182-b131cb94-2cdd-42ea-834a-c9d305e7b429.png)

<br>


<br>

```R
accident_list_sep %>%
    group_by(사고유형메인, 피해운전자.연령대, gid) %>% # 격자별, 사고유형별, 연령대별 수 구하기
        summarise(n=n()) %>% ungroup() -> acci_list
dim(acci_list)
acci_list %>% head(2)
```

![image](https://user-images.githubusercontent.com/97672187/162199336-ea2c6380-3d8e-48e3-9ac3-5874d343b95d.png){: .align-center}

<br>


<br>

```R
acci_list <- acci_list %>% filter(사고유형메인 %in% c('차대사람 ', '차대차 ')) # 차대사람과 차대차에 해당하는 값만 추리기(차량단독 제외)
dim(acci_list)
acci_list %>% head(2)
```

![image](https://user-images.githubusercontent.com/97672187/162199571-b090ece1-e288-4dec-8b46-6ddf376eab83.png){: .align-center}

<br>


<br>

```R
# 피해자의 연령대는 미분류된 것을 제외하고, 20대 미만부터 60대 이상까지의 범주로 재정의
acci_list <- acci_list %>% filter(피해운전자.연령대 != '미분류') 
acci_list$피해운전자.연령대 <- as.character(acci_list$피해운전자.연령대)
acci_list$피해운전자.연령대 <- ifelse(acci_list$피해운전자.연령대 %in% c('60대', '70대', '80대', '90대'), '60대 이상', acci_list$피해운전자.연령대)
acci_list$피해운전자.연령대 <- ifelse(acci_list$피해운전자.연령대 %in% c('10대', '10대미만'), '20대 미만', acci_list$피해운전자.연령대)
dim(acci_list)
acci_list %>% head(2)
```

![image](https://user-images.githubusercontent.com/97672187/162199717-7e79554f-ac45-4227-ae5b-517145d86af8.png){: .align-center}

<br>


<br>


```R
acci_list <- acci_list %>% unite(사고유형, 사고유형메인, 피해운전자.연령대, sep="-") # 사고유형과 연령대를 묶어주기
dim(acci_list)
acci_list %>% head(3)
```

![image](https://user-images.githubusercontent.com/97672187/162199819-e98850db-189d-4ccc-913a-e912af53a919.png){: .align-center}


<br>


<br>


여기서의 사고유형은 기존의 사고유형과 연령대를 합친 것.

```R
acci_list %>%
    group_by(gid, 사고유형) %>% # 격자와 사고유형에 따라서 개수를 구하기
        summarise(n=sum(n)) %>%
            ungroup() -> acci_group
dim(acci_group)
acci_group %>% head(3)
```

![image](https://user-images.githubusercontent.com/97672187/162200104-ce5480ea-f048-4f0d-bb43-8b76a92de712.png){: .align-center}


```R
acci_spread <- acci_group %>% spread(사고유형, n, fill=0) # 위 데이터의 사고유형을 열로 사용하기 위해 spread
dim(acci_spread)
acci_spread %>% head(2)
```

![image](https://user-images.githubusercontent.com/97672187/162200267-a82f4352-2de3-4242-a3a4-7ee3912eadca.png){: .align-center}

<br>


<br>

```R
accident_count_filter_23 <- accident_count_filter_23 %>% left_join(acci_spread, by='gid') # 기존 데이터와 병합

for(i in 34:45){ #새로 추가한 변수에 NA가 있으면 0으로 처리
    accident_count_filter_23[,names(accident_count_filter_23)[i] ] <- ifelse(is.na(accident_count_filter_23[,names(accident_count_filter_23)[i] ]), 0, accident_count_filter_23[,names(accident_count_filter_23)[i] ])
}

dim(accident_count_filter_23)
accident_count_filter_23 %>% head(2)
write.csv(accident_count_filter_23, 'accident_count_filter_24.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수, 정차금지지대수
# 도로속도표시수, 교통안전표지수, 노드개수, 횡단보도수, 건물면적, 자동차대수, 총거주인구수, 연령 및 사고유형에 따라 나눈 데이터까지 merge한 상태.
```

![image](https://user-images.githubusercontent.com/97672187/162206289-7b4c914a-d05c-4c18-aafd-4862d202af29.png){: .align-center}


### Python으로 전처리

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

현재까지 구축된 데이터 불러오기


```python
accident_count_filter_24 = pd.read_csv('accident_count_filter_24.csv', index_col = 0)
accident_count_filter_24.head()
```

![image](https://user-images.githubusercontent.com/97672187/162206713-025f8e82-8d62-4529-8e3e-e9a04c98fc6c.png){: .align-center}

<br>


<br>

### 24. 교통사고내역 데이터로 사망자, 중상자, 경상자의 수를 gid 격자에 따라 현재까지 구축한 데이터와 병합

```python
accident_list = pd.read_csv('1.대전광역시_교통사고내역(2017~2019).csv')
count = accident_list.groupby(['gid', '사망자수', '중상자수', '경상자수']).count()
count.reset_index(inplace=True)
count = count[['gid', '사망자수', '중상자수', '경상자수']]
count.head()
```

![image](https://user-images.githubusercontent.com/97672187/162207788-a9c8ef52-8dc2-43c8-9b05-afdf84ea4e8f.png){: .align-center}

<br>


<br>


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

![image](https://user-images.githubusercontent.com/97672187/162207949-7804dcf0-73d8-4491-b809-85818135d020.png){: .align-center}

<br>


<br>


```python
#기존 데이터와 사망,경상,중상자수 병합
accident_count_filter_25 = accident_count_filter_24.merge(count_all, on='gid', how='left')
accident_count_filter_25 = accident_count_filter_25.fillna(0) # NA 제거
accident_count_filter_25 = accident_count_filter_25.iloc[:, 1:]
print(np.shape(accident_count_filter_25))
accident_count_filter_25.head(3)
accident_count_filter_25.to_csv('accident_count_filter_25.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수, 정차금지지대수, 도로속도표시수
# 교통안전표지수, 노드개수, 횡단보도수, 건물면적, 자동차대수, 총거주인구수, 연령 및 사고유형, 사망&경상&중상자수에 따라 나눈 데이터까지 merge한 상태.
```

![image](https://user-images.githubusercontent.com/97672187/162208582-4e9fded2-94d3-4008-b3de-1a4da4cdbbbb.png){: .align-center}


<br>


<br>


