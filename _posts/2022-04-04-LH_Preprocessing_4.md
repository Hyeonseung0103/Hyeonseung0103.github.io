---
layout: single
title: "전처리 Part.4"
toc: true
toc_sticky: true
category: LH
---

전처리 Part.4에서는 R을 사용해 연도별 날씨 데이터를 불러와서 전처리를 진행했다.

```R
#패키지 불러오기
library(dplyr)
library(tidyr)
library(stringr)
library(sp)
library(rgdal)
library(rgeos)
library(tmap)
library(raster)
library(spdep)
library(gstat)
library(spgwr)
library(GWmodel)
```


## 전처리 Part.4(by using R)
### 연도별 weather 데이터 불러오기
세계기상기구(WMO)는 기온, 강수량 등의 기후요소가 평년값에 비해 현저히 높거나 낮은 수치(90퍼센타일 또는 10퍼센타일 미만의 범위)를 이상 기후로 정의하고 있다.

이 기준에 따라서 2017 ~ 2019년의 weather 데이터의 이상치를 분류하도록 한다.

- 기온과 풍속, 습도, 지면 온도의 데이터 전처리: 1년 별로 비교해서 이상치(하위 10% 미만, 상위 10% 초과) 안에 포함되는 경우 1, 포함되지 않는 경우에는 0

- 강수량, 적설량 데이터 전처리: 당해 연도의 상위 10% 백분위 수에 포함되는 경우 1, 포함되지 않는 경우에는 0

- 안개 데이터 전처리: 안개는 지속 시간이 조금이라도 있으면 1, 없으면 0


1) 2017년의 weather 데이터 처리

```R
weather_2017 <- read.csv('weather_2017.csv')
```

```R
temp_average <- c() # 평균온도
qt <- quantile(weather_2017[, 3], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_average <- ifelse(weather_2017[, 3] < qt[2] | weather_2017[, 3] > qt[3], 1, 0)
# 10퍼센타일 미만이거나 90퍼센타일 이상이면 1 아니면 0

#나머지도 동일하게 적용
temp_min <- c() #최저온도
qt <- quantile(weather_2017[, 4], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_min <- ifelse(weather_2017[, 4] < qt[2] | weather_2017[, 4] > qt[3], 1, 0)

temp_max <- c() #최고온도
qt <- quantile(weather_2017[, 5], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_max <- ifelse(weather_2017[, 5] < qt[2] | weather_2017[, 5] > qt[3], 1, 0)

wind_max <- c() #최대풍속
qt <- quantile(weather_2017[, 7], probs=c(0, 0.1, 0.9, 1), na.rm=T)
wind_max <- ifelse(weather_2017[, 7] < qt[2] | weather_2017[, 7] > qt[3], 1, 0)

wind_average <- c() #평균풍속
qt <- quantile(weather_2017[, 9], probs=c(0, 0.1, 0.9, 1), na.rm=T)
wind_average <- ifelse(weather_2017[, 9] < qt[2] | weather_2017[, 9] > qt[3], 1, 0)

humidity_average <- c() #평균습도
qt <- quantile(weather_2017[, 10], probs=c(0, 0.1, 0.9, 1), na.rm=T)
humidity_average <- ifelse(weather_2017[, 10] < qt[2] | weather_2017[, 10] > qt[3], 1, 0)

road_average <- c() #평균지면온도
qt <- quantile(weather_2017[, 12], probs=c(0, 0.1, 0.9, 1), na.rm=T)
road_average <- ifelse(weather_2017[, 12] < qt[2] | weather_2017[, 12] > qt[3], 1, 0)
```


```R
weather_2017$평균온도 <- temp_average
weather_2017$최저온도 <- temp_min
weather_2017$최고온도 <- temp_max
weather_2017$최대풍속 <- wind_max
weather_2017$평균풍속 <- wind_average
weather_2017$평균습도 <- humidity_average
weather_2017$평균지면온도 <- road_average
```

```R
# NA 값은 비나 눈이 오지 않은 날이므로 0으로 처리
weather_2017[, 6] <- ifelse(is.na(weather_2017[, 6]), 0, weather_2017[, 6])
weather_2017[, 11] <- ifelse(is.na(weather_2017[, 11]), 0, weather_2017[, 11])
```

```R
# NA값 처리 후 quantile 적용
day_rain <- c()
qt <- quantile(weather_2017[, 6], probs=c(0, 0.1, 0.9, 1), na.rm=T)
day_rain <- ifelse(weather_2017[, names(weather_2017)[6]] > qt[3], 1, 0)

day_snow <- c()
qt <- quantile(weather_2017[, 11], probs=c(0, 0.1, 0.9, 1), na.rm=T)
day_snow <- ifelse(weather_2017[, 11] > qt[3], 1, 0)
```

```R
weather_2017$강수량 <- day_rain
weather_2017$적설량 <- day_snow
weather_2017$안개시간 <- ifelse( is.na(weather_2017[, 13]), 0, 1)
```

```R
weather_2017_arrange <- weather_2017[,c(2,14:ncol(weather_2017))]  #새로 만든 변수 확인
dim(weather_2017_arrange)
weather_2017_arrange %>% head()
```

![image](https://user-images.githubusercontent.com/97672187/161553342-5a3c232b-1097-4aa9-a4e5-819f3b7243b0.png){: .align-center}

<br>




</br>

2) 2018년도 마찬가지

```R
weather_2018 <- read.csv('weather_2018.csv')
```

```R
temp_average <- c() # 평균온도
qt <- quantile(weather_2018[, 3], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_average <- ifelse(weather_2018[, 3] < qt[2] | weather_2018[, 3] > qt[3], 1, 0)

temp_min <- c() #최저온도
qt <- quantile(weather_2018[, 4], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_min <- ifelse(weather_2018[, 4] < qt[2] | weather_2018[, 4] > qt[3], 1, 0)

temp_max <- c() #최고온도
qt <- quantile(weather_2018[, 5], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_max <- ifelse(weather_2018[, 5] < qt[2] | weather_2018[, 5] > qt[3], 1, 0)

wind_max <- c() #최대풍속
qt <- quantile(weather_2018[, 7], probs=c(0, 0.1, 0.9, 1), na.rm=T)
wind_max <- ifelse(weather_2018[, 7] < qt[2] | weather_2018[, 7] > qt[3], 1, 0)

wind_average <- c() #평균풍속
qt <- quantile(weather_2018[, 9], probs=c(0, 0.1, 0.9, 1), na.rm=T)
wind_average <- ifelse(weather_2018[, 9] < qt[2] | weather_2018[, 9] > qt[3], 1, 0)

humidity_average <- c() #평균습도
qt <- quantile(weather_2018[, 10], probs=c(0, 0.1, 0.9, 1), na.rm=T)
humidity_average <- ifelse(weather_2018[, 10] < qt[2] | weather_2018[, 10] > qt[3], 1, 0)

road_average <- c() #평균지면온도
qt <- quantile(weather_2018[, 12], probs=c(0, 0.1, 0.9, 1), na.rm=T)
road_average <- ifelse(weather_2018[, 12] < qt[2] | weather_2018[, 12] > qt[3], 1, 0)
```


```R
weather_2018$평균온도 <- temp_average
weather_2018$최저온도 <- temp_min
weather_2018$최고온도 <- temp_max
weather_2018$최대풍속 <- wind_max
weather_2018$평균풍속 <- wind_average
weather_2018$평균습도 <- humidity_average
weather_2018$평균지면온도 <- road_average
```


```R
#NA 값 0으로 처리
weather_2018[, 6] <- ifelse(is.na(weather_2018[, 6]), 0, weather_2018[, 6])
weather_2018[, 11] <- ifelse(is.na(weather_2018[, 11]), 0, weather_2018[, 11])
```

```R
day_rain <- c()
qt <- quantile(weather_2018[, 6], probs=c(0, 0.1, 0.9, 1), na.rm=T)
day_rain <- ifelse(weather_2018[, 6] > qt[3], 1, 0)

day_snow <- c()
qt <- quantile(weather_2018[, 11], probs=c(0, 0.1, 0.9, 1), na.rm=T)
day_snow <- ifelse(weather_2018[, 11] > qt[3], 1, 0)
```


```R
weather_2018$강수량 <- day_rain
weather_2018$적설량 <- day_snow
weather_2018$안개시간 <- ifelse( is.na(weather_2018[, 13]), 0, 1)
```

```R
weather_2018_arrange <- weather_2018[,c(2,14:ncol(weather_2018))]
dim(weather_2018_arrange)
weather_2018_arrange %>% head()
```

![image](https://user-images.githubusercontent.com/97672187/161554196-1f949317-ca44-4c87-a0ef-0cd43fb81484.png){: .align-center}

<br>



</br>

3) 2019년 weather 데이터처리

```R
weather_2019 <- read.csv('weather_2019.csv')
```

```R
temp_average <- c() # 평균온도
qt <- quantile(weather_2019[, 3], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_average <- ifelse(weather_2019[, 3] < qt[2] | weather_2019[, 3] > qt[3], 1, 0)

temp_min <- c() #최저온도
qt <- quantile(weather_2019[, 4], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_min <- ifelse(weather_2019[, 4] < qt[2] | weather_2019[, 4] > qt[3], 1, 0)

temp_max <- c() #최고온도
qt <- quantile(weather_2019[, 5], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_max <- ifelse(weather_2019[, 5] < qt[2] | weather_2019[, 5] > qt[3], 1, 0)

wind_max <- c() #최대풍속
qt <- quantile(weather_2019[, 7], probs=c(0, 0.1, 0.9, 1), na.rm=T)
wind_max <- ifelse(weather_2019[, 7] < qt[2] | weather_2019[, 7] > qt[3], 1, 0)

wind_average <- c() #평균풍속
qt <- quantile(weather_2019[, 9], probs=c(0, 0.1, 0.9, 1), na.rm=T)
wind_average <- ifelse(weather_2019[, 9] < qt[2] | weather_2019[, 9] > qt[3], 1, 0)

humidity_average <- c() #평균습도
qt <- quantile(weather_2019[, 10], probs=c(0, 0.1, 0.9, 1), na.rm=T)
humidity_average <- ifelse(weather_2019[, 10] < qt[2] | weather_2019[, 10] > qt[3], 1, 0)

road_average <- c() #평균지면온도
qt <- quantile(weather_2018[, 12], probs=c(0, 0.1, 0.9, 1), na.rm=T)
road_average <- ifelse(weather_2019[, 12] < qt[2] | weather_2019[, 12] > qt[3], 1, 0)
```


```R
weather_2019$평균온도 <- temp_average
weather_2019$최저온도 <- temp_min
weather_2019$최고온도 <- temp_max
weather_2019$최대풍속 <- wind_max
weather_2019$평균풍속 <- wind_average
weather_2019$평균습도 <- humidity_average
weather_2019$평균지면온도 <- road_average
```


```R
#NA 값 0으로 처리
weather_2019[, 6] <- ifelse(is.na(weather_2019[, 6]), 0, weather_2019[, 6])
weather_2019[, 11] <- ifelse(is.na(weather_2019[, 11]), 0, weather_2019[, 11])
```

```R
day_rain <- c()
qt <- quantile(weather_2019[, 6], probs=c(0, 0.1, 0.9, 1), na.rm=T)
day_rain <- ifelse(weather_2019[, 6] > qt[3], 1, 0)

day_snow <- c()
qt <- quantile(weather_2019[, 11], probs=c(0, 0.1, 0.9, 1), na.rm=T)
day_snow <- ifelse(weather_2019[, 11] > qt[3], 1, 0)
```


```R
weather_2019$강수량 <- day_rain
weather_2019$적설량 <- day_snow
weather_2019$안개시간 <- ifelse( is.na(weather_2019[, 13]), 0, 1)
```

```R
weather_2019 <- weather_2018[,c(2,14:ncol(weather_2019))]
dim(weather_2019_arrange)
weather_2019_arrange %>% head()
```

![image](https://user-images.githubusercontent.com/97672187/161554805-f20fbfe5-6162-43e2-9320-34ee133d2699.png){: .align-center}

<br>



</br>

4) 2017 ~ 2019 날씨 데이터 합치기

```R
weather_arrange <- bind_rows(weather_2017_arrange, weather_2018_arrange, weather_2019_arrange)
dim(weather_arrange)
weather_arrange %>% head()
```

![image](https://user-images.githubusercontent.com/97672187/161555004-90a8d816-29fe-40ba-8fea-aaa57cba5ed7.png){: .align-center}


``R
write.csv(weather_arrange, file='weather_arrange.csv', row.names=F)
```


