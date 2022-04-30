---
layout: single
title: "Modeling Part.2"
toc: true
toc_sticky: true
category: LH
---

비교적 간단한 모델에 속하는 OLS, 포아송 회귀보다 지리적 데이터에 더 적합한 지리적 가중 회귀(GWR) 모델을 사용해보자.

```R
library(sp)
library(dplyr)
library(stringr)
library(rgeos)
library(tmap)
library(raster)
library(spdep)
library(gstat)
library(spgwr)
library(GWmodel)
library(regclass)
library(ggplot2)
library(ggcorrplot)
library(lmtest)
library(leaflet)
library(extrafont)
```

```R
#파일 불러오기
car_under20 <- read.csv("car_under20.csv", row.names = 1)
car_20 <- read.csv("car_20.csv", row.names = 1)
car_30 <- read.csv("car_30.csv", row.names = 1)
car_40 <- read.csv("car_40.csv", row.names = 1)
car_50 <- read.csv("car_50.csv", row.names = 1)
car_over60 <- read.csv("car_over60.csv", row.names = 1)

person_under20 <- read.csv("person_under20.csv", row.names = 1)
person_20 <- read.csv("person_20.csv", row.names = 1)
person_30 <- read.csv("person_30.csv", row.names = 1)
person_40 <- read.csv("person_40.csv", row.names = 1)
person_50 <- read.csv("person_50.csv", row.names = 1)
person_over60 <- read.csv("person_over60.csv", row.names = 1)
```


## Modeling Part.2(by using R)

### 3. GWR(Geographically Weighted Regression, 지리적 가중회귀) 모델링
#### 1) 차 대 사람 사고에 대힌 지리적 가중회귀 모델링

차대사람_60대 group에 대한 지리적 가중 회귀 모델 결과(종속변수가 정규 분포를 따르는 모델)

```R
mod_data2 <- person_over60 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대사람..60대.이상) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod_data3 <- sp::SpatialPointsDataFrame(data=mod_data2, coords = cbind(person_over60$x, person_over60$y))
dst <- GWmodel::gw.dist(dp.locat = cbind(person_over60$x, person_over60$y), longlat = TRUE)
bw2 <- GWmodel::bw.gwr(formula =`차대사람..60대.이상`~.-coords.x1-coords.x2, data=mod_data3, kernel="gaussian", dMat = dst, longlat = TRUE)
model2 <- GWmodel::gwr.basic(차대사람..60대.이상~.-coords.x1-coords.x2, dMat = dst, longlat = TRUE, bw = bw2, kernel="gaussian", data = mod_data3)

fit_val_60_person <- model2$SDF$yhat
coef_60_person <- dplyr::select(as.data.frame(model2$SDF), Intercept:평균혼잡강도)
rownames(coef_60_person) <- person_over60$index
coef_60_person$residuals <- model2$SDF$residual
coef_60_person$fitted_values <- fit_val_60_person
coef_60_person$real_values <- person_over60$`차대사람..60대.이상`
ref <- apply(coef_60_person, 2, function(x){abs(max(x))<0.01 })
bw2 <- GWmodel::bw.gwr(formula =`차대사람..60대.이상`~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도, data=mod_data3, kernel="gaussian", dMat = dst, longlat = TRUE)
model2 <- GWmodel::gwr.basic(차대사람..60대.이상~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도, dMat = dst, longlat = TRUE, bw = bw2, kernel="gaussian", data = mod_data3)
model2
```

![image](https://user-images.githubusercontent.com/97672187/166100376-806423b4-dca1-4c24-b005-f7e9694027a8.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/166100395-91d8dc62-e586-4126-bb99-4de553e59970.png){: .align-center}


<br>


<br>

차대사람_50대 group에 대한 지리적 가중 회귀 모델 결과(종속변수가 정규 분포를 따르는 모델)

```R
mod_data2<-person_50%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대사람..50대)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)
mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(person_50$x,person_50$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(person_50$x,person_50$y),longlat = TRUE)

bw2<-GWmodel::bw.gwr(formula =`차대사람..50대`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대사람..50대~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)

fit_val_50_person<-model2$SDF$yhat
coef_50_person<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_50_person)<-person_50$index
coef_50_person$residuals<-model2$SDF$residual
coef_50_person$fitted_values<-fit_val_50_person
coef_50_person$real_values<-person_50$`차대사람..50대`
ref<-apply(coef_50_person,2,function(x){abs(max(x))<0.01 })
bw2<-GWmodel::bw.gwr(formula =`차대사람..50대`~.-coords.x1-coords.x2-이상안개시간동반사고건수-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대사람..50대~.-coords.x1-coords.x2-이상안개시간동반사고건수-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)
model2
```

![image](https://user-images.githubusercontent.com/97672187/166100422-0e5d5ee9-abbd-4d56-80a9-88ed621ba610.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/166100434-6420cdee-890d-4910-90d4-0fd67975caac.png){: .align-center}

<br>


<br>

차대사람_40대 group에 대한 지리적 가중 회귀 모델 결과(종속변수가 정규분포 따르는 모델)

```R
mod_data2<-person_40%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대사람..40대)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)

mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(person_40$x,person_40$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(person_40$x,person_40$y),longlat = TRUE)

bw2<-GWmodel::bw.gwr(formula =`차대사람..40대`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대사람..40대~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)

fit_val_40_person<-model2$SDF$yhat
coef_40_person<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_40_person)<-person_40$index
coef_40_person$residuals<-model2$SDF$residual
coef_40_person$fitted_values<-fit_val_40_person
coef_40_person$real_values<-person_40$`차대사람..40대`
ref<-apply(coef_40_person,2,function(x){abs(max(x))<0.01 })
bw2<-GWmodel::bw.gwr(formula =`차대사람..40대`~.-coords.x1-coords.x2-이상평균풍속동반사고건수-이상평균습도동반사고건수-이상안개시간동반사고건수-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대사람..40대~.-coords.x1-coords.x2-이상평균풍속동반사고건수-이상평균습도동반사고건수-이상안개시간동반사고건수-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)
model2
```

![image](https://user-images.githubusercontent.com/97672187/166100453-d1b66900-bb12-4949-a9d4-913b83a5b235.png){: .align-center}


<br>


<br>

차대사람_30대 group에 대한 지리적 가중 회귀 모델 결과(종속변수가 정규분포 따르는 모델)

```R
mod_data2<-person_30%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대사람..30대)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)

mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(person_30$x,person_30$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(person_30$x,person_30$y),longlat = TRUE)

bw2<-GWmodel::bw.gwr(formula =`차대사람..30대`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대사람..30대~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)

fit_val_30_person<-model2$SDF$yhat
coef_30_person<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_30_person)<-person_30$index
coef_30_person$residuals<-model2$SDF$residual
coef_30_person$fitted_values<-fit_val_30_person
coef_30_person$real_values<-person_30$`차대사람..30대`
ref<-apply(coef_30_person,2,function(x){abs(max(x))<0.01 })
bw2<-GWmodel::bw.gwr(formula =`차대사람..30대`~.-coords.x1-coords.x2-이상평균풍속동반사고건수-이상평균습도동반사고건수-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대사람..30대~.-coords.x1-coords.x2-이상평균풍속동반사고건수-이상평균습도동반사고건수-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)
model2
```

![image](https://user-images.githubusercontent.com/97672187/166100480-686ee4ab-0261-40c3-b1d3-2da5a308b341.png){: .align-center}

<br>


<br>

차대사람_20대 group에 대한 지리적 가중 회귀 모델 결과(종속변수가 정규분포 따르는 모델)

```R
mod_data2<-person_20%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대사람..20대)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)

mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(person_20$x,person_20$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(person_20$x,person_20$y),longlat = TRUE)

bw2<-GWmodel::bw.gwr(formula =`차대사람..20대`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대사람..20대~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)

fit_val_20_person<-model2$SDF$yhat
coef_20_person<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_20_person)<-person_20$index
coef_20_person$residuals<-model2$SDF$residual
coef_20_person$fitted_values<-fit_val_20_person
coef_20_person$real_values<-person_20$`차대사람..20대`
ref<-apply(coef_20_person,2,function(x){abs(max(x))<0.01 })
bw2<-GWmodel::bw.gwr(formula =`차대사람..20대`~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대사람..20대~.-coords.x1-coords.x2--건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)
model2
```

![image](https://user-images.githubusercontent.com/97672187/166100505-5b6bf904-1e07-473e-89df-cf54f7cafc58.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/166100513-4364f6d1-e1db-4678-b088-668ad5bf26c3.png){: .align-center}

<br>


<br>

차대사람_20대미만 group에 대한 지리적 가중 회귀 모델 결과(종속변수가 정규분포 따르는 모델)

```R
mod_data2<-person_under20%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대사람..20대.미만)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)
mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(person_under20$x,person_under20$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(person_under20$x,person_under20$y),longlat = TRUE)

bw2<-GWmodel::bw.gwr(formula =`차대사람..20대.미만`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대사람..20대.미만~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)

fit_val_under20_person<-model2$SDF$yhat
coef_under20_person<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_under20_person)<-person_under20$index
coef_under20_person$residuals<-model2$SDF$residual
coef_under20_person$fitted_values<-fit_val_under20_person
coef_under20_person$real_values<-person_under20$`차대사람..20대.미만`
ref<-apply(coef_under20_person,2,function(x){abs(max(x))<0.01 })
bw2<-GWmodel::bw.gwr(formula =`차대사람..20대.미만`~.-coords.x1-coords.x2-이상최대풍속동반사고건수-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대사람..20대.미만~.-coords.x1-coords.x2-이상최대풍속동반사고건수-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)
model2
```

![image](https://user-images.githubusercontent.com/97672187/166100550-012fd52e-4311-487b-94b8-8bb1b985edfa.png){: .align-center}


<br>


<br>

#### 2) 차 대 차 사고에 대힌 지리적 가중회귀 모델링

차대차_60대 group에 대한 지리적 가중 회귀 모델 결과(종속변수가 정규분포 따르는 모델)

```R
mod_data2<-car_over60%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대차..60대.이상)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)

mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(car_over60$x,car_over60$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(car_over60$x,car_over60$y),longlat = TRUE)

bw2<-GWmodel::bw.gwr(formula =`차대차..60대.이상`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대차..60대.이상~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)


fit_val_60_car<-model2$SDF$yhat
coef_60_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_60_car)<-car_over60$index
coef_60_car$residuals<-model2$SDF$residual
coef_60_car$fitted_values<-fit_val_60_car
coef_60_car$real_values<-car_over60$`차대차..60대.이상`
ref<-apply(coef_60_person,2,function(x){abs(max(x))<0.01 })
bw2<-GWmodel::bw.gwr(formula =`차대차..60대.이상`~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대차..60대.이상~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)
model2
```

![image](https://user-images.githubusercontent.com/97672187/166100577-066967ab-795b-4b87-baeb-42c4b4ba23e3.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/166100588-aac1c11e-1f0b-4e05-aecf-3ba8053e5d6f.png){: .align-center}

<br>


<br>

차대차_50대 group에 대한 지리적 가중 회귀 모델 결과(종속변수가 정규분포 따르는 모델)

```R
mod_data2<-car_50%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대차..50대)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)
mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(car_50$x,car_50$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(car_50$x,car_50$y),longlat = TRUE)

bw2<-GWmodel::bw.gwr(formula =`차대차..50대`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대차..50대~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)

fit_val_50_car<-model2$SDF$yhat
coef_50_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_50_car)<-car_50$index
coef_50_car$residuals<-model2$SDF$residual
coef_50_car$fitted_values<-fit_val_50_car
coef_50_car$real_values<-car_50$`차대차..50대`
ref<-apply(coef_50_car,2,function(x){abs(max(x))<0.01 })
bw2<-GWmodel::bw.gwr(formula =`차대차..50대`~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대차..50대~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)
model2
```

![image](https://user-images.githubusercontent.com/97672187/166100603-52979a9d-dcef-4eb3-b05b-031c6c7ed671.png){: .align-center}


<br>


<br>

차대차_40대 group에 대한 지리적 가중 회귀 모델 결과(종속변수가 정규분포 따르는 모델)

```R
mod_data2<-car_40%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대차..40대)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)
mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(car_40$x,car_40$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(car_40$x,car_40$y),longlat = TRUE)

bw2<-GWmodel::bw.gwr(formula =`차대차..40대`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대차..40대~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)

fit_val_40_car<-model2$SDF$yhat
coef_40_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_40_car)<-car_40$index
coef_40_car$residuals<-model2$SDF$residual
coef_40_car$fitted_values<-fit_val_40_car
coef_40_car$real_values<-car_40$`차대차..40대`
ref<-apply(coef_40_car,2,function(x){abs(max(x))<0.01 })
bw2<-GWmodel::bw.gwr(formula =`차대차..40대`~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대차..40대~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)
model2
```

![image](https://user-images.githubusercontent.com/97672187/166100616-27cbe3b4-f4f0-429d-a80a-3707422153fc.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/166100619-3ab18dfa-e516-4f05-913f-cc346aff6764.png){: .align-center}

<br>


<br>

차대차_30대 group에 대한 지리적 가중 회귀 모델 결과(종속변수가 정규분포 따르는 모델)

```R
mod_data2<-car_30%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대차..30대)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)

mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(car_30$x,car_30$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(car_30$x,car_30$y),longlat = TRUE)

bw2<-GWmodel::bw.gwr(formula =`차대차..30대`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대차..30대~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)

fit_val_30_car<-model2$SDF$yhat
coef_30_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_30_car)<-car_30$index
coef_30_car$residuals<-model2$SDF$residual
coef_30_car$fitted_values<-fit_val_30_car
coef_30_car$real_values<-car_30$`차대차..30대`
ref<-apply(coef_30_car,2,function(x){abs(max(x))<0.01 })
bw2<-GWmodel::bw.gwr(formula =`차대차..30대`~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대차..30대~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)
model2
```

![image](https://user-images.githubusercontent.com/97672187/166100629-c46c3462-bb7e-4aff-9117-5e965a648f93.png){: .align-center}


<br>


<br>

차대차_20대 group에 대한 지리적 가중 회귀 모델 결과(종속변수가 정규분포 따르는 모델)

```R
mod_data2<-car_20%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대차..20대)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)
mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(car_20$x,car_20$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(car_20$x,car_20$y),longlat = TRUE)

bw2<-GWmodel::bw.gwr(formula =`차대차..20대`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대차..20대~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)

fit_val_20_car<-model2$SDF$yhat
coef_20_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_20_car)<-car_20$index
coef_20_car$residuals<-model2$SDF$residual
coef_20_car$fitted_values<-fit_val_20_car
coef_20_car$real_values<-car_20$`차대차..20대`
ref<-apply(coef_20_car,2,function(x){abs(max(x))<0.01 })
bw2<-GWmodel::bw.gwr(formula =`차대차..20대`~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대차..20대~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)
model2
```

![image](https://user-images.githubusercontent.com/97672187/166100650-366592ac-de79-4543-9eff-58b89c6d9817.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/166100663-2c487332-f2c9-4dd7-bec2-bd5773943996.png){: .align-center}

<br>


<br>

차대차_20대미만 group에 대한 지리적 가중 회귀 모델 결과(종속변수가 정규분포 따르는 모델)

```R
mod_data2<-car_under20%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대차..20대.미만)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)
mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(car_under20$x,car_under20$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(car_under20$x,car_under20$y),longlat = TRUE)

bw2<-GWmodel::bw.gwr(formula =`차대차..20대.미만`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대차..20대.미만~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)

fit_val_under20_car<-model2$SDF$yhat
coef_under20_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_under20_car)<-car_under20$index
coef_under20_car$residuals<-model2$SDF$residual
coef_under20_car$fitted_values<-fit_val_under20_car
coef_under20_car$real_values<-car_under20$`차대차..20대.미만`
ref<-apply(coef_under20_car,2,function(x){abs(max(x))<0.01 })
bw2<-GWmodel::bw.gwr(formula =`차대차..20대.미만`~.-coords.x1-coords.x2-이상평균기온동반사고건수-노드개수-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",dMat = dst,longlat = TRUE)
model2<-GWmodel::gwr.basic(차대차..20대.미만~.-coords.x1-coords.x2-이상평균기온동반사고건수-노드개수-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",data = mod_data3)
model2
```

![image](https://user-images.githubusercontent.com/97672187/166100678-65752c47-aaba-416e-8797-3d938b126c9c.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/166100695-0629634c-63b5-4bb3-9d6b-507b46fcebb7.png){: .align-center}


<br>


<br>

### OLS, 포아송, 지리적 가중회귀 모델의 성능비교

1) 지리적 가중회귀 모델(GWR)의 AIC

![image](https://user-images.githubusercontent.com/97672187/166101356-10d5746d-7e08-49f9-aa6f-ca68d8d7ff4c.png)


2) OLS, 포아송, 지리적 가중회귀 모델 성능비교 그래프

![image](https://user-images.githubusercontent.com/97672187/166101375-67e22db9-8d9d-49e3-87df-22450efc2d1a.png)

지리적 가중 회귀 모형이 전반적으로 OLS보다는 AIC 값이 낮아서 성능이 좋았지만 포아송과 비교하면 여전히 AIC가 높다.

3) 한계점

기존의 지리적 가중 회귀모형은 공간적 이질성을 설명할 수는 있지만, 종속변수가 정규분포를 따르는 연속형 자료일때만 가능하다. 
따라서 이산형 자료이면서 정규분포를 따른다는 보장이 없는 (연령대 및 사고유형별) 사고 건수를 모델링 하는데는 일반 지리적 가중 회귀 보다  
**지리적 가중 포아송 회귀 모형**이 더 적합할 것 같다. 다음 포스팅에서는 **지리적 가중 포아송 회귀 모형** 을 모델링한 결과를 정리해보자.




