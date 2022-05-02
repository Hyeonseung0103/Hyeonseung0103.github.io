---
layout: single
title: "Modeling의 결과인 잔차분석을 통한 최종 위험지역 100개소 도출"
toc: true
toc_sticky: true
category: LH
---

지리적 가중 포아송 회귀를 최종 모델로 선택했고, 2개의 사고유형, 6개의 연령대별을 합쳐서 총 12개의 그룹에 대해 각각 다른 모델을 만들었다. 또한, 이 12개의 모델의 회귀계수를
'coef_사고유형_연령대' 형식의 csv로 저장했다. 이 회귀계수 데이터에는 회귀계수 뿐만 아니라 각 row별로 잔차가 포함되어있는데 이 잔차를 활용하여 추가적인 분석을 진행해보자.

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
acci_count_filter25 <- read.csv('accident_count_filter_25.csv')
coef_60_car <- read.csv("coef_car_over60.csv")
coef_50_car <- read.csv("coef_car_50.csv")
coef_40_car <- read.csv("coef_car_40.csv")
coef_30_car <- read.csv("coef_car_30.csv")
coef_20_car <- read.csv("coef_car_20.csv")
coef_under20_car <- read.csv("coef_car_under20.csv")

coef_60_person <- read.csv("coef_person_over60.csv")
coef_50_person <- read.csv("coef_person_50.csv")
coef_40_person <- read.csv("coef_person_40.csv")
coef_30_person <- read.csv("coef_person_30.csv")
coef_20_person <- read.csv("coef_person_20.csv")
coef_under20_person <- read.csv("coef_person_under20.csv")
```

## Modeling Part.4(by using R)
### 잔차분석
모델이 위험한 지역을 위험하지 않다고 한 것과 위험하지 않은 지역을 위험하다고 한 것중 무엇이 더 중요할까?
만약 위험한 지역을 위험하지 않는다고 예측해서 교통 사고 예방에 대한 조치가 취해지지 않는다면 사람의 생명과 직접적인 연관이 있다.
반면, 위험하지 않은 지역을 위험하다고 예측해서 조치를 취한다면 경제적인 비용이 발생하긴 하지만, 그 지역은 더 위험하지 않은 지역이 될 수 있다.

결국 이와 같은 상황에서는 위험한 지역을 위험하지 않는다고 한 경우가 위험하지 않는 지역을 위험하다고 예측한 경우보다 더 중요하다. 이를 잔차에 비유하자면, 
실제로 위험한 지역을 위험하지 않는다고 한 경우는 양의 잔차(예측한 사고건수보다 실제로 그 지역에서 발생한 사고건수가 큰 것)에 해당한다.
이 양의 잔차의 크기가 크다면 모델이 예측한 것보다 더 위험한 지역 즉, 대책이 필요한 지역일 가능성이 높다. 따라서 이 잔차분석을통해 양의 잔차가 큰 지역의 특성과 사고들의 유형 분석을 통해
적합한 교통 안전시설물 설치와 같은 대책을 제안해볼 수 있다.

1) GWPR로 구한 회귀계수 데이터에 고유 인덱스 부여 및 데이터 병합

도출된 회귀계수 데이터에 고유 인덱스 부여

```R
coef_under20_car$index<-car_under20$X
coef_20_car$index<-car_20$X
coef_30_car$index<-car_30$X
coef_40_car$index<-car_40$X
coef_50_car$index<-car_50$X
coef_60_car$index<-car_over60$X
coef_under20_person$index<-person_under20$X
coef_20_person$index<-person_20$X
coef_30_person$index<-person_30$X
coef_40_person$index<-person_40$X
coef_50_person$index<-person_50$X
coef_60_person$index<-person_over60$X
```

<br>


<br>

각 그룹의 잔차 컬럼명 구분 작업(residual로 되어있는 변수이름에 그룹명을 추가) 

```R
colnames(coef_under20_car)[which(colnames(coef_under20_car)=="residuals")]<-"residuals_under20_car"
colnames(coef_20_car)[which(colnames(coef_20_car)=="residuals")]<-"residuals_20_car"
colnames(coef_30_car)[which(colnames(coef_30_car)=="residuals")]<-"residuals_30_car"
colnames(coef_40_car)[which(colnames(coef_40_car)=="residuals")]<-"residuals_40_car"
colnames(coef_50_car)[which(colnames(coef_50_car)=="residuals")]<-"residuals_50_car"
colnames(coef_60_car)[which(colnames(coef_60_car)=="residuals")]<-"residuals_60_car"
colnames(coef_under20_person)[which(colnames(coef_under20_person)=="residuals")]<-"residuals_under20_person"
colnames(coef_20_person)[which(colnames(coef_20_person)=="residuals")]<-"residuals_20_person"
colnames(coef_30_person)[which(colnames(coef_30_person)=="residuals")]<-"residuals_30_person"
colnames(coef_40_person)[which(colnames(coef_40_person)=="residuals")]<-"residuals_40_person"
colnames(coef_50_person)[which(colnames(coef_50_person)=="residuals")]<-"residuals_50_person"
colnames(coef_60_person)[which(colnames(coef_60_person)=="residuals")]<-"residuals_60_person"
```

<br>


<br>

회귀계수 데이터 병합

```R
residual_join_dta<-coef_under20_car%>%
  dplyr::full_join(coef_20_car,"index")%>%
  dplyr::select(index,residuals_under20_car,residuals_20_car)%>%
  dplyr::full_join(coef_30_car,"index")%>%
  dplyr::select(index,residuals_under20_car,residuals_20_car,residuals_30_car)%>%
  dplyr::full_join(coef_40_car,"index")%>%
  dplyr::select(index,residuals_under20_car,residuals_20_car,residuals_30_car,residuals_40_car)%>%
  dplyr::full_join(coef_50_car,"index")%>%
  dplyr::select(index,residuals_under20_car,residuals_20_car,residuals_30_car,residuals_40_car,residuals_50_car)%>%
  dplyr::full_join(coef_60_car,"index")%>%
  dplyr::select(index,residuals_under20_car,residuals_20_car,residuals_30_car,residuals_40_car,residuals_50_car,residuals_60_car)%>%
  dplyr::full_join(coef_under20_person,"index")%>%
  dplyr::select(index,residuals_under20_car,residuals_20_car,residuals_30_car,residuals_40_car,residuals_50_car,residuals_60_car,residuals_under20_person)%>%
  dplyr::full_join(coef_20_person,"index")%>%
  dplyr::select(index,residuals_under20_car,residuals_20_car,residuals_30_car,residuals_40_car,residuals_50_car,residuals_60_car,residuals_under20_person,residuals_20_person)%>%
  dplyr::full_join(coef_30_person,"index")%>%
  dplyr::select(index,residuals_under20_car,residuals_20_car,residuals_30_car,residuals_40_car,residuals_50_car,residuals_60_car,residuals_under20_person,residuals_20_person,residuals_30_person)%>%
  dplyr::full_join(coef_40_person,"index")%>%
  dplyr::select(index,residuals_under20_car,residuals_20_car,residuals_30_car,residuals_40_car,residuals_50_car,residuals_60_car,residuals_under20_person,residuals_20_person,residuals_30_person,residuals_40_person)%>%
  dplyr::full_join(coef_50_person,"index")%>%
  dplyr::select(index,residuals_under20_car,residuals_20_car,residuals_30_car,residuals_40_car,residuals_50_car,residuals_60_car,residuals_under20_person,residuals_20_person,residuals_30_person,residuals_40_person,residuals_50_person)%>%
  dplyr::full_join(coef_60_person,"index")%>%
  dplyr::select(index,residuals_under20_car,residuals_20_car,residuals_30_car,residuals_40_car,residuals_50_car,residuals_60_car,residuals_under20_person,residuals_20_person,residuals_30_person,residuals_40_person,residuals_50_person,residuals_60_person)
```

<br>


<br>

고유 index를 기준으로 full_join 해서 합친 데이터의 NA 부분을 0으로 바꿔준다.

```R

residual_join_dta$residuals_under20_car<-ifelse(is.na(residual_join_dta$residuals_under20_car),0,residual_join_dta$residuals_under20_car)
residual_join_dta$residuals_20_car<-ifelse(is.na(residual_join_dta$residuals_20_car),0,residual_join_dta$residuals_20_car)
residual_join_dta$residuals_30_car<-ifelse(is.na(residual_join_dta$residuals_30_car),0,residual_join_dta$residuals_30_car)
residual_join_dta$residuals_40_car<-ifelse(is.na(residual_join_dta$residuals_40_car),0,residual_join_dta$residuals_40_car)
residual_join_dta$residuals_50_car<-ifelse(is.na(residual_join_dta$residuals_50_car),0,residual_join_dta$residuals_50_car)
residual_join_dta$residuals_60_car<-ifelse(is.na(residual_join_dta$residuals_60_car),0,residual_join_dta$residuals_60_car)
residual_join_dta$residuals_under20_person<-ifelse(is.na(residual_join_dta$residuals_under20_person),0,residual_join_dta$residuals_under20_person)
residual_join_dta$residuals_20_person<-ifelse(is.na(residual_join_dta$residuals_20_person),0,residual_join_dta$residuals_20_person)
residual_join_dta$residuals_30_person<-ifelse(is.na(residual_join_dta$residuals_30_person),0,residual_join_dta$residuals_30_person)
residual_join_dta$residuals_40_person<-ifelse(is.na(residual_join_dta$residuals_40_person),0,residual_join_dta$residuals_40_person)
residual_join_dta$residuals_50_person<-ifelse(is.na(residual_join_dta$residuals_50_person),0,residual_join_dta$residuals_50_person)
residual_join_dta$residuals_60_person<-ifelse(is.na(residual_join_dta$residuals_60_person),0,residual_join_dta$residuals_60_person)
```

<br>


<br>

#### DGI(Daejeon Gid Index = 대전격자지수)

![image](https://user-images.githubusercontent.com/97672187/166232194-6df49829-38c5-4caa-8874-4dc5ef8cf3b5.png){: .align-center}

위와 같은 수식처럼 각 그룹별로 도출된 가중치 데이터와 잔차를 활용하여 대전광역시 격자별 위험지수를 계산한다.

DGI지수 를 구하는 함수

```R
# DGI(Daejeon Gid Index-대전격자지수)
DGI<-function(weightsvec=NULL,residualsmat=NULL,DIratiovec,threshold_quantiles=NULL,gid=NULL,gid_include=FALSE){
  #residualsmat=>(r1vec,r2vec,r3vec,...,rnvec)
  r_mat<-apply(residualsmat,2,function(x){ifelse(x<=0,0,x)})
  DGIj<-r_mat%*%weightsvec
  if(gid_include){
    DGIj<-as.data.frame(cbind(DGIj,DIratiovec,1:nrow(residualsmat)))
    DGIj$gid<-gid
    colnames(DGIj)<-c("DGI","DI_ratio","index","gid")
  }
  else{
    DGIj<-as.data.frame(cbind(DGIj,DIratiovec,1:nrow(residualsmat)))
    colnames(DGIj)<-c("DGI","DI_ratio","index")
  }
  DGIj_sorted<-DGIj[order(DGIj$DGI,decreasing = T),]
  
  DGIj_sorted$difference<-c(0,diff(DGIj_sorted$DGI))
  DGIj_sorted$false_rank<-1:nrow(residualsmat)
  threshold<-quantile(-diff(DGIj_sorted$DGI),threshold_quantiles)
  s_thres<-0
  switching<-0
  for(i in 1:(nrow(residualsmat)-1) ){
    if((DGIj_sorted[i,"DGI"]-DGIj_sorted[i+1,"DGI"])< threshold){
      if(DGIj_sorted$DI_ratio[i]<DGIj_sorted$DI_ratio[i+1]){
        cache<-DGIj_sorted[i,]
        DGIj_sorted[i,]<-DGIj_sorted[i+1,]
        DGIj_sorted[i+1,]<-cache
        switching<-switching+1
      }
      s_thres<-s_thres+1
    }
    
  }
  DGIj_sorted$true_rank<-1:nrow(residualsmat)  
  return(list(DGI=DGIj_sorted,iterations_of_switching=switching,s_thres=s_thres,weights=weightsvec,residuals_matrix=residualsmat))
}

```

<br>

<br>

격자 데이터(acci_count_filter25)의 index를 기준으로 각 그룹별로 잔차를 병합한 잔차 데이터(residual_join_dta)와 병합한 후 DGI 지수를 계산하기 위해 필요한
변수들만 추출한다.

```R
finalie_last_2<-acci_count_filter25%>%
  mutate(중상자이상수=(사망자수+중상자수) )%>%
  right_join(residual_join_dta,by=c("X"="index"))%>%
  dplyr::select(gid,x,y,중상자이상수,residuals_under20_car:residuals_60_person)

finalie_last_2$index<-as.numeric(rownames(finalie_last_2))
finalie_last_real<-finalie_last_2%>%
  left_join(residual_join_dta,by="index")%>%
  dplyr::select(gid,x,y,중상자이상수,residuals_under20_car.x,residuals_20_car.x,residuals_30_car.x,residuals_40_car.x,residuals_50_car.x,residuals_60_car.x,residuals_under20_person.x,residuals_20_person.x,residuals_30_person.x,residuals_40_person.x,residuals_50_person.x,residuals_60_person.x)
```

<br>

<br>


#### DGI 지수 구하기 and 최종위험지역 100개 도출

```R
#weightsvec->전처리 파트에서 구한 그룹별 가중치 벡터
#residualsmat->잔차 행렬(행:관측치,열:사고유형-연령대 그룹,값:잔차)
#DIratiovec->각 격자가 가지는 사상자 수에 대한 사망자 수 + 중상자 수의 비율
#threshold_quantiles->내림차순으로 정렬한 격자별 DGI 지수의 차이(절댓값)의 하위 n % 백분위수를 threshold로 설정하는데, 여기서 필요한 n 설정   
#함수 리턴 결과가 gid를 포함한 결과이고 싶을 때 gid_include=TRUE 하고 gid 값을 설정해 주면 된다.

dgi<-DGI(weightsvec=c(0.07748547,0.07605981,0.07829668,0.08305307,0.08640373,0.09488054,0.07294598,0.06206129,0.07528174,0.08830434,0.09012788,0.11509948),residualsmat=as.matrix(dplyr::select(finalie_last_2,residuals_under20_car:residuals_60_person),nrow=5138,ncol=12),DIratiovec=finalie_last_real$`중상자이상수`,threshold_quantiles=0.996,gid=finalie_last_2$gid,gid_include = TRUE)
top_100<-dgi$DGI[which(dgi$DGI$true_rank<=100),] # DGI 지수가 가장 높은 즉, 위험지역 100개지역 도출
write.csv(top_100_visu, "top_100_visu.csv", row.names = FALSE)

top_100_visu<-top_100%>% 
  left_join(finalie_last_2,by = c("gid","index"))%>%
  left_join(acci_count_filter25,by=c("gid","x","y"))%>%
  dplyr::select(DGI:residuals_60_person,acci_cnt:중상자수)

plot(finalie_last_real$x,finalie_last_real$y,col="green",pch=19)
lines(top_100_visu$x,top_100_visu$y,type="p",pch=19,col="red")
```

![image](https://user-images.githubusercontent.com/97672187/166233342-47e2eaed-ad16-4f52-b0bc-9cd4d7e5d0dd.png){: .align-center}


<br>


<br>

#### 최종 위험지역 100개 데이터 전처리
위도, 경도 좌표를 활용하여 정확한 주소를 입력하고 반경 범위를 표시한다.

```R
# gid 격자의 시군구 정보 추가
top_100_visu <- read.csv('top_100_visu.csv')
accident_list <- read.csv('1.대전광역시_교통사고내역(2017~2019).csv')
gid_df <- accident_list[, c('시군구', 'gid')]

final_df <- top_100_visu %>% semi_join(accident_list, by='gid') # 기존 데이터와 병합
dim(final_df)
final_df %>% head(3)
```

![image](https://user-images.githubusercontent.com/97672187/166233804-4cfa9774-0240-41a2-8710-a73d0b0eaec2.png){: .align-center}

<br>


<br>


```R
head(final_df[, c('y', 'x')])
```

![image](https://user-images.githubusercontent.com/97672187/166233848-82f4753d-526a-475b-b2f2-b19c51b6cd3c.png){: .align-center}

<br>


<br>

각 격자의 위도, 경도 좌표를 바탕으로 세부적인 주소를 네이버 지도를 통해서 획득

```R
gid_location = c('대전광역시 서구 갈마동 706', '대전광역시 서구 둔산동 1077', '대전광역시 동구 삼성동 458', 
                 '대전광역시 서구 갈마동 307-1', '대전광역시 유성구 봉명동 1063', '대전광역시 서구 관저동 616-10', 
                 '대전광역시 서구 탄방동 544', '대전광역시 유성구 원내동 360-27', '대전광역시 유성구 화암동 216-5', 
                 '대전광역시 서구 둔산동 959-1', '대전광역시 동구 대동 400-1', '대전광역시 대덕구 중리동 450', 
                 '대전광역시 서구 둔산동 911', '대전광역시 서구 둔산동 1204', '대전광역시 동구 판암동 467-13', 
                 '대전광역시 중구 태평동 520-1', '대전광역시 서구 갈마동 1459-1', '대전광역시 유성구 반석동 685', 
                 '대전광역시 동구 가양동 424-6', '대전광역시 유성구 도룡동 8-59', '대전광역시 중구 선화동 383', 
                 '대전광역시 중구 은행동 154', '대전광역시 중구 오류동 165-15', '대전광역시 유성구 봉산동 1002', 
                 '대전광역시 유성구 봉명동 1058', '대전광역시 유성구 장대동 117-13', '대전광역시 중구 대흥동 233-2', 
                 '대전광역시 서구 둔산동 2166', '대전광역시 유성구 구암동 527-61', '대전광역시 중구 대흥동 201', 
                 '대전광역시 서구 둔산동 1162', '대전광역시 동구 신흥동 6', '대전광역시 서구 갈마동 705', 
                 '대전광역시 서구 월평동 282-1', '대전광역시 중구 오류동 196-1', '대전광역시 유성구 지족동 1005', 
                 '대전광역시 유성구 구암동 641', '대전광역시 서구 변동 12-10', '대전광역시 동구 홍도동 838', 
                 '대전광역시 대덕구 중리동 469', '대전광역시 유성구 원내동 711', '대전광역시 서구 월평동 1636', 
                 '대전광역시 유성고 도룡동 465-27', '대전광역시 서구 갈마동 986', '대전광역시 유성구 봉명동 539-1', 
                 '대전광역시 대덕구 읍내동 505-23', '대전광역시 서구 월평동 184-2', '대전광역시 서구 갈마동 1459', 
                 '대전광역시 동구 중동 75-8', '대전광역시 중구 대흥동 508-88', '대전광역시 중구 문화동 1-29', 
                 '대전광역시 유성구 노은동 132', '대전광역시 서구 둔산동 948', '대전광역시 서구 갈마동 685', 
                 '대전광역시 서구 만년동 682', '대전광역시 동구 정동 45-1', '대전광역시 서구 도안동 1493-1', 
                 '대전광역시 서구 가장동 202', '대전광역시 서구 탄방동 544', '대전광역시 대덕구 오정동 43', 
                 '대전광역시 동구 용전동 220', '대전광역시 서구 변동 292', '대전광역시 유성구 노은동 61-13', 
                 '대전광역시 서구 둔산동 1546', '대전광역시 서구 월평동 1503-1', '대전광역시 서구 복수동 611', 
                 '대전광역시 유성구 봉명동 574', '대전광역시 유성구 송강동 171-1', '대전광역시 서구 둔산동 1100', 
                 '대전광역시 서구 둔산동 950', '대전광역시 서구 월평동 1528', '대전광역시 동구 용전동 227', 
                 '대전광역시 대덕구 중리동 150-3', '대전광역시 유성구 구암동 593-2', '대전광역시 유성구 봉명동 1058', 
                 '대전광역시 대덕구 오정동 64-18', '대전광역시 동구 가양동 630', '대전광역시 중구 선화동 862-3', 
                 '대전광역시 대덕구 중리동 424', '대전광역시 중구 용두동 115-2', '대전광역시 서구 도마동 227', 
                 '대전광역시 동구 자양동 67-17', '대전광역시 대덕구 비래동 103-4', '대전광역시 동구 인동 352', 
                 '대전광역시 유성구 구암동 95-5', '대전광역시 유성구 장대동 115-1', '대전광역시 동구 용전동 68-2', 
                 '대전광역시 유성구 반석동 685', '대전광역시 중구 용두동 112-10', '대전광역시 동구 원동 60-10', 
                 '대전광역시 중구 산성동 48-4', '대전광역시 유성구 지족동 1005', '대전광역시 중구 태평동 531', 
                 '대전광역시 동구 성남동 509', '대전광역시 유성구 봉명동 563-23', '대전광역시 대덕구 신탄진동 144-1',  
                 '대전광역시 대덕구 중리동 126-4', '대전광역시 중구 유천동 185-3', '대전광역시 유성구 궁동 220-9', '대전광역시 유성구 봉명동 449-4')
```

<br>

<br>

주소, 반경, 순위 입력

```R
danger_df <- final_df[, c('x', 'y')]
danger_df$'시설명/주소지' <- gid_location
danger_df$'위험순위' <- c(1:100)
danger_df$'반경범위' <- '100M'
dim(danger_df)
danger_df %>% head()
```

![image](https://user-images.githubusercontent.com/97672187/166234008-df0800c5-d0db-4ffb-9c47-1e70cf200ee5.png){: .align-center}

<br>


<br>

```R
danger_df <- danger_df[, c('위험순위', '시설명/주소지', 'x', 'y', '반경범위')]
dim(danger_df)
danger_df %>% head()
```

![image](https://user-images.githubusercontent.com/97672187/166234068-fed5f563-d11e-4ae1-8569-bde926d9c051.png){: .align-center}

<br>


<br>

```R
write.csv(danger_df, "danger_top_100.csv", row.names = F)
```

<br>

<br>

이로써 전처리, 모델링, 잔차분석이라는 긴 과정을 거쳐서 최종 위험지역 100개 지역을 도출했다. 다음 포스팅에서는 이 100개의 교통사고 위험지역에 어떠한 조치를 취하면 좋을지
시각화와 함께 대안책을 정리해보자.

![image](https://user-images.githubusercontent.com/97672187/166234384-3366053f-b1d4-4e2c-bd13-1867ce820a2e.png){: .align-center}



