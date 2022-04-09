---
layout: single
title: "전처리 Part.7"
toc: true
toc_sticky: true
category: LH
---

전처리 Part.7 에서는 다시 한번 국지적 모란지수를 활용해서 연령과 사고유형별로 의미가 떨어지는 사고격자를 제거하고, 사망자수, 중상자 수에
따라 가중치를 도출했다. 이 가중치가 높을 수록 교통사고에 더 치명적인 격자가 될 것이라고 가정했다.

### 전처리 Part.7(by using R)

### 현재까지 구축한 데이터 불러오기
중앙분리대수를 포함하고 있는 격자는 전체 격자에 비해 데이터가 너무 적다고 판단해서 중앙분리대수 변수가 크게 의미가 없을 것 같다.
따라서 중앙 분리대수 변수는 제외했다.

```R
acci_count_filter25 <- read.csv("accident_count_filter_25.csv" ) %>% dplyr::select(-중앙분리대수)
dim(acci_count_filter25)
names(acci_count_filter25)[2:48]
acci_count_filter25 %>% head(1)
```

![image](https://user-images.githubusercontent.com/97672187/162560392-6092bfd3-d1ac-4cab-af1c-c739ee0047cc.png){: .align-center}

<br>


<br>

table 함수를 통해 연령대별 사고유형별 변수의 데이터 분포를 확인할 수 있다. 


```R
for(i in 34:47){
    print(table(acci_count_filter25[, i]))
}
```

![image](https://user-images.githubusercontent.com/97672187/162560478-9e0de2b5-e34b-4bb8-998c-f87685230f5f.png){: .align-center}


다양한 방식으로 사고건수가 0인 격자를 제거했지만, **연령과 사고유형으로 나눈 그룹 안에서는 여전히 0의 개수가 많이 있는 것을 확인** 했다.
따라서 앞서 사용했던 국지적 모란 방식을 통해서 다시 한번 의미성이 떨어지는 0개 사고건수 격자를 제외하는 방법을 수행하도록 한다.

<br>


<br>

### 25. 국지적 모란 방식을 이용하여 각 그룹별로 의미성이 떨어지는 0개 사고건수 격자를 제외하기

local_moran_func 함수는 사고유형별, 연령대별 변수를 입력으로 받아서 해당 변수에 대해 국지적 모란지수를 구하고 다른 격자에 비해
상대적으로 덜 중요하다고 판단되는 격자를 제외한다. 함수의 결과물은 기존 데이터의 분포, 필터링된 데이터의 분포, 사분면을 보여주기 위한
모란지수와 정규화된 사고 건수의 관계 그래프, 기존 데이터의 격자 데이터와 필터링된 격자 데이터의 시각화로 이루어진다. 가장 마지막에
그려지는 시각화는 기존 격자 위에 새롭게 필터링된 격자를 연두색으로 표시한다. 최종적으로는 데이터프레임을 리턴해서 필터링된 격자 데이터를
csv로 저장했다.

```R
local_moran_func <- function(target_group){
  
  ## target_group에 해당하는 변수의 index를 target_col로 할당
  for(i in 34:47){
    if (names(acci_count_filter25[i]) == target_group){
      target_col = i
    }
  }
  
  ## target_group 그룹을 종속변수로 둔 데이터를 필터링
  spdep::set.ZeroPolicyOption(TRUE)
  distt <- spdep::dnearneigh(cbind(acci_count_filter25$x, acci_count_filter25$y), 0, 0.25, longlat=TRUE)
  lw2 <- spdep::nb2listw(distt, style="W", zero.policy=TRUE)
  om <- spdep::localmoran(acci_count_filter25[, target_col], lw2,zero.policy = TRUE)
  pr2 <- data.frame(cbind(om[,1], (acci_count_filter25[, target_col] - mean(acci_count_filter25[, target_col], na.rm = TRUE)) / sd(acci_count_filter25[, target_col], na.rm=T)))
  plot(pr2$X1, pr2$X2, xlab="Local Moran's I", ylab="Standardized number of accidents")
  abline(v=0, h=0, col="red")
  pr3 <- pr2 %>% filter(!X1>0 | !X2<0) # 4사분면 제외. 모란지수가 낮고, 사고건수가 평균 사고건수보다 낮은 데이터들.
  
  ## 분석에 사용할 최종 데이터(target_group을 종속변수로 둔 모델) 구축 완료
  group_df <- acci_count_filter25[rownames(pr3), ]
  
  ## 각 그룹에 맞춘 최종 데이터와 기존 데이터에 대하여 대전시 내의 산포와 비교
  print(table(acci_count_filter25[, target_col])) # 기존 데이터
  print(table(group_df[, target_col]))            # 새롭게 필터링한 데이터
  plot(acci_count_filter25$x, acci_count_filter25$y, xlab="long", ylab="lat", main="Comparison of former and new data")
  lines(group_df$x, group_df$y, col="green", pch=19, cex = 0.5, type="p") # 연두색 데이터가 기존 데이터에서 필터링된 데이터
  return(group_df)
}
```

차대사람_20대 group에 대한 결과

```R
person_20 <- local_moran_func('차대사람..20대')
write.csv(person_20, 'person_20.csv')
```

![image](https://user-images.githubusercontent.com/97672187/162561087-f1a705dc-8f57-42ef-955f-5968f62d36b0.png){: .align-center}

<br>


<br>

차대사람_20대 미만 group에 대한 결과

```R
person_under20 <- local_moran_func('차대사람..20대.미만')
write.csv(person_under20, 'person_under20.csv')
```

![image](https://user-images.githubusercontent.com/97672187/162561154-829e8e4f-eba9-4cf5-996e-006b1c49b9c5.png){: .align-center}

<br>


<br>

차대사람_30대 group에 대한 결과

```R
person_30 <- local_moran_func('차대사람..30대')
write.csv(person_30, 'person_30.csv')
```

![image](https://user-images.githubusercontent.com/97672187/162561213-93e231eb-ee7e-4095-a450-b6b6892cdfa9.png){: .align-center}

<br>


<br>


차대사람_40대 group에 대한 결과

```R
person_40 <- local_moran_func('차대사람..40대')
write.csv(person_40, 'person_40.csv')
```

![image](https://user-images.githubusercontent.com/97672187/162561227-11767b3e-cbcd-410f-9e90-024e65b25bd5.png){: .align-center}

<br>


<br>


차대사람_50대 group에 대한 결과

```R
person_50 <- local_moran_func('차대사람..50대')
write.csv(person_50, 'person_50.csv')
```

![image](https://user-images.githubusercontent.com/97672187/162561242-a0e63410-a07d-4528-bb10-9e4e97cf8d56.png){: .align-center}

<br>


<br>

차대사람_60대 이상 group에 대한 결과

```R
person_over60 <- local_moran_func('차대사람..60대.이상')
write.csv(person_over60, 'person_over60.csv')
```

![image](https://user-images.githubusercontent.com/97672187/162561255-f036633b-5c5c-43a8-ad8c-9912024b98d8.png){: .align-center}

<br>


<br>


차대차_20대 group에 대한 결과

```R
car_20 <- local_moran_func('차대차..20대')
write.csv(car_20, 'car_20.csv')
```

![image](https://user-images.githubusercontent.com/97672187/162561270-693bbafe-0dc2-4e1c-9807-33672ef3109f.png){: .align-center}

<br>


<br>

차대차_20대 미만 group에 대한 결과

```R
car_under20 <- local_moran_func('차대차..20대.미만')
write.csv(car_under20, 'car_under20.csv')
```

![image](https://user-images.githubusercontent.com/97672187/162561297-50f1d215-ef35-4746-a961-cc9e43235e96.png){: .align-center}

<br>


<br>

차대차_30대 group에 대한 결과

```R
car_30 <- local_moran_func('차대차..30대')
write.csv(car_30, 'car_30.csv')
```

![image](https://user-images.githubusercontent.com/97672187/162561322-31532435-036a-4560-bd66-383754ebc3eb.png){: .align-center}

<br>


<br>

차대차_40대 group에 대한 결과

```R
car_40 <- local_moran_func('차대차..40대')
write.csv(car_40, 'car_40.csv')
```

![image](https://user-images.githubusercontent.com/97672187/162561360-6dd5107a-f8fc-441d-9486-dae627281c45.png){: .align-center}

<br>


<br>

차대차_50대 group에 대한 결과

```R
car_50 <- local_moran_func('차대차..50대')
write.csv(car_50, 'car_50.csv')
```

![image](https://user-images.githubusercontent.com/97672187/162561371-094c4fb0-fdda-4c9a-8207-c3248c0e447c.png){: .align-center}

<br>


<br>

차대차_60대 이상 group에 대한 결과

```R
car_over60 <- local_moran_func('차대차..60대.이상')
write.csv(car_over60, 'car_over60.csv')
```

![image](https://user-images.githubusercontent.com/97672187/162561395-7f31e110-1e8f-4d9c-ab0c-b70a13741565.png){: .align-center}

<br>


<br>


### 26. 사망자수와 중상자 및 사망자 수에 따른 가중치 도출 - 범주형 회귀분석

포아송 회귀는 종속변수가 포아송 분포를 따른다고 가정하고 회귀분석을 수행한다. 
포아송 분포란 단위 시간 안에 어떤 사건이 몇 번 발생할 것인지를 표현하는 이산확률 분포로 구간에서 발생하는 사건의 횟수를 추정하는데 매우 유용하다.

사고에 관련된 데이터도 결국 단위 기간 내에 발생한 데이터이기 때문에 이 포아송 회귀를 사용해서 
사고유형별, 연령대별 변수가 사상자수, 중상이상 사상자수에 얼마나 영향을 미치는지 파악하고 이를 통해 가중치를 도출한다.

가중치가 클수록 해당 사고유형과 연령대 조합이 포함된 격자는 교통사고 위험도가 큰 지역이라고 가정할 수 .

```R
accident_list <- read.csv("1.대전광역시_교통사고내역(2017~2019).csv", header=TRUE)
dim(accident_list)
accident_list$피해운전자.연령대 <- as.factor(accident_list$피해운전자.연령대) #연령대변수를 범주형으로 변환
table(accident_list$피해운전자.연령대) # 연령대 변수 데이터 분포
levels(accident_list$`피해운전자.연령대`) <- c("20대미만", "20대미만", "20대", "30대", "40대", "50대", "60대이상", "60대이상", "60대이상", "60대이상", "NULL", "미분류")
```

![image](https://user-images.githubusercontent.com/97672187/162561749-58c74a27-013c-436d-baab-43120c1b8deb.png){: .align-center}

<br>


<br>

```R
accident_list <- accident_list %>% filter(피해운전자.연령대!="NULL" & 피해운전자.연령대!="미분류") # NULL과 미분류 제외
dim(accident_list)
table(accident_list$피해운전자.연령대)
```

![image](https://user-images.githubusercontent.com/97672187/162561783-e18ca5db-6d11-4a00-9b4e-12a99206648c.png){: .align-center}

<br>


<br>

사고유형 변수에서 차대차, 차대사람과 같은 사고유형의 정보만 남도록 text 전처리

사상자수는 사망자 + 중상자 + 경상자수로 계산.

중상자수는 사망자 + 중상자수로 계산

```R
accident_list$`사고유형` <- substr(accident_list$'사고유형', 1, 5)
accident_list$`사고유형` <- str_replace_all(accident_list$`사고유형`, " -", "")
accident_list$`사고유형` <- str_replace_all(accident_list$`사고유형`, " ", "")
accident_list$`사고유형` <- as.factor(accident_list$`사고유형`)
table(accident_list$`사고유형`)

accident_list$사상자수 <- accident_list$`사망자수` + accident_list$`중상자수` + accident_list$`경상자수`
accident_list$중상이상사상자수 <- accident_list$`사망자수` + accident_list$`중상자수`
accident_list %>% head(3)
```
![image](https://user-images.githubusercontent.com/97672187/162561840-7bbe6698-f847-48ef-a264-117354509d30.png){: .align-center}

<br>


<br>


1) 사고유형, 피해운전자 연령대, 그리고 사고유형과 피해운전자 연령대가 결합된 변수가 사상자수에 미치는 영향 정도를 파악.

사고유형(차대차,차대사람) x 연령대(20대 미만, 20대, 30대, 40대, 50대, 60대이상) = 12개의 영향도가 나옴.

```R
acci_num_glm <- glm(사상자수 ~ 사고유형 + 피해운전자.연령대 + 사고유형:피해운전자.연령대, family="poisson", accident_list)
summary(acci_num_glm)
accident_list$pred <- acci_num_glm$fitted.values
unique(acci_num_glm$fitted.values) # 12개의 영향도
```
![image](https://user-images.githubusercontent.com/97672187/162563171-56f4fd4e-0688-4cc5-948e-790e2247842b.png){: .align-center}

<br>


<br>


2) 사고유형, 피해운전자 연령대, 그리고 사고유형과 피해운전자 연령대가 결합된 변수가 중상이상사상자수에 미치는 영향 정도를 파악.

마찬가지로 12개의 영향도가 나옴

```R
serious_acci_num_glm <- glm(중상이상사상자수 ~ 사고유형 + 피해운전자.연령대 + 사고유형:피해운전자.연령대, family="poisson", accident_list)
summary(serious_acci_num_glm)
accident_list$pred_jungsang <- serious_acci_num_glm$fitted.values
unique(serious_acci_num_glm$fitted.values)
```

![image](https://user-images.githubusercontent.com/97672187/162563476-06969bbf-1f9d-4125-bf5d-3b832b42ffe6.png){: .align-center}


<br>


<br>

포아송 회귀로 구한 각각의 영향도를 12개 전체의 영향도의 합으로 나누어서 가중치를 구한다.
사상자수, 중상이상사상자수의 가중치를 각각 구하고 두 가중치를 더해 사고유형별, 연령대별 변수가 사고에 영향을 미치는 정도인
합계 가중치를 도출한다.

```R
accident_list$weights <- accident_list$pred / sum(unique(accident_list$pred)) #사상자수 가중치(규모)
accident_list$weights_jungsang <- accident_list$pred_jungsang / sum(unique(accident_list$pred_jungsang)) #중상자이상수 가중치(신체적 손상, 심각성)
accident_list$total_weight <- (accident_list$weights + accident_list$weights_jungsang) / sum(unique(accident_list$weights + accident_list$weights_jungsang))
dim(accident_list)
accident_list %>% head(3)
```

![image](https://user-images.githubusercontent.com/97672187/162563580-cd872bc3-0a7d-4ed7-ad95-80700a7e0634.png){: .align-center}

<br>


<br>

사고유형, 연령대별 변수를 합친 사고유형_연령대 변수를 만들고 위에서 구한 12개의 합계 가중치를 사고유형_연령대 그룹에 매칭시킨다.

```R
accident_list$사고유형_연령대 <- paste0(accident_list$`사고유형`, "_", accident_list$`피해운전자.연령대`)
table(accident_list$사고유형_연령대)
n <- aggregate(total_weight ~ `사고유형_연령대`, mean, data=accident_list)
weight_df <- n[order(n$total_weight, decreasing = T),]
weight_df
write.csv(weight_df, 'weight_df.csv')
```

![image](https://user-images.githubusercontent.com/97672187/162563891-1be598b7-212c-4d6e-b107-9d50473aa6fb.png){: .align-center}
