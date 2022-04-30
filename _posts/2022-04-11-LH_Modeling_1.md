---
layout: single
title: "Modeling Part.1"
toc: true
toc_sticky: true
category: LH
---

모든 전처리가 끝나고 모델링 파트까지 왔다. 모델링은 Part1,2로 나누어져있고, 모델링은 모두 R을 사용했다. Part.1에서는 가벼운 모델인 OLS와 단위 시간당 특정 사건의 갯수를 예측하는
포아송 회귀 모델을 사용했다. 

사고유형별, 또 연령대별의 사고건수를 파악하기 위해 사고유형_연령대별로 세분화한 데이터를 사용했고, 12개의 그룹에 해당하는 모델을 만들었다.

결국 이 프로젝트에서 알고 싶은 것은 어떤 지역이 사고유형별_연령대별로 교통사고 위험지역인가를 파악하는 것이다.

ex) 20대의 연령대에서 특히 차대사람 사고가 많이나는 지역은?

## Modeling Part.1 (by using R)

현재까지 구축한 데이터 불러오기

데이터 양이 충분하지 않는 '중앙분리대수' 변수는 제외한다.

```R
acci_count_filter25 <- read.csv("accident_count_filter_25.csv" ) %>% dplyr::select(-중앙분리대수)
dim(acci_count_filter25)
acci_count_filter25 %>% head(1)
```

![image](https://user-images.githubusercontent.com/97672187/162734937-d39a35b7-ab38-4835-b125-b6e2aba8da45.png){: .align-center}

```R
names(acci_count_filter25)
```

![image](https://user-images.githubusercontent.com/97672187/162736017-26c45ef7-e418-47eb-91bc-9e5e8d5dad75.png){: .align-center}

<br>


<br>

변수 필터링 과정1- 상관행렬과 Heatmap 확인

```R
# 상관행렬
check_cor <- acci_count_filter25 %>% dplyr::select(`신호등_보행자수`:총거주인구수) %>% cor()

# Heatmap
check_cor2 <- acci_count_filter25 %>% dplyr::select(`신호등_보행자수`:총거주인구수) %>% 
                cor()%>% ggcorrplot(tl.srt = 90)
check_cor2
```

![image](https://user-images.githubusercontent.com/97672187/162736586-07c32da4-64bf-4b37-a1e1-3a833d42371d.png){: .align-center}


<br>


<br>

변수 필터링 과정2 - 분산팽창인자(VIF) 확인

```R
check_cor_df <- acci_count_filter25 %>% dplyr::select(`신호등_보행자수`:총거주인구수, 전체_추정교통량,사고건수)

mod1 <- lm(사고건수 ~ ., data=check_cor_df)
vif <- VIF(mod1)
print("다중공선성 존재")
names(vif)[vif > 5]; # 다중공선성 있다고 판단되는 독립변수들

print("다중공선성이 존재하지 않음")
names(vif)[vif < 5]; # 다중공선성이 있다고 판단되지는 않는 변수들(정상적인 변수들)
```

![image](https://user-images.githubusercontent.com/97672187/162736887-75319a69-73e1-458d-b989-f3275eee3cf6.png){: .align-center}


<br>


<br>

혼잡빈도강도, 혼잡시간강도, 이상평균기온동반사고건수, 이상최저온도동반사고건수, 이상최고온도동반사고건수, 이상평균지면온도동반사고건수는 vif가 5 이상으로 판별되었다.

- 데이터 변형1 : 혼잡빈도강도, 혼잡시간강도 데이터의 평균을 "평균혼잡강도" 데이터로 수정한다.

- 데이터 변형2 : 평균기온동반사고건수, 최저온도동반사고건수, 최고온도동반사고건수, 평균지면온도동반사고건수 변수들의 상관관계가 높아 "이상평균기온동반사고건수"만 사용한다.


```R
revised_df <- acci_count_filter25 %>% mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수, -혼잡빈도강도, -혼잡시간강도)

# VIF 확인
check_cor_df2 <- revised_df %>% dplyr::select(`신호등_보행자수`:총거주인구수, 전체_추정교통량, 사고건수)
mod2 <- lm(사고건수 ~ ., data=check_cor_df2)
vif_final <- VIF(mod2)

print("다중공선성 존재")
names(vif_final)[vif_final>5]; # 다중공선성 있다고 판단되는 독립변수들

print("다중공선성이 존재하지 않음")
names(vif_final)[vif_final<5]; # 다중공선성이 있다고 판단되지는 않는 변수들(정상적인 변수들)
```

![image](https://user-images.githubusercontent.com/97672187/162737543-fb72a190-2915-48f3-b447-56e8da0f3289.png){: .align-center}

<br>


<br>

### 1. OLS 모델링
회귀에서 가장 기본적인 모델 중 하나인 OLS 모델로 사고유형별, 연령대별 사고 건수를 예측한다. 독립변수는 위에서 제거한 것처럼 다중공선성이 없는 변수들만 사용한다.

각 모델의 최적의 변수 조합은 stepwise 방법을 사용했다.

데이터 불러오기

```R
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

OLS 모델 적용 - 차대사람..60대 이상

```R
mod_data2 <- person_over60 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대사람..60대.이상) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- lm(`차대사람..60대.이상`~., mod_data2)
# 회귀분석의 변수선택(stepwise 방법)
summary(step(mod,direction = 'both')) # 결과 코드가 너무 길어져서 해당부분은 출력 생략.
mod1 <- lm(formula = 차대사람..60대.이상 ~ 이상평균기온동반사고건수 + 
           이상평균풍속동반사고건수 + 이상평균습도동반사고건수 + 이상안개시간동반사고건수 + 
           안전지대수 + 횡단보도수 + 자동차대수 + 평균혼잡강도, data = mod_data2)
summary(mod1)
paste("AIC:", round(AIC(mod1), 2))

# OLS의 기본 가정들을 만족하는지 확인 - 등분산성, 독립성, 정규성
bptest(mod1) # 등분산성은 만족하지 않음(유의수준 0.05)
dwtest(mod1) # 독립성은 만족(유의수준=0.05)
shapiro.test(x=residuals(mod1)) # 정규성 만족하지 않음(유의수준=0.05)
```

![image](https://user-images.githubusercontent.com/97672187/162740125-f25a7859-b1b9-458c-9909-4553c026496b.png){: .align-center}

<br>


<br>

OLS 모델 적용 - 차대사람..50대

```R
mod_data2 <- person_50 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대사람..50대) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- lm(`차대사람..50대`~., mod_data2)
# summary(step(mod,direction = 'both'))
mod1 <- lm(formula = 차대사람..50대 ~ 이상평균기온동반사고건수 + 이상최대풍속동반사고건수 + 
           이상평균풍속동반사고건수 + 이상평균습도동반사고건수 + 이상강수량동반사고건수 + 
           안전지대수 + 횡단보도수, data = mod_data2)
summary(mod1)
paste("AIC:", round(AIC(mod1), 2))

# OLS의 기본 가정들을 만족하는지 확인 - 등분산성, 독립성, 정규성
bptest(mod1) # 등분산성은 만족하지 않음(유의수준 0.05)
dwtest(mod1) # 독립성은 만족(유의수준=0.05)
shapiro.test(x=residuals(mod1)) # 정규성 만족하지 않음(유의수준=0.05)
```

![image](https://user-images.githubusercontent.com/97672187/162740562-d89d5279-4c91-4a99-950f-5ef86a26a62a.png){: .align-center}

<br>


<br>


OLS 모델 적용 - 차대사람..40대

```R
mod_data2 <- person_40 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대사람..40대) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- lm(`차대사람..40대`~., mod_data2)
# summary(step(mod,direction = 'both'))
mod1 <- lm(formula = 차대사람..40대 ~ 신호등_보행자수 + 신호등_차량등수 + 
           이상평균기온동반사고건수 + 이상최대풍속동반사고건수 + 이상강수량동반사고건수 + 
           이상적설량동반사고건수 + 도로속도표시수 + 횡단보도수 + 전체_추정교통량 + 
           평균혼잡강도, data = mod_data2)
summary(mod1)
paste("AIC:", round(AIC(mod1), 2))

# OLS의 기본 가정들을 만족하는지 확인 - 등분산성, 독립성, 정규성
bptest(mod1) # 등분산성은 만족하지 않음(유의수준 0.05)
dwtest(mod1) # 독립성은 만족(유의수준=0.05)
shapiro.test(x=residuals(mod1)) # 정규성 만족하지 않음(유의수준=0.05)
```

![image](https://user-images.githubusercontent.com/97672187/162740708-20585910-8571-4435-aead-5f5d47a0a5c1.png){: .align-center}

<br>


<br>

OLS 모델 적용 - 차대사람..30대

```R
mod_data2 <- person_30 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대사람..30대) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- lm(`차대사람..30대`~., mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))

mod1 <- lm(formula = 차대사람..30대 ~ 신호등_차량등수 + cctv수 + 이상평균기온동반사고건수 + 
           이상최대풍속동반사고건수 + 이상평균습도동반사고건수 + 이상강수량동반사고건수 + 
           이상적설량동반사고건수 + 정차금지지대수 + 횡단보도수 + 총거주인구수 + 
           전체_추정교통량, data = mod_data2)
summary(mod1)
paste("AIC:", round(AIC(mod1), 2))

# OLS의 기본 가정들을 만족하는지 확인 - 등분산성, 독립성, 정규성
bptest(mod1) # 등분산성은 만족하지 않음(유의수준 0.05)
dwtest(mod1) # 독립성은 만족(유의수준=0.05)
shapiro.test(x=residuals(mod1)) # 정규성 만족하지 않음(유의수준=0.05)
```

![image](https://user-images.githubusercontent.com/97672187/162740775-428f6af2-c5fb-4f6d-b476-cd802c7c4ba8.png){: .align-center}

<br>


<br>

OLS 모델 적용 - 차대사람..20대

```R
mod_data2 <- person_20 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대사람..20대) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- lm(`차대사람..20대`~., mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))

mod1 <- lm(formula = 차대사람..20대 ~ 신호등_보행자수 + 신호등_차량등수 + 
           cctv수 + 이상평균기온동반사고건수 + 이상최대풍속동반사고건수 + 
           이상평균풍속동반사고건수 + 이상평균습도동반사고건수 + 이상강수량동반사고건수 + 
           이상적설량동반사고건수 + 정차금지지대수 + 도로속도표시수 + 
           노드개수, data = mod_data2)
summary(mod1)
paste("AIC:", round(AIC(mod1), 2))

# OLS의 기본 가정들을 만족하는지 확인 - 등분산성, 독립성, 정규성
bptest(mod1) # 등분산성은 만족하지 않음(유의수준 0.05)
dwtest(mod1) # 독립성은 만족(유의수준=0.05)
shapiro.test(x=residuals(mod1)) # 정규성 만족하지 않음(유의수준=0.05)
```

![image](https://user-images.githubusercontent.com/97672187/162740850-dde3ab4b-823e-42ac-9e2c-e871092ccd60.png){: .align-center}

<br>


<br>

OLS 모델 적용 - 차대사람..20대미만

```R
mod_data2 <- person_under20 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대사람..20대.미만) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- lm(`차대사람..20대.미만`~., mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))

mod1 <- lm(formula = 차대사람..20대.미만 ~ 신호등_보행자수 + 신호등_차량등수 + 
           cctv수 + 이상평균기온동반사고건수 + 이상평균풍속동반사고건수 + 
           이상평균습도동반사고건수 + 이상적설량동반사고건수 + 이상안개시간동반사고건수 + 
           안전지대수 + 정차금지지대수 + 노드개수 + 횡단보도수 + 총거주인구수 + 
           전체_추정교통량, data = mod_data2)
summary(mod1)
paste("AIC:", round(AIC(mod1), 2))

# OLS의 기본 가정들을 만족하는지 확인 - 등분산성, 독립성, 정규성
bptest(mod1) # 등분산성은 만족하지 않음(유의수준 0.05)
dwtest(mod1) # 독립성은 만족(유의수준=0.05)
shapiro.test(x=residuals(mod1)) # 정규성 만족하지 않음(유의수준=0.05)
```

![image](https://user-images.githubusercontent.com/97672187/162740945-1ab50d37-bee6-41cf-bb7a-f4597cdc64c4.png){: .align-center}

<br>


<br>


OLS 모델 적용 - 차대차-60대이상

```R
mod_data2 <- car_over60 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대차..60대.이상) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- lm(`차대차..60대.이상`~., mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))
mod1 <- lm(차대차..60대.이상 ~ 신호등_차량등수 + cctv수 + 이상평균기온동반사고건수 + 
              이상최대풍속동반사고건수 + 이상평균풍속동반사고건수 + 이상평균습도동반사고건수 + 
              이상적설량동반사고건수 + 이상안개시간동반사고건수 + 안전지대수 + 
              정차금지지대수 + 도로속도표시수 + 횡단보도수 + 전체_추정교통량, data = mod_data2)
summary(mod1)
paste("AIC:", round(AIC(mod1), 2))

# OLS의 기본 가정들을 만족하는지 확인 - 등분산성, 독립성, 정규성
bptest(mod1) # 등분산성은 만족하지 않음(유의수준 0.05)
dwtest(mod1) # 독립성은 만족(유의수준=0.05)
shapiro.test(x=residuals(mod1)) # 정규성 만족하지 않음(유의수준=0.05)
```

![image](https://user-images.githubusercontent.com/97672187/162741142-35f5f053-019b-4624-bc9e-caac9e9c9692.png){: .align-center}

<br>


<br>

OLS 모델 적용 - 차대차-50대

```R
mod_data2 <- car_50 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대차..50대) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- lm(`차대차..50대`~., mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))
mod1 <- lm(차대차..50대 ~ 신호등_보행자수 + 신호등_차량등수 + 
              cctv수 + 이상평균기온동반사고건수 + 이상최대풍속동반사고건수 + 
              이상평균풍속동반사고건수 + 이상평균습도동반사고건수 + 이상강수량동반사고건수 + 
              이상적설량동반사고건수 + 이상안개시간동반사고건수 + 안전지대수 + 
              정차금지지대수 + 도로속도표시수 + 교통안전표지수 + 노드개수 + 
              횡단보도수 + 평균혼잡강도, data = mod_data2)
summary(mod1)
paste("AIC:", round(AIC(mod1), 2))

# OLS의 기본 가정들을 만족하는지 확인 - 등분산성, 독립성, 정규성
bptest(mod1) # 등분산성은 만족하지 않음(유의수준 0.05)
dwtest(mod1) # 독립성은 만족(유의수준=0.05)
shapiro.test(x=residuals(mod1)) # 정규성 만족하지 않음(유의수준=0.05)
```

![image](https://user-images.githubusercontent.com/97672187/162741206-5c407be6-9734-4e3e-a0b1-fee0aa5483dd.png){: .align-center}

<br>


<br>

OLS 모델 적용 - 차대차-40대

```R
mod_data2 <- car_40 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대차..40대) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- lm(`차대차..40대`~., mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))
mod1 <- lm(차대차..40대 ~ 신호등_차량등수 + 이상평균기온동반사고건수 + 
              이상최대풍속동반사고건수 + 이상평균풍속동반사고건수 + 이상평균습도동반사고건수 + 
              이상강수량동반사고건수 + 이상안개시간동반사고건수 + 안전지대수 + 
              도로속도표시수 + 교통안전표지수 + 노드개수 + 횡단보도수 + 
              총거주인구수, data = mod_data2)
summary(mod1)
paste("AIC:", round(AIC(mod1), 2))

# OLS의 기본 가정들을 만족하는지 확인 - 등분산성, 독립성, 정규성
bptest(mod1) # 등분산성은 만족하지 않음(유의수준 0.05)
dwtest(mod1) # 독립성은 만족(유의수준=0.05)
shapiro.test(x=residuals(mod1)) # 정규성 만족하지 않음(유의수준=0.05)
```


![image](https://user-images.githubusercontent.com/97672187/162741283-d9920c0b-162d-4340-953d-9892c8024a7e.png){: .align-center}

<br>


<br>


```R

```

{: .align-center}

<br>


<br>

OLS 모델 적용 - 차대차-30대

```R
mod_data2 <- car_30 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대차..30대) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- lm(`차대차..30대`~., mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))
mod1 <- lm(차대차..30대 ~ 신호등_보행자수 + cctv수 + 이상평균기온동반사고건수 + 
              이상최대풍속동반사고건수 + 이상평균풍속동반사고건수 + 이상평균습도동반사고건수 + 
              이상강수량동반사고건수 + 이상적설량동반사고건수 + 안전지대수 + 
              정차금지지대수 + 도로속도표시수 + 교통안전표지수 + 노드개수 + 
              횡단보도수 + 평균혼잡강도, data = mod_data2)
summary(mod1)
paste("AIC:", round(AIC(mod1), 2))

# OLS의 기본 가정들을 만족하는지 확인 - 등분산성, 독립성, 정규성
bptest(mod1) # 등분산성은 만족하지 않음(유의수준 0.05)
dwtest(mod1) # 독립성은 만족(유의수준=0.05)
shapiro.test(x=residuals(mod1)) # 정규성 만족하지 않음(유의수준=0.05)
```

![image](https://user-images.githubusercontent.com/97672187/162741353-0e85a914-c94b-4a08-a31f-e2397d9cdd46.png){: .align-center}

<br>


<br>

OLS 모델 적용 - 차대차-20대

```R
mod_data2 <- car_20 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대차..20대) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- lm(`차대차..20대`~., mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))
mod1 <- lm(차대차..20대 ~ 신호등_보행자수 + cctv수 + 이상평균기온동반사고건수 + 
              이상최대풍속동반사고건수 + 이상평균풍속동반사고건수 + 이상평균습도동반사고건수 + 
              이상적설량동반사고건수 + 이상안개시간동반사고건수 + 안전지대수 + 
              정차금지지대수 + 교통안전표지수 + 노드개수, data = mod_data2)
summary(mod1)
paste("AIC:", round(AIC(mod1), 2))

# OLS의 기본 가정들을 만족하는지 확인 - 등분산성, 독립성, 정규성
bptest(mod1) # 등분산성은 만족하지 않음(유의수준 0.05)
dwtest(mod1) # 독립성은 만족(유의수준=0.05)
shapiro.test(x=residuals(mod1)) # 정규성 만족하지 않음(유의수준=0.05)
```

![image](https://user-images.githubusercontent.com/97672187/162741436-d7d9be4a-036a-47ae-91ba-439cbc52dda0.png){: .align-center}

<br>


<br>

OLS 모델 적용 - 차대차-20대미만

```R
mod_data2 <- car_under20 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대차..20대.미만) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- lm(`차대차..20대.미만`~., mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))
mod1<-lm(차대차..20대.미만 ~ 신호등_보행자수 + cctv수 + 이상최대풍속동반사고건수 + 
              이상평균풍속동반사고건수 + 이상평균습도동반사고건수 + 이상강수량동반사고건수 + 
              이상적설량동반사고건수 + 안전지대수 + 횡단보도수 + 총거주인구수, data = mod_data2)
summary(mod1)
paste("AIC:", round(AIC(mod1), 2))

# OLS의 기본 가정들을 만족하는지 확인 - 등분산성, 독립성, 정규성
bptest(mod1) # 등분산성은 만족하지 않음(유의수준 0.05)
dwtest(mod1) # 독립성은 만족(유의수준=0.05)
shapiro.test(x=residuals(mod1)) # 정규성 만족하지 않음(유의수준=0.05)
```

![image](https://user-images.githubusercontent.com/97672187/162741508-ac316cbf-54c4-4d13-bfbb-b15123c0fa4d.png){: .align-center}

<br>


<br>

### 2. 포아송 회귀 모델링

로지스틱 회귀 중 포아송회귀를 적용했다. 프로젝트에서 알고자 하는 것이 결국, 단위 기간 내에 사고유형별, 연령대별로 사고건수가 얼마나 발생하는 것을 예측하는 것이기 때문에
포아송 회귀를 사용하는 것이 적합하다고 판단했다.

각 모델의 최적의 변수 조합은 stepwise 방법을 사용했다.


포아송회귀 (GLM) 모델 적용 - 차대사람..60대 이상

```R
mod_data2 <- person_over60 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대사람..60대.이상) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- glm(`차대사람..60대.이상`~., family="poisson", mod_data2)
#회귀분석의 변수선택(stepwise 방법)
summary(step(mod,direction = 'both')) #결과 코드가 너무 길어져서 해당 부분은 출력 생략.
mod1 <- glm(formula = 차대사람..60대.이상 ~ cctv수 + 이상평균기온동반사고건수 + 
            이상평균습도동반사고건수 + 이상안개시간동반사고건수 + 안전지대수 + 
            횡단보도수 + 자동차대수 + 평균혼잡강도, family = "poisson", data = mod_data2)
summary(mod1)
paste("AIC:",round(AIC(mod1),2))
```

![image](https://user-images.githubusercontent.com/97672187/162742402-9170e9a5-ed0a-4f00-bd7e-936d1165045f.png){: .align-center}

<br>


<br>

포아송회귀 (GLM) 모델 적용 - 차대사람..50대

```R
mod_data2 <- person_50 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대사람..50대) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- glm(`차대사람..50대`~., family="poisson", mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))
mod1 <- glm(formula = 차대사람..50대 ~ cctv수 + 이상평균기온동반사고건수 + 
            이상최대풍속동반사고건수 + 이상평균풍속동반사고건수 + 이상평균습도동반사고건수 + 
            안전지대수 + 횡단보도수 , family = "poisson", data = mod_data2)
summary(mod1)
paste("AIC:",round(AIC(mod1),2))
```

![image](https://user-images.githubusercontent.com/97672187/162742560-89d7282e-e678-4a41-8144-fe3d58385f41.png){: .align-center}

<br>


<br>

포아송회귀 (GLM) 모델 적용 - 차대사람..40대

```R
mod_data2 <- person_40 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대사람..40대) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- glm(`차대사람..40대`~., family="poisson", mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))
mod1<-glm(formula = 차대사람..40대 ~ cctv수 + 이상평균기온동반사고건수 + 
            이상강수량동반사고건수 + 횡단보도수 + 평균혼잡강도 , family = "poisson", data = mod_data2)
summary(mod1)
paste("AIC:",round(AIC(mod1),2))
```

![image](https://user-images.githubusercontent.com/97672187/162742641-f0fbb26a-3021-4545-89f8-4620a6eaab33.png){: .align-center}

<br>


<br>

포아송회귀 (GLM) 모델 적용 - 차대사람..30대

```R
mod_data2 <- person_30 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대사람..30대) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- glm(`차대사람..30대`~., family="poisson", mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))
mod1 <- glm(formula = 차대사람..30대 ~ 신호등_차량등수 + cctv수 + 이상평균기온동반사고건수 + 
            이상평균습도동반사고건수 + 이상강수량동반사고건수 + 횡단보도수 + 
            총거주인구수 , family = "poisson", data = mod_data2)
summary(mod1)
paste("AIC:",round(AIC(mod1),2))
```

![image](https://user-images.githubusercontent.com/97672187/162742695-d6697eb8-f613-4deb-8749-c1d986ce8309.png){: .align-center}

<br>


<br>

포아송회귀 (GLM) 모델 적용 - 차대사람..20대

```R
mod_data2 <- person_20 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대사람..20대) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- glm(`차대사람..20대`~., family="poisson", mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))
mod1 <- glm(formula = 차대사람..20대 ~ 신호등_보행자수 + 신호등_차량등수 + 
            cctv수 + 이상평균기온동반사고건수 + 이상최대풍속동반사고건수 + 
            이상평균습도동반사고건수 + 이상강수량동반사고건수 + 정차금지지대수 + 
            도로속도표시수 + 노드개수 + 횡단보도수 + 총거주인구수 , family = "poisson", data = mod_data2)
summary(mod1)
paste("AIC:",round(AIC(mod1),2))
```

![image](https://user-images.githubusercontent.com/97672187/162742813-acc0e3cd-9253-485d-bdaa-3c6562e41d4f.png){: .align-center}

<br>


<br>

포아송회귀 (GLM) 모델 적용 - 차대사람..20대미만

```R
mod_data2 <- person_under20 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대사람..20대.미만) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod<-glm(`차대사람..20대.미만`~., family="poisson", mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))
mod1<-glm(formula = 차대사람..20대.미만 ~ 신호등_보행자수 + cctv수 + 
            이상평균기온동반사고건수 + 이상평균풍속동반사고건수 + 이상강수량동반사고건수 + 
            이상안개시간동반사고건수 + 안전지대수 + 노드개수 + 횡단보도수 + 
            총거주인구수 , family = "poisson", data = mod_data2)
summary(mod1)
paste("AIC:",round(AIC(mod1),2))
```

![image](https://user-images.githubusercontent.com/97672187/162742888-f9550651-6034-4845-aaec-6dbee4762061.png){: .align-center}

<br>


<br>

포아송회귀 (GLM) 모델 적용 - 차대차..60대이상

```R
mod_data2 <- car_over60 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대차..60대.이상) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod<-glm(`차대차..60대.이상`~., family="poisson", mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))
mod1<-glm(formula = 차대차..60대.이상 ~ 신호등_차량등수 + 이상평균기온동반사고건수 + 
            이상평균풍속동반사고건수 + 이상평균습도동반사고건수 + 이상적설량동반사고건수 + 
            안전지대수 + 도로속도표시수 + 노드개수 + 횡단보도수 + 자동차대수 + 
            총거주인구수 + 전체_추정교통량 + 평균혼잡강도 , family = "poisson", data = mod_data2)
summary(mod1)
paste("AIC:",round(AIC(mod1),2))
```

![image](https://user-images.githubusercontent.com/97672187/162743246-1aeea5cb-eb33-4706-bf76-f8407d10a0d7.png){: .align-center}

<br>


<br>

포아송회귀 (GLM) 모델 적용 - 차대차..50대

```R
mod_data2 <- car_50 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대차..50대) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- glm(`차대차..50대`~., family="poisson", mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))
mod1 <- glm(formula = 차대차..50대 ~ 신호등_차량등수 + cctv수 + 이상평균기온동반사고건수 + 
            이상평균풍속동반사고건수 + 이상평균습도동반사고건수 + 이상강수량동반사고건수 + 
            안전지대수 + 정차금지지대수 + 도로속도표시수 + 노드개수 + 
            건물면적 + 총거주인구수 + 전체_추정교통량 + 평균혼잡강도 , family = "poisson", data = mod_data2)
summary(mod1)
paste("AIC:",round(AIC(mod1),2))
```

![image](https://user-images.githubusercontent.com/97672187/162743335-6cdeb88f-0dbd-4b28-9d7e-d184c9868702.png){: .align-center}

<br>


<br>

포아송회귀 (GLM) 모델 적용 - 차대차..40대

```R
mod_data2 <- car_40 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대차..40대) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- glm(`차대차..40대`~., family="poisson", mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))
mod1<-glm(formula = 차대차..40대 ~ 신호등_보행자수 + cctv수 + 이상평균기온동반사고건수 + 
            이상평균풍속동반사고건수 + 이상평균습도동반사고건수 + 이상강수량동반사고건수 + 
            이상적설량동반사고건수 + 이상안개시간동반사고건수 + 도로속도표시수 + 
            교통안전표지수 + 노드개수 + 횡단보도수 + 총거주인구수 + 전체_추정교통량 , family = "poisson", data = mod_data2)
summary(mod1)
paste("AIC:",round(AIC(mod1),2))
```

![image](https://user-images.githubusercontent.com/97672187/162743530-ee960bc3-d312-4ff8-bf9a-31bc0993fbc4.png){: .align-center}

<br>


<br>


포아송회귀 (GLM) 모델 적용 - 차대차..30대

```R
mod_data2 <- car_30 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대차..30대) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod<-glm(`차대차..30대`~., family="poisson", mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))
mod1<-glm(formula = 차대차..30대 ~ 이상평균기온동반사고건수 + 이상평균풍속동반사고건수 + 
            이상평균습도동반사고건수 + 이상강수량동반사고건수 + 이상적설량동반사고건수 + 
            이상안개시간동반사고건수 + 안전지대수 + 도로속도표시수 + 
            교통안전표지수 + 노드개수 + 횡단보도수 + 전체_추정교통량 , family = "poisson", data = mod_data2)
summary(mod1)
paste("AIC:",round(AIC(mod1),2))
```

![image](https://user-images.githubusercontent.com/97672187/162743611-283a5ea6-0eb8-4f26-8e90-ac8138125b2f.png){: .align-center}

<br>


<br>

포아송회귀 (GLM) 모델 적용 - 차대차..20대

```R
mod_data2 <- car_20 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대차..20대) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod<-glm(`차대차..20대`~., family="poisson", mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))
mod1<-glm(formula = 차대차..20대 ~ 신호등_보행자수 + 신호등_차량등수 + 
            이상평균기온동반사고건수 + 이상최대풍속동반사고건수 + 이상평균풍속동반사고건수 + 
            이상평균습도동반사고건수 + 이상강수량동반사고건수 + 이상적설량동반사고건수 + 
            이상안개시간동반사고건수 + 정차금지지대수 + 도로속도표시수 + 
            교통안전표지수 + 노드개수 + 전체_추정교통량 , family = "poisson", data = mod_data2)
summary(mod1)
paste("AIC:",round(AIC(mod1),2))
```

![image](https://user-images.githubusercontent.com/97672187/162743743-670f3d81-697f-4d66-8b8a-26854d74e4ec.png){: .align-center}

<br>


<br>


포아송회귀 (GLM) 모델 적용 - 차대차..20대미만

```R
mod_data2 <- car_under20 %>%
  dplyr::select(신호등_보행자수:총거주인구수, 차대차..20대.미만) %>%
  mutate(평균혼잡강도=(혼잡빈도강도 + 혼잡시간강도)/2) %>%
  dplyr::select(-혼잡빈도강도, -혼잡시간강도, -이상최저온도동반사고건수, -이상최고온도동반사고건수, -이상평균지면온도동반사고건수)

mod <- glm(`차대차..20대.미만`~., family="poisson", mod_data2)
#회귀분석의 변수선택(stepwise 방법)
# summary(step(mod,direction = 'both'))
mod1 <- glm(formula = 차대차..20대.미만 ~ 신호등_보행자수 + 이상평균풍속동반사고건수 + 
            이상평균습도동반사고건수 + 이상적설량동반사고건수 + 교통안전표지수 + 
            횡단보도수 + 총거주인구수 , family = "poisson", data = mod_data2)
summary(mod1)
paste("AIC:",round(AIC(mod1),2))
```

![image](https://user-images.githubusercontent.com/97672187/162743833-0792566e-7c16-4a39-bbf1-6cff4aa806db.png){: .align-center}

<br>


<br>

### OLS와 포아송 회귀 성능 비교(AIC)

1) 사고유형별 연령대별 OLS 결과 AIC

![image](https://user-images.githubusercontent.com/97672187/166101059-afffb36d-79b4-4abf-bd1b-a8e4b6c7d4c0.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/166101155-870889b3-a74c-41c2-bc49-3349793f824f.png){: .align-center}

<br>

<br>

2) 사고유형별 연령대별 포아송 회귀 결과 AIC

![image](https://user-images.githubusercontent.com/97672187/166101194-79ac232c-6e62-4ab6-b811-51d1855466fc.png){: .align-center}

<br>

<br>

3) OLS와 포아송 회귀 성능 비교 그래프

![image](https://user-images.githubusercontent.com/97672187/166101019-f2e5131b-2455-44f3-905c-749b290a204b.png){: .align-center}


<br>

<br>

포아송 회귀의 AIC가 OLS의 AIC보다 전반적으로 모두 낮기 때문에 두 모델 중 포아송 회귀가 더 적합한 모델이라고 할 수 있다.

4) 한계점

기존의 포아송 회귀 모형은 전역적(Global)인 회귀모형이기 때문에 종속변수와 독립변수들 간의 지리적으로 변화하는 관계(공간적 이질성)를 탐색하여 회귀계수를 도출할 수가 없다.

즉, 회귀 모형에 대한 대략적인 피드백은 줄 수 있지만, **각 지역에 따른 맞춤형 피드백**을 줄 순 없다는 한계점이 존재한다. 따라서 다음 포스팅에서는 **각 지역의 특성과 유사성을 고려한** **'지리적 가중 회귀 모델(GWR)'** 을 사용한 모델링을 정리해보고자 한다.

