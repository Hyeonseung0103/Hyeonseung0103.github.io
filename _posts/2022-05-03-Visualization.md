---
layout: single
title: "시각화와 결론"
toc: true
toc_sticky: true
category: LH
---

지난 포스팅까지 해서 대전광역시 교통사고 위험 예상지역 100개지역을 도출했고, 마지막으로 이 지역들에게 어떠한 조치가 취해지면 좋을지 대안책을 제시해보자.

### 전처리부터 결론까지의 과정 요약
1) 격자별로 교통안전시설물 갯수, 혼잡강도, 시간대별 추정 교통량 등의 변수를 추가하는 전처리

2) 6개의 연령대와 2개의 교통사고 유형을 합친 12개의 그룹을 만들어서 그룹별 사고건수를 종속변수로 활용하여 위험지역을 도출

- OLS, 포아송, 지리적 가중 회귀, 지리적 가중 포아송 회귀 중 **지리적 가중 포아송 회귀 모델** 사용

- 위험 지역을 판별하는 기준으로 DDGI(Daejeon Danger Gid Index, 대전격자지수)를 활용

3) 도출된 위험지역을 사고의 빈번성, 규모성, 심각성에 적합한 변수를 활용하여 대안책 제시

### 최종분석을 위한 데이터 정제(by using R)

최종적으로 도출된 위험지역 100개에 대한 데이터 불러오기

```R
top100 <- read.csv('top_100_visu.csv')
dim(top100)
top100 %>% head(3)
```

![image](https://user-images.githubusercontent.com/97672187/166439766-e20da8ec-1d03-4693-a2c6-420ce2cf701d.png){: .align-center}

<br>


<br>

사망자, 중상자, 경상자 수를 합하여 '사상자 수'라는 변수 추가

```R
top100$사상자수 <- rowSums(top100[ ,64:66])
rank_df <- top100[,c(1,4,10,25,26)]
rank_df %>% head(3)
```

![image](https://user-images.githubusercontent.com/97672187/166445615-62cba1c0-2796-42e5-a35e-0ba29a72b451.png){: . align-center}

<br>


<br>

빈번성: 사고건수

심각성: 중상자이상수

규모성: 사상자수

빈번성, 심각성, 규모성에 해당되는 변수들을 추출하고, '저, 중, 고' 라는 값의 factor 형 변수 추가

```R
rank_df$심각성 <- cut(rank_df$중상자이상수, breaks = c(0,4,8,23), include.lowest = TRUE, right = FALSE, labels = c("저","중","고"))
rank_df$빈번성 <- cut(rank_df$사고건수, breaks = c(0,18,34,63), include.lowest = TRUE, right = FALSE, labels = c("저","중","고"))
rank_df$규모성 <- cut(rank_df$사상자수, breaks = c(0,11,25,51), include.lowest = TRUE, right = FALSE, labels = c("저","중","고"))
rank_df %>% head(3)
```

![image](https://user-images.githubusercontent.com/97672187/166445802-587833a2-86c4-4299-9952-090f5d4064ac.png){: . align-center}

<br>


<br>

연령대별 데이터도 고려하기 위하여 top100 데이터에서 연령과 관련된 변수도 추가

```R
rank_df2 <- cbind(rank_df,top100[,53:64])
rank_df2 %>% head(3)
```

![image](https://user-images.githubusercontent.com/97672187/166445862-52e29b64-6714-4f8d-82d8-d3c93623122f.png){: . align-center}

<br>


<br>

교통안전시설물과의 인과 관계를 고려하기 위하여 시설물과 관련된 변수도 추가

```R
rank_df3 <- cbind(rank_df2,top100[,names(top100)[c(27:29,42:45,47, 49)]])
names(rank_df3)
```

![image](https://user-images.githubusercontent.com/97672187/166445946-301f760e-c262-4e60-90e5-d81c096f15e0.png){: . align-center}

<br>


<br>

사고유형을 차대차, 차대사람으로 단순화시키기 위해 사고유형 변수를 자르기

```R
accident <- read.csv("1.대전광역시_교통사고내역(2017~2019).csv")
accident2 <- accident  %>% separate(col = 사고유형, sep = " ", into = c("사고유형2","a","사고유형3"))
accident2$a <- NULL
accident$사고유형2 <- accident2$사고유형2
accident$사고유형3 <- accident2$사고유형3
accident$사고유형3 <- ifelse(accident$사고유형 == "차대차 - 기타","기타2",
                         ifelse(accident$사고유형 == "차대사람 - 기타","기타1",
                                ifelse(accident$사고유형 == "차대단독 - 기타","기타3", accident$사고유형3)))

dim(accident)
accident %>% head(3)
```

![image](https://user-images.githubusercontent.com/97672187/166446478-019a96a2-5ced-46f7-ab30-a24f23b972aa.png){: . align-center}

<br>


<br>

위험지역 100개소에 해당하는 지역의 격자중 사고유형별로 사고건수를 계산해서 저장(accident2)

```R
accident[accident$gid %in% rank_df3$gid,]  %>% group_by(gid,사고유형) %>% count() %>% spread(사고유형,n) -> accident2
accident2[is.na(accident2)] <- 0
dim(accident2)
accident2 %>% head(3)
```

![image](https://user-images.githubusercontent.com/97672187/166446855-43c0dd13-1967-4291-8c9b-a85899685e90.png){: . align-center}

<br>


<br>

위험지역 100개소 격자의 사고유형별 사고건수를 DGI와 함께보기 위해 병합

```R
rank_df4 <- rank_df3 %>% left_join(accident2, by = "gid")
rank_df4 <- rank_df4[,c(-(39:42), -24)] # 차량단독 유형은 제거
dim(rank_df4)
rank_df4 %>% head(3)
```

![image](https://user-images.githubusercontent.com/97672187/166447529-1b9b04af-e670-40f0-8eba-d8989f06fc2e.png){: . align-center}

<br>


<br>

빈번성, 심각성, 규모성 변수에서 고가 2개 이상인 행만 추출

```R
rank_last <- rank_df4[rank_df4$gid %in% c('다바840114', '다바890176', '다바905178', '다바960132', '다바831220'), ]  
rank_last
```

![image](https://user-images.githubusercontent.com/97672187/166447637-8f535df9-078f-4c52-9a5d-4e83ed6a9948.png){: . align-center}

<br>


<br>

해당격자에 어떤 교통 안전시설물이 몇개 있는가를 파악 가능

```R
top100_count <- top100[,names(top100)[c(4,27:29,43:47,49)]] # 교통안전시설물 관련 변수들만 추출
top100_count %>% head()
```

![image](https://user-images.githubusercontent.com/97672187/166448210-5178efef-d245-4b72-a257-2c5f1699c1fc.png){: . align-center}

<br>


<br>

가장 심각하다고 판단되는 지역

```R
rank_last <- grid[grid$gid %in% rank_last$gid,]
rank_last.head()
```

![image](https://user-images.githubusercontent.com/97672187/166447869-ce85ff19-28e1-4d24-a0af-c3aea56ee28b.png){: . align-center}

<br>


<br>

6개의 심각지역 지도 시각화

```R
m <- leaflet() %>% addTiles() %>%
setView(lng = 127.390594, lat = 36.348315, zoom = 12) %>% 
addProviderTiles("CartoDB.Positron") %>%
addPolygons(data = base, color = "gray") %>% 
addCircles(data = rank_last,lng=~longitude, lat=~latitude, color='navy',radius = 500)
m
```

![image](https://user-images.githubusercontent.com/97672187/166448148-53645a9c-3cd2-4c4d-98cd-ca4d5cf668b5.png){: . align-center}

<br>


<br>


### 결론
1) 빈번성과 심각성이 상대적으로 더 높았던 지역 분석
100개의 위험 지역에서 사고 건수와 중상자 이상 수(사망자, 중상자 수의 합)가 상위 10%에 해당하는 지역을 각각 빈번성과 심각성이 높은 지역으로 선정하여 해당 지역의 위치 정보를
확인해보았다.

- 사고의 빈번성이 높았던 지역

![image](https://user-images.githubusercontent.com/97672187/166448439-cf15d589-dadf-499e-b8f5-c2c5a561465d.png){: .align-center}

사고가 자주 발생하면, 지자체적(거시적) 관점에서 사고를 관리하고 처리해야 하는 많은 비용이 발생한다. 따라서 사고가 빈번하게 발생하는 지역의 사고를 감소시킨다면, 
경제적 비용을 감소시킬 수 있다.


- 사고의 심각성이 높았던 지역

![image](https://user-images.githubusercontent.com/97672187/166448523-53872e59-b32c-412b-90f9-dd22d50e8591.png){: .align-center}

교통사고 건수는 거시적 관점에서 피해를 반영하지만, 개개인의 피해 정도(얼만큼 다쳤는지)를 반영하진 않는다. 따라서 심각한 사고가 발생되는 지역을 감소시킨다면, 
시민들의 정신적, 육체적 피해를 감소시킬 수 있다.

2) 많은 사고 건수가 발생한 빈번성이 높은 지역의 특성

![image](https://user-images.githubusercontent.com/97672187/166448790-8b7c7fc3-c58d-425f-8131-3671f5242cba.png){: .align-center}

전반적으로 차대사람 사고보다 차대차 사고가 더 빈번하게 일어났고, 20대의 경우에는 차대사람 사고가 차대차 사고보다 빈번하게 발생했다(왼쪽 그래프 하얀색 막대 참고)
사고 유형에서는 차대차 측면충돌 사고가 가장 많이 발생한 것을 볼 수 있다.


3) 중상자 및 사망자 수가 많이 발생한 심각성이 높은 지역의 특성

![image](https://user-images.githubusercontent.com/97672187/166449750-647f3ab4-007d-430c-8d0c-051f4b883844.png){: .align-center}

차대사람 사고보다 차대차 사고에서 심각성이 높았고, 차대사람 사고의 경우에는 60대 이상이 가장 심각성이 높은 것을 볼 수 있다.
사고 유형에서는 전반적으로 차대차 사고가 심각성이 높았고, 차대사람 사고의 경우에는 횡단중일때 가장 심각성이 높았다.


4) 빈번성과 심각성이 높은 지역에서의 교통안전시설물 현황


![image](https://user-images.githubusercontent.com/97672187/166450065-4e692ab1-c8b9-4245-8fe3-388a17348f9f.png){: .align-center}

빈번성이 높은 지역에서는 전반적으로 교통안전시설물이 일반 위험 지역보다 더 많은 것을 확인할 수 있었다.
심각성이 높은 지역에서도 전반적으로 교통안전시설물이 잘 설치 되었고, 특히 도로속도표시수는 빈번성이 높은 곳보다 심각성이 높은 곳에 잘 설치되어있다.

5) 심층분석

빈번성, 규모성, 심각성 중 고가 2개 이상에 해당되어서 가장 위험하다고 판단되는 6개의 지역에 대해 심층분석을 진행했다.
글이 너무 길어지기 때문에 모든 지역에 대해 설명하지 않고, 하나의 지역에 대해 예시로 설명해보자.

![image](https://user-images.githubusercontent.com/97672187/166450571-d86a885b-9154-41c2-9b63-a6b940fecdb1.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/166450613-d2cf91ce-0a79-4dfa-90cd-79629534818e.png){: .align-center}

해당 지역을 구글의 거리뷰를 활용하여 주변을 분석하고, 주어진 데이터로 교통 안전시설물 수를 파악했다. 그 결과 주변에 아파트 단지, 초등학교, 중학교가 있는
스쿨존이라는 것을 알게 되었고, 교통 안전시설물도 전반적으로 잘 설치가 된 것을 확인했다. 하지만 CCTV가 존재하지 않아서 차량이 신호를 어기면서 측면충돌, 정면충돌 등의
차대차 사고가 많이 발생했고, 스쿨존 지역이다보니 횡단중 차대사람 사고가 많이 발생한다는 사실을 알 수 있었다. 사거리 지역이라 차량 통행이 많고, 스쿨존이 있기 때문에
CCTV를 설치하거나 학교 측에서 등하교시에만이라도 횡단시 안전요원을 배치하는 방법이 대안책이 될 수 있을 것 같다.

다음과 같은 방법으로 6개지역에 대한 심층분석을 진행했다.

### 의의 및 한계
1) 의의

- 교통사고 위험 지역을 ‘교통사고’와 ‘위험 지역’ 각각의 속성을 활용하여 구체화 및 다각화 했다. 

- 교통사고 속성 정의를 위해서 차대차, 차대사람 사고를 유형화했다.

- 위험 지역 속성 정의를 위해서 빈번성, 규모성, 심각성 지표를 활용했다.

- 공간 데이터 성격을 가지고 있으므로 공간적 이질성을 충분히 고려한 지리적 가중 포아송 회귀 모형을 사용했다.

- 다양한 통계 방법을 토대로 위험 정도를 판단할 수 있는 정량화된 DDGI 지표를 만들었습니다.

2) 한계

- 교통안전시설물 외 요인으로 교통사고에 중요한 요인으로 고려되는 장소별 유동 인구 데이터를 고려하지 못했다.

- 교통안전시설물 설치 날짜에 대한 데이터가 없어서 해당 시설물이 교통사고에 유의미한 영향을 미치는지 확인하는데 어려움이 있었다.

- 도로가 존재하는 지역에 교통사고가 주로 발생한다고 가정하여 도로 부분 이외(아파트 단지 등)에 발생하는 사고를 분석에 적용하지 못했다.

- 지리적 가중 포아송 회귀 모형의 모델링 과정에서 독립변수를 구축 할 때, 해석의 어려움 때문에 교호작용항을 포함하지 않았다.

3) 개인적인 느낀점

5명의 사람들끼리 프로젝트를 진행했는데 각자 잘 사용할 수 있는 언어가 달라서 R과 파이썬이 따로 작업되었다. 다른 프로그래밍 언어로 인해 너무나 많은 파일들이 만들어져서 이것을 합치는 과정이
힘들었다. 또한, 프로젝트 특성상 사람마다 파트를 나눠서 작업을 했는데 나는 주로 전처리 파트에서 복잡하고 오래걸리는 일을 담당하고, 모델링 이후 시각화를 통한 인사이트 도출에
힘을 많이 쏟아서 처음 접해봤던 지리적 가중 회귀 모형 모델링에 어려움을 겪었다.

약 한달이라는 시간동안 프로젝트를 진행하면서 어려운 점도 많았지만, geojson이라는 새로운 데이터 형식도 접해보았고, R과 파이썬 모두를 사용하여 프로젝트를 진행해서 두 가지 언어에 대해
더 익숙해졌다. 무엇보다도 LH주관 전국규모 공모전에서 장려상이라는 상을 수상할 수 있어서 그동안의 고생이 모두 보람찼다. 모든 프로젝트에서 수상한건 아니지만, 앞으로도 내가했던 프로젝트들에
대해 꾸준히 정리해봐야겠다.




