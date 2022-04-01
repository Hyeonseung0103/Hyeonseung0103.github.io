---
layout: single
title:  "전처리 Part.7"
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
library(dplyr)
library(tidyr)
```

<pre>

Attaching package: ‘dplyr’


The following objects are masked from ‘package:stats’:

    filter, lag


The following objects are masked from ‘package:base’:

    intersect, setdiff, setequal, union


</pre>
## 현재까지 구축했던 데이터 불러오기



```python
accident_count_filter_23 <- read.csv('accident_count_filter_23.csv')
dim(accident_count_filter_23)
accident_count_filter_23 %>% head(2)
```

<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>5556</li><li>34</li></ol>


<table>
<caption>A data.frame: 2 × 34</caption>
<thead>
	<tr><th></th><th scope=col>X</th><th scope=col>gid</th><th scope=col>acci_cnt</th><th scope=col>geometry</th><th scope=col>사고건수</th><th scope=col>사상자수</th><th scope=col>x</th><th scope=col>y</th><th scope=col>신호등_보행자수</th><th scope=col>신호등_차량등수</th><th scope=col>⋯</th><th scope=col>안전지대수</th><th scope=col>중앙분리대수</th><th scope=col>정차금지지대수</th><th scope=col>도로속도표시수</th><th scope=col>교통안전표지수</th><th scope=col>노드개수</th><th scope=col>횡단보도수</th><th scope=col>건물면적</th><th scope=col>자동차대수</th><th scope=col>총거주인구수</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>⋯</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0</td><td>다바931203</td><td>2</td><td>MULTIPOLYGON (((127.4230710131166 36.38013455218083, 127.4230701251944 36.38103608833949, 127.4241850500282 36.38103680113275, 127.424185925082 36.38013526495075, 127.4230710131166 36.38013455218083)))</td><td>2</td><td>2</td><td>127.4236</td><td>36.38059</td><td>1</td><td>3</td><td>⋯</td><td>1</td><td>0</td><td>0</td><td>2</td><td>5</td><td>0</td><td>1</td><td>1291.19</td><td>409</td><td> 0</td></tr>
	<tr><th scope=row>2</th><td>1</td><td>다바861174</td><td>0</td><td>MULTIPOLYGON (((127.3450791441312 36.35391426501025, 127.3450773577309 36.35481580264442, 127.3461919054753 36.3548172424657, 127.346193679024 36.35391570478436, 127.3450791441312 36.35391426501025))) </td><td>0</td><td>0</td><td>127.3456</td><td>36.35437</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1159.69</td><td> 16</td><td>24</td></tr>
</tbody>
</table>


## 23. 교통사고내역 데이터를 활용하여 사고유형과 연령대를 현재까지 구축한 데이터와 병합



사고유형(차대차, 차대사람)과 연령대 변수(20대 미만, 20대, 30대, 40대, 50대, 60대 이상)를 모두 고려하기 위하여<br/>

각각의 데이터를 병합하여 **총 12개의 새로운 그룹을 만들고, 전체 데이터와 병합**하도록 한다.



```python
accident_list <- read.csv('1.대전광역시_교통사고내역(2017~2019).csv')
accident_list_sep <- accident_list %>% separate('사고유형', sep='-', into=c('사고유형메인', '사고유형디테일')) # 사고유형 변수를 종류와 디테일로 나누기
dim(accident_list_sep)
accident_list_sep %>% head(2)
```

<pre>
Warning message:
“Expected 2 pieces. Additional pieces discarded in 62 rows [618, 651, 1355, 1391, 1492, 1667, 1889, 2035, 2054, 2377, 3088, 3205, 3360, 3448, 3878, 4303, 4405, 5203, 6775, 7388, ...].”
</pre>
<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>23652</li><li>17</li></ol>


<table>
<caption>A data.frame: 2 × 17</caption>
<thead>
	<tr><th></th><th scope=col>사고일</th><th scope=col>시군구</th><th scope=col>사고유형메인</th><th scope=col>사고유형디테일</th><th scope=col>법규위반</th><th scope=col>사고내용</th><th scope=col>사망자수</th><th scope=col>중상자수</th><th scope=col>경상자수</th><th scope=col>부상신고자수</th><th scope=col>가해운전자.차종</th><th scope=col>가해운전자.연령대</th><th scope=col>가해운전자.성별</th><th scope=col>피해운전자.차종</th><th scope=col>피해운전자.연령대</th><th scope=col>피해운전자.성별</th><th scope=col>gid</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>2017-01-01</td><td>대전광역시 서구 용문동</td><td>차대차 </td><td> 측면충돌</td><td>안전운전불이행</td><td>중상사고</td><td>0</td><td>1</td><td>1</td><td>0</td><td>승용</td><td>50대</td><td>남</td><td>승용</td><td>60대</td><td>남</td><td>다바905151</td></tr>
	<tr><th scope=row>2</th><td>2017-01-01</td><td>대전광역시 서구 탄방동</td><td>차대차 </td><td> 추돌    </td><td>안전운전불이행</td><td>경상사고</td><td>0</td><td>0</td><td>1</td><td>0</td><td>승용</td><td>50대</td><td>남</td><td>승용</td><td>30대</td><td>남</td><td>다바905166</td></tr>
</tbody>
</table>



```python
accident_list_sep %>%
    group_by(사고유형메인, 피해운전자.연령대, gid) %>% # 사고유형과 연령대 변수에 대하여 격자별 개수 구하기
        summarise(n=n()) %>% ungroup() -> acci_list
dim(acci_list)
acci_list %>% head(2)
```

<pre>
`summarise()` regrouping output by '사고유형메인', '피해운전자.연령대' (override with `.groups` argument)

</pre>
<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>15767</li><li>4</li></ol>


<table>
<caption>A tibble: 2 × 4</caption>
<thead>
	<tr><th scope=col>사고유형메인</th><th scope=col>피해운전자.연령대</th><th scope=col>gid</th><th scope=col>n</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><td>차대사람 </td><td>10대</td><td>다바823162</td><td>1</td></tr>
	<tr><td>차대사람 </td><td>10대</td><td>다바827180</td><td>1</td></tr>
</tbody>
</table>



```python
acci_list <- acci_list %>% filter(사고유형메인 %in% c('차대사람 ', '차대차 ')) # 차대사람과 차대차에 해당하는 값만 추리기(차량단독 제외)
dim(acci_list)
acci_list %>% head(2)
```

<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>15145</li><li>4</li></ol>


<table>
<caption>A tibble: 2 × 4</caption>
<thead>
	<tr><th scope=col>사고유형메인</th><th scope=col>피해운전자.연령대</th><th scope=col>gid</th><th scope=col>n</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><td>차대사람 </td><td>10대</td><td>다바823162</td><td>1</td></tr>
	<tr><td>차대사람 </td><td>10대</td><td>다바827180</td><td>1</td></tr>
</tbody>
</table>



```python
# 피해자의 연령대는 미분류된 것을 제외하고, 20대 미만부터 60대 이상까지의 범주로 재정의
acci_list <- acci_list %>% filter(피해운전자.연령대 != '미분류') 
acci_list$피해운전자.연령대 <- as.character(acci_list$피해운전자.연령대)
acci_list$피해운전자.연령대 <- ifelse(acci_list$피해운전자.연령대 %in% c('60대', '70대', '80대', '90대'), '60대 이상', acci_list$피해운전자.연령대)
acci_list$피해운전자.연령대 <- ifelse(acci_list$피해운전자.연령대 %in% c('10대', '10대미만'), '20대 미만', acci_list$피해운전자.연령대)
dim(acci_list)
acci_list %>% head(2)
```

<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>15110</li><li>4</li></ol>


<table>
<caption>A tibble: 2 × 4</caption>
<thead>
	<tr><th scope=col>사고유형메인</th><th scope=col>피해운전자.연령대</th><th scope=col>gid</th><th scope=col>n</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><td>차대사람 </td><td>20대 미만</td><td>다바823162</td><td>1</td></tr>
	<tr><td>차대사람 </td><td>20대 미만</td><td>다바827180</td><td>1</td></tr>
</tbody>
</table>



```python
acci_list <- acci_list %>% unite(사고유형, 사고유형메인, 피해운전자.연령대, sep="-") # 사고유형과 연령대를 묶어주기
dim(acci_list)
acci_list %>% head(3)
```

<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>15110</li><li>3</li></ol>


<table>
<caption>A tibble: 3 × 3</caption>
<thead>
	<tr><th scope=col>사고유형</th><th scope=col>gid</th><th scope=col>n</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><td>차대사람 -20대 미만</td><td>다바823162</td><td>1</td></tr>
	<tr><td>차대사람 -20대 미만</td><td>다바827180</td><td>1</td></tr>
	<tr><td>차대사람 -20대 미만</td><td>다바828179</td><td>1</td></tr>
</tbody>
</table>



```python
acci_list %>%
    group_by(gid, 사고유형) %>% # 격자와 사고유형에 따라서 개수를 구하기
        summarise(n=sum(n)) %>%
            ungroup() -> acci_group
dim(acci_group)
acci_group %>% head(3)
```

<pre>
`summarise()` regrouping output by 'gid' (override with `.groups` argument)

</pre>
<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>14626</li><li>3</li></ol>


<table>
<caption>A tibble: 3 × 3</caption>
<thead>
	<tr><th scope=col>gid</th><th scope=col>사고유형</th><th scope=col>n</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><td>다바780093</td><td>차대차 -50대</td><td>1</td></tr>
	<tr><td>다바781090</td><td>차대차 -50대</td><td>1</td></tr>
	<tr><td>다바781091</td><td>차대차 -40대</td><td>1</td></tr>
</tbody>
</table>



```python
acci_spread <- acci_group %>% spread(사고유형, n, fill=0) # 위 데이터의 행과 열을 바꿔주기
dim(acci_spread)
acci_spread %>% head(2)
```

<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>5915</li><li>13</li></ol>


<table>
<caption>A tibble: 2 × 13</caption>
<thead>
	<tr><th scope=col>gid</th><th scope=col>차대사람 -20대</th><th scope=col>차대사람 -20대 미만</th><th scope=col>차대사람 -30대</th><th scope=col>차대사람 -40대</th><th scope=col>차대사람 -50대</th><th scope=col>차대사람 -60대 이상</th><th scope=col>차대차 -20대</th><th scope=col>차대차 -20대 미만</th><th scope=col>차대차 -30대</th><th scope=col>차대차 -40대</th><th scope=col>차대차 -50대</th><th scope=col>차대차 -60대 이상</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>다바780093</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td></tr>
	<tr><td>다바781090</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td></tr>
</tbody>
</table>



```python
accident_count_filter_23 <- accident_count_filter_23 %>% left_join(acci_spread, by='gid') # 기존 데이터와 병합

for(i in 34:45){
    accident_count_filter_23[,names(accident_count_filter_23)[i] ] <- ifelse(is.na(accident_count_filter_23[,names(accident_count_filter_23)[i] ]), 0, accident_count_filter_23[,names(accident_count_filter_23)[i] ])
}

dim(accident_count_filter_23)
accident_count_filter_23 %>% head(2)
```

<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>5556</li><li>46</li></ol>


<table>
<caption>A data.frame: 2 × 46</caption>
<thead>
	<tr><th></th><th scope=col>X</th><th scope=col>gid</th><th scope=col>acci_cnt</th><th scope=col>geometry</th><th scope=col>사고건수</th><th scope=col>사상자수</th><th scope=col>x</th><th scope=col>y</th><th scope=col>신호등_보행자수</th><th scope=col>신호등_차량등수</th><th scope=col>⋯</th><th scope=col>차대사람 -30대</th><th scope=col>차대사람 -40대</th><th scope=col>차대사람 -50대</th><th scope=col>차대사람 -60대 이상</th><th scope=col>차대차 -20대</th><th scope=col>차대차 -20대 미만</th><th scope=col>차대차 -30대</th><th scope=col>차대차 -40대</th><th scope=col>차대차 -50대</th><th scope=col>차대차 -60대 이상</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>⋯</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0</td><td>다바931203</td><td>2</td><td>MULTIPOLYGON (((127.4230710131166 36.38013455218083, 127.4230701251944 36.38103608833949, 127.4241850500282 36.38103680113275, 127.424185925082 36.38013526495075, 127.4230710131166 36.38013455218083)))</td><td>2</td><td>2</td><td>127.4236</td><td>36.38059</td><td>1</td><td>3</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td> 0</td></tr>
	<tr><th scope=row>2</th><td>1</td><td>다바861174</td><td>0</td><td>MULTIPOLYGON (((127.3450791441312 36.35391426501025, 127.3450773577309 36.35481580264442, 127.3461919054753 36.3548172424657, 127.346193679024 36.35391570478436, 127.3450791441312 36.35391426501025))) </td><td>0</td><td>0</td><td>127.3456</td><td>36.35437</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>NA</td></tr>
</tbody>
</table>



```python
write.csv(accident_count_filter_23, 'accident_count_filter_24.csv')
# 신호등(보행등), 신호등(차량등), cctv수, 혼잡빈도, 혼잡시간, 교통추정량, 날씨, 안전지대수, 중앙분리대수, 정차금지지대수
# 도로속도표시수, 교통안전표지수, 노드개수, 횡단보도수, 건물면적, 자동차대수, 총거주인구수, 연령 및 사고유형에 따라 나눈 데이터까지 merge한 상태.
```
