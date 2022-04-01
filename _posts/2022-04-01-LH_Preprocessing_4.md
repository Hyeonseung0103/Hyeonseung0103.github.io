---
layout: single
title:  "전처리 Part.4"
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
# install.packages("sp")
# install.packages("rgdal")
# install.packages("rgeos")
# install.packages("ggmap")
# install.packages("tmap")
# install.packages("raster")
# install.packages("spdep")
# install.packages("gstat")
# install.packages("spgwr")
# install.packages("GWmodel")

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

<pre>

Attaching package: ‘dplyr’


The following objects are masked from ‘package:stats’:

    filter, lag


The following objects are masked from ‘package:base’:

    intersect, setdiff, setequal, union


rgdal: version: 1.5-16, (SVN revision 1050)
Geospatial Data Abstraction Library extensions to R successfully loaded
Loaded GDAL runtime: GDAL 3.1.2, released 2020/07/07
Path to GDAL shared files: /usr/local/gdal-3.1.2/share/gdal
GDAL binary built with GEOS: FALSE 
Loaded PROJ runtime: Rel. 6.1.0, May 15th, 2019, [PJ_VERSION: 610]
Path to PROJ shared files: /usr/local/proj-6.1.0/share/proj
Linking to sp version:1.4-2
To mute warnings of possible GDAL/OSR exportToProj4() degradation,
use options("rgdal_show_exportToProj4_warnings"="none") before loading rgdal.

rgeos version: 0.5-3, (SVN revision 634)
 GEOS runtime version: 3.8.0-CAPI-1.13.1 
 Linking to sp version: 1.4-2 
 Polygon checking: TRUE 



Attaching package: ‘raster’


The following object is masked from ‘package:tidyr’:

    extract


The following object is masked from ‘package:dplyr’:

    select


Loading required package: spData

To access larger datasets in this package, install the spDataLarge
package with: `install.packages('spDataLarge',
repos='https://nowosad.github.io/drat/', type='source')`

Loading required package: sf

Linking to GEOS 3.8.0, GDAL 3.1.2, PROJ 6.1.0

NOTE: This package does not constitute approval of GWR
as a method of spatial analysis; see example(gwr)

Loading required package: maptools

Checking rgeos availability: TRUE

Loading required package: robustbase

Loading required package: Rcpp

Loading required package: spatialreg

Loading required package: Matrix


Attaching package: ‘Matrix’


The following objects are masked from ‘package:tidyr’:

    expand, pack, unpack


Registered S3 methods overwritten by 'spatialreg':
  method                   from 
  residuals.stsls          spdep
  deviance.stsls           spdep
  coef.stsls               spdep
  print.stsls              spdep
  summary.stsls            spdep
  print.summary.stsls      spdep
  residuals.gmsar          spdep
  deviance.gmsar           spdep
  coef.gmsar               spdep
  fitted.gmsar             spdep
  print.gmsar              spdep
  summary.gmsar            spdep
  print.summary.gmsar      spdep
  print.lagmess            spdep
  summary.lagmess          spdep
  print.summary.lagmess    spdep
  residuals.lagmess        spdep
  deviance.lagmess         spdep
  coef.lagmess             spdep
  fitted.lagmess           spdep
  logLik.lagmess           spdep
  fitted.SFResult          spdep
  print.SFResult           spdep
  fitted.ME_res            spdep
  print.ME_res             spdep
  print.lagImpact          spdep
  plot.lagImpact           spdep
  summary.lagImpact        spdep
  HPDinterval.lagImpact    spdep
  print.summary.lagImpact  spdep
  print.sarlm              spdep
  summary.sarlm            spdep
  residuals.sarlm          spdep
  deviance.sarlm           spdep
  coef.sarlm               spdep
  vcov.sarlm               spdep
  fitted.sarlm             spdep
  logLik.sarlm             spdep
  anova.sarlm              spdep
  predict.sarlm            spdep
  print.summary.sarlm      spdep
  print.sarlm.pred         spdep
  as.data.frame.sarlm.pred spdep
  residuals.spautolm       spdep
  deviance.spautolm        spdep
  coef.spautolm            spdep
  fitted.spautolm          spdep
  print.spautolm           spdep
  summary.spautolm         spdep
  logLik.spautolm          spdep
  print.summary.spautolm   spdep
  print.WXImpact           spdep
  summary.WXImpact         spdep
  print.summary.WXImpact   spdep
  predict.SLX              spdep


Attaching package: ‘spatialreg’


The following objects are masked from ‘package:spdep’:

    anova.sarlm, as_dgRMatrix_listw, as_dsCMatrix_I, as_dsCMatrix_IrW,
    as_dsTMatrix_listw, as.spam.listw, bptest.sarlm, can.be.simmed,
    cheb_setup, coef.gmsar, coef.sarlm, coef.spautolm, coef.stsls,
    create_WX, deviance.gmsar, deviance.sarlm, deviance.spautolm,
    deviance.stsls, do_ldet, eigen_pre_setup, eigen_setup, eigenw,
    errorsarlm, fitted.gmsar, fitted.ME_res, fitted.sarlm,
    fitted.SFResult, fitted.spautolm, get.ClusterOption,
    get.coresOption, get.mcOption, get.VerboseOption,
    get.ZeroPolicyOption, GMargminImage, GMerrorsar, griffith_sone,
    gstsls, Hausman.test, HPDinterval.lagImpact, impacts, intImpacts,
    Jacobian_W, jacobianSetup, l_max, lagmess, lagsarlm, lextrB,
    lextrS, lextrW, lmSLX, logLik.sarlm, logLik.spautolm, LR.sarlm,
    LR1.sarlm, LR1.spautolm, LU_prepermutate_setup, LU_setup,
    Matrix_J_setup, Matrix_setup, mcdet_setup, MCMCsamp, ME, mom_calc,
    mom_calc_int2, moments_setup, powerWeights, predict.sarlm,
    predict.SLX, print.gmsar, print.ME_res, print.sarlm,
    print.sarlm.pred, print.SFResult, print.spautolm, print.stsls,
    print.summary.gmsar, print.summary.sarlm, print.summary.spautolm,
    print.summary.stsls, residuals.gmsar, residuals.sarlm,
    residuals.spautolm, residuals.stsls, sacsarlm, SE_classic_setup,
    SE_interp_setup, SE_whichMin_setup, set.ClusterOption,
    set.coresOption, set.mcOption, set.VerboseOption,
    set.ZeroPolicyOption, similar.listw, spam_setup, spam_update_setup,
    SpatialFiltering, spautolm, spBreg_err, spBreg_lag, spBreg_sac,
    stsls, subgraph_eigenw, summary.gmsar, summary.sarlm,
    summary.spautolm, summary.stsls, trW, vcov.sarlm, Wald1.sarlm


Welcome to GWmodel version 2.2-3.
The new version of GWmodel 2.2-4 now is ready


Attaching package: ‘GWmodel’


The following objects are masked from ‘package:stats’:

    BIC, fitted


</pre>
## 연도별 weather 데이터 불러오기



세계기상기구(WMO)는 기온, 강수량 등의 기후요소가 평년값에 비해 현저히 높거나 낮은 수치(90 퍼센타일 또는 10 퍼센타일 미만의 범위)를 이상기후로 정의하고 있다.<br/>

이 기준에 따라서 2017~2019년의 weather 데이터의 이상치를 분류하도록 한다.



- 기온과 풍속, 습도, 지면 온도의 데이터 전처리: 1년 별로 비교해서 이상치(하위 10% 미만, 상위 10% 초과) 안에 포함되는 경우 1, 포함되지 않는 경우에는 0

- 강수량, 적설량 데이터 전처리: 당해 연도의 상위 10% 백분위 수에 포함되는 경우 1, 포함되지 않는 경우에는 0

- 안개 데이터 전처리: 안개는 지속 시간이 조금이라도 있으면 1, 없으면 0


### 2017년의 weather 데이터 처리



```python
weather_2017 <- read.csv('weather_2017.csv')
```


```python
temp_average <- c()
qt <- quantile(weather_2017[, names(weather_2017)[3]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_average <- ifelse(weather_2017[, names(weather_2017)[3]] < qt[2] | weather_2017[, names(weather_2017)[3]] > qt[3], 1, 0)

temp_min <- c()
qt <- quantile(weather_2017[, names(weather_2017)[4]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_min <- ifelse(weather_2017[, names(weather_2017)[4]] < qt[2] | weather_2017[, names(weather_2017)[4]] > qt[3], 1, 0)

temp_max <- c()
qt <- quantile(weather_2017[, names(weather_2017)[5]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_max <- ifelse(weather_2017[, names(weather_2017)[5]] < qt[2] | weather_2017[, names(weather_2017)[5]] > qt[3], 1, 0)

wind_max <- c()
qt <- quantile(weather_2017[, names(weather_2017)[7]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
wind_max <- ifelse(weather_2017[, names(weather_2017)[7]] < qt[2] | weather_2017[, names(weather_2017)[7]] > qt[3], 1, 0)

wind_average <- c()
qt <- quantile(weather_2017[, names(weather_2017)[9]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
wind_average <- ifelse(weather_2017[, names(weather_2017)[9]] < qt[2] | weather_2017[, names(weather_2017)[9]] > qt[3], 1, 0)

humidity_average <- c()
qt <- quantile(weather_2017[, names(weather_2017)[10]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
humidity_average <- ifelse(weather_2017[, names(weather_2017)[10]] < qt[2] | weather_2017[, names(weather_2017)[10]] > qt[3], 1, 0)

road_average <- c()
qt <- quantile(weather_2017[, names(weather_2017)[12]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
road_average <- ifelse(weather_2017[, names(weather_2017)[12]] < qt[2] | weather_2017[, names(weather_2017)[12]] > qt[3], 1, 0)
```


```python
weather_2017$평균온도 <- temp_average
weather_2017$최저온도 <- temp_min
weather_2017$최고온도 <- temp_max
weather_2017$최대풍속 <- wind_max
weather_2017$평균풍속 <- wind_average
weather_2017$평균습도 <- humidity_average
weather_2017$평균지면온도 <- road_average
```


```python
# NA 값은 이상치가 없는 경우이므로 0으로 처리
weather_2017[, names(weather_2017)[6]] <- ifelse(is.na(weather_2017[, names(weather_2017)[6]]), 0, weather_2017[, names(weather_2017)[6]])
weather_2017[, names(weather_2017)[11]] <- ifelse(is.na(weather_2017[, names(weather_2017)[11]]), 0, weather_2017[, names(weather_2017)[11]])
```


```python
day_rain <- c()
qt <- quantile(weather_2017[, names(weather_2017)[6]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
day_rain <- ifelse(weather_2017[, names(weather_2017)[6]] > qt[3], 1, 0)

day_snow <- c()
qt <- quantile(weather_2017[, names(weather_2017)[11]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
day_snow <- ifelse(weather_2017[, names(weather_2017)[11]] > qt[3], 1, 0)
```


```python
weather_2017$강수량 <- day_rain
weather_2017$적설량 <- day_snow
weather_2017$안개시간 <- ifelse( is.na(weather_2017[, names(weather_2017)[13]]), 0, 1)
```


```python
weather_2017_arrange <- weather_2017[,c(2,14:ncol(weather_2017))] 
dim(weather_2017_arrange)
weather_2017_arrange %>% head()
```

<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>365</li><li>11</li></ol>


<table>
<caption>A data.frame: 6 × 11</caption>
<thead>
	<tr><th></th><th scope=col>일시</th><th scope=col>평균온도</th><th scope=col>최저온도</th><th scope=col>최고온도</th><th scope=col>최대풍속</th><th scope=col>평균풍속</th><th scope=col>평균습도</th><th scope=col>평균지면온도</th><th scope=col>강수량</th><th scope=col>적설량</th><th scope=col>안개시간</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>2017-01-01</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>2</th><td>2017-01-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>3</th><td>2017-01-03</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>4</th><td>2017-01-04</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>5</th><td>2017-01-05</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>6</th><td>2017-01-06</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
</tbody>
</table>


### 2018년의 weather 데이터 처리



```python
weather_2018 <- read.csv('weather_2018.csv')
```


```python
temp_average <- c()
qt <- quantile(weather_2018[, names(weather_2018)[3]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_average <- ifelse(weather_2018[, names(weather_2018)[3]] < qt[2] | weather_2018[, names(weather_2018)[3]] > qt[3], 1, 0)

temp_min <- c()
qt <- quantile(weather_2018[, names(weather_2018)[4]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_min <- ifelse(weather_2018[, names(weather_2018)[4]] < qt[2] | weather_2018[, names(weather_2018)[4]] > qt[3], 1, 0)

temp_max <- c()
qt <- quantile(weather_2018[, names(weather_2018)[5]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_max <- ifelse(weather_2018[, names(weather_2018)[5]] < qt[2] | weather_2018[, names(weather_2018)[5]] > qt[3], 1, 0)

wind_max <- c()
qt <- quantile(weather_2018[, names(weather_2018)[7]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
wind_max <- ifelse(weather_2018[, names(weather_2018)[7]] < qt[2] | weather_2018[, names(weather_2018)[7]] > qt[3], 1, 0)

wind_average <- c()
qt <- quantile(weather_2018[, names(weather_2018)[9]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
wind_average <- ifelse(weather_2018[, names(weather_2018)[9]] < qt[2] | weather_2018[, names(weather_2018)[9]] > qt[3], 1, 0)

humidity_average <- c()
qt <- quantile(weather_2018[, names(weather_2018)[10]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
humidity_average <- ifelse(weather_2018[, names(weather_2018)[10]] < qt[2] | weather_2018[, names(weather_2018)[10]] > qt[3], 1, 0)

road_average <- c()
qt <- quantile(weather_2018[, names(weather_2018)[12]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
road_average <- ifelse(weather_2018[, names(weather_2018)[12]] < qt[2] | weather_2018[, names(weather_2018)[12]] > qt[3], 1, 0)
```


```python
weather_2018$평균온도 <- temp_average
weather_2018$최저온도 <- temp_min
weather_2018$최고온도 <- temp_max
weather_2018$최대풍속 <- wind_max
weather_2018$평균풍속 <- wind_average
weather_2018$평균습도 <- humidity_average
weather_2018$평균지면온도 <- road_average
```


```python
weather_2018[, names(weather_2018)[6]] <- ifelse(is.na(weather_2018[, names(weather_2018)[6]]), 0, weather_2018[, names(weather_2018)[6]])
weather_2018[, names(weather_2018)[11]] <- ifelse(is.na(weather_2018[, names(weather_2018)[11]]), 0, weather_2018[, names(weather_2018)[11]])
```


```python
day_rain <- c()
qt <- quantile(weather_2018[, names(weather_2018)[6]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
day_rain <- ifelse(weather_2018[, names(weather_2018)[6]] > qt[3], 1, 0)

day_snow <- c()
qt <- quantile(weather_2018[, names(weather_2018)[11]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
day_snow <- ifelse(weather_2018[, names(weather_2018)[11]] > qt[3], 1, 0)
```


```python
weather_2018$강수량 <- day_rain
weather_2018$적설량 <- day_snow
weather_2018$안개시간 <- ifelse( is.na(weather_2018[, names(weather_2018)[13]]), 0, 1)
```


```python
weather_2018_arrange <- weather_2018[,c(2,14:ncol(weather_2018))]
dim(weather_2018_arrange)
weather_2018_arrange %>% head()
```

<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>365</li><li>11</li></ol>


<table>
<caption>A data.frame: 6 × 11</caption>
<thead>
	<tr><th></th><th scope=col>일시</th><th scope=col>평균온도</th><th scope=col>최저온도</th><th scope=col>최고온도</th><th scope=col>최대풍속</th><th scope=col>평균풍속</th><th scope=col>평균습도</th><th scope=col>평균지면온도</th><th scope=col>강수량</th><th scope=col>적설량</th><th scope=col>안개시간</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>2018-01-01</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>2</th><td>2018-01-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>3</th><td>2018-01-03</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>4</th><td>2018-01-04</td><td>1</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>5</th><td>2018-01-05</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>6</th><td>2018-01-06</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
</tbody>
</table>


### 2019년의 weather 데이터 처리



```python
weather_2019 <- read.csv('weather_2019.csv')
```


```python
qt <- quantile(weather_2019[, names(weather_2017)[3]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_average <- ifelse(weather_2019[, names(weather_2019)[3]] < qt[2] | weather_2019[, names(weather_2019)[3]] > qt[3], 1, 0)

temp_min <- c()
qt <- quantile(weather_2019[, names(weather_2019)[4]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_min <- ifelse(weather_2019[, names(weather_2019)[4]] < qt[2] | weather_2019[, names(weather_2019)[4]] > qt[3], 1, 0)

temp_max <- c()
qt <- quantile(weather_2019[, names(weather_2019)[5]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
temp_max <- ifelse(weather_2019[, names(weather_2019)[5]] < qt[2] | weather_2019[, names(weather_2019)[5]] > qt[3], 1, 0)

wind_max <- c()
qt <- quantile(weather_2019[, names(weather_2019)[7]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
wind_max <- ifelse(weather_2019[, names(weather_2019)[7]] < qt[2] | weather_2019[, names(weather_2019)[7]] > qt[3], 1, 0)

wind_average <- c()
qt <- quantile(weather_2019[, names(weather_2019)[9]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
wind_average <- ifelse(weather_2019[, names(weather_2019)[9]] < qt[2] | weather_2019[, names(weather_2019)[9]] > qt[3], 1, 0)

humidity_average <- c()
qt <- quantile(weather_2019[, names(weather_2019)[10]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
humidity_average <- ifelse(weather_2019[, names(weather_2019)[10]] < qt[2] | weather_2019[, names(weather_2019)[10]] > qt[3], 1, 0)

road_average <- c()
qt <- quantile(weather_2019[, names(weather_2019)[12]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
road_average <- ifelse(weather_2019[, names(weather_2019)[12]] < qt[2] | weather_2019[, names(weather_2019)[12]] > qt[3], 1, 0)
```


```python
weather_2019$평균온도 <- temp_average
weather_2019$최저온도 <- temp_min
weather_2019$최고온도 <- temp_max
weather_2019$최대풍속 <- wind_max
weather_2019$평균풍속 <- wind_average
weather_2019$평균습도 <- humidity_average
weather_2019$평균지면온도 <- road_average
```


```python
weather_2019[, names(weather_2019)[6]] <- ifelse(is.na(weather_2019[, names(weather_2019)[6]]), 0, weather_2019[, names(weather_2019)[6]])
weather_2019[, names(weather_2019)[11]] <- ifelse(is.na(weather_2019[, names(weather_2019)[11]]), 0, weather_2019[, names(weather_2019)[11]])
```


```python
day_rain <- c()
qt <- quantile(weather_2019[, names(weather_2019)[6]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
day_rain <- ifelse(weather_2019[, names(weather_2019)[6]] > qt[3], 1, 0)

day_snow <- c()
qt <- quantile(weather_2019[, names(weather_2019)[11]], probs=c(0, 0.1, 0.9, 1), na.rm=T)
day_snow <- ifelse(weather_2019[, names(weather_2019)[11]] > qt[3], 1, 0)
```


```python
weather_2019$강수량 <- day_rain
weather_2019$적설량 <- day_snow
weather_2019$안개시간 <- ifelse( is.na(weather_2019[, names(weather_2019)[13]]), 0, 1)
```


```python
weather_2019_arrange <- weather_2019[,c(2, 14:ncol(weather_2019))]
dim(weather_2019_arrange)
weather_2019_arrange %>% head()
```

<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>365</li><li>11</li></ol>


<table>
<caption>A data.frame: 6 × 11</caption>
<thead>
	<tr><th></th><th scope=col>일시</th><th scope=col>평균온도</th><th scope=col>최저온도</th><th scope=col>최고온도</th><th scope=col>최대풍속</th><th scope=col>평균풍속</th><th scope=col>평균습도</th><th scope=col>평균지면온도</th><th scope=col>강수량</th><th scope=col>적설량</th><th scope=col>안개시간</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>2019-01-01</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>2</th><td>2019-01-02</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>3</th><td>2019-01-03</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>4</th><td>2019-01-04</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>5</th><td>2019-01-05</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>6</th><td>2019-01-06</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td></tr>
</tbody>
</table>



```python
weather_arrange <- bind_rows(weather_2017_arrange, weather_2018_arrange, weather_2019_arrange)
dim(weather_arrange)
weather_arrange %>% head()
```

<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>1095</li><li>11</li></ol>


<table>
<caption>A data.frame: 6 × 11</caption>
<thead>
	<tr><th></th><th scope=col>일시</th><th scope=col>평균온도</th><th scope=col>최저온도</th><th scope=col>최고온도</th><th scope=col>최대풍속</th><th scope=col>평균풍속</th><th scope=col>평균습도</th><th scope=col>평균지면온도</th><th scope=col>강수량</th><th scope=col>적설량</th><th scope=col>안개시간</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>2017-01-01</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>2</th><td>2017-01-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>3</th><td>2017-01-03</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>4</th><td>2017-01-04</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>5</th><td>2017-01-05</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>6</th><td>2017-01-06</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
</tbody>
</table>



```python
write.csv(weather_arrange, file='weather_arrange.csv', row.names=F)
```
