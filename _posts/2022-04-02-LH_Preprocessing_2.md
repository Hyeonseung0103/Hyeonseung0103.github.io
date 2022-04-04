---
layout: single
title: '전처리 Part.2'
toc: true
toc_sticky: true
category: LH
---

전처리 두번째 파트는 '국지적 모란지수'라는 개념을 사용하기 위해 R로 작업했다. 자세한 내용은 뒤에서 다뤄보도록 하자.

## 전처리 Part.2(by using R)
```R
# 패키지 불러오기
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
library(ggplot2)
library(geojsonio)
library(broom)
library(gridExtra)
```

사고건수 및 사상자수를 포함한 유의미한 교통격자와 전체 교통격자 데이터 불러오기
```R
accident_count <- read.csv('accident_count.csv', row.names = 1) # 이전 전처리에서 추출한 유의미한 교통격자 데이터
dim(accident_count)
accident_count %>% head(2)
```

![image](https://user-images.githubusercontent.com/97672187/161379431-26588b79-3319-4ed4-ad04-35e3f06d5b33.png){: .align-center}


<br>


</br>




```R
accident_grid <- read.csv('accident_grid.csv', row.names = 1) # 위도, 경도를 추가한 대전시 전체 교통격자 데이터
dim(accident_grid)
accident_grid %>% head(2)
```

![image](https://user-images.githubusercontent.com/97672187/161379523-691b1206-6ce8-4a82-9f46-eb4c08c45f8b.png){: .align-center}


<br>


</br>





### 4. '국지적 모란지수'를 통해 사고건수가 0인 값 중 의미없는 데이터를 제거
이전 포스팅의 전처리에서 사고건수가 0인 격자가 매우 많아서 데이터가 불균형하다는 것을 알 수 있었다. 이 불균형을 해소하기 위해 국지적 모란지수라는 개념을 활용했다.

국지적 모란지수(Local Moran's)는 **인접지역 간 속성 값의 수치적 유사성을 근거로 양의 방향으로 공간적 상관이 형성되는 것을 확인하는 통계량**이다.
여기서 
공간적 자기 상관이란, 한 위치에서 발생하는 사건과 그 주변지역에서 발생하는 사건과는 높은 상관관계를 보이는데 이는 공간상에 인접함으로서 나타나는 파급효과 때문이라는 것이다.
즉, 해당 사건이 발생한 위치에서 가까이 있을 수록 그 사건과 그 지역은 높은 상관관계를 보이게 된다.

이 국지적 모란지수를 통해 특정 지역이 전체 지역의 공간적 자기 상관에 얼마나 영향을 미치는지 확인이 가능하다. 
간단히 말하면, 모란지수가 클수록 해당 지역은 전체 지역의 공간적 자기 상관에 큰 영향을 미치는 중요한 지역이라는 것이다.

이 개념을 활용하여 교통사고라는 사건에 별로 중요하지 않은 지역을 제거함으로써 데이터 불균형을 해소할 수 있다.

```R
# 중심점으로부터 1.5km 반경의 이웃을 고려하여 이웃통계량을 계산, longlat=TRUE는 큰 원 거리를 계산하고 거리 단위로 킬로미터를 반환
S.dist <- spdep::dnearneigh(cbind(accident_count$x, accident_count$y), 0, 1.5, longlat=TRUE)
S.dist
```

![image](https://user-images.githubusercontent.com/97672187/161379636-f1d75fb3-b5a8-4ddd-8f31-b69832b4c207.png){: .align-center}




<br>


</br>



```R
spdep::set.ZeroPolicyOption(TRUE)
lw <- spdep::nb2listw(S.dist, style="W", zero.policy=TRUE)
lom <- spdep::localmoran(accident_count$`사고건수`, lw, zero.policy=TRUE)
pr <- data.frame(cbind(lom[,1], (accident_count$`사고건수` - mean(accident_count$`사고건수`)) / sd(accident_count$`사고건수`)))
plot(pr, xlab="Moran's I", ylab="Standarized num of accidents", pch=19)
abline(h=0, col="red")
abline(v=0, col="red")
title("Plot: Local Moran's I")
```

![image](https://user-images.githubusercontent.com/97672187/161379699-5e8cf1ed-5c09-4320-8335-9b9029d03fda.png){: .align-center}



<br>


</br>


위의 그래프를 보면 빨간색 선을 통해 1,2,3,4분면을 알 수 있다.

- 1사분면: 이웃 지역과 특성 및 속성이 유사하면서 사고가 평균보다 많이 발생하는 지역

- 2사분면: 이웃 지역과 특성 및 속성이 유사하지는 않지만, 사고가 평균보다 많이 발생하는 지역

- 3사분면: 이웃 지역과 특성 및 속성이 유사하지 않으면서 사고가 평균보다 적게 발생하는 지역

- 4사분면: 이웃 지역과 특성 및 속성이 유사하지만, 사고가 평균보다 적게 발생하는 지역

데이터 불균형을 해소하기 위해 특정 사분면의 데이터를 제거해야하는데 4사분면의 데이터를 제거하기로 했다.
그 이유는 4사분면은 이 구역의 사고건수의 Moran 통계량은 양의 값으로 특정 지역과 그 이웃 지역이 유사한 경향성을 갖는다고 할 수 있는데 표준화된 사고 건수는 평균보다 낮은 경우이다.
사고 건수가 0 또는 작은 관측치들이 이 구역에 유사한 경향성(군집)을 가지고 있을 것이라고  4사분면의 데이터를 제거한다.

```R
# 4사분면 부분이 아닌 부분을 필터링하여 최종 데이터 도출
pr_new <- pr[which(!pr$X1>0 | !pr$X2<0),]
accident_count_filter <- accident_count[as.numeric(rownames(pr_new)), ]
nrow(accident_count_filter)
table(accident_count_filter$`사고건수`) # 총 5556개
```

![image](https://user-images.githubusercontent.com/97672187/161380376-6d62feaa-5ff0-47cc-a467-2acf35b60145.png){: .align-center}


<br>


</br>


<br>


</br>




빨간색 지역이 사용할 수 있는(중요하지 않은 지역을 제거한) 유의미한 데이터이다.

```R
plot(accident_grid$x, accident_grid$y, col="green", pch=19, xlab="long", ylab="lat") # 전체 격자 데이터
lines(accident_count$x, accident_count$y, col="yellow", type="p", pch=19) # 사고 격자 데이터
lines(accident_count_filter$x, accident_count_filter$y, col = "red", type="p", pch=19) # 중요하지 않은 지역을 제거한 데이터
```


<br>


</br>





4사분면이 아닌 3사분면의 데이터를 제거했을 때 어떤 결과가 나오는지 비교해보자.

```R
# 대조하기 위하여 4사분면이 아닌 3사분면을 기준으로 필터링한 데이터
pr_new2 <- pr[which(pr$X1>0 | pr$X2>0), ]
accident_count_filter_2 <- accident_count[as.numeric(rownames(pr_new2)), ]
plot(accident_grid$x, accident_grid$y, col="green", pch=19, xlab="long", ylab="lat")
lines(accident_count$x, accident_count$y, col="yellow", type="p", pch=19)
lines(accident_count_filter_2$x, accident_count_filter_2$y, col = "red", type="p", pch=19)
table(accident_count_filter_2$`사고건수`)
```

![image](https://user-images.githubusercontent.com/97672187/161380492-d012a48a-ab94-470e-a605-7e295fbce819.png){: .align-center}

<br>


</br>


3사분면의 데이터는 이웃 지역과 특성 및 속성이 유사하지 않으면서 사고가 평균보다 적게 발생하는 지역이었다. 데이터 불균형을 처리하려는 목적이 사고건수가 0인 지역을
많이 제거하는 것이었는데 3사분면의 데이터를 제거했기 때문에 사고가 적은 지역과의 유사성이 약한 지역들이 제거가 되어서 여전히 불균형 문제가 존재하는 것으로 보인다.

4사분면의 데이터를 제거했을 때는 사고건수가 0인 데이터가 1500개로 줄었지만, 3사분면의 데이터를 제거하니까 0인 데이터가 5300개까지 밖에 줄지 않았다. 기존에는 
사고건수가 0인 데이터가 약 6900개정도였는데 5300개면 여전히 데이터 불균형 문제가 해소되지 않은 것이다.

밑의 결과를 통해 4사분면을 필터링한 지역이 데이터 불균형을 잘 해소할 것이라는 즉, 가장 유의미할 것이라는 가정이 충족된 것을 알 수 있다.

```R
table(accident_count$acci_cnt == 0) # 전체 데이터에서 0의 개수는 6959개로 절반 이상

except_4quadrant <- pr[which(!pr$X1 > 0 | !pr$X2 < 0),]
except_4quadrant_df <- accident_count[as.numeric(rownames(except_4quadrant)), ]
table(except_4quadrant_df$acci_cnt == 0) # 4사분면을 제외한 데이터에서 0의 개수는 1590개

except_3quadrant <- pr[which(!pr$X1 < 0 | !pr$X2 < 0),]
except_3quadrant_df <- accident_count[as.numeric(rownames(except_3quadrant)), ]
table(except_3quadrant_df$acci_cnt == 0) # 3사분면을 제외한 데이터에서 0의 개수는 5369개
```

![image](https://user-images.githubusercontent.com/97672187/161380843-5e810509-73e1-4221-87c4-3849ef0b9e4c.png){: .align-center}


<br>


</br>




위의 과정들을 통해 **기존의 54,912개의 격자를 최종적으로 5,556개의 고유한 격자 데이터**로 필터링했다.

```R
rownames(accident_count_filter) <- NULL # reset_index
dim(accident_count_filter)
accident_count_filter %>% head(2)
```

![image](https://user-images.githubusercontent.com/97672187/161380953-239af15f-1d98-4753-93df-2af1fed79789.png)


<br>


</br>





추려진 데이터를 시각화를 사용해 확인해보자.

```R
# 전체 격자와 4사분면의  제거했을 때의 격자비교
road_filter_map <- 
    ggplot() + geom_polygon(data=daejeon_tidy, aes(x=long, y =lat, group=group), fill='white', color='black', size=1) +
    coord_map() + theme_void() + geom_point(data = accident_count,aes(x= x, y= y),color= 'purple',size=2,, alpha=0.3)+
    scale_color_gradient(high='red', low='skyblue', guide_legend(title="Coefs for PSC_cnt"))

moran_filter_map <- 
    ggplot() + geom_polygon(data=daejeon_tidy, aes(x=long, y =lat, group=group), fill='white', color='black', size=1) +
    coord_map() + theme_void() + geom_point(data = accident_count_filter,aes(x= x, y= y),color= 'navy',size=2,, alpha=0.3)+
    scale_color_gradient(high='red', low='skyblue', guide_legend(title="Coefs for PSC_cnt"))

grid.arrange(road_filter_map, moran_filter_map,nrow = 1, ncol = 2)
```

![image](https://user-images.githubusercontent.com/97672187/161381015-76197153-e00b-4ff9-ba3a-7df6aee7f27a.png){: .align-center}




<br>


</br>




```R
#전체 격자와 3사분면의 데이터를 제거했을 때의 격자비교
road_filter_map2 <- 
    ggplot() + geom_polygon(data=daejeon_tidy, aes(x=long, y =lat, group=group), fill='white', color='black', size=1) +
    coord_map() + theme_void() + geom_point(data = accident_count,aes(x= x, y= y),color= 'purple',size=2,, alpha=0.3)+
    scale_color_gradient(high='red', low='skyblue', guide_legend(title="Coefs for PSC_cnt"))

moran_filter_map2 <- 
    ggplot() + geom_polygon(data=daejeon_tidy, aes(x=long, y =lat, group=group), fill='white', color='black', size=1) +
    coord_map() + theme_void() + geom_point(data = accident_count_filter_2,aes(x= x, y= y),color= 'navy',size=2,, alpha=0.3)+
    scale_color_gradient(high='red', low='skyblue', guide_legend(title="Coefs for PSC_cnt"))

grid.arrange(road_filter_map2, moran_filter_map2, nrow = 1, ncol = 2)
```

![image](https://user-images.githubusercontent.com/97672187/161381028-61034f5d-01d8-4d41-992d-f3a8f0a7cd5a.png){: .align-center}

