---
layout: single
title: "Modeling Part.3"
toc: true
toc_sticky: true
category: LH
---

지리적 데이터에 적합하면서 이산형 자료를 잘 예측할 수 있는 지리적 가중 포아송 회귀 모델을 사용하여 사고유형별, 연령대별 사고건수를 예측해보자.

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

## Modeling Part.3(by using R)

### 4. GWPR(Geographically Weighted Poisson Regression, 지리적 가중 포아송회귀) 모델링

#### GWPR 모델 만들기

```R
#패키지 GWmodels의 ggwr.basic 함수를 수정-일반화 선형 지리적 가중 회귀모형을 도출하는 함수가 AIC를 전통적인 log-likelihood에 기반하지 않고 
#Deviance에 기반하여 측정하여 이 함수의 소스 코드를 찾아 AIC를 다른 회귀 모형과 비교할 수 있게 log-likelihood 기반으로 수정하였다.
ggwr.basic2<-function(formula, data, regression.points, bw, family ="poisson", kernel="bisquare",
                      adaptive=FALSE, cv=T, tol=1.0e-5, maxiter=20, p=2, theta=0, longlat=TRUE, dMat,dMat1)
{
  ##Record the start time
  timings <- list()
  timings[["start"]] <- Sys.time()
  ###################################macth the variables
  this.call <- match.call()
  p4s <- as.character(NA)
  #####Check the given data frame and regression points
  #####Regression points
  if (missing(regression.points))
  {
    rp.given <- FALSE
    regression.points <- data
    hatmatrix<-T
  }
  else
  {
    rp.given <- TRUE
    hatmatrix<-F
  }
  ##Data points{
  if (is(data, "Spatial"))
  {
    p4s <- proj4string(data)
    dp.locat<-coordinates(data)
    data <- as(data, "data.frame")
  }
  else
  {
    stop("Given regression data must be Spatial*DataFrame")
  }
  
  ####################
  ######Extract the data frame
  ####Refer to the function lm
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data"), names(mf), 0L)
  
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- as.name("model.frame")
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms")
  y <- model.extract(mf, "response")
  x <- model.matrix(mt, mf)
  ############################################
  var.n<-ncol(x)
  if(is(regression.points, "Spatial"))
    rp.locat<-coordinates(regression.points)
  else if(is.numeric(regression.points)&&dim(regression.points)[2]==2)
  {
    rp.locat <- regression.points
  }
  else
    stop("Please use the correct regression points for model calibration!")
  
  rp.n<-nrow(rp.locat)
  dp.n<-nrow(data)
  betas <-matrix(nrow=rp.n, ncol=var.n)
  betas1<- betas
  if(hatmatrix)
  {
    betas.SE <-matrix(nrow=rp.n, ncol=var.n)
    betas.TV <-matrix(nrow=rp.n, ncol=var.n)
    ##S: hatmatrix
    S<-matrix(nrow=dp.n,ncol=dp.n)
  }
  #C.M<-matrix(nrow=dp.n,ncol=dp.n)
  idx1 <- match("(Intercept)", colnames(x))
  if(!is.na(idx1))
    colnames(x)[idx1]<-"Intercept"
  colnames(betas) <- colnames(x)
  #colnames(betas)[1]<-"Intercept"
  ####################################################GWR
  #########Distance matrix is given or not
  
  if (missing(dMat))
  {
    DM.given<-F
    if(dp.n + rp.n <= 10000)
    {
      dMat <- gw.dist(dp.locat=dp.locat, rp.locat=rp.locat, p=p, theta=theta, longlat=longlat)
      DM.given<-T
    }
  }
  else
  {
    DM.given<-T
    dim.dMat<-dim(dMat)
    if (dim.dMat[1]!=dp.n||dim.dMat[2]!=rp.n)
      stop("Dimensions of dMat are not correct")
  }
  if(missing(dMat1))
  {
    DM1.given<-F
    if(hatmatrix&&DM.given)
    {
      dMat1 <- dMat
      DM1.given<-T
    }
    else
    {
      if(dp.n < 8000)
      {
        dMat1 <- gw.dist(dp.locat=dp.locat, rp.locat=dp.locat, p=p, theta=theta, longlat=longlat)
        DM1.given<-T
      }
    }
  }
  else
  {
    DM1.given<-T
    dim.dMat1<-dim(dMat1)
    if (dim.dMat1[1]!=dp.n||dim.dMat1[2]!=dp.n)
      stop("Dimensions of dMat are not correct")
  }
  ####Generate the weighting matrix
  #############Calibration the model
  W1.mat<-matrix(numeric(dp.n*dp.n),ncol=dp.n)
  W2.mat<-matrix(numeric(dp.n*rp.n),ncol=rp.n)
  for (i in 1:dp.n)
  {
    if (DM1.given)
      dist.vi<-dMat1[,i]
    else
    {
      dist.vi<-gw.dist(dp.locat=dp.locat, focus=i, p=p, theta=theta, longlat=longlat)
    }
    W.i<-gw.weight(dist.vi,bw,kernel,adaptive)
    W1.mat[,i]<-W.i
  }
  if (rp.given)
  {
    for (i in 1:rp.n)
    {
      if (DM.given)
        dist.vi<-dMat[,i]
      else
      {
        dist.vi<-gw.dist(dp.locat, rp.locat, focus=i, p, theta, longlat)
      }
      W.i<-gw.weight(dist.vi,bw,kernel,adaptive)
      W2.mat[,i]<-W.i
    }
  }
  else
    W2.mat<-W1.mat
  
  ##model calibration
  if(family=="poisson")
    res1<-gwr.poisson(y,x,regression.points,W1.mat,W2.mat,hatmatrix,tol, maxiter)
  if(family=="binomial")
    res1<-gwr.binomial(y,x,regression.points,W1.mat,W2.mat,hatmatrix,tol, maxiter)
  ####################################
  CV <- numeric(dp.n)
  if(hatmatrix && cv)
  {
    CV <- ggwr.cv.contrib(bw, x, y,family, kernel,adaptive, dp.locat, p, theta, longlat,dMat)
  }
  ####encapsulate the GWR results
  GW.arguments<-list()
  GW.arguments<-list(formula=formula,rp.given=rp.given,hatmatrix=hatmatrix,bw=bw, family=family,
                     kernel=kernel,adaptive=adaptive, p=p, theta=theta, longlat=longlat,DM.given=DM1.given)
  
  timings[["stop"]] <- Sys.time()
  ##############
  res<-list(GW.arguments=GW.arguments,GW.diagnostic=res1$GW.diagnostic,glms=res1$glms,SDF=res1$SDF,CV=CV,timings=timings,this.call=this.call)
  class(res) <-"ggwrm"
  invisible(res) 
}

# This version of this function is kept to make the code work with the early versions of GWmodel (before 2.0-1)
gwr.generalised<-function(formula, data, regression.points, bw, family ="poisson", kernel="bisquare",
                          adaptive=FALSE, cv=T, tol=1.0e-5, maxiter=20, p=2, theta=0, longlat=F, dMat,dMat1)
{
  ##Record the start time
  timings <- list()
  timings[["start"]] <- Sys.time()
  ###################################macth the variables
  this.call <- match.call()
  p4s <- as.character(NA)
  #####Check the given data frame and regression points
  #####Regression points
  if (missing(regression.points))
  {
    rp.given <- FALSE
    regression.points <- data
    hatmatrix<-T
  }
  else
  {
    rp.given <- TRUE
    hatmatrix<-F
  }
  ##Data points{
  if (is(data, "Spatial"))
  {
    p4s <- proj4string(data)
    dp.locat<-coordinates(data)
    data <- as(data, "data.frame")
  }
  else
  {
    stop("Given regression data must be Spatial*DataFrame")
  }
  
  ####################
  ######Extract the data frame
  ####Refer to the function lm
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data"), names(mf), 0L)
  
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- as.name("model.frame")
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms")
  y <- model.extract(mf, "response")
  x <- model.matrix(mt, mf)
  ############################################
  var.n<-ncol(x)
  if(is(regression.points, "Spatial"))
    rp.locat<-coordinates(regression.points)
  else if(is.numeric(regression.points)&&dim(regression.points)[2]==2)
  {
    rp.locat <- regression.points
  }
  else
    stop("Please use the correct regression points for model calibration!")
  
  rp.n<-nrow(rp.locat)
  dp.n<-nrow(data)
  betas <-matrix(nrow=rp.n, ncol=var.n)
  betas1<- betas
  if(hatmatrix)
  {
    betas.SE <-matrix(nrow=rp.n, ncol=var.n)
    betas.TV <-matrix(nrow=rp.n, ncol=var.n)
    ##S: hatmatrix
    S<-matrix(nrow=dp.n,ncol=dp.n)
  }
  #C.M<-matrix(nrow=dp.n,ncol=dp.n)
  idx1 <- match("(Intercept)", colnames(x))
  if(!is.na(idx1))
    colnames(x)[idx1]<-"Intercept"
  colnames(betas) <- colnames(x)
  #colnames(betas)[1]<-"Intercept"
  ####################################################GWR
  #########Distance matrix is given or not
  
  if (missing(dMat))
  {
    DM.given<-F
    if(dp.n + rp.n <= 10000)
    {
      dMat <- gw.dist(dp.locat=dp.locat, rp.locat=rp.locat, p=p, theta=theta, longlat=longlat)
      DM.given<-T
    }
  }
  else
  {
    DM.given<-T
    dim.dMat<-dim(dMat)
    if (dim.dMat[1]!=dp.n||dim.dMat[2]!=rp.n)
      stop("Dimensions of dMat are not correct")
  }
  if(missing(dMat1))
  {
    DM1.given<-F
    if(hatmatrix&&DM.given)
    {
      dMat1 <- dMat
      DM1.given<-T
    }
    else
    {
      if(dp.n < 8000)
      {
        dMat1 <- gw.dist(dp.locat=dp.locat, rp.locat=dp.locat, p=p, theta=theta, longlat=longlat)
        DM1.given<-T
      }
    }
  }
  else
  {
    DM1.given<-T
    dim.dMat1<-dim(dMat1)
    if (dim.dMat1[1]!=dp.n||dim.dMat1[2]!=dp.n)
      stop("Dimensions of dMat are not correct")
  }
  ####Generate the weighting matrix
  #############Calibration the model
  W1.mat<-matrix(numeric(dp.n*dp.n),ncol=dp.n)
  W2.mat<-matrix(numeric(dp.n*rp.n),ncol=rp.n)
  for (i in 1:dp.n)
  {
    if (DM1.given)
      dist.vi<-dMat1[,i]
    else
    {
      dist.vi<-gw.dist(dp.locat=dp.locat, focus=i, p=p, theta=theta, longlat=longlat)
    }
    W.i<-gw.weight(dist.vi,bw,kernel,adaptive)
    W1.mat[,i]<-W.i
  }
  if (rp.given)
  {
    for (i in 1:rp.n)
    {
      if (DM.given)
        dist.vi<-dMat[,i]
      else
      {
        dist.vi<-gw.dist(dp.locat, rp.locat, focus=i, p, theta, longlat)
      }
      W.i<-gw.weight(dist.vi,bw,kernel,adaptive)
      W2.mat[,i]<-W.i
    }
  }
  else
    W2.mat<-W1.mat
  
  ##model calibration
  if(family=="poisson")
    res1<-gwr.poisson(y,x,regression.points,W1.mat,W2.mat,hatmatrix,tol, maxiter)
  if(family=="binomial")
    res1<-gwr.binomial(y,x,regression.points,W1.mat,W2.mat,hatmatrix,tol, maxiter)
  ####################################
  CV <- numeric(dp.n)
  if(hatmatrix && cv)
  {
    CV <- ggwr.cv.contrib(bw, x, y,family, kernel,adaptive, dp.locat, p, theta, longlat,dMat)
  }
  ####encapsulate the GWR results
  GW.arguments<-list()
  GW.arguments<-list(formula=formula,rp.given=rp.given,hatmatrix=hatmatrix,bw=bw, family=family,
                     kernel=kernel,adaptive=adaptive, p=p, theta=theta, longlat=longlat,DM.given=DM1.given)
  
  timings[["stop"]] <- Sys.time()
  ##############
  res<-list(GW.arguments=GW.arguments,GW.diagnostic=res1$GW.diagnostic,glms=res1$glms,SDF=res1$SDF,CV=CV,timings=timings,this.call=this.call,yhat_GWPR=yhat,residuals=residual)
  class(res) <-"ggwrm"
  invisible(res)
}



############ Possipon GWGLM
gwr.poisson<-function(y,x,regression.points,W1.mat,W2.mat,hatmatrix,tol=1.0e-5, maxiter=500)
{
  p4s <- as.character(NA)
  if (is(regression.points, "Spatial"))
  {
    p4s <- proj4string(regression.points)
  }
  ############################################
  ##Generalized linear regression
  glms<-glm.fit(x, y, family = poisson()) 
  null.dev <- glms$null.deviance
  glm.dev <-glms$deviance
  glm.pseudo.r2 <- 1- glm.dev/null.dev 
  glms$pseudo.r2 <- glm.pseudo.r2
  var.n<-ncol(x)
  dp.n<-nrow(x)
  ########change the aic
  #glms$aic <- glm.dev + 2*var.n
  #glms$aicc <- glm.dev + 2*var.n + 2*var.n*(var.n+1)/(dp.n-var.n-1)
  ############################################
  if(is(regression.points, "Spatial"))
    rp.locat<-coordinates(regression.points)
  else
    rp.locat <- regression.points
  rp.n<-nrow(rp.locat)
  betas <- matrix(nrow=rp.n, ncol=var.n)
  betas1<- matrix(nrow=dp.n, ncol=var.n)
  betas.SE <-matrix(nrow=dp.n, ncol=var.n)
  betas.TV <-matrix(nrow=dp.n, ncol=var.n)
  ##S: hatmatrix
  S<-matrix(nrow=dp.n,ncol=dp.n)
  #C.M<-matrix(nrow=dp.n,ncol=dp.n)
  colnames(betas) <- colnames(x)
  # colnames(betas)[1]<-"Intercept" 
  ####################################
  ##model calibration
  
  it.count <- 0
  llik <- 0.0
  mu <- y + 0.1
  nu <- log(mu)
  cat(" Iteration    Log-Likelihood\n=========================\n")
  wt2 <- rep(1,dp.n)
  repeat {
    y.adj <- nu + (y - mu)/mu
    for (i in 1:dp.n)
    {
      W.i<-W1.mat[,i]
      gwsi<-gw_reg(x,y.adj,W.i*wt2,hatmatrix=F,i)
      betas1[i,]<-gwsi[[1]]
    }
    nu <- gw.fitted(x,betas1)
    mu <- exp(nu)
    old.llik <- llik
    #llik <- sum(y*nu - mu - log(gamma(y+1)))
    llik <- sum(dpois(y, mu, log = TRUE))
    cat(paste("   ",formatC(it.count,digits=4,width=4),"    ",formatC(llik,digits=4,width=7),"\n"))
    if (abs((old.llik - llik)/llik) < tol) break
    wt2 <- as.numeric(mu)
    it.count <- it.count+1
    if (it.count == maxiter) break}
  GW.diagnostic <- NULL
  gw.dev <- 0
  for(i in 1:dp.n)
  {
    if(y[i]!=0)
      gw.dev <- gw.dev + 2*(y[i]*(log(y[i]/mu[i])-1)+mu[i])
    else
      gw.dev <- gw.dev + 2* mu[i]
  }
  
  #gw.dev <- 2*sum(y*log(y/mu)-(y-mu))     
  #local.dev <- numeric(dp.n)     
  #local.null.dev <- numeric(dp.n)
  #local.pseudo.r2 <- numeric(dp.n) 
  if(hatmatrix)
  { 
    for (i in 1:dp.n)
    { 
      W.i<-W2.mat[,i]
      gwsi<-gw_reg(x,y.adj,W.i*wt2,hatmatrix,i)
      betas[i,]<-gwsi[[1]]
      ##Add the smoother y.adjust, see equation (30) in Nakaya(2005)
      #S[i,]<-gwsi[[2]]
      S[i,]<-gwsi[[2]]
      Ci<-gwsi[[3]]      
      #betas.SE[i,]<-diag(Ci%*%t(Ci)) 
      invwt2 <- 1.0 /as.numeric(wt2)
      betas.SE[i,] <- diag((Ci*invwt2) %*% t(Ci))# diag(Ci/wt2%*%t(Ci))  #see Nakaya et al. (2005)
    }
    tr.S<-sum(diag(S))
    ####trace(SWS'W^-1) is used here instead of tr.StS
    #tr.StS<-sum(S^2)
    tr.StS<- sum(diag(S%*%diag(wt2)%*%t(S)%*% diag(1/wt2)))
    ###edf is different from the definition in Chris' code
    #edf<-dp.n-2*tr.S+tr.StS
    yhat<-gw.fitted(x, betas)
    residual<-y-exp(yhat)
    ########rss <- sum((y - gwr.fitted(x,b))^2)
    #rss <- sum((y-exp(yhat))^2)
    #sigma.hat <- rss/edf
    #sigma.aic <- rss/dp.n
    for(i in 1:dp.n)
    {
      #betas.SE[i,]<-sqrt(sigma.hat*betas.SE[i,])
      betas.SE[i,]<-sqrt(betas.SE[i,])
      betas.TV[i,]<-betas[i,]/betas.SE[i,]  
    }
    #AICc <- -2*llik + 2*tr.S*dp.n/(dp.n-tr.S-2) 
    AICc <- -2*llik + 2*tr.S + 2*tr.S*(tr.S+1)/(dp.n-tr.S-1)  # This is generic form of AICc (TN)
    #AIC <- gw.dev + 2*tr.S
    AIC<- -2*llik+2*length(diag(x))
    #AICc <- gw.dev + 2*tr.S + 2*tr.S*(tr.S+1)/(dp.n-tr.S-1) 
    #yss.g <- sum((y - mean(y))^2)
    #gw.R2<-1-rss/yss.g; ##R Square valeu
    #gwR2.adj<-1-(1-gw.R2)*(dp.n-1)/(edf-1) #Adjusted R squared valu
    
    pseudo.R2 <- 1- gw.dev/null.dev
    GW.diagnostic<-list(gw.deviance=gw.dev,AICc=AICc,AIC=AIC,pseudo.R2 =pseudo.R2)        
  }
  else
  {
    for (i in 1:rp.n)
    { 
      W.i<-W2.mat[,i]
      gwsi<-gw_reg(x,y.adj,W.i*wt2,hatmatrix,i)
      betas[i,]<-gwsi[[1]] ######See function by IG
    }
  }
  if (hatmatrix)                                         
  {
    gwres.df<-data.frame(betas,y,exp(yhat),residual,betas.SE,betas.TV)
    colnames(gwres.df)<-c(c(c(colnames(betas),c("y","yhat","residual")),paste(colnames(betas), "SE", sep="_")),paste(colnames(betas), "TV", sep="_"))
  }
  else
  {
    gwres.df<-data.frame(betas)
  }
  rownames(rp.locat)<-rownames(gwres.df)
  griddedObj <- F
  if (is(regression.points, "Spatial"))
  { 
    if (is(regression.points, "SpatialPolygonsDataFrame"))
    {
      polygons<-polygons(regression.points)
      #SpatialPolygons(regression.points)
      #rownames(gwres.df) <- sapply(slot(polygons, "polygons"),
      #  function(i) slot(i, "ID"))
      SDF <-SpatialPolygonsDataFrame(Sr=polygons, data=gwres.df, match.ID=F)
    }
    else
    {
      griddedObj <- gridded(regression.points)
      SDF <- SpatialPointsDataFrame(coords=rp.locat, data=gwres.df, proj4string=CRS(p4s), match.ID=F)
      gridded(SDF) <- griddedObj 
    }
  }
  else
    SDF <- SpatialPointsDataFrame(coords=rp.locat, data=gwres.df, proj4string=CRS(p4s), match.ID=F)
  ##############
  if(hatmatrix)
    res<-list(GW.diagnostic=GW.diagnostic,glms=glms,SDF=SDF,yhat_GWPR=yhat,residuals=residual)
  else
    res <- list(glms=glms,SDF=SDF,yhat_GWPR=yhat,residuals=residual)
}

############ Binomial GWGLM

gwr.binomial <- function(y,x,regression.points,W1.mat,W2.mat,hatmatrix,tol=1.0e-5, maxiter=20)
{
  p4s <- as.character(NA)
  if (is(regression.points, "Spatial"))
  {
    p4s <- proj4string(regression.points)
  }
  
  ############################################
  ##Generalized linear regression
  glms<-glm.fit(x, y, family = binomial())
  null.dev <- glms$null.deviance
  glm.dev <-glms$deviance
  glm.pseudo.r2 <- 1- glm.dev/null.dev
  glms$pseudo.r2 <- glm.pseudo.r2
  var.n<-ncol(x)
  dp.n<-nrow(x)
  glms$aic <- glm.dev + 2*var.n
  glms$aicc <- glm.dev + 2*var.n + 2*var.n*(var.n+1)/(dp.n-var.n-1)
  ############################################
  rp.locat<-coordinates(regression.points)
  rp.n<-nrow(rp.locat)
  betas <-matrix(nrow=rp.n, ncol=var.n)
  betas1<- matrix(nrow=dp.n, ncol=var.n)
  betas.SE <-matrix(nrow=rp.n, ncol=var.n)
  betas.TV <-matrix(nrow=rp.n, ncol=var.n)
  ##S: hatmatrix
  S<-matrix(nrow=dp.n,ncol=dp.n)
  #C.M<-matrix(nrow=dp.n,ncol=dp.n)
  colnames(betas) <- colnames(x)
  #colnames(betas)[1]<-"Intercept"
  ####################################
  ##model calibration
  n=rep(1,length(y))
  it.count <- 0
  llik <- 0.0
  mu <- 0.5
  nu <- 0
  cat(" Iteration    Log-Likelihood\n=========================\n")
  wt2 <- rep(1,dp.n)
  repeat {
    y.adj <- nu + (y - n*mu)/(n*mu*(1 - mu))
    for (i in 1:dp.n)
    {
      W.i<-W1.mat[,i]
      gwsi<-gw_reg(x,y.adj,W.i*wt2,hatmatrix=F,i)
      betas1[i,]<-gwsi[[1]]
    }
    nu <- gw.fitted(x,betas1)
    mu <- exp(nu)/(1 + exp(nu))
    old.llik <- llik
    llik <- sum(lchoose(n,y) + (n-y)*log(1 - mu/n) + y*log(mu/n))
    if(is.na(llik)) llik <-old.llik
    cat(paste("   ",formatC(it.count,digits=4,width=4),"    ",formatC(llik,digits=4,width=7),"\n"))
    if (abs((old.llik - llik)/llik) < tol) break
    wt2 <- n*mu*(1-mu)
    #print(length(wt2))
    it.count <- it.count+1
    if (it.count == maxiter) break}
  
  if(hatmatrix)
  {
    for (i in 1:rp.n)
    {
      W.i<-W1.mat[,i]
      gwsi<-gw_reg(x,y.adj,W.i*wt2,hatmatrix,i)
      betas[i,]<-gwsi[[1]] ######See function by IG
      #S[i,]<-gwsi[[2]]
      S[i,]<-gwsi[[2]][1,]
      Ci<-gwsi[[3]]
      #betas.SE[i,]<-diag(Ci%*%t(Ci))
      invwt2 <- 1.0 /as.numeric(wt2)
      betas.SE[i,] <- diag((Ci*invwt2) %*% t(Ci))   #see Nakaya et al. (2005)
    }
    tr.S<-sum(diag(S))
    #tr.StS<-sum(S^2)
    
    #tr.StS<- sum(diag(S%*%diag(wt2)%*%t(S)%*% diag(1/wt2)))
    ###edf is different from the definition in Chris' code
    #edf<-dp.n-2*tr.S+tr.StS
    yhat<-gw.fitted(x, betas)
    residual<-y-exp(yhat)/(1+exp(yhat))
    ########rss <- sum((y - gwr.fitted(x,b))^2)
    rss <- sum(residual^2)
    #sigma.hat <- rss/edf
    #sigma.aic <- rss/dp.n   ### can be omitted? (TN)
    gw.dev <- sum(log(1/((y-n+exp(yhat)/(1+exp(yhat))))^2))
    for(i in 1:dp.n)
    {
      #betas.SE[i,]<-sqrt(sigma.hat*betas.SE[i,])
      betas.SE[i,]<-sqrt(betas.SE[i,])
      betas.TV[i,]<-betas[i,]/betas.SE[i,]
    }
    AIC <- -2*llik + 2*length(diag(S))
    #AICc <- -2*llik + 2*tr.S + 2*tr.S*(tr.S+1)/(dp.n-tr.S-1)
    AICc <- gw.dev + 2*tr.S + 2*tr.S*(tr.S+1)/(dp.n-tr.S-1)
    #AIC <- gw.dev + 2*tr.S
    #yss.g <- sum((y - mean(y))^2)
    #gw.R2<-1-rss/yss.g; ##R Square valeu  ### is R2 needed? (TN)
    #gwR2.adj<-1-(1-gw.R2)*(dp.n-1)/(edf-1) #Adjusted R squared value
    pseudo.R2 <- 1 - gw.dev/null.dev
    #GW.diagnostic<-list(rss=rss,AICc=AICc,edf=edf,gw.R2=gw.R2,gwR2.adj=gwR2.adj)
    GW.diagnostic<-list(gw.deviance=gw.dev,AICc=AICc,AIC=AIC,pseudo.R2 =pseudo.R2)
  }
  else
  {
    for (i in 1:rp.n)
    {
      W.i<-W2.mat[,i]
      gwsi<-gw_reg(x,y.adj,W.i*wt2,hatmatrix,i)
      betas[i,]<-gwsi[[1]]
    }
  }
  if(hatmatrix)
  {
    gwres.df<-data.frame(betas,y,exp(yhat)/(1+exp(yhat)),residual,betas.SE,betas.TV)
    colnames(gwres.df)<-c(c(c(colnames(betas),c("y","yhat","residual")),paste(colnames(betas), "SE", sep="_")),paste(colnames(betas), "TV", sep="_"))
  }
  else
  {
    gwres.df<-data.frame(betas)
  }
  rownames(rp.locat)<-rownames(gwres.df)
  
  griddedObj <- F
  if (is(regression.points, "Spatial"))
  {
    if (is(regression.points, "SpatialPolygonsDataFrame"))
    {
      polygons<-polygons(regression.points)
      #SpatialPolygons(regression.points)
      #rownames(gwres.df) <- sapply(slot(polygons, "polygons"),
      #  function(i) slot(i, "ID"))
      SDF <-SpatialPolygonsDataFrame(Sr=polygons, data=gwres.df,match.ID =F)
    }
    else
    {
      griddedObj <- gridded(regression.points)
      SDF <- SpatialPointsDataFrame(coords=rp.locat, data=gwres.df, proj4string=CRS(p4s), match.ID=F)
      gridded(SDF) <- griddedObj
    }
  }
  else
    SDF <- SpatialPointsDataFrame(coords=rp.locat, data=gwres.df, proj4string=CRS(p4s), match.ID=F)
  ##############
  if(hatmatrix)
    res<-list(GW.diagnostic=GW.diagnostic,glms=glms,SDF=SDF)
  else
    res <- list(glms=glms,SDF=SDF)
}

############################Layout function for outputing the GWR results
##Author: BL	
print.ggwrm<-function(x, ...)
{
  if(class(x) != "ggwrm") stop("It's not a gwm object")
  cat("   ***********************************************************************\n")
  cat("   *                       Package   GWmodel                             *\n")
  cat("   ***********************************************************************\n")
  cat("   Program starts at:", as.character(x$timings$start), "\n")
  cat("   Call:\n")
  cat("   ")
  print(x$this.call)
  vars<-all.vars(x$GW.arguments$formula)
  var.n<-length(x$glms$coefficients)
  cat("\n   Dependent (y) variable: ",vars[1])
  cat("\n   Independent variables: ",vars[-1])
  dp.n<-length(x$glms$residuals)
  cat("\n   Number of data points:",dp.n)
  cat("\n   Used family:",x$GW.arguments$family)
  ################################################################ Print Linear
  cat("\n   ***********************************************************************\n")
  cat("   *              Results of Generalized linear Regression               *\n")
  cat("   ***********************************************************************\n")
  print(summary.glm(x$glms))
  #cat("\n AICc: ", x$glms$aicc)
  cat("\n Pseudo R-square value: ", x$glms$pseudo.r2)
  #########################################################################
  cat("\n   ***********************************************************************\n")
  cat("   *          Results of Geographically Weighted Regression              *\n")
  cat("   ***********************************************************************\n")
  cat("\n   *********************Model calibration information*********************\n")
  cat("   Kernel function:", x$GW.arguments$kernel, "\n")
  if(x$GW.arguments$adaptive)
    cat("   Adaptive bandwidth: ", x$GW.arguments$bw, " (number of nearest neighbours)\n", sep="") 
  else
    cat("   Fixed bandwidth:", x$GW.arguments$bw, "\n")
  if(x$GW.arguments$rp.given) 
    cat("   Regression points: A seperate set of regression points is used.\n")
  else
    cat("   Regression points: the same locations as observations are used.\n")
  if (x$GW.arguments$DM.given) 
    cat("   Distance metric: A distance matrix is specified for this model calibration.\n")
  else
  {
    if (x$GW.arguments$longlat)
      cat("   Distance metric: Great Circle distance metric is used.\n")
    else if (x$GW.arguments$p==2)
      cat("   Distance metric: Euclidean distance metric is used.\n")
    else if (x$GW.arguments$p==1)
      cat("   Distance metric: Manhattan distance metric is used.\n") 
    else if (is.infinite(x$GW.arguments$p))
      cat("   Distance metric: Chebyshev distance metric is used.\n")
    else 
      cat("   Distance metric: A generalized Minkowski distance metric is used with p=",x$GW.arguments$p,".\n")
    if (x$GW.arguments$theta!=0&&x$GW.arguments$p!=2&&!x$GW.arguments$longlat)
      cat("   Coordinate rotation: The coordinate system is rotated by an angle", x$GW.arguments$theta, "in radian.\n")   
  } 
  
  cat("\n   ************Summary of Generalized GWR coefficient estimates:**********\n")      
  df0 <- as(x$SDF, "data.frame")[,1:var.n, drop=FALSE]
  if (any(is.na(df0))) {
    df0 <- na.omit(df0)
    warning("NAs in coefficients dropped")
  }
  CM <- t(apply(df0, 2, summary))[,c(1:3,5,6)]
  if(var.n==1) 
  { 
    CM <- matrix(CM, nrow=1)
    colnames(CM) <- c("Min.", "1st Qu.", "Median", "3rd Qu.", "Max.")
    rownames(CM) <- names(x$SDF)[1]
  }
  rnames<-rownames(CM)
  for (i in 1:length(rnames))
    rnames[i]<-paste("   ",rnames[i],sep="")
  rownames(CM) <-rnames 
  printCoefmat(CM)
  cat("   ************************Diagnostic information*************************\n")
  
  if (x$GW.arguments$hatmatrix) 
  {	
    cat("   Number of data points:", dp.n, "\n")
    cat("   GW Deviance:", x$GW.diagnostic$gw.deviance, "\n")
    cat("   AIC :",
        x$GW.diagnostic$AIC, "\n")
    cat("   AICc :",
        x$GW.diagnostic$AICc, "\n")
    cat("   Pseudo R-square value: ",x$GW.diagnostic$pseudo.R2,"\n")
  }
  cat("\n   ***********************************************************************\n")
  cat("   Program stops at:", as.character(x$timings$stop), "\n")
  invisible(x)
}
```

```R
names(person_over60)
```

![image](https://user-images.githubusercontent.com/97672187/166218548-9ad78190-4d14-451c-8826-289cac72e962.png){: .align-center}


<br>


<br>

지리적 가중 포아송 회귀의 회귀계수를 최종 분석 결과해석에 사용하기 위해 csv 파일의 형태로 저장했다.

#### 1) 차 대 사람 사고에 대한 지리적 가중 포아송 회귀 모델링

차대사람_20대 미만 group에 대한 지리적 가중 포아송 회귀모델(GWPR) 결과

```R
mod_data2<-person_under20%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대사람..20대.미만)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)

mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(person_under20$x,person_under20$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(person_under20$x,person_under20$y),longlat = TRUE)
bw2<-GWmodel::bw.ggwr(formula =차대사람..20대.미만~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대사람..20대.미만~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_under20_person<-model2$SDF$yhat
coef_under20_person<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_under20_person)<-person_under20$X
coef_under20_person$residuals<-model2$SDF$residual
coef_under20_person$fitted_values<-fit_val_under20_person
coef_under20_person$real_values<-person_under20$`차대사람..20대.미만`
ref<-apply(coef_under20_person,2,function(x){abs(max(x))<0.01 })

bw2<-GWmodel::bw.ggwr(formula =차대사람..20대.미만~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대사람..20대.미만~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_under20_person<-model2$SDF$yhat
coef_under20_person<-dplyr::select(as.data.frame(model2$SDF),Intercept:횡단보도수)
rownames(coef_under20_person)<-person_under20$X
coef_under20_person$residuals<-model2$SDF$residual
coef_under20_person$fitted_values<-fit_val_under20_person
coef_under20_person$real_values<-person_under20$`차대사람..20대.미만`
model2

write.csv(coef_under20_person, "coef_person_under20.csv", row.names = F)
```

![image](https://user-images.githubusercontent.com/97672187/166218921-799ead28-a1e9-48c6-ab8b-21aeba3a77c4.png){: .align-center}

<br>


<br>


차대사람_20대 group에 대한 지리적 가중 포아송 회귀모델(GWPR) 결과

```R
mod_data2<-person_20%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대사람..20대)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)

mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(person_20$x,person_20$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(person_20$x,person_20$y),longlat = TRUE)
#정차금지지대수(데이터 안에 7개밖에 존재 안함)를 변수에 포함하면 변수 행렬이 singular 해져 역행렬을 구할 수 없어 고정 bandwidth를 구하는 데 장애가 되므로 이 경우에는 불가피하게 제거
bw2<-GWmodel::bw.ggwr(formula =차대사람..20대~.-coords.x1-coords.x2-정차금지지대수,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대사람..20대~.-coords.x1-coords.x2-정차금지지대수,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_20_person<-model2$SDF$yhat
coef_20_person<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_20_person)<-person_20$X
coef_20_person$residuals<-model2$SDF$residual
coef_20_person$fitted_values<-fit_val_20_person
coef_20_person$real_values<-person_20$`차대사람..20대`
ref<-apply(coef_20_person,2,function(x){abs(max(x))<0.01 })

bw2<-GWmodel::bw.ggwr(formula =차대사람..20대~.-coords.x1-coords.x2-정차금지지대수-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대사람..20대~.-coords.x1-coords.x2-정차금지지대수-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_20_person<-model2$SDF$yhat
coef_20_person<-dplyr::select(as.data.frame(model2$SDF),Intercept:횡단보도수)
rownames(coef_20_person)<-person_20$X
coef_20_person$residuals<-model2$SDF$residual
coef_20_person$fitted_values<-fit_val_20_person
coef_20_person$real_values<-person_20$`차대사람..20대`
model2

write.csv(coef_20_person, "coef_person_20.csv", row.names = F)
```

![image](https://user-images.githubusercontent.com/97672187/166218983-a5c1e4ac-1272-49dd-bdc5-0879e77ca4d3.png){: .align-center}

<br>


<br>


차대사람_30대 group에 대한 지리적 가중 포아송 회귀모델(GWPR) 결과

```R
mod_data2<-person_30%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대사람..30대)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)

mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(person_30$x,person_30$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(person_30$x,person_30$y),longlat = TRUE)
bw2<-GWmodel::bw.ggwr(formula =`차대사람..30대`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대사람..30대~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_30_person<-model2$SDF$yhat
coef_30_person<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_30_person)<-person_30$X
coef_30_person$residuals<-model2$SDF$residual
coef_30_person$fitted_values<-fit_val_30_person
coef_30_person$real_values<-person_30$`차대사람..30대`
ref<-apply(coef_30_person,2,function(x){abs(max(x))<0.01 })

bw2<-GWmodel::bw.ggwr(formula =`차대사람..30대`~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대사람..30대~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_30_person<-model2$SDF$yhat
coef_30_person<-dplyr::select(as.data.frame(model2$SDF),Intercept:횡단보도수)
rownames(coef_30_person)<-person_30$X
coef_30_person$residuals<-model2$SDF$residual
coef_30_person$fitted_values<-fit_val_30_person
coef_30_person$real_values<-person_30$`차대사람..30대`
model2

write.csv(coef_30_person, "coef_person_30.csv", row.names = F)
```

![image](https://user-images.githubusercontent.com/97672187/166219041-46cb30cd-15f7-4d11-8362-0fbe1753980f.png){: .align-center}

<br>


<br>


차대사람_40대 group에 대한 지리적 가중 포아송 회귀모델(GWPR) 결과

```R
mod_data2<-person_40%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대사람..40대)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)

mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(person_40$x,person_40$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(person_40$x,person_40$y),longlat = TRUE)
bw2<-GWmodel::bw.ggwr(formula =`차대사람..40대`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대사람..40대~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_40_person<-model2$SDF$yhat
coef_40_person<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_40_person)<-person_40$X
coef_40_person$residuals<-model2$SDF$residual
coef_40_person$fitted_values<-fit_val_40_person
coef_40_person$real_values<-person_40$`차대사람..40대`
ref<-apply(coef_40_person,2,function(x){abs(max(x))<0.01 })

bw2<-GWmodel::bw.ggwr(formula =`차대사람..40대`~.-coords.x1-coords.x2-이상평균풍속동반사고건수-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대사람..40대~.-coords.x1-coords.x2-이상평균풍속동반사고건수-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_40_person<-model2$SDF$yhat
coef_40_person<-dplyr::select(as.data.frame(model2$SDF),Intercept:횡단보도수)
rownames(coef_40_person)<-person_40$X
coef_40_person$residuals<-model2$SDF$residual
coef_40_person$fitted_values<-fit_val_40_person
coef_40_person$real_values<-person_40$`차대사람..40대`
model2

write.csv(coef_40_person, "coef_person_40.csv", row.names = F)
```

![image](https://user-images.githubusercontent.com/97672187/166219101-2973b9d1-8f39-41fc-b1e4-b11e33d5614b.png){: .align-center}

<br>


<br>


차대사람_50대 group에 대한 지리적 가중 포아송 회귀모델(GWPR) 결과

```R
mod_data2<-person_50%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대사람..50대)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)

mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(person_50$x,person_50$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(person_50$x,person_50$y),longlat = TRUE)
bw2<-GWmodel::bw.ggwr(formula =`차대사람..50대`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대사람..50대~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_50_person<-model2$SDF$yhat
coef_50_person<-dplyr::select(as.data.frame(model2$SDF),Intercept:`전체_추정교통량`)
rownames(coef_50_person)<-person_50$X
coef_50_person$residuals<-model2$SDF$residual
coef_50_person$fitted_values<-fit_val_50_person
coef_50_person$real_values<-person_50$`차대사람..50대`
ref<-apply(coef_50_person,2,function(x){abs(max(x))<0.01 })

#관측치 회귀계수의 최댓값이 0.01 미만인 변수들을 제거(단, 교통안전시설물 관련 변수는 제외)
bw2<-GWmodel::bw.ggwr(formula =`차대사람..50대`~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-노드개수,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대사람..50대~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-노드개수,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_50_person<-model2$SDF$yhat
coef_50_person<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_50_person)<-person_50$X
coef_50_person$residuals<-model2$SDF$residual
coef_50_person$fitted_values<-fit_val_50_person
coef_50_person$real_values<-person_50$`차대사람..50대`
model2

write.csv(coef_50_person, "coef_person_50.csv", row.names = F)
```

![image](https://user-images.githubusercontent.com/97672187/166219167-5e665288-5aad-4ed4-8aa6-08fb293a784d.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/166219190-0f2e2417-dc7b-4366-899a-9e6524953270.png){: .align-center}

<br>


<br>

차대사람_60대 이상 group에 대한 지리적 가중 포아송 회귀모델(GWPR) 결과


```R
mod_data2<-person_over60%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대사람..60대.이상)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)

mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(person_over60$x,person_over60$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(person_over60$x,person_over60$y),longlat = TRUE)
bw2<-GWmodel::bw.ggwr(formula =`차대사람..60대.이상`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대사람..60대.이상~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_60_person<-model2$SDF$yhat
coef_60_person<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_60_person)<-person_over60$X
coef_60_person$residuals<-model2$SDF$residual
coef_60_person$fitted_values<-fit_val_60_person
coef_60_person$real_values<-person_over60$`차대사람..60대.이상`
ref<-apply(coef_60_person,2,function(x){abs(max(x))<0.01 })

#관측치 회귀계수의 최댓값이 0.01 미만인 변수들을 제거(단, 교통안전시설물 관련 변수는 제거하지 않음) 
bw2<-GWmodel::bw.ggwr(formula =`차대사람..60대.이상`~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대사람..60대.이상~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_60_person<-model2$SDF$yhat
coef_60_person<-dplyr::select(as.data.frame(model2$SDF),Intercept:횡단보도수)
rownames(coef_60_person)<-person_over60$X
coef_60_person$residuals<-model2$SDF$residual
coef_60_person$fitted_values<-fit_val_60_person
coef_60_person$real_values<-person_over60$`차대사람..60대.이상`
model2

write.csv(coef_60_person, "coef_person_over60.csv", row.names = F)
```

![image](https://user-images.githubusercontent.com/97672187/166219272-1b30ce11-f356-4a0d-b990-b8324f39cc17.png){: .align-center}

<br>


<br>

#### 2) 차 대 차 사고에 대한 지리적 가중 포아송 회귀 모델링

차대차_20대 미만 group에 대한 지리적 가중 포아송 회귀모델(GWPR) 결과

```R
mod_data2<-car_under20%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대차..20대.미만)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)

mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(car_under20$x,car_under20$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(car_under20$x,car_under20$y),longlat = TRUE)
bw2<-GWmodel::bw.ggwr(formula =`차대차..20대.미만`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대차..20대.미만~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_under20_car<-model2$SDF$yhat
coef_under20_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_under20_car)<-car_under20$X
coef_under20_car$residuals<-model2$SDF$residual
coef_under20_car$fitted_values<-fit_val_under20_car
coef_under20_car$real_values<-car_under20$`차대차..20대.미만`
ref<-apply(coef_under20_car,2,function(x){abs(max(x))<0.01 })

#관측치 회귀계수의 최댓값이 0.01 미만인 변수들을 제거(단, 교통안전시설물 관련 변수는 제거하지 않음)
bw2<-GWmodel::bw.ggwr(formula =`차대차..20대.미만`~.-coords.x1-coords.x2-이상평균기온동반사고건수-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대차..20대.미만~.-coords.x1-coords.x2-이상평균기온동반사고건수-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_under20_car<-model2$SDF$yhat
coef_under20_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:횡단보도수)
rownames(coef_under20_car)<-car_under20$X
coef_under20_car$residuals<-model2$SDF$residual
coef_under20_car$fitted_values<-fit_val_under20_car
coef_under20_car$real_values<-car_under20$`차대차..20대.미만`
model2

write.csv(coef_under20_car, "coef_car_under20.csv", row.names = F)
```

![image](https://user-images.githubusercontent.com/97672187/166219313-0da2d8ad-f879-4c74-a592-8dd54541a7e9.png){: .align-center}

<br>


<br>


차대차_20대 group에 대한 지리적 가중 포아송 회귀모델(GWPR) 결과

```R
mod_data2<-car_20%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대차..20대)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)
mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(car_20$x,car_20$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(car_20$x,car_20$y),longlat = TRUE)
bw2<-GWmodel::bw.ggwr(formula =`차대차..20대`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대차..20대~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)

fit_val_20_car<-model2$SDF$yhat
coef_20_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_20_car)<-car_20$X
coef_20_car$residuals<-model2$SDF$residual
coef_20_car$fitted_values<-fit_val_20_car
coef_20_car$real_values<-car_20$`차대차..20대`
ref<-apply(coef_20_car,2,function(x){abs(max(x))<0.01 })

#관측치 회귀계수의 최댓값이 0.01 미만인 변수들을 제거(단, 교통안전시설물 관련 변수는 제거하지 않음) 
bw2<-GWmodel::bw.ggwr(formula =`차대차..20대`~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대차..20대~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_20_car<-model2$SDF$yhat
coef_20_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:횡단보도수)
rownames(coef_20_car)<-car_20$X
coef_20_car$residuals<-model2$SDF$residual
coef_20_car$fitted_values<-fit_val_20_car
coef_20_car$real_values<-car_20$`차대차..20대`
model2

write.csv(coef_20_car, "coef_car_20.csv", row.names = F)
```

![image](https://user-images.githubusercontent.com/97672187/166219343-2b92b1cb-0f07-4824-a507-0ceade22a753.png){: .align-center}

<br>


<br>

차대차_30대 group에 대한 지리적 가중 포아송 회귀모델(GWPR) 결과


```R
mod_data2<-car_30%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대차..30대)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)

mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(car_30$x,car_30$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(car_30$x,car_30$y),longlat = TRUE)
bw2<-GWmodel::bw.ggwr(formula =`차대차..30대`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대차..30대~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_30_car<-model2$SDF$yhat
coef_30_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_30_car)<-car_30$X
coef_30_car$residuals<-model2$SDF$residual
coef_30_car$fitted_values<-fit_val_30_car
coef_30_car$real_values<-car_30$`차대차..30대`
ref<-apply(coef_30_car,2,function(x){abs(max(x))<0.01 })

#관측치 회귀계수의 최댓값이 0.01 미만인 변수들을 제거(단, 교통안전시설물 관련 변수는 제거하지 않음) 
bw2<-GWmodel::bw.ggwr(formula =`차대차..30대`~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대차..30대~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_30_car<-model2$SDF$yhat
coef_30_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:횡단보도수)
rownames(coef_30_car)<-car_30$X
coef_30_car$residuals<-model2$SDF$residual
coef_30_car$fitted_values<-fit_val_30_car
coef_30_car$real_values<-car_30$`차대차..30대`
model2

write.csv(coef_30_car, "coef_car_30.csv", row.names = F)
```

![image](https://user-images.githubusercontent.com/97672187/166219599-1df788e0-a15e-4e48-8771-d184268e1cec.png){: .align-center}

<br>


<br>


차대차_40대 group에 대한 지리적 가중 포아송 회귀모델(GWPR) 결과

```R
mod_data2<-car_40%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대차..40대)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)

mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(car_40$x,car_40$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(car_40$x,car_40$y),longlat = TRUE)
bw2<-GWmodel::bw.ggwr(formula =`차대차..40대`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대차..40대~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)

fit_val_40_car<-model2$SDF$yhat
coef_40_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_40_car)<-car_40$X
coef_40_car$residuals<-model2$SDF$residual
coef_40_car$fitted_values<-fit_val_40_car
coef_40_car$real_values<-car_40$`차대차..40대`
ref<-apply(coef_40_car,2,function(x){abs(max(x))<0.01 })

#관측치 회귀계수의 최댓값이 0.01 미만인 변수들을 제거(단, 교통안전시설물 관련 변수는 제거하지 않음) 
bw2<-GWmodel::bw.ggwr(formula =`차대차..40대`~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대차..40대~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_40_car<-model2$SDF$yhat
coef_40_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:횡단보도수)
rownames(coef_40_car)<-car_40$X
coef_40_car$residuals<-model2$SDF$residual
coef_40_car$fitted_values<-fit_val_40_car
coef_40_car$real_values<-car_40$`차대차..40대`
model2

write.csv(coef_40_car, "coef_car_40.csv", row.names = F)
```

![image](https://user-images.githubusercontent.com/97672187/166219680-a306bc9d-a919-47a2-9953-59a83c3e14c7.png){: .align-center}

<br>


<br>


차대차_50대 group에 대한 지리적 가중 포아송 회귀모델(GWPR) 결과

```R
mod_data2<-car_50%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대차..50대)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)

mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(car_50$x,car_50$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(car_50$x,car_50$y),longlat = TRUE)
bw2<-GWmodel::bw.ggwr(formula =`차대차..50대`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대차..50대~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_50_car<-model2$SDF$yhat
coef_50_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_50_car)<-car_50$X
coef_50_car$residuals<-model2$SDF$residual
coef_50_car$fitted_values<-fit_val_50_car
coef_50_car$real_values<-car_50$`차대차..50대`
ref<-apply(coef_50_car,2,function(x){abs(max(x))<0.01 })

#관측치 회귀계수의 최댓값이 0.01 미만인 변수들을 제거(단, 교통안전시설물 관련 변수는 제거하지 않음) 
bw2<-GWmodel::bw.ggwr(formula =`차대차..50대`~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대차..50대~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_50_car<-model2$SDF$yhat
coef_50_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:횡단보도수)
rownames(coef_50_car)<-car_50$X
coef_50_car$residuals<-model2$SDF$residual
coef_50_car$fitted_values<-fit_val_50_car
coef_50_car$real_values<-car_50$`차대차..50대`
model2

write.csv(coef_50_car, "coef_car_50.csv", row.names = F)
```

![image](https://user-images.githubusercontent.com/97672187/166219748-c5ea9956-90c2-485e-bf3a-8b1e82903759.png){: .align-center}


![image](https://user-images.githubusercontent.com/97672187/166219790-ce47394c-f278-4ff9-8ec8-3dcd0ad4af24.png){: .align-center}

<br>


<br>

차대차_60대 이상 group에 대한 지리적 가중 포아송 회귀모델(GWPR) 결과


```R
mod_data2<-car_over60%>%
  dplyr::select(신호등_보행자수:총거주인구수,차대차..60대.이상)%>%
  mutate(평균혼잡강도=(혼잡빈도강도+혼잡시간강도)/2)%>%
  dplyr::select(-혼잡빈도강도,-혼잡시간강도,-이상최저온도동반사고건수,-이상최고온도동반사고건수,-이상평균지면온도동반사고건수)

mod_data3<-sp::SpatialPointsDataFrame(data=mod_data2,coords = cbind(car_over60$x,car_over60$y))
dst<-GWmodel::gw.dist(dp.locat = cbind(car_over60$x,car_over60$y),longlat = TRUE)
bw2<-GWmodel::bw.ggwr(formula =`차대차..60대.이상`~.-coords.x1-coords.x2,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대차..60대.이상~.-coords.x1-coords.x2,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_60_car<-model2$SDF$yhat
coef_60_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:평균혼잡강도)
rownames(coef_60_car)<-car_over60$X 
coef_60_car$residuals<-model2$SDF$residual
coef_60_car$fitted_values<-fit_val_60_car
coef_60_car$real_values<-car_over60$`차대차..60대.이상`
ref<-apply(coef_60_car,2,function(x){abs(max(x))<0.01 })

#관측치 회귀계수의 최댓값이 0.01 미만인 변수들을 제거(단, 교통안전시설물 관련 변수는 제거하지 않음) 
bw2<-GWmodel::bw.ggwr(formula =`차대차..60대.이상`~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,data=mod_data3,kernel="gaussian",family="poisson",dMat = dst,longlat = TRUE)
model2<-ggwr.basic2(차대차..60대.이상~.-coords.x1-coords.x2-건물면적-자동차대수-총거주인구수-전체_추정교통량-평균혼잡강도,dMat = dst,longlat = TRUE,bw = bw2,kernel="gaussian",family="poisson",data = mod_data3)
fit_val_60_car<-model2$SDF$yhat
coef_60_car<-dplyr::select(as.data.frame(model2$SDF),Intercept:횡단보도수)
rownames(coef_60_car)<-car_over60$X
coef_60_car$residuals<-model2$SDF$residual
coef_60_car$fitted_values<-fit_val_60_car
coef_60_car$real_values<-car_over60$`차대차..60대.이상`
model2

write.csv(coef_60_car, "coef_car_over60.csv", row.names = F)
```

![image](https://user-images.githubusercontent.com/97672187/166220283-e7612c3b-a8c4-4384-bdde-a89de03d8094.png){: .align-center}

<br>


<br>


### OLS, 포아송, 지리적 가중회귀 모델(GWR), 지리적 가중 포아송 회귀 모델(GWPR)의 성능비교

1) OLS, 포아송, GWR, GWPR 가중회귀 모델 성능 비교 그래프

![image](https://user-images.githubusercontent.com/97672187/166221206-ec221885-52a8-4f73-b7e5-a327692fe499.png){: .align-center}

GWPR 모델을 사용했을 때가 4개의 모델 중 가장 낮은 AIC를 보인다. 즉, 지리적 가중 포아송 회귀 모델의 성능이 가장 좋다. 따라서 4개의 모델 중 **'지리적 가중 포아송 회귀 모형'** 을 최종
모델로 사용한다.



다음 포스팅에서는 모델링 결과를 바탕으로 진행한 추가 분석을 정리해보자.

