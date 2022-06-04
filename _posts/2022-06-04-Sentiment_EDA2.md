---
layout: single
title: "피파온라인 댓글 감성분석 프로젝트 전처리 및 EDA Part.2"
toc: true
toc_sticky: true
category: Sentiment
---

피파 인벤 홈페이지에서 수집한 댓글 데이터에는 이미 평점 데이터가 주어져서 이 평점 데이터를 기반으로 해당 댓글이 긍정인지 부정인지를 판단할 수 있다. 하지만, 사용자마다 매기는 평점은
주관적이기 때문에 같은 4점의 평점이라도 이 4점이 높은 편인지, 낮은 편인지를 파악하기가 힘들다. 따라서 보다 객관적인 방법으로 긍정,부정 여부를 판단하기 위해 군산대에서 개발한
KNU 감성사전을 사용해서 댓글의 감성 점수를 매겼다.

KNU 감성사전에는 여러 단어들의 점수가 매겨져 있다. 각 댓글에 있는 단어와 사전의 점수를 매칭시키고 한 댓글에 있는 단어들의 점수를 모두 합해서 해당 댓글의 감성점수를 매겼다. 
게임 데이터의 특성상 신조어가 많이 등장하는데 KNU 감성사전가 모든 신조어들을 다 포함할 순 없기 때문에 사전에 없는 단어의 점수는 모두 중립을 의미하는 0으로 대체했다.

KNU 감성 사전을 기반으로 감성 점수를 도출하는 전처리는 R언어를 사용했다.

## Comments Sentiment Analysis 프로젝트 전처리 및 EDA Part.2

### 3. 감성점수 도출

```R
library(rJava)
library(memoise)
library(multilinguer)
library(KoNLP)
library(stringi)
library(tibble)

pkgs <- c(
  # Data munipulate packages   
  "dplyr",
  "stringr",
  
  # text mining packages
  "tidytext",
  "KoNLP",
  "tidyr",
  
  # graphic packages
  "ggplot2",
  
  'httr',
  'readr'
)

# 패키지 동시에 적용
sapply(pkgs, require, character.only = T)
```

```R
# 감성사전 불러오기
useNIADic()
url = "https://drive.google.com/u/0/uc?id=14t6CqgzfNpJjU45W8HFmLBlHaI_FjOUw&export=download"
GET(url, write_disk(tf <- tempfile(fileext = ".csv")))

dic <- read_csv(tf)
rm(url,tf)

#감성 사전 저장
write.csv(dic, file = 'KNU_Dict.csv')
```

```R
#감성 사전
dic <- read.csv('KNU_Dict.csv', header = T)

#댓글 데이터
raw_comment <- fread('fifa_comment_data2.csv', encoding = 'UTF-8')

#tibble 형태로 변환
raw_comment <- tibble(raw_comment$comment)
colnames(raw_comment) <- 'comment'
```

```R
# 댓글 데이터 전처리
fifa_comment <- raw_comment %>% 
  mutate(id = row_number(), #댓글마다 번호 붙이기
         reply = str_squish(comment)) # 공백을 하나로만 줄여줌
```

```R
# 데이터 토큰화 
word_comment <- fifa_comment %>% 
  unnest_tokens(input = reply,
                output = word,
                token = "words",
                drop = F)%>% 
  filter(str_detect(word, "[가-힣]")) %>%  # 한글만 추출
  filter(str_count(word) >= 2) # 2번이상 등장한 단어만 사용
```

```R
# KNU 감성사전을 기반으로 단어에 감정점수 부여
# 사전에 없는 단어는 모두 0(중립)으로 처리
word_comment <- word_comment %>% 
  left_join(dic, by ="word") %>% 
  mutate(polarity = ifelse(is.na(polarity), 0 , polarity))
```

```R
# 감정 분류 하기 
word_comment <- word_comment %>% 
  mutate(sentiment = ifelse(polarity ==  2, "pos",
                            ifelse(polarity == -2, "neg", "neu")))
```

```R
#댓글별 감정점수
#한 댓글을 이루는 단어의 감성점수를 모두 합침
score_comment <- word_comment %>% 
  group_by(id, reply) %>% 
  summarise(score = sum(polarity)) %>% 
  ungroup()
```

```R
#위에서 번호를 매겨놓은 댓글 데이터와
#감성점수가 포함된 댓글 데이터를 병합
df <- fifa_comment %>% 
  left_join(score_comment, by ="id") %>% 
  select(id,comment,score)
```

```R
#원본 데이터
origin <- fread('fifa_comment_data2.csv', encoding = 'UTF-8')

#원본 데이터에 댓글의 감성 점수를 합침
df_score <- cbind(origin,df$score)
colnames(df_score)[6] <- 'score'
```

```R
#점수가 NA가 아닌 것들만 데이터로 저장
df_score <- df_score[!is.na(df_score$score),]
write.csv(df_score, file = 'sentiment_scores.csv', row.names = F)
```

**감성 점수가 추가된 데이터의 형태는 다음과 같다(score 변수).**

![image](https://user-images.githubusercontent.com/97672187/171997914-d892cf43-4fc2-46c1-a667-7001f19f541a.png){: .align-center}

### 4. label 분류
감성 점수가 도출 되었기 때문에 이 감성 점수를 기반으로 평점(rating)보다는 더 객관적인 기준으로 긍정, 부정을 분류해서 모델링을 위한 label 데이터를 만들었다.
감성 점수가 양수면 긍정을 나타내는 1, 음수면 부정을 나타내는 0으로 라벨링했고, 이진 분류를 위해 점수가 0이면 모두 부정으로 라벨링했다.

점수가 0인 댓글을 모두 부정으로 라벨링해서 0인 데이터가 1인 데이터보다 약 3배 정도 많은 불균형 데이터가 되었고, 이를 고려해서 모델링을 진행했다.

```python
df = pd.read_csv('..../sentiment_scores.csv')
#0보다 크면 긍정(1), 같거나 작으면 부정(0)
df.loc[df.score > 0, 'label'] = 1
df.loc[df.score <= 0, 'label'] = 0
df.label = df.label.astype(int)
display(df.tail())
df.label.value_counts().plot(kind = 'bar')
```

![image](https://user-images.githubusercontent.com/97672187/171998157-b2576c81-01c2-4013-8773-858b7f791320.png){: .align-center}

<br>

<br>

```python
#데이터 저장
df.to_csv('..../sentiment_scores.csv', index = False)
```

댓글마다 감성 점수를 도출하고 이 감성 점수를 기반으로 label을 분류해서 이진 분류 모델을 위한 타겟 변수를 만들었다. 다음 포스팅에서는 RNN 계열의 딥러닝 모델 중 하나인
LSTM 모델을 사용한 모델링 과정을 정리해보겠다.


