---
layout: single
title: "피파온라인 댓글 감성분석 프로젝트 추천시스템 및 결론"
toc: true
toc_sticky: true
category: Sentiment
---

지난 포스팅까지 데이터 수집, 전처리, 모델링의 과정을 모두 마쳤다. 이번 포스팅에서는 전처리가 완료된 데이터를 기반으로 사용자가 특정 선수를 입력하면 그 선수와 비슷한 유형의 선수나
관련 있는 선수를 추천해주는 추천시스템의 개발 과정을 정리해보자. 또한, 프로젝트의 분석 결과와 한계에 대해서도 다뤄보자.

## Comments Sentiment Analysis 프로젝트 추천시스템 및 결론

### 1. Recommendation System about related player

#### 1) 감성 점수를 기반으로 한 새로운 평점 점수 생성

```python
import numpy as np
import pandas as pd
```

데이터 불러오기

```python
DATA_PATH = '....'
df = pd.read_csv(f'{DATA_PATH}sentiment_scores.csv')
df.tail()
```

![image](https://user-images.githubusercontent.com/97672187/172293782-3d895c42-3b5f-4a84-a0b0-4ae7c7cf8f90.png){: .align-center}

<br>


<br>

각 댓글의 최소 감성 점수가 -11점, 최대 감성 점수는 44점이다. 

```python
df['score'].describe()
```

![image](https://user-images.githubusercontent.com/97672187/172293808-a63c0e44-60f9-4876-921a-b7274ed4c85d.png){: .align-center}

<br>


<br>

보통 사람들이 매기는 평점에는 음수가 존재하지 않는다. 영화나 음식, 쇼핑 등에 평점을 매길 때 0 ~ 5점, 0 ~ 10점과 같이
양수로 평점을 매긴다. 피파 인벤 홈페이지에서 제공하는 평점데이터도 0 ~ 5점 사이의 평점으로 이루어져 있는 만약 감성 점수를 기존 평점에 단순히 곱해서 새로운 평점을 구한다면 부정의
감정을 나타내는 댓글의 평점은 음수가 되어버린다. 따라서 음수의 평점을 만들지 않으면서 감성점수가 반영된 새로운 평점 점수를 만들기 위해 감성 점수를 0 ~ 1 사이의 수로 scaling하고
부정, 긍정에 따라 다른 방법을 사용하여 새로운 평점을 계산했다. scaling된 감성 점수를 사용하여 새로운 평점을 구한 방법은 다음과 같다.

$$ new_{negative} = (1 - new\;score) * original\;rating $$

$$ new_{positive} = (1 + new\;score) * original\;rating $$

위와 같은 식으로 계산함으로써 부정적인 댓글이면 기존의 평점보다 조금 더 낮은 평점으로, 긍정적인 댓글이면 기존의 평점보다 더 높은 평점으로 새로운 평점 변수를 만들었다.

```python
from sklearn.preprocessing import MinMaxScaler
s = MinMaxScaler()

# 감성점수 정규화
df['new_score'] = s.fit_transform(df[['score']])
df.tail()
```

![image](https://user-images.githubusercontent.com/97672187/172294853-1b06fa5e-ef48-4deb-9162-bb6948b10757.png){: .align-center}

<br>


<br>


```python
#긍정이면 정규화된 감성점수에 + 1을 해서 점수가 더 커지게 하고,
#부정이면 정규화된 감성점수를 1에서 빼서 점수가 더 작아지게 바꿈.

df.loc[df['score'] > 0, 'new_rating'] = df['rating'] * (1 + df['new_score'])
df.loc[df['score'] <= 0, 'new_rating'] = df['rating'] * (1 - df['new_score'])
df = df[['userId', 'player_name', 'new_rating']]
display(df.tail())
```

![image](https://user-images.githubusercontent.com/97672187/172294951-534b9579-e1c9-4e72-b584-a187ab03a07e.png){: .align-center}

<br>


<br>

#### 2) 추천시스템 개발(feats. KNN)

사용자 ID를 행으로 하면 사용자가 기준이 되어서 사용자 기반 협업 필터링, 선수이름(Item)을 행으로 하면
아이템 기반 협업 필터링이 되는데 좀 더 성능이 좋다고 알려져있는 아이템 기반 협업 필터링을 사용하기 위해 선수이름을(Item) 행, 사용자 ID를 열로 하는 데이터 프레임을 만들었다. 

```python
#평점행렬 만들기
ratings = df.pivot_table('new_rating', index = 'player_name', columns = 'userId')
ratings.fillna(0, inplace = True)
ratings.head()
```

![image](https://user-images.githubusercontent.com/97672187/172295343-c532acd9-7e1b-4082-b4b3-e3e8e55d32c0.png){: .align-center}

<br>


<br>

2906명의 선수와 28003의 사용자 ID가 존재한다.

```python
print(ratings.shape)
```

![image](https://user-images.githubusercontent.com/97672187/172295379-050535c1-a122-4398-a1c4-d92f205cd81e.png){: .align-center}

<br>


<br>

추천시스템은 KNN 알고리즘을 사용하고, 유사도 계산은 코사도 유사도를 사용한다. 평점 데이터에 음수가 존재하지 않기 때문에 유사도가 1에 가까울 수록 유사하고, 0에 가까울 수록
유사하지 않은 선수가 된다. 

```python
from sklearn.neighbors import NearestNeighbors

n = 5
cosine_knn = NearestNeighbors(n_neighbors=n, algorithm='brute', metric = 'cosine')
item_cosine_knn_fit = cosine_knn.fit(ratings.values)
item_distances, item_indices = item_cosine_knn_fit.kneighbors(ratings.values)
#유사한 거리와, 가까운 선수들의 index
```

0번째 인덱스의 선수와 유사한 선수의 인덱스

```python
print(item_indices[0])
```

![image](https://user-images.githubusercontent.com/97672187/172295615-6dd5658f-97eb-4042-9f3a-435e8c6039ec.png){: .align-center}

<br>


<br>


```python
#거리(코사인 유사도)가 가장 가까운 선수들을 매칭시켜서 저장(자기 포함 top 5)
items_dic = {}
for i in range(len(ratings.index)):
  item_idx = item_indices[i]
  col_names = ratings.index[item_idx].tolist()
  items_dic[ratings.index[i]] = col_names
```

```python
print(items_dic['크리스티아누 호날두'])
print(items_dic['리오넬 메시'])
print(items_dic['루카 모드리치'])
print(items_dic['손흥민'])
```

![image](https://user-images.githubusercontent.com/97672187/172295723-edc53178-b0dc-4e13-b9cb-fa94bf7cfac3.png){: .align-center}

<br>


<br>

사용자가 원하는 선수를 입력하면 입력한 선수와 가장 유사하거나 관련있는 선수를 출력해주는 함수. 자신을 포함하여 가장 유사한 선수를 5명 추천 해준다. 자신을 제외하면 4명의 선수가
추천된다.

```python
def get_recommendation(name):
  print(name + ' 선수와 가장 연관있는 선수리스트: \n')
  for idx, val in enumerate(items_dic[name][1:]):
    print(f'{idx+1}. ' + val)
```

```python
get_recommendation('박지성')
```

![image](https://user-images.githubusercontent.com/97672187/172295885-0e681efb-34b7-4b75-ba71-2e5837a800f1.png){: .align-center}

<br>


<br>


```python
get_recommendation('리오넬 메시')
```

![image](https://user-images.githubusercontent.com/97672187/172295908-5ca83d87-13ab-42cd-86ad-f6a1a47ad249.png){: .align-center}

<br>


<br>

### 2. 분석결과
#### 1) Word Cloud
NLP에서는 Word Cloud 패키지를 사용하면 어떤 단어가 가장 많이 등장했는지 시각적으로 표현할 수 있다. Word Cloud 형식으로 등장한 단어들을 보고 긍정, 부정에 등장하는 단어들의
차이나 이번 프로젝트를 예시로 들면, 선수들의 포지션마다 자주 등장하는 단어들이 어떻게 다른지 파악 가능하다.

데이터를 전처리 하는 과정에서 mecab 패키지를 사용하여 댓글을 형태소 단위로 토큰화했었다. 토큰화까지 진행된 파일을 사용하여 Word Cloud를 그려보자.

```python
#pickle 상태로 저장된 파일
df = pd.read_pickle(f"{DATA_PATH}df_tokenized.pkl")
df.tail()
```

![image](https://user-images.githubusercontent.com/97672187/172297219-b72f9a1b-4d87-4e3e-81a9-518c637965d4.png){: .align-center}

<br>


<br>


```python
#긍정과 부정 단어는 어떤 단어들이 많이 나올까
negative_words = np.hstack(df[df.label == 0]['tokenized'].values)
positive_words = np.hstack(df[df.label == 1]['tokenized'].values)
```

```python
#포지션별로 등장하는 단어들은 어떻게 다를까
fw_words = np.hstack(df[df.position == 'fw']['tokenized'].values)
mf_words = np.hstack(df[df.position == 'mf']['tokenized'].values)
df_words = np.hstack(df[df.position == 'df']['tokenized'].values)
```

Word Cloud를 그리는 함수. 가장 많이 등장한 단어 50개를 그려주고, top 10 단어를 텍스트로 출력해준다.

```python
def draw_wc(word_list):
  count_list = Counter(word_list)
  most_common_word = count_list.most_common(50)
  wc = WordCloud(font_path= path ,background_color="white", max_font_size=60,
                 stopwords = new_stopwords).generate_from_frequencies(dict(most_common_word))
  print(most_common_word[:10])
  plt.figure(figsize=(12,8))
  return plt.imshow(wc)
```

- 긍정, 부정 댓글의 단어 Word Cloud

```python
#긍정 word_cloud
draw_wc(positive_words)
```

![image](https://user-images.githubusercontent.com/97672187/172297528-86b48db5-507b-4649-ac1a-fee9e06fa736.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/172297358-ec0af40a-1168-4297-a5ac-026d378c5a26.png){: .align-center}

<br>


<br>


```python
#부정 word_cloud
draw_wc(negative_words)
```

![image](https://user-images.githubusercontent.com/97672187/172297513-f6b8d97e-a857-4ba4-b83e-d36bc516d678.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/172297426-f2409979-0081-4973-95ec-6970ab633ac9.png){: .align-center}

<br>


<br>

긍정과 부정에 등장한 Word Cloud의 를 보니 차이가 존재한다. 가장 많이 등장한 10개의 단어들에서는 같은 단어라도 긍정이냐 부정이냐에 따라 등장한 빈도나
순위에 차이가 존재하는 것을 확인할 수 있다.

- 포지션별 Word Cloud

```python
#fw word_cloud
draw_wc(fw_words)
```

![image](https://user-images.githubusercontent.com/97672187/172297938-cfba15e1-2916-4c30-a9f8-ff64ca7da979.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/172297973-6186fc6b-bf41-4f5f-9178-e5a663e1e092.png){: .align-center}

<br>


<br>


```python
#mf word_cloud
draw_wc(mf_words)
```

![image](https://user-images.githubusercontent.com/97672187/172298031-5dc9fc22-5c30-4f35-98cc-6ca3ed05b893.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/172298043-ee3b3b7a-5c79-4ed8-b6ea-4c9ffba29e52.png){: .align-center}

<br>


<br>


```python
#df word_cloud
draw_wc(df_words)
```

![image](https://user-images.githubusercontent.com/97672187/172298072-534f3b1b-29f1-4f30-b115-a89713931cad.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/172298081-1a9d39ff-0d28-4847-9820-0d64dcb467e4.png){: .align-center}

<br>


<br>

포지션별로도 자주 등장하는 단어들에 차이가 존재하는 것을 확인할 수 있다. 게임 특성상 공격수는 스피드와 침투 스탯이 중요하고, 미드필더는 패스와 중거리, 수비수는 급여와 수비 능력이
중요한데 포지션별 댓글마다 이러한 특성들이 잘 반영되어 있는 것 같다. '그냥', '진짜' 와 같은 일반적으로 너무 많이 등장하는 단어들을 제거하고 그리면 더 명확한 차이가 보였을 것이다.


이처럼 Word Cloud를 사용하여 댓글 데이터를 분석한 결과 긍정, 부정에 따라 자주 사용되는 단어들과 포지션별로 자주 사용되는 단어들의 차이가 존재한다는 것을 알 수 있었다. 

### 3. 결론

오늘날에는 SNS나 쇼핑 플랫폼이 매우 발달해서 이미지 데이터, 자연어 데이터를 활용해 다양한 분야를 분석할 수 있다. 특히, 자연어 관점에서는 사용자들이 
제품이나 기능에 대해 평가한 댓글 데이터를 감성 분석 함으로써 소비자의 선호도를 파악할 수 있고, 파악한 정보를 기반으로 상품을 보완하고, 기능을 업데이트 시킬 수 있다.

이번 프로젝트에서는 평소에 즐겨하던 게임인 피파 온라인의 댓글 데이터를 감성 분석함으로써 선수를 평가한 댓글이 긍정적인지, 부정적인지를 평가하고 감성 점수를 기반으로 
관련 선수를 추천해주는 시스템을 개발했다. 게임회사의 관점에서는 댓글 데이터만 가지고도 선수의 선호도를 평가할 수 있어, 다음에 다른 시즌의 선수를 만들 때 선수의 스탯을 
더 현실적으로 보완할 수 있고, 사용자의 관점에서는 비슷하거나 관련 있는 유형의 선수를 추천받아서 게임 내에서 자신의 경제적인 수준에 맞춰 선수를 구매할 수 있다는 장점이 생길 것으로 
기대 할 수 있다.

딥러닝의 RNN 계열 모델인 LSTM 모델을 사용하여 댓글의 긍정, 부정을 분류하는 이진 분류를 수행한 결과 정확도가 거의 90% 달하는 모델을 만들었다. 또한, 긍정,부정 여부나 포지션에 따라
자주 사용되는 단어들의 차이가 존재한다는 분석 결과도 얻었다. 개인적으로 의미있는 분석결과와 준수한 성능을 내는 모델을 만들긴 했지만, 프로젝트 특성 상 시간이 부족해서 몇가지 한계가 존재했다.

첫째, 같은 선수라도 시즌별로 스탯이 다르기 때문에 다양한 선수처럼 존재할 수 있는데 선수를 추천해줄 때 시즌을 고려하지 않고, 모든 시즌을 하나의 선수로 취급했다. 
이 문제는 만약 시간이 좀 더 주어졌다면 시즌별로 선수를 세분화하여 추천시스템 모델을 개발함으로써 해결할 수 있다. 

둘째, 감성 점수를 기반으로 한 새로운 평점 점수를 만들었는데 논리적으로 문제가 없는지 타당성을 입증하기 힘들었다. 
이 문제는 감성 분석과 추천 시스템이 결합된 관련된 연구를 더 찾아본다면 보다 설득력 있는 점수를 만들 수 있을 것이다. 

셋째, 게임 데이터의 특성상 댓글에 다양한 신조어들이 등장하는데 감성점수를 계산했던 KNU 한국어 감성 사전에 존재하지 않는 신조어들은 모두 0으로 계산해서 
신조어들의 특징을 잘 살릴 수 없었다. 만약, 빈도수가 높은 신조어들을 10개, 20개 정도라도 감성 사전에 추가했다면 좀 더 신조어의 감성점수를 반영한 모델을 만들 수 있을 것 같다. 

나중에 기회가 된다면 pytorch를 기반으로 BERT나 KOBERT모델을 사용하여 똑같은 데이터에 감성분석을 진행해봄으로써 어떤 모델의 성능이 가장 뛰어나는지 비교해보며 성능을 높여봐야겠다.
딥러닝 모델을 사용해보았고, 흥미로운 분석결과를 얻을 수 있어서 개인적으로 재미있었던 프로젝였다.


