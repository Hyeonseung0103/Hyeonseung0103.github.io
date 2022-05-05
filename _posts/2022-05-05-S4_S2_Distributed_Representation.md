---
layout: single
title: "분산 표현, Word2Vec, FastText"
toc: true
toc_sticky: true
category: Section4
---

NLP에서 단어를 벡터화 하기 위해서는 등장 횟수 기반과 분산 기반 표현 이 2가지 방법이 있다. 등장 횟수 기반은 단어의 빈도수만 고려하고 다른 단어와의 상관관계는 고려하지 않는다는
단점이 존재한다. 또한, 단어의 갯수가 많아질수록 계산해야 하는 차원이 많아져서 저장 공간이나 계산량이 많아진다. 이러한 문제를 해결하기 위해 분산 기반 표현
으로 단어를 벡터화 할 수 있다.

### 분포가설(Distribution Hypothesis)
분포가설은 비슷한 위치에서 등장하는 단어들은 비슷한 의미를 가진다는 가설이다. 반대로 비슷한 의미를 지닌 단어는 주변 단어의 분포도 비슷하다는 것이다.

예를 들어, "He is a **good** man" , "He is a **great** man" 이라는 문장에서 **good**과 **great**는 비슷한 위치에 있고 주변 단어의 분포가 같아서 단어의 의미나 쓰임새가
비슷할 것이라는 가정이다. 이 분포 가설을 기반으로 단어의 벡터 표현이 결정되기 때문에 분산 표현이라고 부른다.

### One Hot Encoding(원 핫 인코딩) vs Embedding(임베딩)
원핫인코딩은 매우 간단하게 단어를 벡터화 할 수 있는 방법이다. 하지만, 단어의 갯수만큼 차원이 증가하고, 모든 단어를 0또는 1로 표현해서 단어간의 유사도를 계산할 수 없다는 단점이 있다.

Embedding을 사용하면 이 문제를 해결할 수 있다. 임베딩은 단어를 고정 길이의 벡터, 즉 차원이 일정한 벡터로 나타낸다. 사용자가 정한 차원만큼 단어를 연속적인 값으로 표현하기 때문에 0과
1로만 표현되어 유사도를 계산할 수 없고, 차원이 단어의 갯수만큼 늘어나는 문제를 해결한다. Word2Vec은 가장 널리 사용되는 임베딩 방법 중 하나이다.

### Word2Vec
Word2Vec은 단어를 벡터로(Word to Vector) 나타내는 방법이다. 특정 단어 양 옆에 이는 단어(window size)의 관계를 활용해서 분포가설이 잘 반영되어 있다.
Word2Vec에는 CBoW와 Skip-gram이라는 2가지 방법이 있다.

#### 1) CBow vs Skip-gram
CBow는 주변 단어에 대한 정보를 바탕으로 중심단어를 예측하는 모델이다. 반면, Skip-gram은 중심 단어를 기반으로 주변 단어를 예측하는 모델이다. 

![image](https://user-images.githubusercontent.com/97672187/166857753-5c1c6b7e-ebbf-4357-974e-a4a7c1e99aa4.png){: .align-center}

이미지 출처: https://www.researchgate.net/figure/Illustration-of-the-Skip-gram-and-Continuous-Bag-of-Word-CBOW-models_fig1_281812760

위의 그림을 보면 CBoW는 주변 단어를 통해 하나의 중심 단어를 예측하고, Skip-gram은 하나의 중심 단어를 입력으로 받아 주변단어를 예측하는 것을 알 수 있다.
CBoW가 입력이 더 많기 때문에 Skip-gram보다 성능이 좋을 것 같지만, 역전파 관점에서 보면 Skip-gram이 훨씬 더 많은 학습이 일어나기 때문에 Skip-gram의 성능이 조금 더 좋다고 한다.
CBoW는 주변 단어로부터 1개의 중심단어를 학습하지만, Skip-gram은 중심 단어로부터 여러 문맥의 단어를 여러 번 예측 및 학습 하기 때문이다.

예시)"I am a good student"

CBow: am a good student -> I, I a good student -> am ...... (이렇게 중심 단어는 한 번만 학습된다.)

Skip-gram: I -> am a good student, am -> I a good student .... (이렇게 중심 단어가 아닌 단어들은 여러번 학습될 수 있다.)

하지만, 그만큼 Skip-gram의 계산량이 조금 더 많다.

#### 2) Word2Vec 모델 구조
성능이 조금 더 좋은 Skip-gram을 기준으로 Word2Vec 모델의 구조를 살펴보자.

Word2Vec은 원핫인코딩 된 단어 벡터를 입력으로 받아서 사용한다. 하지만, 이 원핫인코딩 된 단어 벡터로는 유사도를 계산할 수 없기 때문에 은닉층에 임베딩 벡터라는 것을 만들어서
분산표현을 사용하여 유사도를 계산할 수 있게 한다. 은닉층은 사용자가 정한 임베딩 벡터의 차원 수만큼의 노드로 구성된 1개의 층을 사용한다. Skip-gram은 출력층에서 중심단어를 기반으로
주변 단어를 출력해야 하기 때문에 다중 분류 활성화 함수인 softmax 함수를 사용한다. 

![image](https://user-images.githubusercontent.com/97672187/166858363-db801e69-30ff-49b5-955e-97b26949b1fc.png){: .align-center}

이미지출처: https://wooono.tistory.com/244

위의 그림을 보면 "the"라는 단어의 원핫 벡터를 넣어서 은닉층에서 임베딩 벡터로 변환하여 표현 한 뒤 이 임베딩 벡터를 다시 입력에서 사용한 차원으로 만들어서 출력한다(주변단어들도
원핫인코딩 벡터로 표현되어 있을 것이고, 입력에서 사용한 차원과 똑같은 차원으로 되어있기 때문에 사용자가 임의로 정한 임베딩 벡터의 차원을 원핫인코딩 벡터의 차원으로 바꿔줘야만
오차를 계산할 수 있다).
원핫인코딩으로 이루어져있던 입력이, 입력과 같은 차원이지만 임베딩 벡터로 변환되어 있기 때문에 출력층에서 주변 단어를 예측하고 손실을 계산할 수 있다.
softmax에서는 출력된 임베딩 벡터를 각 클래스(여러개의 주변단어)에 속할 확률로 변환시켜서 원핫인코딩으로 이루어진 벡터(여러개의 주변단어)와의 오차를 측정 한다. 이 확률과 실제
타겟값인 원핫인코딩 벡터의 차이는 손실함수로 cross entropy를 활용하고, 계산된 손실정보를 역전파 시켜서 임베딩 벡터값을 업데이트 하게 될 것이다.

#### 3) Window Size
Window size는 중심 단어를 기반으로 주변 단어를 몇개까지 고려할 것인지의 범위이다. Word2Vec에서 window size가 2라면 주변 단어는 중심단어로부터 2 단어 떨어진 단어까지 구성된다.

예시) "He is the best player in the world"

- 중심 단어: He -> 주변단어: (He, is) , (He, the)

- 중심 단어: is -> 주변단어: (is, He), (is, the) , (is, best)

......

- 중심 단어: world -> 주변단어: (world, the), (world, in)

|중심단어|주변단어|
|:-:|:-:|
|He|is|
|He|the|
|is|He|
|is|the|
|is|best|
|...|...|
|world|the|
|world|in| {: .align-center}

위와 같이 학습데이터를 만들고 중심단어를 입력으로 사용해서 문맥단어를 레이블로 분류하는 학습을 진행한다.

### gensim 패키지로 Word2Vec 실습
gensim 패키지는 사전 학습된 임베딩 벡터를 제공함으로써 Word2Vec을 실습하기에 편리하다.

```python
# colab을 사용한다면 업그레이드 시키고 런타임 종료후 다시 실행하자.
!pip install gensim --upgrade
```

```python
import gensim

gensim.__version__ #제대로 업그레이드 되었는지 버전확인
```

gensim 패키지에서 제공하는 구글 뉴스가 말뭉치로 학습된 사전훈련된 word2vec다운로드

```python
import gensim.downloader as api

# 오래 걸리니까 저장하고 불러오자
wv = api.load('word2vec-google-news-300')

# 저장
# wv.save('word2vec-google-news-300.bin')

# 불러오기
#from gensim.models import KeyedVectors
#wv = KeyedVectors.load("word2vec-google-news-300.bin")

#구글 드라이브에서 불러오기
from google.colab import drive
drive.mount('/content/drive')
wv = KeyedVectors.load('/content/drive/MyDrive/AI_camp_data/Section4/Sprint2/word2vec-google-news-300.bin')
```

```python
# king이라는 단어의 임베딩 벡터값
print(wv['soccer'])
```

![image](https://user-images.githubusercontent.com/97672187/166865883-1da7fb28-9132-454e-8aca-bf0006397a93.png){: .align-center}


```python
#말뭉치에 없는 카메룬이라는 단어는 임베딩 벡터가 없기 때문에 에러 발생
print(wv['cameroon'])
```

![image](https://user-images.githubusercontent.com/97672187/166865948-09409752-f1d4-4c47-94e2-5a0d948ffd63.png){: .align-center}


단어간 유사도 파악

- similarity함수로 단어 간 유사도를 파악할 수 있다.

```python
pairs = [
    ('soccer', 'sports'),   
    ('soccer', 'football'),  
    ('soccer', 'Korea'),
]

# 축구와 관련된 스포츠, 풋볼이라는 단어는 유사도가 높다. 
# 하지만 Korea와는 유사도가 거의 없는 것을 확인할 수 있다.
for w1, w2 in pairs:
    print(f'{w1} 과 {w2}의 유사도 \t  {wv.similarity(w1, w2):.2f}')
```

![image](https://user-images.githubusercontent.com/97672187/166866153-99e5232a-a40d-4e8a-8c0f-7708ce392a4e.png){: .align-center}

- most_similar 함수로 가장 비슷한 단어를 볼 수 있다.

```python
#soccer의 벡터와 player의 벡터를 더해서 가장 유사한 단어 3개를 출력
#더할 때는 positive
for i, (word, similarity) in enumerate(wv.most_similar(positive=['soccer', 'player'], topn=3)):
    print(f"Top {i+1} : {word}, {similarity}")
```

![image](https://user-images.githubusercontent.com/97672187/166866349-bb1aac4d-4bea-4d2f-a7a2-d01320770906.png){: .align-center}

두 벡터를 더하고, 다른 벡터를 빼줄 때 나오는 값

```python
# band와 guitar를 더해주고 sing을 빼니까
# guitalist가 출력
print(wv.most_similar(positive=['band', 'guitar'], negative=['sing'], topn=1))
```
![image](https://user-images.githubusercontent.com/97672187/166866656-01884900-2760-4f52-9d36-37fb4231d506.png){: .align-center}

- 가장 관계없는 단어 출력

doesnt_match 함수 사용

```python
# 스포츠 중에 수학이 껴있으니까
# 가장 관련없는 수학이 출력
print(wv.doesnt_match(['soccer', 'baseball', 'basetball', 'swimming', 'math']))
```

![image](https://user-images.githubusercontent.com/97672187/166866787-6f68d7b3-26e5-4874-b27e-1212296108f4.png){: .align-center}

### fastText



```python

```

```python

```

```python

```

```python

```

```python

```

```python

```



