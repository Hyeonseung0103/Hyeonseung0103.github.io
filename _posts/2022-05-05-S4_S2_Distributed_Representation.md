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
|world|in|

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

### Word2Vec을 사용하여 문장 분류하기(사전에 정의된 임베딩 벡터 사용)
문장 분류를 사용하는 방법 중 가장 간단한 것은 단어 벡터를 모두 더한 뒤 평균을 내는 방법이다. 임베딩 벡터를 사용하여 문장분류를 해보자.


- 데이터 불러오기 및 EDA

```python
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.datasets import imdb
```

```python
# 시드 고정
tf.random.set_seed(42)
```

```python
# 빈도수 상위 20000개의 단어만 사용해서 불러오기
# 각 문장들은 이 20000개의 단어들 중 몇개로 이루어져있음
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=20000)
```

```python
# 단어가 이렇게 정수 인코딩 되어 표현되어 있다.
X_train[0][:10]
```

![image](https://user-images.githubusercontent.com/97672187/166893057-923882e5-859c-49b6-8c0a-f00a0f9acc69.png){: .align-center}

```python
np.unique(y_train) #0 아니면 1의 문장 카테고리로 분류
```

```python
# key:단어, val:인덱스 로 되어있는걸
# key: 인덱스, val: 단어로 바꿈.
# 나중에 어떤 문장인지 확인하기 위해서
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

print(word_index['king'])
print(reverse_word_index[708])

#이 함수를 실행하면 인덱스로 이루어진 문장을 받아
# 알맞은 단어로 변환하여 출력
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
```

![image](https://user-images.githubusercontent.com/97672187/166893148-b806ffd1-4cfd-4b78-a95a-21dedf89d189.png){: .align-center}


```python
decode_review(X_train[0][:100])
```

![image](https://user-images.githubusercontent.com/97672187/166893233-6577fab6-9a9a-4308-862c-485640bf9e04.png){: .align-center}

- 토큰화

```python
#정수 인코딩 되어 있는 문장을 단어 문장으로 변환
sentences = [decode_review(idx) for idx in X_train]

#토큰화
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
```

```python
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size) # 19999
```

- Padding

padding은 모든 벡터의 차원을 동일하게 맞춰주는 것으로, 모든 문장의 길이가 같아져서 컴퓨터가 이를 하나의 행렬로 보고 병렬 처리를 할 수 있다. 3단어로 이루어진 문장이 있다고 했을때
만약 해당 문서에서 가장 긴 문장이 100단어로 이루어진 문장이라면 두 문장의 길이가 다르기 때문에 3단어의 문장에 97개의 일정한 수로 인코딩 함으로써(주로 0, zero padding) 길이를
맞춰준다. 꼭 긴 문장의 길이만큼 패딩을 할 필요는 없지만, 만약 100개의 단어 길이에 50개로만 padding을 한다고 하면 50개 이후의 단어가 사라지기 때문에 적절한 패딩의 크기를 정해야한다. 주로 가장 긴 단어의 길이를 사용하거나 평균 단어 길이보다 조금 더 큰 값을 크기로 사용한다. 컴퓨터가 병렬처리를 할 때 패딩으로 표현한 수는 연산없이 바로 넘기기 때문에 차원이
커졌다고 해서 계산량이 엄청 증가하는 것은 아니다.

```python
# 토큰화된 단어에 정수 인코딩
X_encoded = tokenizer.texts_to_sequences(sentences)

# 문서에서 가장 긴 문장의 길이
max_len = max(len(sent) for sent in X_encoded)
print(max_len) # 2494

# 문장들의 평균길이
print(np.mean([len(sent) for sent in X_train]))  # 238
```

```python
#평균길이보다 조금 더 긴 400개의 차원으로 패딩
#패딩은 0을 사용하고 길이가 짧으면 존재하는 단어 뒤에 모두 0으로 채워넣음
X_train=pad_sequences(X_encoded, maxlen=400, padding='post')
y_train=np.array(y_train)
```

```python
X_train[0]
```

![image](https://user-images.githubusercontent.com/97672187/166896951-e8a92b05-cc7a-4be1-b9ee-ee70143295f8.png){: .align-center}


- 임베딩 가중치 행렬 만들기

미리 학습된 300만개의 단어를 모두 쓰면 행렬이 너무 커지니까 현재 사용하는 단어인 19999개(vocab_size)의 단어만 임베딩 가중치 행렬로 만듬.

```python
#차원은 300
embedding_matrix = np.zeros((vocab_size, 300))

print(np.shape(embedding_matrix)) # (19999, 300)
```


```python
#imdb에서 불러온 단어가 gensim에서 불러온 임베딩 벡터의 단어이면 불러옴
def get_vector(word):
    if word in wv:
        return wv[word]
    else:
        return None

for word, i in tokenizer.word_index.items():
    temp = get_vector(word)
    if temp is not None:
        embedding_matrix[i] = temp
```

- 신경망 만들기 & 학습

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
```

```python
model = Sequential()

#trainable = False는 사전에 훈련된 임베딩 벡터가 아니니까 모델링하면서 업데이트 시키라는 뜻
model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=400, trainable=False))
model.add(GlobalAveragePooling1D()) # 입력되는 단어 벡터의 평균을 구하는 함수
model.add(Dense(1, activation='sigmoid'))
```

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, y_train, batch_size=64, epochs=20, validation_split=0.2)
```

![image](https://user-images.githubusercontent.com/97672187/166898620-965f078d-ec11-4404-bcf5-335f1ead3e48.png){: .align-center}


- 검증

```python
test_sentences = [decode_review(idx) for idx in X_test]

X_test_encoded = tokenizer.texts_to_sequences(test_sentences)

X_test=pad_sequences(X_test_encoded, maxlen=400, padding='post')
y_test=np.array(y_test)

model.evaluate(X_test, y_test)
```

![image](https://user-images.githubusercontent.com/97672187/166898661-0e452d9a-8642-49de-b446-87797315dcc8.png){: .align-center}


### fastText
Word2Vec에서는 임베딩 벡터에 존재하지 않은 단어의 벡터를 찾으려고 하면 에러가 발생한다. 위의 예시에서는 'cameroon'과 같은 단어다.
아무리 모든 단어를 다 수집하려고 노력해도 세상에 존재하는 모든 단어가 들어있는 말뭉치를 구하는 것은 불가능하다. 

이렇게 말뭉치에 등장하지 않는 단어가 등장하는 **문제를 OOV(Out of Vocabulary)** 문제 라고 한다. 또한, Word2Vec은 적게 등장하는 단어에 대해서 학습이 적게 일어나기 때문에
적절한 임베딩 벡터를 생성해내지 못한다는 단점이 존재한다.

이 문제를 해결하기 위해 등장한 것이 철자 단위 임베딩Character level Embedding)이다. fastText는 철자 단위의 임베딩을 보조 정보로 사용해서 OOV의 문제를 해결했다. 즉,
철자 단위로 쪼개고, 단어의 의미를 파악해서 처음 보는단어도 임베딩 벡터로 표현할 수 있게 된다.

- fastText가 철자 단위 임베딩을 적용하는 법: Chracter n-gram

fastText는 3 ~ 6개의 철자로 묶은 3~6 grams의 단위를 사용한다. 또한, 모델은 해당 단어에 앞뒤로 "<", ">"를 붙여서 접두사와 접미사를 인식할 수 있도록 한다.

예시: playing

|word|Length|Character n-grams
|:-:|:-:|:-:
|playing|3|<pl, pla, lay, ayi, yin, ing, ng>
|playing|4|<pla, play, layi, ayin, ying, ing>
|playing|5|<play, playi, layin, aying, ying>
|playing|6|<playi, playin, laying, aying>
|...|...|

다른 단어가 있다면 다른 단어에 대해서도 다음과 같이 n-gram을 수행하고, fastText에서는 이렇게 얻어진 n-gram들의 임베딩 벡터를 모두 구하게 된다.
경우의 수가 많지만, 알고리즘이 매우 효율적으로 구현되어 있어서 시간상으로 Word2Vec과 엄청난 차이가 나진 않는다.

만약 playing이라는 단어가 기존의 말뭉치에 있었다면 skip-gram으로 학습한 임베딩 벡터에 위에서 얻은 22개의 n-gram들의 벡터를 더해 준다. 반대로, 존재하지 않은 단어라면
해당 단어는 이 22개의 n-gram들의 벡터로만 구성된다.

### gensim 패키지로 fastText 실습

위에서 imdb를 단어를 맵핑하여 문장으로 바꿔놓은 sentences 리스트를 활용하여 FastText모델에 학습시켜 보자.

```python
sentences[0]
```

![image](https://user-images.githubusercontent.com/97672187/166908049-c13078d6-9ab7-4f99-a20d-3f20284d3714.png){: .align-center}


```python
import spacy
from spacy.tokenizer import Tokenizer

nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)
tokens = []
for sentence in tokenizer.pipe(sentences):
  current_tokens = [word.text for word in sentence if word.text not in ['\n', '\n\n', ' ']]
  tokens.append(current_tokens)
tokens[0][0:10]
```

![image](https://user-images.githubusercontent.com/97672187/166908103-e2b940c9-6c28-4958-9428-0adbfe8d65cb.png){: .align-center}


```python
# 위에서 사용한 sentences를 토큰화 한 tokens리스트를 FastText모델에서 학습 시켜보자.
model = FastText(tokens,vector_size=100)
```

```python
ft = model.wv
print(ft)

print(f"soccer 이라는 단어가 있을까?  {'soccer' in ft.key_to_index}")
print(f"electronicsoccer 이라는 단어가 있을까?  {'electronicsoccer' in ft.key_to_index}")
```

![image](https://user-images.githubusercontent.com/97672187/166908165-2186f666-ffdf-4162-a74e-7cc95b2c9061.png){: .align-center}


```python
print(ft['soccer'])
```

![image](https://user-images.githubusercontent.com/97672187/166908193-9bf6bc89-64cd-48e0-b789-60ea3aeb7344.png){: .align-center}


```python
#존재하지 않는 electronicsoccer라는 단어에도 임베딩 벡터가 있다. 
print(ft['electronicsoccer'])
```

![image](https://user-images.githubusercontent.com/97672187/166908300-ecf08772-20f4-41ce-9979-9575ce79645a.png){: .align-center}

```python
#유사도
print(ft.similarity("soccer", "electronicsoccer"))
```

```python
#비슷한 단어 3개
print(ft.most_similar("electronicsoccer")[:3])
```

![image](https://user-images.githubusercontent.com/97672187/166908378-bfee8255-db0f-4ae0-9fd3-54ac20a90a10.png){: .align-center}


```python
#비슷한 단어 3개
print(ft.most_similar("electronicsoccer")[:3])
```

![image](https://user-images.githubusercontent.com/97672187/166908427-d575abcf-6d87-44fb-b8d1-adf9c183450e.png){: .align-center}


```python
# 유사하지 않은 단어도 생각보다 잘 걸러낸다.
print(ft.doesnt_match("soccer baseball player worker".split()))
```

![image](https://user-images.githubusercontent.com/97672187/166908447-fa3f893a-6430-48b8-a161-dca7176ee2c2.png){: .align-center}


하지만, FastText는 철자단위로 단어를 임베딩 하기 때문에 단어의 **의미보다는 생김새나 구조**에 더 비중을 둬서 단어의 의미를 잘 분류하지 못할 수도 있다.


### Negative Sampling
Word2Vec에서는 역전파 과정에서 기준 단어나 문맥 단어와 전혀 상관 없는 단어의 임베딩 벡터값도 업데이트 된다. window size가 2라고 한다면, 중심 단어로부터 3단어가 떨어진 단어는
주변 단어가 아니기 때문에 굳이 멀리있는 단어까지 꼭 임베딩 벡터값을 조정하지 않아도 되는데 만약 사전의 크기가 매우 크면 모든 단어의 임베딩 벡터를 조정하는 것은무거운 작업이 될 것이다. 이를 해결하기 위해 Negative Sampling은 임베딩 조절시에 전체 단어 집합이 아닌, 일부 단어집합만 조정한다. 기준 단어 주변에 등장한 문맥단어를 positive sample, 기준 단어 주변에
등장하지 않은 단어를 negative sample로 나눌 수 있다. 
Negative sampling은 기준단어와 관련이 없는 negative sample들은 굳이 다 업데이트 하는 것이 아니라 문맥 단어수의 + 20개를 빈도수가 높은 단어 순으로 뽑는다.

예를 들면, "I like play soccer with my friends" 라는 문장이 있으면

window size가 2이고, soccer 이라는 단어를 학습할 때 주변단어는 like, play, with, my이다. 이 4단어가 positive sample에 해당하고 negative sample은 이 friends, I와 같은 
주변 단어가 아닌 단어들 중 문서에서 빈도수가 높은 단어를 20개를 추가한다고 한다. negative sample 중 20개의 단어만 벡터값을 업데이트 하기 때문에 역전파 과정에서 모든 단어의 임베딩 벡터를 업데이트 시켜서 발생할 수 있는 연산량을 훨씬 줄일 수 있다.



