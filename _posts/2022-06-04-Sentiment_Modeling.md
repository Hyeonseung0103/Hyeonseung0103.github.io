---
layout: single
title: "피파온라인 댓글 감성분석 프로젝트 모델링"
toc: true
toc_sticky: true
category: Sentiment
---

모델링을 진행하기전에 각 댓글을 형태소 기반으로 토큰화하고 불필요한 특수문자나 불용어들을 제거할 것이다. 모델이 단어를 이해할 수 있게끔 각 단어를 정수로 맵핑해주는 정수 인코딩과
모든 댓글의 길이를 동등하게 맞춰주는 패딩의 과정을 거쳐 모델링을 진행했다. 초기 모델링 후에는 성능을 높이기 위해 하이퍼 파라미터 튜닝을 진행했다.

## Comments Sentiment Analysis 프로젝트 모델링

### 1. 토큰화 및 정수 인코딩
구글의 형태소 분석기인 Mecab을 사용해서 댓글을 토큰화 했다.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import time
import os
import urllib.request
from konlpy.tag import Mecab
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from collections import Counter
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout,BatchNormalization
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
```

```python
DATA_PATH = '.....'
df = pd.read_csv(f'{DATA_PATH}sentiment_scores.csv')

#감성분석에 필요한 comment와 label만 남겨두기
df = df[['comment', 'label']]
df.tail()
```

![image](https://user-images.githubusercontent.com/97672187/171999443-76d207dd-92da-4f69-bac0-bcbc8827a9b6.png){: .align-center}

<br>


<br>


```python
# 중복과 결측치 확인
print(df.duplicated().sum())
print(df.isnull().sum().sum())
```

![image](https://user-images.githubusercontent.com/97672187/171999476-ddeafddf-89ba-429f-8884-5c28f7cf19ee.png){: .align-center}

<br>


<br>


```python
#중복제거
df.drop_duplicates(inplace = True)
print(df.duplicated().sum())
df.shape
```

![image](https://user-images.githubusercontent.com/97672187/171999491-15fc0e06-3b90-498b-827a-8b4bef23ca2e.png){: .align-center}

<br>


<br>


부정 댓글이 더 많은 불균형 데이터이기 때문에 stratify 파라미터를 사용해서 label 데이터가 한쪽에만 치우치지 않게 한다.

```python
X_train, X_test = train_test_split(df, test_size = 0.2, 
                                   random_state = 42, stratify = df['label'])
display(X_train.tail())
print(X_train.shape)
display(X_test.tail())
print(X_test.shape)
```

![image](https://user-images.githubusercontent.com/97672187/171999544-f1a2b463-23d1-4524-ad87-88f8807727ce.png){: .align-center}

<br>


<br>

```python
#숫자, 알파벳, 한글, 공백 외에는 모두 제거
X_train['comment'] = X_train['comment'].str.replace('[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣 ]','')
X_test['comment'] = X_test['comment'].str.replace('[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣 ]','')

#중복 확인 및 제거
display(X_train[X_train.duplicated(subset = 'comment')].head())
display(X_test[X_test.duplicated(subset = 'comment')].head())

X_train.drop_duplicates(subset = 'comment',inplace = True)
X_test.drop_duplicates(subset = 'comment',inplace = True)
print(X_train.duplicated().sum())
print(X_test.duplicated().sum())
```

![image](https://user-images.githubusercontent.com/97672187/171999620-bce852e8-ddde-4def-bf9c-0dcc54edfbe5.png){: .align-center}

<br>


<br>


```python
X_train.shape, X_test.shape
```

![image](https://user-images.githubusercontent.com/97672187/171999628-94be9e2a-da8d-4c62-9a73-da921fe63b78.png){: .align-center}

<br>


<br>


불용어 제거 및 토큰화

```python
# 불용어 사전
# 남아있는 거래 관련 단어, 필요없는 접속사 관사 등
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', 
             '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게', '구매합니다', '올렸습니다',
             '팔렸나요', '팔아주신분', '얼만가요', '판매합니다']
```

```python
mecab = Mecab()

# 토큰화, 불용어 제거
X_train['tokenized'] = X_train['comment'].apply(mecab.morphs)
X_train['tokenized'] = X_train['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

X_test['tokenized'] = X_test['comment'].apply(mecab.morphs)
X_test['tokenized'] = X_test['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
```

```python
X_train_tokenized = X_train['tokenized'].values
X_test_tokenized = X_test['tokenized'].values
y_train = X_train['label'].values
y_test = X_test['label'].values

X_train_tokenized[0][:10]
```

![image](https://user-images.githubusercontent.com/97672187/171999785-14ca7b3d-e5f7-43c2-a576-1c0322ab8602.png){: .align-center}

<br>


<br>


```python
t = Tokenizer()
t.fit_on_texts(X_train_tokenized)
```

희귀 단어의 비율 확인

```python
threshold = 2
total_cnt = len(t.word_index) #단어 수
rare_cnt = 0 # 2번보다 적게 나온 단어 갯수
total_freq = 0 #학습 데이터 전체 단어의 빈도수 합
rare_freq = 0 # 희귀 단어 전체의 빈도수 합

for key,value in t.word_counts.items():
  total_freq = total_freq + value

  if value < threshold:
    rare_cnt = rare_cnt + 1
    rare_freq = rare_freq + value

print('전체 단어의 수', total_cnt)
print('2번 이하로 등장한 희귀 단어 수', rare_cnt)
print('전체 단어 중 희귀 단어 비율', (rare_cnt / total_cnt) * 100)
print('전체 단어의 빈도에서 희귀 단어 빈도의 비율', (rare_freq / total_freq) * 100)

```

![image](https://user-images.githubusercontent.com/97672187/171999827-51d82a7c-b584-450e-9b0f-131fcc7117fc.png){: .align-center}

<br>


<br>

전체 단어 중 희귀 단어가 46퍼센트지만 등장 빈도는 0.5% 밖에 되지 않기 때문에 별로 중요하지 않을 것 같다. 희귀단어를 제거해보자.

```python
# 전체 단어 - 희귀 단어 = 2번 이상 등장한 단어.
# 빈도수가 높은 단어순으로 저장되기 때문에 희귀 단어가 제거 된다.
# +2는 패딩과 OOV 토큰 고려

vocab_size = total_cnt - rare_cnt + 2
print(vocab_size)
```

![image](https://user-images.githubusercontent.com/97672187/171999863-ec198a55-3552-4457-ad0b-cffa1fddae67.png){: .align-center}

<br>


<br>

정수 인코딩

```python
t = Tokenizer(vocab_size, oov_token = 'OOV') 
t.fit_on_texts(X_train_tokenized)
X_train_encoded = t.texts_to_sequences(X_train_tokenized)
X_test_encoded = t.texts_to_sequences(X_test_tokenized)

print(X_train_encoded[:2])
print(X_test_encoded[:2])
```

![image](https://user-images.githubusercontent.com/97672187/171999928-69877c34-ea0e-467e-b6ec-8240130792ff.png){: .align-center}

<br>


<br>

### 2. 패딩

패딩을 위한 최대 길이와 평균 길이 파악

```python
print('리뷰 최대 길이 ', max(len(review) for review in X_train_encoded))
print('리뷰 평균 길이 ', sum(map(len, X_train_encoded)) / len( X_train_encoded))
```

![image](https://user-images.githubusercontent.com/97672187/172000196-0a1062b9-4503-41ba-93ba-e2ca6e09b1dd.png){: .align-center}

<br>


<br>


```python
def below_threshold_len(max_len, data):
  count = 0
  for sentence in data:
    if(len(sentence) <= max_len):
        count = count + 1
  print(count / len(data)*100)
```

```python
# 길이를 200으로 하면 약 98퍼센트 이상 데이터를 보존할 수 있다.
# 200의 길이로 패딩하자.
below_threshold_len(200, X_train_encoded)
```

![image](https://user-images.githubusercontent.com/97672187/172000292-4f109e55-c8d8-47ab-b9cc-07a1bc75eeea.png){: .align-center}

<br>


<br>

```python
#패딩
max_len = 200
X_train_padded = pad_sequences(X_train_encoded, maxlen = max_len)
X_test_padded = pad_sequences(X_test_encoded, maxlen = max_len)
X_train_padded[0][100:150]
```

![image](https://user-images.githubusercontent.com/97672187/172000301-edbc418c-3b05-40f0-9093-a7ca3a24b9b2.png){: .align-center}

<br>


<br>

### 3. 초기 모델링
패딩까지 완료된 토큰화된 댓글데이터를 input으로 사용하고 케라스에서 제공하는 임베딩 벡터, 하나의 LSTM 층을 쌓아서 
긍정 혹은 부정을 예측하는 이진분류 LSTM 모델을 만들었다. 검증 데이터는 학습 데이터의 20%를 사용했고, 과적합을 방지하기 위해  5번 동안 검증 데이터의 성능 개선이 없으면
조기 종료가 되도록 했다. 가장 좋은 성능을 낸 모델은 'best_lstm_model_fifa' 라는 이름의 파일로 저장한다.

```python
np.random.seed(42)
tf.random.set_seed(42)

embedding_dim = 300
hidden_units = 256

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim)) # 임베딩층
model.add(LSTM(hidden_units)) # LSTM
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('/content/drive/MyDrive/AI_camp_data/Section4/project/best_lstm_model_fifa.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_padded, y_train, epochs=100, callbacks=[es, mc], batch_size=128, validation_split=0.2)
```

![image](https://user-images.githubusercontent.com/97672187/172000455-c4c3770b-fd6f-496e-8a4a-07d6e941f97f.png){: .align-center}

<br>


<br>


```python
loaded_model = load_model('/content/drive/MyDrive/AI_camp_data/Section4/project/best_lstm_model_fifa.h5')
print("\n 학습 정확도: %.4f" % (loaded_model.evaluate(X_train_padded, y_train)[1]))
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test_padded, y_test)[1]))
```

![image](https://user-images.githubusercontent.com/97672187/172000550-1b9de55c-cb54-4192-ac82-95f7d1437e10.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/172000567-8b271eb6-08d1-49a2-9693-90169b7d4c5e.png){: .align-center}

<br>


<br>

### 4. 하이퍼 파라미터 튜닝
Dense 은닉층 따로 설계하지 않고 임베딩 층과 LSTM층, 출력층만 사용해서 초기 모델링을 진행했을 때 학습 데이터의 성능은 정확도가 **0.8974**, 테스트 데이터는 **0.8883**이었다.
과적합은 딱히 존재하지 않는다고 판단했다. 


#### 1) Randomized Search CV

성능을 높이기 위해 임베딩층과 LSTM층 사이에 Dense층을 하나 더 추가하고 Batch 정규화를 사용하여 학습이 더 잘 되도록 한다. 또 노드의 갯수, 에포크수, 배치 사이즈, 활성화 함수,
옵티마이저와 같은 하이퍼 파라미터들을 Randomized Search CV를 사용해서 튜닝을 진행해보자.


```python
#모델 설계
def make_model_rscv(units = 128, activation = 'relu', optimizer = 'adam'):
  embedding_dim = 300
  model = Sequential()
  model.add(Embedding(vocab_size, embedding_dim)) # 임베딩층
  model.add(Dense(units, activation = activation, kernel_initializer='he_normal')) # he 가중치 초기화
  model.add(BatchNormalization())
  model.add(LSTM(units)) # LSTM
  model.add(Dense(1, activation = 'sigmoid'))

  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])  

  return model
```

```python
np.random.seed(42)
tf.random.set_seed(42)

units = np.arange(32,256,32)
epochs = [10,20,25]
batch_size = [32,64,128,256]
activation = ['relu', 'sigmoid', 'tanh']
optimizer = ['adam', 'adagrad', 'rmsprop']

param_grid = dict(batch_size = batch_size, units = units, activation = activation, optimizer = optimizer, epochs = epochs)
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)

model_rscv = KerasClassifier(build_fn=make_model_rscv, verbose=0, callbacks = [es])
```

Randomized Search CV 실행

```python
start = time.time()
with tf.device('/GPU:0'):
    rscv = RandomizedSearchCV(
        estimator=model_rscv,
        param_distributions=param_grid,
        n_iter=5,
        cv=2,
        scoring='accuracy',
        verbose=1,
        n_jobs=2,
        random_state=42)

    rscv_result = rscv.fit(X_train_padded, y_train)

print(time.time() - start))
```

![image](https://user-images.githubusercontent.com/97672187/172003291-137f0896-3b5d-4fbf-a810-7e7b1df3781b.png){: .align-center}

<br>


<br>


```python
print(f"Best: {rscv_result.best_score_} using {rscv_result.best_params_}")
```

![image](https://user-images.githubusercontent.com/97672187/172003320-79d258dc-41b9-4642-a492-d6ea5e3ea514.png){: .align-center}

<br>


<br>

최적의 하이퍼 파라미터로 재학습

```python
np.random.seed(42)
tf.random.set_seed(42)

units = rscv_result.best_params_.get('units')
optimizer = rscv_result.best_params_.get('optimizer')
batch_size = rscv_result.best_params_.get('batch_size')
activation = rscv_result.best_params_.get('activation')
epochs = rscv_result.best_params_.get('epochs')

best_model = make_model_rscv2(units = units, activation = activation,optimizer = optimizer)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint(f'{DATA_PATH}best_lstm_model_fifa2.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

with tf.device('/GPU:0'):
    history = best_model.fit(X_train_padded, y_train, epochs = epochs, batch_size = batch_size, callbacks = [es,mc], 
                             validation_split=0.2)
```

![image](https://user-images.githubusercontent.com/97672187/172003331-205b2b69-9d89-4cd6-a553-940742a2ff78.png){: .align-center}

<br>


<br>

테스트 데이터의 정확도가 0.8883에서 **0.8938**로 증가했다.

```python
loaded_model = load_model(f'{DATA_PATH}best_lstm_model_fifa2.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test_padded, y_test)[1]))
```

![image](https://user-images.githubusercontent.com/97672187/172003344-8a5d7568-aabf-488d-8c46-b39002e8e46a.png){: .align-center}

<br>


<br>

#### 2) Grid Search CV
Randomized Search CV로 하이퍼 파라미터의 범위가 어느 정도일 때 성능이 높은지 확인했고, 이 범위를 좀 더 좁혀서 Grid Search CV를 진행했다.

```python
units = [64,128,200,256]
epochs = [20]
batch_size = [128,256]
activation = ['relu', 'sigmoid']
optimizer = ['rmsprop', 'adam']

param_grid = dict(batch_size = batch_size, units = units, activation = activation, optimizer = optimizer, epochs = epochs)
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)

model_grid = KerasClassifier(build_fn=make_model_rscv, verbose=0, callbacks = [es])
```

```python
np.random.seed(42)
tf.random.set_seed(42)

start = time.time()
with tf.device('/GPU:0'):
    grid = GridSearchCV(
        estimator=model_grid,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1,
        n_jobs=2)

    grid_result = grid.fit(X_train_padded, y_train)

print((time.time() - start)/60)
```

![image](https://user-images.githubusercontent.com/97672187/172008217-9844d753-3fc3-4bae-9160-ffc66dd7c815.png){: .align-center}

<br>


<br>


```python
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
```

![image](https://user-images.githubusercontent.com/97672187/172008107-41ae7f59-9e8e-46a6-bb34-8c8b8b9f8ed5.png){: .align-center}

<br>


<br>


```python
np.random.seed(42)
tf.random.set_seed(42)

units = grid_result.best_params_.get('units')
optimizer = grid_result.best_params_.get('optimizer')
batch_size = grid_result.best_params_.get('batch_size')
activation = grid_result.best_params_.get('activation')
epochs = grid_result.best_params_.get('epochs')

best_model2 = make_model_rscv2(units = units, activation = activation,optimizer = optimizer)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint(f'{DATA_PATH}best_lstm_model_fifa3.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

with tf.device('/GPU:0'):
    history = best_model2.fit(X_train_padded, y_train, epochs = epochs, batch_size = batch_size, callbacks = [es,mc], 
                             validation_split=0.2)
```

![image](https://user-images.githubusercontent.com/97672187/172008276-3e2dc680-eff5-4603-badf-5abefcb44a90.png){: .align-center}

<br>


<br>

```python
loaded_model2 = load_model(f'{DATA_PATH}best_lstm_model_fifa3.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model2.evaluate(X_test_padded, y_test)[1]))
```

![image](https://user-images.githubusercontent.com/97672187/172008306-593d32de-c030-4280-ba0b-5b4d63739417.png){: .align-center}

<br>


<br>

Randomzied Search CV로 범위를 잡고 Grid Search CV로 세부적으로 탐색한 결과 Grid Search CV를 사용했을 때 테스트 데이터의 정확도가 **0.8954**가 나오면서 기존의
0.8883과 Randomized Search CV의 0.8938보다 높은 정확도를 기록했다.

다음 포스팅에서는 감성 점수 데이터를 기반으로 한 추천 시스템과 분석 결과해 대해 정리해보겠다.



