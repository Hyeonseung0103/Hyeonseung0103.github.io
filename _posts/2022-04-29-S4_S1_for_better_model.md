---
layout: single
title: "Note 413 학습률, 가중치 초기화, 정규화"
toc: true
toc_sticky: true
category: Section4
---

딥러닝도 머신러닝의 일부라고 할 수 있기 때문에 성능을 더 높이거나 과적합을 해소하기 위한 방법들이 머신러닝과 유사하다. 이번 포스팅에서는 학습률, 가중치 초기화, 정규화 등의 방법을 사용하여 
신경망의 성능을 어떻게 높이고, 과적합을 해소할 수 있는지 다뤄보자.

### 학습률(Learning rate)
학습률은 매 가중치에 대해 구해진 기울기 값을 경사 하강법에 얼마나 적용할 지 결정하는 하이퍼 파라미터이다. 쉽게 말해, 경사하강법에서 경사를 얼만큼 내려갈 것인가의 보폭을 결정한다. 
지난 포스팅(Note 412)에서 다뤘던 것처럼 학습률이 너무 크면 경사 하강 과정에서 발산해버려서 모델의 최적값을 찾을 수 없게 되고, 학습률이 너무 작으면 해당 최저점을 찾기까지 시간이
매우 오래걸릴 뿐만 아니라 주어진 iteration 내에 최저점에 도달하지 못하고 학습이 종료될 수 있다. 이때 사용할 수 있는 것이 학습률 감소법과 학습률 계획법이다.

1) 학습률 감소(Learning rate Decay)

학습률 감소는 처음에는 빨리 내려가다가 최저점에 가까이 왔다가 생각하면 그때부터 학습률을 감소시켜 천천히 하강하는 것을 말한다. Adagrad, RMSprop, Adam과 같은 옵티마이저에 이미 구현되어
있고, 하이퍼파라미터를 조정하면서 감소 정도를 변화시킬 수 있다.

```python
#optimizer 하이퍼 파라미터에 적용
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1 = 0.89), 
              loss = 'sparse_categorical_crossentropy', 
              metrics = ['acc'])          
```              

2) 학습률 계획(Learning rate Scheduling)

학습률 계획법은 처음에 학습률은 0에서 시작해서 일정단계까지 웜업 스텝을 가진후 학습률을 감소 시키는 방법이다. 즉, 웜업스텝에는 학습률을 오히려 증가시키는데 천천히 증가시키면서
변화를 보다가 웜업 스텝이 끝나고 학습 속도가 느려지면 학습률을 낮춰서 최적의 고정 학습률보다 좋은 솔루션을 더 빨리 발견할 수 있도록 한다.

![image](https://user-images.githubusercontent.com/97672187/165868350-0cc2abec-b3fd-404a-855b-7d1d7cecf91b.png){: .align-center}

이미지출처: https://hwk0702.github.io/ml/dl/deep%20learning/2020/08/28/learning_rate_scheduling/

```python
initial_learning_rate = 0.01 #매 주기의 최초 학습률
first_decay_steps = 1000 # 학습률 감소의 주기. 1000 step마다 학습률 갑소
t_mul = 2.0 #주기 T를 늘려갈 비율 (첫 주기가 100step 이면 다음은 200step, 400step...)
m_mul = 1.0 # 최초 학습률로 설정한 값에 매 주기마다 곱해줄 값
#(0.9라고 학면 매 주기 시작마다 initial_learning_rate에 0.9 * 주기 순서 를 곱한 값을 주기 시작 학습률로 사용.
alpha = 0.0 # 학습률의 하한을 설정. 학습률의 감소 하한은 initial_learning_rate * alpha. 최대 이만큼 까지 감소할 것이다?

lr_decayed_fn = (
  tf.keras.experimental.CosineDecayRestarts(
      initial_learning_rate,
      first_decay_steps,
      t_mul,
      m_mul,
      alpha))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn)
             , loss='sparse_categorical_crossentropy'
             , metrics=['accuracy'])
```

### 가중치 초기화(Weight Initialization)
초기에 가중치를 어떻게 설정하느냐에 따라 신경망의 성능이 크게 달라질 수 있다. 가중치 초기화는 경사하강법에서 어디서부터 출발할지 시작점을 결정하는 것.
매 Iteration마다 하는 것이 아니라 첫 에포크를 시작할 때 처음 한 번만 가중치값을 초기화 해주고 그 뒤로는 하지 않는다. 만약 매 Iteration마다 가중치가 초기화 되면 가중치를
업데이트 하는 의미가 없어진다. 또한, 보통 가중치 초기화는 정규분포를 사용하는데 만약 분포를 사용하지 않고 매번 한 숫자로 가중치를 초기화 시키면 모든 가중치가 같은 값으로 업데이트
될 것이기 때문에 의미가 없어진다. 

정규분포로 가중치를 초기화 하면 활성화 값이 고르게 분포하기 때문에 정규분포를 사용한다.

1) 표준편차가 1인 정규분포로 가중치를 초기화 할 때

밑의 그래프는 **노드의 수와 상관없이** 표준편차가 1인 일정한 정규분포로 가중치를 초기화 해주고, 활성화 함수는 시그모이드를 사용한 활성화 값을 나타낸 그래프입니다.
대부분의 출력값이 0과 1에 몰려있는 것을 볼 수 있는데 시그모이드 함수에서 활성화 값이 0과 1 이라는 것은 미분값이 0에 가깝다고 할 수 있기 때문에
기울기 소실(Gradient Vanishing) 문제가 발생할 수 있습니다. 또한, 활성화 값이 고르지 않을때 학습이 제대로 이루어지지 않기 때문에 간단하지만 잘 사용되지 않는다.

![image](https://user-images.githubusercontent.com/97672187/165870133-9a140cc3-e35b-4220-a3c8-b642cec3bd69.png){: .align-center}

2) Xavier 초기화를 사용했을 때

사비에르 초기화를 사용하면 가중치를 초기화할 때 표준편차가 일정해서 발생하는 문제점을 해소할 수 있다. Xavier 초기화는 이전 층의 노드가 n개 일 때, 현재 층의 가중치를 표준편차가
![image](https://user-images.githubusercontent.com/97672187/165869987-b3e51a7c-0435-4db0-8413-22cafe0ecf43.png) 인 정규분포로 초기화 한다.

(케라스에서의 Xavier 초기화는 이전 층의 노드가 n개, 현재 층의 노드가 m개라면 현재 층의 가중치를 표준편차가 ![image](https://user-images.githubusercontent.com/97672187/165870053-c97760fe-038e-40e0-a027-ce3e0d16b548.png)
인 정규분포로 초기화 한다.

쉽게 말해 각 층마다 노드의 수가 다를텐데, 이 다른 노드의 수를 반영해서 표준편차에 다른 값을 사용한다.

밑의 그래프도 시그모이드 활성화 함수에 Xavier 초기화 방법을 사용했을 때다. 활성화 값이 고르게 분포한다는 것을 알 수 있다.

![image](https://user-images.githubusercontent.com/97672187/165870148-c82fe21f-1d59-45d8-b65d-597599819cf6.png){: .align-center}

3) He(Kiming He) 초기화를 사용했을 때

![image](https://user-images.githubusercontent.com/97672187/165870431-e41bda05-be23-4831-9d7d-ba0b85dc38e6.png){: .align-center}

위의 그래프는 ReLU함수에 Xavier 초기화를 적용했을 때 활성화 값의 분포이다.

Xavier 초기화는 시그모이드 함수를 사용한 신경망에서는 잘 동작하지만, ReLU일 경우 활성값이 고르지 못하는 문제가 있다. He는 이전 층의 노드가 n개일 때, 현재 층의 가중치를 표준편차가
![image](https://user-images.githubusercontent.com/97672187/165870292-db921695-4915-49dc-a25b-f49fca72c720.png)인 정규분포로 초기화하면서 이 문제를 해결한다.

밑에 그래프는 활성화 함수를 ReLU로 사용하고, He 초기화 방법을 사용했을 때 활성값의 분포이다. Xavier를 사용 했을 때보다 활성값이 고르게 분포하는 것을 알 수 있다. 보통 활성값이
0인 분포가 가장 큰데 그 이유는 ReLU는 가중합값이 음수로 나오면 모두 0으로 취급하기 때문에 0의 비율이 높은 경우가 많다.

![image](https://user-images.githubusercontent.com/97672187/165870383-3d7a19c9-147b-4056-9bc1-4cb1f92d2629.png){: .align-center}

이미지출처: https://yngie-c.github.io/deep%20learning/2020/03/17/parameter_init/


### 과적합 방지
딥러닝은 은닉층이 2개이상인 신경망이기 때문에 노드수에 따라 모델에 사용되는 파라미터 수가 매우 많아질 수 있다. 이로인해 모델은 더 복잡해지고, 과적합이 발생할 수 있는데 신경망 모델에서도
과적합을 해소할 수 있는 방법들이 존재한다.

1) Weight Decay(가중치 감소)

가중치 값이 크면 과적합이 발생할 수 있다. 가중치 감소는 가중치가 너무 커지지 않도록 규제를 주는 것이다. Ridge 회귀(Note231)을 보면 알 수 있듯이 이 규제는 오차항과 관련이 있다.
즉, 손실 함수에 가중치와 관련된 항을 추가하는 것이다. 조건을 어떻게 적용할 지에 따라 L1(Lasso), L2(Ridge) Regulization 으로 나뉜다. 이 둘의 목적은 손실 함수가 최소화 되는
가중치와 편향을 찾는 동시에 가중치에 관한 항(L1 or L2)의 합이 최소화 시키는 것이다. 즉, 가중치의 모든 원소가 0이 되거나 0에 가깝게 되는 것.

![image](https://user-images.githubusercontent.com/97672187/165872391-c40bbadb-c0c2-40a6-b005-5851d66a8a4d.png){: .align-center}

라쏘 회귀는 패널티 항에 가중치의 절대값의 합을 추가해 규제를 준다. 유의미하지 않는 가중치를 0에 가깝게 혹은 0으로 만들어서 특정 변수를 모델에서 삭제하고 모델을 단순화 할 수 있다. 즉, 변수를 자동으로
제거해주기때문에 Feature Selection이 가능하다. 하지만, 미분이 불가능해서 일반적인 경사하강법으로는 최적화를 하긴 힘들다. 


<br>


<br>


![image](https://user-images.githubusercontent.com/97672187/165872360-05166576-2bdb-411f-aac1-8f3cca013f54.png){: .align-center}

이미지출처: https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/

반면, 릿지 회귀는 패널티 항에 가중치의 제곱을 추가해서 규제를 한다. 라쏘와는 달리 유의미하지 않는 가중치를 0으로 만들진 않고 0에 가깝게 함으로써 가중치의 영향력을 낮춘다. 
가중치의 영향력이 낮아지면 해당 가중치와 연결된 변수의 영향력도 낮아지게 될 것이다. 유의미하지 않은 변수가 0이 되지 않아 자동으로 Feature Selection이 되진
않지만, 미분이 가능해서 경사하강법을 사용할 수 있다는 장점이 있다. 

보통 특성이 많은데 그 중 일부분만 중요하면 라쏘, 전체적으로 중요도가 비슷하면 릿지를 사용한다.

```python
Dense(64,
      kernel_regularizer=regularizers.l2(0.02),
      activity_regularizer=regularizers.l1(0.01))
```

kernel_regularzier는 레이어의 가중치에 패널티를 적용 -> 가중치에 패널티를 적용하는 것이니까 제곱을 사용하여 큰 가중치에 더 큰 패널티를 주고, 작은 가중치에 적은 패널티를 주는
즉, 가중치마다 패널티를 달리 주는 L2를 주로 사용한다.

activity_regularizer는 레이어의 출력에 패널티를 적용 -> 출력된 값에 패널티를 적용하는 것이니까 절대값으로 가중치를 모두 동일하게 낮춰주는 L1을 주로 사용한다.

2) Dropout(드롭아웃)

드롭아웃은 Iteration마다 각 층의 노드 중 일부를 사용하지 않으면서 학습을 진행하는 방법이다. Iteration마다 다른 노드가 학습되기 때문에 항상 똑같은 노드의 가중치가 학습되어서
과적합되는 것을 막을 수 있다.

0과 1 사이의 실수를 입력하게 되고 모델 내의 특정 층의 노드를 입력한 비율만큼 강제로 버린다. 매 Iteration 마다 랜덤하게 노드가 차단된다.

```python
Dense(64, activation = 'relu')
Dropout(0.5) # 위에서 존재하는 64개의 노드 중 32개는 버리겠다.
#32개만 랜덤하게 사용하겠다.
```

3) Early Stopping(조기 종료)

조기 종료는 보통 검증 데이터의 성능이 더 안 좋아지는 지점에서 학습을 종료시키는 방법을 사용한다.

```python
import tensorflow as tf
import keras

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1) # 10번동안 성능개선 없으면 종료

model.fit(X_train, y_train, batch_size=32, epochs=30, verbose=1, 
          validation_data=(X_test,y_test), 
          callbacks=[early_stop])
```

### best 모델 저장(feat. Checkpoint)

Checkpoint 함수를 사용하면 모델의 가중치 혹은 best 모델 자체를 저장할 수 있다.

```python
import tensorflow as tf
import keras

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
save_best = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_models.hdf5', monitor='val_loss', verbose=1, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch', options=None)

model.fit(X_train, y_train, batch_size=32, epochs=30, verbose=1, 
          validation_data=(X_test,y_test), 
          callbacks=[early_stop, save_best])
```

!조기종료 직전의 모델과 best 모델의 성능은 다르다. 만약 학습 후 바로 model.evaluate를 한다면 이 모델은 조기 종료 직전의 모델이지 best 모델이 아닐 수 있다. best 모델이 아니라면
성능이 best 모델일 때보다 좀 떨어질 것.

따라서 만약에 모델이 현재 메모리에 위치해있고 가중치를 저장해두었다면 load_weights 함수를 사용해 지금 존재하는 모델에 best 가중치만 불러온다.

메모리에 위치해있지 않다면(즉, 프로그램을 종료했다가 다시 새로 시작한 상태) load_model 함수를 사용해 best 모델을 불러온 후 성능을 테스트 한다.

### 신경망 모델 만들기
베이스 모델에 은닉층을 2개 추가하고, 출력층에서는 100개의 클래스를 분류하기 위해서 활성화 함수로는 softmax를 사용한다. cifar100 데이터를 사용한다.
타겟이 정수 라벨링이 되어있어서 손실함수는 sparse_categorical_entropy를 사용하고 성능은 정확도로 평가한다.

```python
import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# 신경망을 여러번 돌려도 같은 결과가 나오도록 seed 고정
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# 데이터 불러오기
(X_train, y_train), (X_test, y_test) = cifar100.load_data()

# 픽셀이 0~255까지 표현된 픽셀값을 0과 1 사이로 정규화
# 정규화를 시켜야 더 빠르고, 손실을 최소화 하여 학습할 수 있다.
X_train, X_test = X_train / 255, X_test / 255 # 255로 나눠주어서 0과 1 사이로 전처리

# 변수 설정을 따로 하는 방법을 적용하기 위한 코드입니다. 
batch_size = 100
epochs_max = 20

# model
model = Sequential()
model.add(Flatten(input_shape= (32,32,3))) # 고차원 데이터를 1차원으로 변형. 입력층
model.add(Dense(128, activation='relu', 
                kernel_regularizer=regularizers.l2(0.00001),   
                activity_regularizer=regularizers.l1(0.00001))) # 은닉층1. 과적합 방지위해 L1, L2 정규화 적용               
model.add(Dense(128, activation='relu')) # 은닉층2
Dropout(0.1) # 128개의 노드 중 10%를 버리고 학습. 과적합 방지.
model.add(Dense(100, activation='softmax')) # 출력층

# 컴파일 단계, 옵티마이저와 손실함수, 측정지표를 연결해서 계산 그래프를 구성함
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1) # 5회이상 검증데이터의 성능개선없으면 조기 종료

# Validation Set을 기준으로 가장 최적의 모델을 찾기
save_best = keras.callbacks.ModelCheckpoint(filepath = checkpoint_filepath,monitor = 'val_loss', verbose = 1, 
                                            save_best_only = True, save_weights_only=True, mode ='auto', save_freq = 'epoch', options = None)

results = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs_max, validation_data=(X_test,y_test), 
                    callbacks=[early_stop, save_best], verbose = 1)
                    
```


