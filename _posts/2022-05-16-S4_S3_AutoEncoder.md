---
layout: single
title: "Note 433 AutoEncoder"
toc: true
toc_sticky: true
category: Section4
---

AutoEncoder는 입력된 데이터를 저차원의 벡터로 압축시키고 다시 원본 크기의 데이터로 복원하는 신경망으로, 유용하게 쓰일 수 있다. 이 AutoEncoder를 활용하면 차원을 축소할 수 있고,
데이터의 어느 정도 형태는 유지하면서 데이터를 압축시킬 수 있기 때문에 저장면에서도 메모리 공간을 절약할 수 있다. 또한, 데이터를 압축시킬 때 노이즈들을 제거함으로써
필요한 데이터만 추출할 수 있고, 이 과정에서 이상치를 탐지할 수 있다는 장점을 가지고 있다. 이번 포스팅에서는 이렇게 유용하게 사용되는 AutoEncoder에 대해 알아보자.

### AutoEncoder의 구조

![image](https://user-images.githubusercontent.com/97672187/168503097-1ceabff9-c895-4878-86bf-fc0c31d8e25e.png){: .align-center}

이미지출처: https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798

AutoEncoder는 입력데이터를 저차원으로 축소시키는 인코더 부분과 축소된 데이터를 다시 원본 크기로 복구하는 디코더 부분으로 나눌 수 있다. 특히 위 그림에서 Code라고 표현된 가장
저차원의 벡터를 Latent(잠재)벡터라고 하는데 Latent 벡터는 원본 데이터보다 저차원이지만, 원본 데이터의 특징을 잘 보존하는 벡터이다. 결국 AutoEncoder의 학습목표는 원본 데이터의
특징을 가장 잘 보존하는 이 Latent 벡터를 잘 얻기 위함이다.  Latent 벡터는 원본 데이터를 압축했다가 복원하는 과정에서 원본 데이터의 특징일 최대한 보존할 수 있도록 학습된다.

Latent벡터의 차원수는 사용자가 정할 수 있지만, 사용되는 피처는 학습을 통해 데이터의 특징을 가장 잘 반영한 피처가 자동으로 선택된다. 원본 데이터보다 차원이 꼭 작을 필요는 없고, 정말
특수한 경우에 원본 데이터보다 큰 경우도 있다.

AutoEncoder는 label이 없는 비지도 학습이지만 결국 원본 데이터와 비교하기 위해 사용되기 때문에 지도 학습처럼 풀어낼 수 있다.

### 매니폴드 학습(Manifold Learning)
Manifold란 고차원의 데이터가 이루는 저차원의 공간이고, Manifold 학습은 고차원 데이터를 데이터 공간에 뿌렸을 때 sample들을 가장 잘 표현하는 Manifold 공간을 찾는 것이다. 즉,
기존 sample들의 정보를 최대한 잘 반영한 고차원 데이터가 가지고 있는 저차원 공간을 찾는 것이다. 

차원의 저주는 데이터의 차원이 커질수록 데이터 분석이나 좋은 모델을 만들기 위해 필요한 샘플 데이터 수가 증가하는 것을 말한다. 데이터의 차원이 커질수록 데이터의 밀도가 낮아지는데
이 고차원의 데이터는 저차원의 매니폴드를 포함하고 있고, 이 저차원의 매니폴드를 벗어나는 순간 밀도는 급격히 낮아진다. 밀도가 낮으면(샘플들끼리 거리가 멀리 떨어져있으면) 유사한 데이터를
찾기 어렵고 데이터 간의 관계를 파악하기가 힘들 것이다. 따라서, Manifold 학습을 차원의 저주 관점에서 보면 고차원의 데이터를
잘 표현하는 manifold 공간을 찾으면서 샘플 데이터의 특징을 파악할 수 있고, 샘플 데이터의 특징을 잘 파악한다면 차원이 커져서 더 많은 샘플들이 필요로한 차원의 저주 문제를 해결할 수 있다.

차원의 크기는 Feature(특성)의 수로 표현되는데 결국 데이터를 잘 표현한 manifold를 찾는다는것은 중요한 특성을 찾고, 이를 기반으로 manifold 좌표를 조금씩 변경해가며 데이터를 유의미하게
변화시키는 것이다. 반대로, manifold를 잘 찾았다는 것은 중요한 특성이 유사하게 표현된 sample을 찾을 수 있다는 것이다. 

![image](https://user-images.githubusercontent.com/97672187/168505598-7cae8777-6279-4c9b-bc4f-85bfbdec9a96.png){: .align-center}

이미지출처: https://www.analyticsvidhya.com/blog/2021/02/a-quick-introduction-to-manifold-learning/

아래의 그림처럼, 고차원으로 복잡하게 표현되어 있는 데이터를 저차원 평면으로 바꿔서 차원은 줄이면서, 고차원에 표현된 데이터의 정보를 잘 반영한 저차원 공간을 찾을 수 있다. 
AutoEncoder의 Latent 벡터는 원본 데이터를 잘 표현한 저차원 데이터를 의미하는데 이 관점에서 매니폴드 학습이 잘 이루어지면 기존 샘플들의 정보를 잘 유지하면서 더 적은 차원으로 
데이터를 표현할 수 있기 때문에 Manifold는 AutoEncoder에서 핵심이 되는 개념이 된다.

### AutoEncoder를 사용한 이미지 비교 실습

```python
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
```

```python
# 단순히 이미지에 AutoEncoder를 적용하고, 입력에 사용했던 원본 이미지와 비교하기 때문에
# label이 굳이 필요하지 않다.

(X_train, _), (X_test, _) = fashion_mnist.load_data()
X_train[0][0]
```

```python
#정규화
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

print(X_train.shape)
print(X_test.shape)
```

![image](https://user-images.githubusercontent.com/97672187/168541707-f57ce42a-abcb-4539-bfd3-82e617380f85.png){: .align-center}

```python
# 잠재 벡터 차원수
LATENT_DIM = 64
```

```python
class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__() # 기존의 파이썬의 Autoencoder를 상속받음
        self.latent_dim = latent_dim   # 잠재 벡터
        self.encoder = tf.keras.Sequential([
            layers.Flatten(), # 차원을 1차원까지 축소
            layers.Dense(latent_dim, activation='relu'), # 1차원으로 축소된 벡터를 잠재벡터로 변환, 비선형성 추가
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(784, activation='sigmoid'), # 차원이 축소된 잠재 벡터를 28*28 1차원(784) 벡터로 다시 복원
            layers.Reshape((28, 28)) # 데이터의 원래 크기인 28 , 28 크기로 복원
        ])

    def call(self, x):
        encoded = self.encoder(x) # class 내부에서 인코더 및 디코더 호출할 수 있도록
        decoded = self.decoder(encoded)
        return decoded # 변환된 이미지 출력
```

```python
model = Autoencoder(LATENT_DIM)
model.compile(optimizer='adam', loss='mse') # 기존 이미지 데이터와의 오차(픽셀값은 숫자)
model.fit(X_train, X_train, # 입력에 사용한 데이터를 변환후 다시 입력과 비교
          epochs=10,
          shuffle=True,
          validation_split=0.2)
```

```python
#이미지가 잘 복원되었는지 확인
encoded_imgs = model.encoder(X_test).numpy()
decoded_imgs = model.decoder(encoded_imgs).numpy()
```


10개의 이미지를 출력해본 결과, 거의 비슷한 형태로 잘 복원된 것을 확인할 수 있다.

```python
n = 10

plt.figure(figsize=(20, 4))

for i in range(n):
    # 원본 이미지를 출력합니다.
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i])
    plt.title("Original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # AutoEncoder에 의해서 복원된 이미지를 출력합니다.
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("Reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
```

![image](https://user-images.githubusercontent.com/97672187/168541863-cd6aa4b2-998a-4f6f-b658-4dc873a435ae.png){: .align-center}

### 2. DAE(Denoising AutoEncoder)

노이즈 제거

```python
(X_train, _), (X_test, _) = fashion_mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = X_train[..., tf.newaxis] # 합성곱 연산을 위해 채널을 추가
X_test = X_test[..., tf.newaxis]

print(X_train.shape)
```

![image](https://user-images.githubusercontent.com/97672187/168542141-d5bd8df5-ce82-4093-9fe5-b0b197f976d1.png){: .align-center}


```python
# Random Noise 추가
noise_factor = 0.2 # 주로 0.2나 0.25정도로 노이즈를 준다.

X_train_noisy = X_train + (noise_factor * tf.random.normal(shape=X_train.shape))
X_test_noisy = X_test + (noise_factor * tf.random.normal(shape=X_test.shape))

#Scaling
X_train_noisy = tf.clip_by_value(X_train_noisy, clip_value_min=0., clip_value_max=1.)
X_test_noisy = tf.clip_by_value(X_test_noisy, clip_value_min=0., clip_value_max=1.)
```

노이즈가 추가된 것을 확인할 수 있다.

```python
# 노이즈 확인
n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.title("Original + Noise")
    plt.imshow(tf.squeeze(X_test_noisy[i]))
    plt.gray()
plt.show()
```

![image](https://user-images.githubusercontent.com/97672187/168542225-ca5a149b-c86c-48ac-ba28-748c1d0290e3.png){: .align-center}


```python
# Denoising AutoEncoder 구축

class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)), 
            layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=2), # 합성곱
            layers.Conv2D(8, (3,3), activation='relu', padding='same', strides=2)])
        
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'), # 합성곱 연산으로 줄어든 size 복원
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')]) # 최종출력 이미지는 이미지 벡터 1개
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

```python
model = Denoise()
model.compile(optimizer='adam', loss='mse')
model.fit(X_train_noisy, X_train, #노이즈 데이터를 원본과 비교
          epochs=10,
          shuffle=True,
          validation_split= 0.2)
```

노이즈가 잘 제거된 것을 확인할 수 있다.

```python
# 검증
encoded_imgs = model.encoder(X_test).numpy()
decoded_imgs = model.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):

    # Noise가 추가된 원본 이미지를 출력합니다.
    ax = plt.subplot(2, n, i + 1)
    plt.title("Original + Noise")
    plt.imshow(tf.squeeze(X_test_noisy[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # AutoEncoder에 의해서 복원된 이미지를 출력합니다.
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("Reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()
```

![image](https://user-images.githubusercontent.com/97672187/168542385-26085512-fc63-45a9-a9b2-6421229a7708.png){: .align-center}

### 이상치 탐지(Anomaly Detection)

```python
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
```

```python
df = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
df_value=df.values # 값들만 추출(데이터 프레임 형식 말고)
df.head()
```

![image](https://user-images.githubusercontent.com/97672187/168542621-e4b821a5-c246-4957-a2cb-88c558ea9354.png){: .align-center}


```python
label = df_value[:, -1]
data = df_value[:, 0:-1]

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
X_train[0][:10]
```

![image](https://user-images.githubusercontent.com/97672187/168542675-0928b03f-1221-4260-86d5-ff064dafb867.png){: .align-center}


```python
# 이상치에 민감하기 반응시키기 위해 MinMaxScaling
min_val = tf.reduce_min(X_train)
max_val = tf.reduce_max(X_train)

X_train = (X_train - min_val) / (max_val - min_val)
X_test = (X_test - min_val) / (max_val - min_val)

X_train = tf.cast(X_train, tf.float32)
X_test = tf.cast(X_test, tf.float32)
```

```python
# 모델 학습시에 레이블이 1로 지정된 즉, 정상 데이터만 사용
y_train = y_train.astype(bool)
y_test = y_test.astype(bool)

X_train_normal = X_train[y_train] # 1인 데이터만 남김
X_test_normal = X_test[y_test]

X_train_abnormal = X_train[~y_train]
X_test_abnormal = X_test[~y_test]
```

```python
# 모델 구축
class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")])
        
        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(140, activation="sigmoid")]) # 마지막 층은 입력에 사용되었던 원본 차원과 동일하게 맞춤.
            # 입력은 140개의 feature로 이루어져 있다.
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

```python
model = AnomalyDetector()
model.compile(optimizer='adam', loss='mae')
history = model.fit(X_train_normal, X_train_normal,  # 정상 데이터만 사용하여 원본과 비교
          epochs=20, 
          batch_size=512,
          validation_split= 0.2,
          shuffle=True)
```

```python
# 복원할 때 임계값보다 오류가 큰 데이터는 비정상 데이터로 분류한다.
# 임계값은 보통 손실의 평균 + 손실의 표준편차이다.
# 손실은 mae를 사용했다.

reconstructions = model.predict(X_train_normal)
train_loss = tf.keras.losses.mae(reconstructions, X_train_normal)

threshold = np.mean(train_loss) + np.std(train_loss)
print(threshold)
```

![image](https://user-images.githubusercontent.com/97672187/168542910-f837b28f-42d4-4d1c-a9ab-d71861a5a034.png){: .align-center}

복원 할 때 오류가 생기는 이미지를 비정상 데이터로(False)로 잘 분류했는지 성능 평가

```python
def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold) # 각 데이터마다 손실이 임계값보다 작으면 true, 크면 false
    # 이 T,F 가지고 정확도, 정밀도, 재현율을 지표로 사용가능
    # 해당 이미지의 손실이 임계값보다 크면 비정상 데이터, 작으면 정상 데이터

def print_stats(predictions, labels):
    print(f"Accuracy = {accuracy_score(labels, predictions)}")
    print(f"Precision = {precision_score(labels, predictions)}")
    print(f"Recall = {recall_score(labels, predictions)}")
```

```python
preds = predict(model, X_test, threshold)
print_stats(preds, y_test)
```

![image](https://user-images.githubusercontent.com/97672187/168542973-cab7efba-2761-4b7a-ad3b-09834842e33e.png){: .align-center}











