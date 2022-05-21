---
layout: single
title: "Note 414 교차검증과 하이퍼파라미터 튜닝"
toc: true
toc_sticky: true
category: Section4
---

머신러닝에서 일반화 능력을 향상 시키기 위해 교차검증을 사용했고, 모델을 최적화 시키기 위해 하이퍼파라미터 튜닝을 진행했다. 머신러닝에서는 하이퍼파라미터가 많아야 20가지 정도가 되었지만,
신경망에서는 층이 깊어질수록 튜닝해야 하는 하이퍼파라미터가 정말 많아진다. 이를 손으로 직접 조정하기에는 너무나 많은 시간이 걸릴 것이기 때문에 Grid Search CV, Randomized Search CV,
Bayesian Methods 등을 사용할 수 있다.

K-Fold, Stratified K-Fold CV는 머신러닝과 사용하는 방법이 같아서 이번 포스팅에서는 따로 다루지않겠다. (Note224 참고)

### Grid Search CV
Note224를 참고하면 알겠지만, Grid Search CV는 주어진 범위 내의 모든 조합을 다 사용해서 튜닝을 진행하기 때문에 정확하지만, 범위를 넓게 잡을수록 시간이 오래걸린다는 단점이 있다.

```python
import numpy
import pandas as pd
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
```

```python
def create_model():
  model = Sequential()
  model.add(Dense(100, input_dim = 8, activation = 'relu'))
  model.add(Dense(1, activation = 'sigmoid'))
  
  model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['acc'])
  return model
```

```python
model = KerasClassifier(build_fn=create_model, verbose=0)
batch_size = [8, 16, 32, 64, 128]
param_grid = dict(batch_size=batch_size)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
result = grid.fit(X, Y)
```

```python
print(f"Best Score: {result.best_score_}, 사용한 하이퍼파라미터: {result.best_params_}")

means = result.cv_results_['mean_test_score']
stds = result.cv_results_['std_test_score']
params = result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print(f"means: {mean}, std: {stdev} 사용한 하이퍼파라미터: {param}")
```

### Keras Tuner를 사용한 튜닝

```python
pip install -U keras-tuner
import kerastuner as kt
```

```python
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

import tensorflow as tf
import IPython
```

1) Model 만들기

```python
def model_builder(hp):
  model = Sequential()
  
  # 은닉층에 사용할 노드수를 32~512까지 32씩 증가시키며 튜닝
  hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)
  model.add(Dense(units = hp_units, activation = 'relu'))
  model.add(Dense(10, activation = 'softmax'))
  
  #학습률은 0.01, 0.001, 0.0001 3개 사용
  hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])

  model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate),
                loss = 'sparse_categorical_crossentropy',
                metrics = ['acc'])
  
  return model
```

2) 튜터(Tuner) 지정하기

Keras Tuner에는 Random Search, Bayesian Optimization, Hyperband 등의 튜너가 존재하지만 예시로 Hyperband 튜너를 사용해보자.

Hyperband를 사용하려면 max_epochs와 model을 만드는 함수(위에서 만든 model_builder)가 필요하다. 리소스를 알아서 조절하고 조기 종료 기능을 사용하여 높은 성능을 보이는 조합을 신속하게
찾을 수 있다는 장점이 있다.

```python
tuner = kt.Hyperband(model_builder, 
                     objective = 'val_accuracy', 
                     max_epochs = 10,
                     factor = 3,
                     directory = 'my_dir',
                     project_name = 'intro_to_kt')
```

3) Callback 함수 지정

학습 단계에서 매 에포크마다 특정 함수를 실행시키기 위해 Callback에 사용될 함수를 만든다. 
여기서는 ClearTrainingOutput class를 만들어서 매 에포크마다 학습이 끝나면 이전 출력이 지워지도록 한다.

```python
class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)
```

4) 하이퍼파라미터 탐색

```python
tuner.search(X_train, y_train, epochs = 10, validation_data = (X_val, y_val), callbacks = [ClearTrainingOutput()])

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

#최적의 노드수와 학습률 출력
print(best_hps.get('units'), best_hps.get('larning_rate'))
```

5) 최고 성능을 낸 파라미터 사용하여 재학습

```python
best_model = tuner.hypermodel.build(best_hps)
model.summary()
```

```python
model.fit(X_train,y_train, epochs = 10, validation_data = (X_val, y_val))
```

### Randomized Search CV
Randomized Search CV는 주어진 조합 내에서 랜덤으로 조합을 찾아서 하이퍼 파라미터를 튜닝시키는 방법이다. 모든 조합을 보지 않기 때문에 Grid Search CV보다는 빠르면서,
어느 정도의 성능을 내는 튜닝을 할 수 있다는 장점이 있다. 따라서, Grid Search CV비해 넓은 범위의 탐색도 적은 시간으로 가능하다.

1) 모델 만들기(기본값 정의)

```python
def make_model_rscv(units = 64, batch_size = 64, activation = 'relu', dropout_rate = 0.3, C = 0.01, optimizer = 'adam'):
  model = Sequential()
  model.add(Dense(units, activation = activation))
  model.add(Dense(256, activation = 'relu'))
  model.add(Dropout(dropout_rate))
  model.add(Dense(512, activation = 'relu',
            kernel_regularizer = regularizers.l2(C),
            activity_regularizer = regularizers.l1(C)))
  model.add(Dense(1, activation = 'sigmoid'))

  model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['acc'])
  return model
```

2) 하이퍼파라미터 범위 지정

```python
from sklearn.model_selection import RandomizedSearchCV

tf.random.set_seed(42)
units = np.arange(32,512,32)
batch_size = [32,64,128,256,512]
epochs = [5,10,20,30,50]
activation = ['relu', 'sigmoid']
optimizer = ['adam', 'adagrad', 'rmsprop']
dropout_rate = [0.2,0.3,0.4,0.5]
C = [0.01,0.1,1]

param_grid = dict(batch_size = batch_size, units = units, activation = activation, optimizer = optimizer, dropout_rate = dropout_rate, C = C, epochs = epochs)

model_rscv = KerasClassifier(build_fn=make_model_rscv, verbose=0)
```

3) Randomized Search CV

```python
rscv = RandomizedSearchCV(
    estimator=model_rscv,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    verbose=3,
    n_jobs=1,
    random_state=12)

rscv_result = rscv.fit(X_train_scaled, y_train)
```

4) 결과출력

```python
rs = pd.DataFrame(rscv_result.cv_results_).sort_values(by='rank_test_score').head()
rs.T
```

5) best 파라미터로 재학습

```python
print(f"Best: {rscv_result.best_score_} using {rscv_result.best_params_}")

units = rscv_result.best_params_.get('units')
optimizer = rscv_result.best_params_.get('optimizer')
dropout_rate = rscv_result.best_params_.get('dropout_rate')
batch_size = rscv_result.best_params_.get('batch_size')
activation = rscv_result.best_params_.get('activation')
C = rscv_result.best_params_.get('C')
epochs = rscv_result.best_params_.get('epochs')
best_model = make_model_rscv(units = units, batch_size = batch_size, activation = activation, dropout_rate = dropout_rate, C = C, optimizer = optimizer)

tf.random.set_seed(42)
best_model.fit(X_train_scaled, y_train, epochs = epochs)
```

보통 Randomized Search CV로 어느 정도 하이퍼 파라미터의 범위를 잡고, Grid Search CV로 세부적으로 튜닝하는 방법을 사용한다.
