---
layout: single
title: "Note 334 시간과 부호화"
toc: true
toc_sticky: true
category: Section3
---

데이터를 저장할 때 날짜나 시간을 표현해야 하는 경우가 있다. 하지만, 시간을 표현하는 방법과 기준이 다 제각각이면 데이터를 읽고, 처리하는데 번거로움이 발생한다. 
예를 들어, 시간별로 주가 정보를 표현하는 데이터가 있다고 할 때, 날짜와 시간을 나타내는 데이터가 매우 중요할텐데 데이터를 수집하고 저장할 때마다 표현방법을 다르게 한다면 
통일된 시간으로 데이터를 읽고, 분석하기 위해서 추가적인 전처리가 또 필요할 것이다.

이번 포스팅에서는 시간을 표현하는 여러가지 방법과 추가로 부호화에 대해 다뤄보자.

### 시간

컴퓨터에서 시간을 표기하는 여러가지 방법

- **UTC**

```bash
date -u
```

![image](https://user-images.githubusercontent.com/97672187/163297232-3480e2c8-8757-49bf-9cae-bbf7bff875e1.png){: .align-center}


UTC는 영국과 프랑스가 협정하여 만든 협정 세계시로 영국을 기준으로 시차를 규정한 시간이다. 각자의 컴퓨터나 서버의 시간을 localtime이라고 하고, UTC는 시차에 따라 localtime이
다르게 표현되는 것을 UTC±0을 기준으로 표현한다.

- **KST**

KST(Korea Standard Time) 한국 표준시간을 의미한다. UTC±0 기준으로 9시간 차이가 나기 떄문에 표기할 때는 UTC±9 나 UTC±09:00 으로 사용된다.

```bash
date
```

![image](https://user-images.githubusercontent.com/97672187/163297345-e4101434-22ba-4dca-a2f3-627b8c0dd25d.png){: .align-center}

이미지 출처: https://metomi.github.io/rose/2019.01.2/html/tutorial/cylc/scheduling/datetime-cycling.html

- **ISO 8601**

날짜나 시간을 표현하는 방법은 매우 다양하다(2022-04-14 10:38), 202204141038). 각 나라마다 시간을 표현하는 방법이 달라서 발생할 수 있는 문제를 해결하기위해 ISO 8601이라는
국제 표준 시간 표현 방법을 사용할 수 있다. 

기본 표기법은 '20220414' 처럼 날짜를 표기하는 것이지만, 여러가지 확장표기법이 존재한다.

![image](https://user-images.githubusercontent.com/97672187/163297449-5d3fd0af-3f38-42fb-817e-3a39057674c5.png){: .align-center}

연도는 4개의 숫자, 그 외의 날짜 및 시간은 2개의 숫자로 구성된다. 확장형은 구분자인 (-,:,Z,T)가 사용되고 가장 끝에 있는 Time Zone Z는 UTC+0인 영국시간을 의미한다.
만약 한국시간을 TimeZone으로 표현하고 싶다면 2022-04-14T10:46:+0900 으로 표현할 수 있다.

- **date +%s** 와 **Unix Time**

UTC, KST가 사람이 쉽게 읽을 수 있는 시간이라면, date +%s은컴퓨터가 해석하기 쉬운시간표현이다.

![image](https://user-images.githubusercontent.com/97672187/163297906-03f67589-5f63-4a20-b45f-97714a18eeae.png){: .align-center}

Unix Time은 특정 시간을 기준으로 시간이 표현된다. 특정 시간은 영국의 1970년 1월 1일 0시 0분 0초이고 이를 Epoch라고 부른다.
Epoch로부터 1초 이후는 1을 더해서 표기하고, 1초 이전은 -1로 표기한다.

Unix Time은 특정 시간을 기준으로 시간을 표현하기 때문에 절대적인 시간보다는 주로 시간들간의 차이를 표현할 때 사용된다.
따라서 DB나 네트워크 상에서 시간차를 계산할 때 많이 사용한다.

### 스케줄링
결과값을 출력하거나 데이터를 수집하는 등의 코딩 작업을 할 때 한번에 모든 데이터를 불러오게 할 수도 있지만, 날짜나 시간을 지정해서 지정한 시간에 원하는 작업이 이루어지게
할 수도 있다. 예를 들어 서버의 부하를 줄이기 위해 트래픽이 가장 적은 시간대에 데이터를 수집하거나 매일 자정마다 데이터를 수집해야 하는 경우가 있을 수 있다.
파이썬의 APScheduler 라이브러리는 운영체제에 상관없이 스케줄러를 만들어서 원하는 시간에 원하는 작업을 할 수 있게 한다.

APScheduler는 다른 어플리케이션과 연동되지 않고 독자적으로 사용되는 스케줄러와 다른 어플리케이션과 연동되어 사용되는 스케줄러로 나눌 수 있다.
먼저, 다른 어플리케이션과 연동되지 않을 때는 이 스케줄러 자체가 해당 프로그램의 주기능이 될 것이다. 스케줄러로 실행하려고 하는 것이 해당 프로그램 내에서
주기능일 것이기 때문이다. 이런 경우에는 BlockingScheduler를 사용할 수 있다.

만약 스케줄러가 다른 어플리케이션과 연동이 되어 사용된다면 다른 어플리케이션을 실행시키기 위한 부가적인 기능이 된다. 스케줄링이 포함된 프로그램 내에서
어떠한 주기능을 수행하는 것이 아니라 다른 어플리케이션을 실행시키기 위한 스케줄러가 되는 것이다. 럴 때 사용하는 것이 BackgroundScheduler, GeventScheduler, AtScheduler 등등 이 있다.

이처럼 다양한 스케줄러가 있지만 이번에는 BlockingScheduler로 간단하게 예시를 들어보자.

**스케줄링 순서**

1) 스케줄러 선언

2) 스케줄러의 Job 선언

3) 스케줄러 시작

```python
from apscheduler.schedulers.blocking import BlockingScheduler

#스케줄러 선언
scheduler = BlockingScheduler({'apscheduler.timezone':'UTC'}) # 영국시간 기준 스케줄링

# 스케줄러에 사용될 job 선언
def hello():
    print("안녕하세요 저는 5초마다 실행됩니다.")

scheduler.add_job(func=hello, trigger = 'interval', seconds = 5) # triggers를 intervals로 하면  구간을 주, 시간, 분, 초, 등으로 입력 가능

# 스케줄러 시작하기
scheduler.start()
```

위 코드는 5초에 한번씩 hello함수를 실행한다. trigger 옵션에는 여러가지 인자값을 사용할 수 있지만, 자세한 내용은 APScheduler 공식 문서를 참고하자.

https://apscheduler.readthedocs.io/en/3.x/userguide.html

APScheduler는 어플리케이션 단위로 실행되는 파이썬 라이브러리이기 때문에 파이썬이 구동되고 있을 때만 스케줄링이 가능하다. 
cron이라는 스케줄링을 사용하면 운영체제 단위로 실행되어 파이썬의 여부와 상관없이 스케줄링 할 수 있다는 장점이 있긴하지만, 최근 들어 여러 문제들이 발생하고 있다고 한다.

# 객체 부호화(Object Encoding)

파이썬에서 사용되는 모든 것은 객체로 표현되고 이 객체는 크게 두 가지 방식으로 존재할 수 있다.

1) 인메모리 방식

인메모리 방식은 파이썬 코드가 실행되고 있을 때 메모리 안에서 표현되는 방식이고, CPU가 데이터를 효율적으로 처리하기 편한 상태로 저장되어 있다.

2) 바이트열 방식

바이트열 방식은 데이터를 파일에 쓰거나, 네트워크에서 전송되기 위해 표현되는 방식이다.

인메모리 방식 -> 바이트열 방식: 부호화, 직렬화, 인코딩, 피클링(파이썬), 마샬링

바이트열 방식 -> 인메모리 방식: 복호화, 역직렬화, 디코딩, 역피클링(파이썬), 언마샬링

파이썬에서 객체를 만들면(예를 들어 모델) 이 객체는 메모리 안에서 CPU가 쉽게 처리할 수 있는 인메모리형태로 저장되어 있다. 
하지만, 이것을 다른사람에게 전달하기 위해서는 메모리 밖으로 나와야 하기 때문에 형태를 바이트열 형태로 변환해주어야 하는데 이렇게 인메모리의 객체를 바이트열 객체로 변환시키는 것을 
파이썬에서는 피클링이라고 한다. pickle 이라는 라이브러리를 사용하고 반대 작업은 역피클링이라고 한다.

만약 간단한 선형회귀 모델을 돌렸다고 해보자. 이 모델을 다른 사람에게 전달해주는 방법은 두 가지로 정리할 수 있을 것이다. 첫번째는 코드 전체를 전달해줘서 그 사람이 똑같이
모델을 만들 수 있도록 하는 것과 두번째는 해당 모델만 전달해주는 것이다. 학습할 때 걸리는 시간을 생각한다면 받는 사람 입장에서는 모델만 전달 받는 것이 훨씬 효율적일 것이다.

모델만 전달하기 위해서는 모델을 파일형태로(바이트열) 추출해서 전달해주어야 한다. 즉, 부호화(피클링) 해야하고, 받은 모델 파일을 복호화(역피클링) 해서 모델로 사용할 수 있다.

1) 모델객체 부호화, 복호화

```python
#1. 부호화
import pandas as pd
from sklearn.linear_model import LinearRegression

#house prices 데이터를 가져오기
df = pd.read_csv('https://ds-lecture-data.s3.ap-northeast-2.amazonaws.com/house-prices/house_prices_train.csv')
df_t = pd.read_csv('https://ds-lecture-data.s3.ap-northeast-2.amazonaws.com/house-prices/house_prices_test.csv')

model = LinearRegression()

feature = ['GrLivArea']
target = ['SalePrice']
X_train = df[feature]
y_train = df[target]

## 학습
model.fit(X_train, y_train)

## 한 샘플을 선택해 학습한 모델로 예측
X_test = [[4000]]
y_pred = model.predict(X_test)

print(f'{X_test[0][0]} sqft GrLivArea를 가지는 주택의 예상 가격은 ${int(y_pred)} 입니다.')

# 인메모리 형태로 저장되어 있는 모델을 피클을 통해 밖으로 꺼내기
import pickle

with open('model.pkl', 'wb') as pickle_file: # 바이트 형태로 write 할 거니까 wb를 사용
    pickle.dump(model, pickle_file) #dump 함수를 사용하여 모델을 부호화 시켜서 저장 한다. 피클링
인메모리 형태의 객체를 바이트열 형태로 변환하는 것을 파이썬에서는 피클링이라고 한다.
```

```python
import pickle

반대로 전달받은 바이트열 객체를 역으로 내 코드에 인메모리 형태로 변환하는 것을 역피클링이라고 한다.
역피클링(바이트열 형태 -> 인메모리 형태)

import pickle

model = None

with open('model.pkl','rb') as pickle_file: # 바이트열 형태의 객체를 읽을거니까 rb
    model = pickle.load(pickle_file) # load 함수로 바이트열 형태의 피클 객체를 읽어옴. 역피클링

X_test = [[4000]]
y_pred = model.predict(X_test)

#기존과 똑같은 예측 결과를 얻을 수 있다.
print(f'{X_test[0][0]} sqft GrLivArea를 가지는 주택의 예상 가격은 ${int(y_pred)} 입니다.')
```

2) JSON 부호화, 복호화

```python
import json
data = {
    "player": {
        "name": "Messi",
        "goals": "20"
    }
}

with open('json_data.json', 'w') as json_file: # w는 string 형태로 문자를 쓸것이라는 것. json은 바이트열이 아니라 str형태로 객체를 만들어야 한다.
    json.dump(data, json_file) # dict -> json
```

```python
import json

str_1 = None

with open('json_data.json', 'r') as json_file:
    str_1 = json.load(json_file) # json -> dict

print(str_1) # 똑같은 결과를 얻을 수 있다.
```


