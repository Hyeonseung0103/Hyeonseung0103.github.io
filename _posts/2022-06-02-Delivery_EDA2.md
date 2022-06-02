---
layout: single
title: 배달 데이터 프로젝트 전처리 및 EDA Part.2
toc: true
toc_sticky: true
category: Delivery
---

데이터의 변수를 늘려서 모델의 성능을 올리기 위해 여러 변수들을 추가해보자. 날씨, 요일, 공휴일, 축구, 구별 연령대별 인구수, 미세먼지 등의 변수를 추가했다.

## Delivery Project 전처리 및 EDA Part.2

### 4. 날씨 변수 추가(기온, 체감온도, 강수량, 적설량 등등)
날씨가 배달에 미치는 영향이 있을 것이라고 생각해서 기상청 데이터를 활용해서 날짜, 시간대에 맞는 날씨를 크롤링 했다.

```python
df_all['시간'] = df_all['시간'].apply(lambda x: str('0' + f'{x}') if (x < 10) else str(x) )
df_all['시간'].unique()
```

![image](https://user-images.githubusercontent.com/97672187/171610896-a95615b6-e14e-477f-b254-9dc962f6d75a.png){: .align-center}

<br>


<br>

크롤링을 위해 기상청 사이트에 있는 날짜 형식으로 날짜 데이터를 변환한다.

```python
df_all['time'] = (df_all['날짜'] + '-' + df_all['시간']).str.replace('-','.')
print(df_all.time.apply(len).unique()) #자릿수가 모두 맞춰졌다.
df_all.head()
```

![image](https://user-images.githubusercontent.com/97672187/171611324-18ce33ce-cd4d-45c3-af54-7d1d6858422f.png){: .align-center}

<br>


<br>


```python
#날씨 데이터를 크롤링 하는 함수
def get_weather(time):
    time_url = f'&tm={time}%3A00'
    html = requests.get(url+time_url).text
    soup = BeautifulSoup(html, "html.parser")
    table = soup('table', 'table-col')[0]

    #테이블 내 모든 <tr> 태그
    table_rows = table.find_all('tr')
    #최초 두 태그는 테이블 헤더, 나머지가 데이터
    if len(table_rows) > 6: #날씨 정보가 있는 날
        table_headers = table_rows[:2]
        table_data = [table_rows[5]]
        table_data_elements = [x.find_all('td') for x in table_data]
    else:
        table_data_elements = [''] # 날씨 정보가 없는 날도 있음

    #모든 데이터 행에 대해 <td> 항목 추출
    data = []
    
    for elem in table_data_elements:
        if len(elem) == 14:
            elem = [elem[0],elem[5], elem[7],elem[8], elem[9],elem[10],elem[12]]
            data.append([x.text if idx != 6 else x.text.split('(')[1].split(',')[0].replace('\'','') for idx, x in enumerate(elem)])
        elif len(elem) == 13: #여름에는 적설량 변수가 없어서 index를 좀 바꿔서 해야함.
            elem = [elem[0],elem[5], elem[7],elem[8], elem[9],elem[11]]
            data.append([x.text if idx != 5 else x.text.split('(')[1].split(',')[0].replace('\'','') for idx, x in enumerate(elem)])
        else:
            data.append([np.nan] * 7)
    data[0].insert(0,time)
    if len(data[0]) == 7:
        data[0].insert(6,np.nan)
    data = pd.DataFrame(data, columns = ['time', '지점', '기온', '체감온도', '일강수량', '상대습도', '적설', '풍속'])
    df = pd.concat([df_weather, pd.DataFrame(data)],axis = 0)
    return df

```

```python
url = 'https://www.weather.go.kr/w/obs-climate/land/city-obs.do?auto_man=m&stn=0&dtm=&type=t99&reg=109'
df_weather = pd.DataFrame(columns = ['time', '지점', '기온', '체감온도','일강수량', '상대습도', '적설', '풍속'] )
for time in tqdm(df_all['time'].unique()):
    df_weather = get_weather(time)

#날씨 데이터
df_weather.head()
```

![image](https://user-images.githubusercontent.com/97672187/171626895-cf5c728c-206b-4ba3-b9d7-f9b305ac5fce.png){: .align-center}

<br>


<br>


```python
#값이 이상한 부분 전처리
df_weather.loc[df_weather['적설'].isnull(),'적설'] = 0
df_weather.loc[df_weather['일강수량'] == '\xa0','일강수량'] = 0
df_weather.loc[df_weather['상대습도'] == '\xa0','상대습도'] = 0
df_weather.loc[df_weather['체감온도'] == '\xa0','체감온도'] = 0
df_weather.loc[df_weather['풍속'] == '&amp;nbsp;','풍속'] = 0

df_weather['년'] = df_weather['time'].str.split('.').apply(lambda x: x[0])
df_weather['월'] = df_weather['time'].str.split('.').apply(lambda x: x[1])
```

```python
#NA인 부분을 각 년도별 날씨 평균으로 대체하자.
df_weather[['체감온도', '일강수량', '상대습도', '풍속']] = df_weather[['체감온도', '일강수량', '상대습도', '풍속']].astype(float)
df_mean = df_weather.groupby(['년','월'])[['기온', '체감온도','일강수량','상대습도','풍속']].mean().reset_index()
df_mean = df_weather.merge(df_mean, on = ['년', '월'])
df_mean.head()
```

![image](https://user-images.githubusercontent.com/97672187/171627237-c5c84da5-0001-4599-98a3-cf76986390cc.png){: .align-center}

<br>


<br>


```python
df_mean.loc[df_mean['지점'].isnull(), '지점'] = '서울'
col1 = ['기온_x', '체감온도_x', '일강수량_x', '상대습도_x', '풍속_x']
col2 = ['기온_y', '체감온도_y', '일강수량_y', '상대습도_y', '풍속_y']

# NA인 부분을 미리 계산해놓은 년,월별 평균으로 대체
for i in range(len(col1)):
    df_mean.loc[df_mean[col1[i]].isnull(), col1[i]] = df_mean.loc[df_mean[col1[i]].isnull(), col2[i]]
print(df_mean.isnull().sum()) # NA가 모두 제거되었다.
```

![image](https://user-images.githubusercontent.com/97672187/171627551-1f0aadbe-0adc-479d-903e-bdbec6d2d689.png){: .align-center}

<br>


<br>


```python
#불필요한 변수 제거 및 NA가 제거된 날씨 데이터
df_mean = df_mean[['time','지점','기온_x','체감온도_x','일강수량_x','상대습도_x','적설','풍속_x']]
df_mean.columns = df_weather.drop(columns = ['년','월']).columns                  
df_weather = df_mean.copy()
display(df_weather.head())
```

![image](https://user-images.githubusercontent.com/97672187/171627815-66f21e80-0334-4418-96d7-7e34d9e8c733.png){: .align-center}

<br>


<br>

기존 배달 데이터에 날씨 데이터 합치기

```python
df_new = df_all.merge(df_weather, on = 'time')
display(df_new.head())
print(df_new.shape)
print(df_new.isnull().sum().sum())
df_new.drop(columns = ['지점'],inplace = True)
df_new2 = df_new.copy()
```

![image](https://user-images.githubusercontent.com/97672187/171627901-c4f50d50-f621-433a-bd07-a1dd07422614.png){: .align-center}

<br>


<br>


### 5. 공휴일, 요일 ,주말 변수 만들기
공휴일은 공공 데이터 포털에서 제공하는 데이터를 사용했다. 19,20,21년도의 공휴일 데이터를 사용했다.

```python
holiday = pd.read_csv(f'{DATA_PATH}holiday.csv')
holiday.head()
```

![image](https://user-images.githubusercontent.com/97672187/171628036-8659e836-bd9b-4f30-95bb-8739a98dd12e.png){: .align-center}

<br>


<br>


```python
#배달 데이터의 형식으로 월과 일을 맞춰줌
holiday['월'] = holiday['월'].apply(lambda x: '0' + str(x) if x < 10 else str(x))
holiday['일'] = holiday['일'].apply(lambda x: '0' + str(x) if x < 10 else str(x))

holiday = holiday.astype(str)
holiday['날짜'] = holiday['년'] + '-' + holiday['월'] + '-' + holiday['일']
holiday.head()
```

![image](https://user-images.githubusercontent.com/97672187/171628164-4d763362-e39a-4af3-b865-7aaad0164330.png){: .align-center}

<br>


<br>


```python
#공휴일이면 1 아니면 0
df_new2.loc[df_new2['날짜'].isin(holiday['날짜'].unique()), '공휴일'] = 1
df_new2.loc[~df_new2['날짜'].isin(holiday['날짜'].unique()), '공휴일'] = 0
df_new2.head()
```

![image](https://user-images.githubusercontent.com/97672187/171628262-a16cc90b-2af7-482a-bc23-981bd8ead878.png){: .align-center}

<br>


<br>


```python
#요일과 주말
days = ['월','화','수','목','금','토','일']
df_new2['요일'] = pd.to_datetime(df_new2['날짜']).apply(lambda x: days[x.weekday()])
df_new2.loc[(df_new2['요일'] == '토') | (df_new2['요일'] == '일'), '주말'] = 1
df_new2.loc[(df_new2['요일'] != '토') & (df_new2['요일'] != '일'), '주말'] = 0
df_new2.head()
```

![image](https://user-images.githubusercontent.com/97672187/171628343-94aef785-6b78-4aa7-a038-4c278f79799e.png){: .align-center}

<br>


<br>


### 6. 축구 변수 만들기
국가대표 축구경기가 있으면 치킨을 자주 시켜먹곤한다. 따라서 축구 국가대표 경기가 있는 날을 1, 아닌 날을 0으로 하는 축구변수를 만들어보자. 네이버 스포츠에 있는
데이터를 크롤링했다.

```python
#축구 데이터 크롤링
def get_soccer(year):
    new_url = url + year + '&category=amatch&month=12'
    html = requests.get(new_url).text
    soup = BeautifulSoup(html, "html.parser")
    table1 = soup.find_all('script')
    table2 = table1[20].get_text().split('scheduleList')
    df1 = pd.DataFrame(columns = ['date', 'time'])
    for i in range(1,len(table2)):
        t3 = table2[i].split('\"')
        date_idx = [idx for idx,x in enumerate(t3) if x == 'gameStartDate']
        for idx in date_idx:
            date = t3[idx+2]
            time = t3[idx+6]
            df_c = pd.DataFrame([date,time]).transpose()
            df_c.columns = ['date', 'time']
            df1 = pd.concat([df1,df_c])
    return df1
```

```python
url = 'https://sports.news.naver.com/kfootball/schedule/index?year='
df_s = pd.DataFrame(columns = ['date', 'time'])

#축구 데이터 크롤링
for year in tqdm(['2019', '2020', '2021']):
    df_s = pd.concat([df_s,get_soccer(year)])
```

```python
#배달 데이터와 합치기 위해 시간 데이터 형식 변환
df_s['time'] = df_s['time'].str.split(':').apply(lambda x: x[0])
df_s['time2'] = df_s['date'] + '-' + df_s['time']
df_s['time2'] = df_s['time2'].str.replace('-','.')
df_s.head() # 축구 경기가 있는 날과 경기 시간
```

![image](https://user-images.githubusercontent.com/97672187/171629034-29e62a74-086b-40cb-aead-abb6d1b7a6c2.png){: .align-center}

<br>


<br>


```python
df_new2.loc[df_new2.time.isin(df_s['time2'].unique()),'축구'] = 1
df_new2.loc[~df_new2.time.isin(df_s['time2'].unique()),'축구'] = 0
df_new2.head()
```

![image](https://user-images.githubusercontent.com/97672187/171629165-63353263-c788-4a71-81b0-3b6ee9986972.png){: .align-center}

<br>


<br>


### 7. 구별, 연령대별 인구수 데이터 추가
공공 데이터 포털에서 제공하는 데이터 참고. 구별, 연령대별 인구수에 따라 배달을 시키는 정도가 다를 것이라고 판단했다.

```python
population = pd.read_csv(f'{DATA_PATH}population.csv')
population.head()
```

![image](https://user-images.githubusercontent.com/97672187/171629806-1bde4c73-c0bd-4ec3-badc-b44f13e15c5f.png){: .align-center}

<br>


<br>


```python
# , 제거 함수
def to_int(s):
    return int(s.replace(',',''))
```

```python
col = population.columns[3:].tolist()
for i in col:
    population[i] = population[i].apply(to_int)

#10대, 20대처럼 10살 단위로 묶기
col = population.columns[4:].tolist()
age = 0
for i in range(0,len(col),2):
    globals()['population_{}'.format(age*10)] = population.iloc[:,i+4] + population.iloc[:,i+5] # 글로벌 변수로 한 번에 할당.
    age = age + 1
```

```python
population['0대'] = population_0
population['10대'] = population_10
population['20대'] = population_20
population['30대'] = population_30
population['40대'] = population_40
population['50대'] = population_50
population['60대'] = population_60
population['70대'] = population_70
population['80대'] = population_80
population['90대이상'] = population_90
```

```python
#성별이랑 불필요한 나이대 제거
population = population.loc[(population['성별'] == '계') & (population['행정구역별'] != '합계')]
population.drop(columns = population.columns[2:24].tolist(),inplace =True)
population.head()
```

![image](https://user-images.githubusercontent.com/97672187/171630219-0d541d86-7588-4856-afbe-b4f8f966e477.png){: .align-center}

<br>


<br>

배달 데이터에 구별, 연령대 데이터 병합

```python
df_new2['년'] = df_new2['날짜'].str.split('-').apply(lambda x: x[0])
population['기간'] = population['기간'].astype(str)
df_population = df_new2.merge(population, left_on = ['년','구'], right_on = ['기간','행정구역별'])
print(df_population.shape)
df_population.head()
```

![image](https://user-images.githubusercontent.com/97672187/171630593-d777f1b8-cd22-4656-88a5-3f544319606c.png){: .align-center}

<br>


<br>

### 8. 미세먼지 데이터
미세먼지가 많은 날에는 사람들이 외출을 삼가해서 배달음식을 시켜먹진 않을까라는 생각이 들었다. 공공데이터 포털에서 데이터를 다운 받을 수 있었지만, 2021년의 데이터는
5월까지만 존재했다. 기존 배달데이터에는 2021년의 데이터가 7월까지 나타나있는데 미세먼지 데이터는 5월까지 밖에 제공되지 않았기 때문에 공공 데이터 API를 활용해서 2021년 6월,7월의
데이터도 크롤링해서 데이터 손실을 없게 했다. 

API를 활용하여 날짜별 시간별 구별 미세먼지 데이터를 크롤링해보자.

```python
#공공 데이터 포털에서 제공된 데이터
#2021년은 5월까지 밖에 없다.
pm_19 = pd.read_csv(f'{DATA_PATH}pm_2016_19.csv', encoding = 'cp949')
pm_20s = pd.read_csv(f'{DATA_PATH}pm_2020_21.csv',encoding = 'cp949')
display(pm_19.head())
display(pm_20s.head())
```

![image](https://user-images.githubusercontent.com/97672187/171630974-14aacc57-0d17-4c64-8024-29ec1061dae4.png){: .align-center}

<br>


<br>


```python
pm_19['날짜'] = pm_19['일시'].str.split(' ').apply(lambda x: x[0])
pm_19['시간'] = pm_19['일시'].str.split(' ').apply(lambda x: x[1])
pm_19 = pm_19.loc[pd.to_datetime(pm_19['날짜']) > datetime.strptime('2018-12-31', '%Y-%m-%d')]
pm_20s['날짜'] = pm_20s['일시'].str.split(' ').apply(lambda x: x[0])
pm_20s['시간'] = pm_20s['일시'].str.split(' ').apply(lambda x: x[1])

df_pm = pd.concat([pm_19, pm_20s], axis = 0)
df_pm = df_pm.loc[df_pm['구분'] != '평균']
df_pm.head()
```

![image](https://user-images.githubusercontent.com/97672187/171631114-33ef5eb7-c660-4222-acd5-ba57fa5cf95c.png){: .align-center}

<br>


<br>

데이터 병합을 위해 시간형식을 배달 데이터의 시간형식으로 바꿔준다.

```python
df_pm['시간'] = df_pm['시간'].str.split(':').apply(lambda x: x[0])
df_pm['시간'] = df_pm['시간'].apply(lambda x: '0' + x if len(x) == 1 else x)
df_pm['time'] = df_pm['날짜'] + '-' + df_pm['시간']
df_pm['time'] = df_pm['time'].str.replace('-','.')
df_pm.drop(columns = ['일시'], inplace = True)
df_pm.head()
```

![image](https://user-images.githubusercontent.com/97672187/171631206-b564c88e-fd11-4bf4-8fe1-c96e4373b701.png){: .align-center}

<br>


<br>


```python
#기존 배달 데이터에 있는 2021년 6,7월 데이터의 날짜만 가져오고
#API를 호출할 때 필요한 날짜 형식에 맞춰 .을 제거해준 time2 변수를 만든다.
df2 = df_new.copy()
df2['년'] = df2['날짜'].str.split('-').apply(lambda x: x[0])
df2 = df2.loc[(df2['년'] == '2021') & (df2['월'].astype(int) > 5)]

df2['time2'] = df2['time'].str.replace('.','')
df2.head()
```

![image](https://user-images.githubusercontent.com/97672187/171633150-5dffd2cd-6ba2-42f1-9c65-fa5f4b15852b.png){: .align-center}

<br>


<br>


```python
#미세먼지 데이터 크롤링
def get_pm(time):
    new_url = url + time + '00'
    res = requests.get(new_url)
    html = requests.get(new_url).text
    soup = BeautifulSoup(html, "html.parser")
    time = pd.DataFrame(soup.find_all('msrdt'))
    gu = pd.DataFrame(soup.find_all('msrste_nm'))
    pm10 = pd.DataFrame(soup.find_all('pm10'))
    pm25 = pd.DataFrame(soup.find_all('pm25'))
    pm_c = pd.concat([time,gu,pm10,pm25], axis = 1)
    pm_c.columns = df_pm2.columns.tolist()
    df = pd.concat([df_pm2,pm_c],axis = 0)
    return df
```

```python
url = 'http://openAPI.seoul.go.kr:8088/발급받은key.../xml/TimeAverageAirQuality/1/25/' 
df_pm2 = pd.DataFrame(columns = ['time', '구', '미세', '초미세'])

#2021년 6월, 7월의 미세먼지 가져오기
for time in tqdm(df2['time2'].unique()):
    df_pm2 = get_pm(time)
df_pm2.head()
```

![image](https://user-images.githubusercontent.com/97672187/171633326-293cd446-1fd6-431d-b559-6c286b39a566.png){: .align-center}

<br>


<br>

공공 데이터 포털에서 제공받았던 2019년~2021년 5월까지의 데이터와 API로 크롤링했던 2021년 6,7월 데이터를 합치고, 위에서 만들었던
인구수가 추가된 df_population 데이터와 병합.

```python
#<> 형식으로 표현된 데이터 전처리
df_pm2['l1'] = df_pm2['미세'].apply(lambda x: len(x))
df_pm2['미세'] = df_pm2['미세'].str.replace('<pm10>','')
df_pm2['미세'] = df_pm2['미세'].str.replace('</pm10>','')
df_pm2.loc[df_pm2['미세'] == '', '미세'] = 0
print(sum(df_pm2['미세'] == ''))
df_pm2['l1'] = df_pm2['초미세'].apply(lambda x: len(x))
df_pm2['초미세'] = df_pm2['초미세'].str.replace('<pm25>','')
df_pm2['초미세'] = df_pm2['초미세'].str.replace('</pm25>','')
df_pm2.loc[df_pm2['초미세'] == '', '초미세'] = 0
```

```python
# 시간관련 데이터 재정의
df_pm2.time = df_pm2.time.astype(str)
df_pm2.time = df_pm2.time.apply(lambda x: x[:10])
df_pm2.drop(columns = ['l1'], inplace= True)
df_pm2['년'] = df_pm2.time.apply(lambda x: x[:4])
df_pm2['월'] = df_pm2.time.apply(lambda x: x[4:6])
df_pm2['일'] = df_pm2.time.apply(lambda x: x[6:8])
df_pm2['시간'] = df_pm2.time.apply(lambda x: x[8:])
```

```python
#데이터 병합을 위해 기존 데이터의 시간 형식과 동일하게 변경
df_pm2.time = df_pm2['년'] + '.' + df_pm2['월'] + '.' + df_pm2['일'] + '.' + df_pm2['시간']
df_pm2.drop(columns = ['년','월', '일','시간'], inplace = True)
df_pm = df_pm[['time', '구분','미세먼지(PM10)','초미세먼지(PM25)']]
#변수 이름 재정의
df_pm.columns = df_pm2.columns
display(df_pm.head())        
df_pm.isnull().sum() #미세,초미세 변수에 결측치 존재
```

![image](https://user-images.githubusercontent.com/97672187/171634311-87f88a1c-aca7-4ee1-b852-c11efd908131.png){: .align-center}

<br>


<br>

결측치는 구별 평균으로 대체

```python
pm_mean = df_pm.groupby('구')['미세'].mean().to_frame().reset_index()
pm_mean.columns = ['구', '미세평균']
pm25_mean = df_pm.groupby('구')['초미세'].mean().to_frame().reset_index()
pm25_mean.columns = ['구', '초미세평균']
pm_mean.head()
```

![image](https://user-images.githubusercontent.com/97672187/171634455-7c066169-9e9a-483b-8af3-6dbacff816ee.png){: .align-center}

<br>


<br>


```python
#결측치 대체
df_n = df_pm.copy()
df_n = df_n.merge(pm_mean, on = '구')
df_n = df_n.merge(pm25_mean, on = '구')
df_n.loc[df_n['미세'].isnull(), '미세'] = df_n.loc[df_n['미세'].isnull(), '미세평균']
df_n.loc[df_n['초미세'].isnull(), '초미세'] = df_n.loc[df_n['초미세'].isnull(), '초미세평균']
df_n.isnull().sum()
```

![image](https://user-images.githubusercontent.com/97672187/171634588-dc642f3c-439f-4a6e-86b8-8a6c55a2cc15.png){: .align-center}

<br>


<br>


```python
#데이터 병합 전 불필요 변수 모두 제거
df_n.drop(columns = ['미세평균', '초미세평균'], inplace = True)
df_pm_all = pd.concat([df_n, df_pm2], axis = 0)
df_pm_all.head()
```

![image](https://user-images.githubusercontent.com/97672187/171634647-a07f3a35-10e4-4163-88fa-525d3257963e.png){: .align-center}

<br>


<br>

미세먼지 데이터와 요일, 주말, 공휴일, 축구, 날씨, 인구 데이터 병합

```python
#최종으로 인구까지 붙여진 데이터에 미세먼지까지 추가
df_final = df_population.merge(df_pm_all, on = ['time', '구'])
df_final.drop(columns = ['주문건수'],inplace = True)
df_final = df_final[['날짜', '시간', '업종', '시도', '구', '월','time', '기온', '체감온도', '일강수량', '상대습도', '적설', '풍속', '공휴일', '요일', '주말', '축구', '0대', '10대', '20대', '30대', '40대', '50대', '60대', '70대', '80대', '90대이상', '미세', '초미세','주문정도']]
df_final.head()
```

![image](https://user-images.githubusercontent.com/97672187/171634884-ead1a943-0cd9-4d47-b2f2-450a9e6ff085.png){: .align-center}

<br>


<br>


```python
df_final.shape
```

![image](https://user-images.githubusercontent.com/97672187/171634920-a85aca27-9043-43b0-a76d-100aedc5b25f.png){: .align-center}

<br>


<br>

이로써 모델의 성능을 높이기 위해 최대한 많은 변수들을 추가해보았다. 최종 데이터는 기존데이터에 날씨, 요일, 주말, 공휴일, 축구, 구별 연령대별 인구수, 미세먼지 데이터가 추가 되었다.

다음 포스팅에서는 본격적으로 모델링을 진행하고, 하이퍼 파라미터 튜닝을 통한 최적화 과정을 정리해보겠다.


