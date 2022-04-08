---
layout: single
title: "Note 322 HTML, CSS, Web Scraping"
toc: true
toc_sticky: true
category: Section3
---

데이터 분석을 할 때 항상 데이터가 주어지는 것은 아니다. 파일이나 DB가 주어지지 않더라도 연구자가 직접 데이터를 수집해야 할 상황이 생길 수 있다. 웹 스크레이핑을 사용하면
데이터가 주어져있지 않더라도, 웹 상에 있는 데이터를 심지어는 과거의 데이터까지 찾아서 데이터를 수집할 수 있다.

웹 스크레이핑을 위한 HTML, CSS와 같은 프론트엔드의 기본 개념들과 파이썬 패키지를 활용한 스크레이핑을 정리해보자.

### HTML(Hyper Text Markup Language)
HTML은 웹 페이지에서 보여지는 것들이 어떻게, 어떤 방식으로 보여져야 하는지 알려주는 마크업 언어이다. 이 HTML이 프로그래밍 언어다, 아니다에는 의견이 갈리지만,
웹과 관련된 공식문서인 MDN에 의하면 프로그래밍 언어는 아니다.

HTNL에는 다양한 요소(element)들이 존재(ex) head, body, div, li 등)한다. 각 요소는 <head> </head> 과 같이 태그 형식으로 표현이 된다. '/'가 포함된 태그가 요소를 닫는 태그인데
모든 태그가 다 닫는 태그를 가지고 있는 것이 아니다. 예를 들어 줄 바꿈을 나타내는 '<br>' 은 닫는 태그가 없다.

만약 요소 안에 다른 요소가 있다면, 안에 있는 요소를 자식 요소, 밖에 있는 요소를 부모 요소라고 한다.

```html
<ul> #부모 요소
    <li>Hello</li> #자식 요소
</ul>
```

<br>



<br>

### CSS(Cascading Style Sheets)
CSS는 HTML이 표현한 문서가 어떻게 표현 되는지 알려주는 언어이다. HTML은 웹이 어떻게 구성되는지 틀을 짜고, CSS는 어떻게 표현이 되는지 살을 붙인다. 또, Java Script는 붙여진 살이
동작하도록 기능을 추가한다. HTML에서도 스타일을 적용할 수 있지만, 코드가 길어지면 가독성이 좋지 않기 때문에 보통 HTML과 CSS는 분리해서 사용한다.

- CSS Selector

1) Type selector: CSS 타입에 따라서 선택할 수 있는 것.(p, div, h1)

2) Class selector: 클래스에 따라 선택할 수 있는 것 (class = 'Messi' 라고 하면 Messi라고 부여된 클래스를 조회가능)

3) ID selector: ID에 따라 선택할 수 있는 것 (id = '123' 이면, id가 '123' 태그의 내용을 조회 할 수 있다.)

클래스와 ID는 비슷한 개념으로 사용되지만, 보통 ID는 고유한 번호로써 중복이 되지 않도록 사용한다. 하지만, 꼭 고유하게 쓰여야할 필요는 없다.

- CSS 상속

자식 태그는 부모 태그의 영향을 받는다. 하지만 스타일은 자식 태그부터 적용이 되기 때문에, 만약 자식 태그에 이미 적용된 스타일이 있다면, 부모 태그의 스타일은 무시할 수 있다.

- CSS 클래스

클래스는 특정 요소들에게 공통적으로 스타일을 적용하기 위해서 사용된다. 보통 해당 클래스와 관련된 요소들을 상속 받게 한다.

한 요소에 여러개의 클래스를 부여할 수도 있다.

```html
<p class = 'Messi Argentina PSG'>He is the best player</p>
```

class는 CSS에서 '.' 표현한다.

```css
.Messi {
  color: "yellow";
}
```

- CSS ID

클래스와 비슷하게 사용되지만,  '#'으로 표현된다. 그리고 보통 ID는 태그를 식별하기 위해 사용하기 때문에 여러 개의 요소에 사용되진 않는다.

```html
<p id = 'Leo'>He is the best player</p>
```

```css
#Leo {
  color: "red";
}
```

<br>




<br>


### DOM(Document Object Model)
문서 객체 모델인 DOM은 HTML, XML 등, 문서의 프로그래밍 인터페이스다. 프로그래밍 언어를 통해서 HTML 문서 등에 접근할 수 있도록 한다. 자바스크립트에서 사용하는 객체 라는 개념을 이용해 문서를 객체화하여 표현하고,
문서를 하나의 구조화된 형식으로 표현을 하기 때문에 이 구조를 사용해 원하는 동작을 할 수 있다.

DOM을 사용할 수 있는 쉬운 방법은 웹 브라우저 -> 개발자 도구 -> 콘솔 창을 활용하는 것이다. 많이 쓰는 함수들을 정리해보자.

```javascript
document // 화면에 보이는 정보를 객체화 시켜서 HTML 문서로 보여준다.
document.querySelector('p') // p태그들 중 젤 위에 태그 조회
document.querySelectorAll('p') // p태그 모두 조회
document.getElementsById('123') // 아이디가 일치하는 태그 조회
document.getElementsByClassNmae('aaa') // 클래스가 일치하는 태그조회
```


<br>



<br>


### Web Scraping
웹 스크레이핑은 웹에서 필요한 정보를 긁어오는 것을 말한다. 비슷한 개념으로 웹 크롤링이 있는데 이 둘을 굳이 분류하자면 동적이냐, 정적이냐의 차이가 될 것 같다.
웹 스크레이핑은 화면에 보이는 데이터만 가져오는 정적인 스크레이핑이고, 웹 크롤링을 화면에 보이지 않은 정보도 돌아다니면서 데이터를 가져오는 동적인 스크레이핑이다.

또한, 웹 스크레이핑은 자동화에 초점이 맞춰져 있고 특정 정보를 가져오는 것이 목적이라면, 웹 크롤링은 자동화를 사용하긴하지만 주로 인터넷에 있는 사이트들을 인덱싱하는데에
목적을 두고 있다.

차이점이 존재하지만, 결국 두 가지 모두 정보를 수집하기 위한 목적을 가지고 있다.

- 파이썬 requests, bs4 패키지를 활용한 웹 스크레이핑

```python
import requests
from requests.exceptions import HTTPError
from bs4 import BeautifulSoup

url = 'https://google.com'
try:
  req = requests.get(url) # get method를 통해서 url접근을 요청하고 결과를 가져온다.
  #결과는 response 객체로 저장된다.
  
  req.raise_for_status() # 요청이 성공적이면 status code가 200이지만, 실패하면 에러가 발생하게 한다.

except HTTPError as Err:
  print('HTTP 에러')

except Exception as Err:
  print('다른 에러 발생')

else:
  print('성공')

#만약 성공적으로 불러와졌다면 req에는 응답 객체가 저장되어 있다.
#print(req.text) # 응답 객체를 text로 변환하면 html형태로 보이는 것을 알 수 있다.

#parser 쪼개는 것을 의미하고 보통 html parser를 많이 사용한다.
#응답객체를 텍스트로 변환한 것을 html 구조를 기준으로 쪼갠다.
soup = BeautifulSoup(req.text, 'html.parser') # 응답 객체를 text로 변환 후 parsing
#soup = BeautifulSoup(req.content, 'html.parser') # 응답 객체의 내용을 parsing
# 큰 차이는 없음.
```

```python
# 1) req 응답 객체를 text로 변환 후 parsing 했을 때
# select를 사용함.
e = soup.select('messi') # messi 라는 태그 다 불러옴.
e1 = soup.select('messi .argentina') # messi태그에서 argentina 클래스 다 불러옴
e2 = soup.select('messi .argentina > abc > a) # '>'를 사용해서 태그를 계속 세분화해서 정보를 불러올 수 있음.
e2.get_text() # 위에서 불러온 e2에서 내가 원하는 text 정보만 불러올 수 있다.
#만약 예쁘게 잘 불러와지지 않는다면 string을 쪼개고, 불필요한 부분 없애고 이런식으로 해서 다듬어야 한다.
```

```python
# 2) req 응답 객체를 content로 변환해서 parsing 했을 때
# find를 사용함.

#messi라는 id를 가진 태그들 중 가장 위의 태그의 정보만 저장
e3 = soup.find(id = 'messi')

#messi class 중 특히 div 태그에 있는 태그 정보를 저장
e4 = soup.find_all('div',class_ = 'messi') # class는 파이썬의 class 개념과 비교하기 위해 _ 를 붙여야함

#for 문 활용하여 필요한 값 저장하기
e_list  = []
for e in soup.find_all(class_ = 'ronaldo'):
  e_list.append(e.get_text()) # get_text()함수를 써서 해당 태그에서 원하는 text만 불러옴

#string을 활용
e5 = soup.find_all(string =lambda text: 'raining' in text.lower()) # 대소문자 실수가 있어도 정보를 잘 가져올 수 있게끔 소문자화
e5 = soup.find_all('h3', string =lambda text: 'raining' in text.lower()) #위에처럼 하면 요소가 아닌 string이 리턴된다. 따라서 요소로 리턴을 받으려면 태그를 추가해줘야한다.

# 공백 제거
soup.find('p', class_ = 'messi').text.strip()
```

# 실습
피파게임 손흥민 선수 2022 TOTY 시즌 댓글 가져오기

```python
url = 'https://fifaonline4.inven.co.kr/dataninfo/rate/?&searchword=%EC%86%90%ED%9D%A5%EB%AF%BC&season=258'
req = requests.get(url)
req.status_code

soup = BeautifulSoup(req.content, 'html.parser')
#soup.select('table').text
son = soup.find_all(class_ = 'fifa4 comment')
for i in range(10):
    print(f'{i}번째 리뷰', end = ': ')
    print(son[i].get_text())
```

