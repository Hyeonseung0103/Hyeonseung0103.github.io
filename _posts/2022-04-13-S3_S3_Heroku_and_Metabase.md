---
layout: single
title: "Note 333 Heroku, Metabase"
toc: true
toc_sticky: true
category: Section3
---

내가 만든 어플리케이션을 나만 사용하려면 local 서버에 결과를 올려두어도 상관없다. 이것을 다른 사람도 사용하게 하려면 이 어플리케이션을 '배포'해주어야 한다. 나의 
어플리케이션을 로컬 서버에 올리면 내 컴퓨터로만 사용할 수 있기 때문에 보통 AWS, Oracle과 같은 클라우드 서버를 활용하여 많은 사람들이 사용할 수 있게 한다.  

하지만, 클라우드 서버는 빈 서버를 생성하는 것이기 때문에 필요한 파일, 환경변수 등 개발환경을 사용자가 모두 설정해야 한다는 번거로움이 있다. 서버 통제권이 사용자에게 강력하게 존재하긴하지만,
그만큼 원하는 어플리케이션을 작동시키기 위해 서버를 구축하는데에 많은 시간이 소요된다. 여기서 사용할 수 있는 것이 'Heroku'다. 

이번 포스팅에서는 Heroku와 대시보드인 Metabase에 대해 알아보자.

### WSGI(Web Server Gateway Interface), gunicorn
WSGI란 서버나 게이트웨이와 어플리케이션이나 프레임워크를 이어주는 middleware이다. 쉽게 말하면 서버와 어플리케이션을 이어주는 중간 다리 역할이라고 할 수 있다. 예를 들면, Flask와 같은
마이크로 프레임워크를 서버로 연결해 외부에서 접속할 수 있도록 도와주는 역할을 한다.

파이썬은 WSGI HTTP Server로 gunicorn이라는 것을 사용한다. gunicorn은 HTTP에 관련된 웹 요청이 들어오게 되면 flask와 같은 어플리케이션을 이용해서 요청을 처리해주는 역할을 한다.
이 gunicorn methods를 활용해서 Flask로 만든 WSGI 어플리케이션을 실행시킬 수 있다.

예시) gunicorn 실행

- 어플리케이션 팩토리 형태가 아닐 때

```bash
gunirocn --workers=1 FLASK앱의 상위 폴더:플라스크 앱 파일이름
```

- 어플리케이션 팩토리 형태 일 때(따옴표를 해줘야함)

```bash
gunirocn --workers=1 'FLASK앱의 상위 폴더:함수이름'
```

### Heroku
Heroku는 클라우드 플랫폼을 제공하는 서비스(Platform as a Service)이다. 

장점:

1) AWS, Oracle과 같은 서버에 비해 많은 설정들이 이미 사전에 정해져 있기 때문에 사용자가 설정해야 하는 개발환경이 적다. 

2) 어플리케이션을 만들고 Heroku와 연동만 시키면 배포가 이루어져서 간단하게 어플리케이션 배포를 테스트 해볼 때 사용하기 좋다. 

3) Git을 사용하기 때문에 레포 단위로 배포를 진행,관리 할 수 있고, 코드가 준비만 되어 있으면 코드를 배포할 서버에 올리고(git으로 push하면 올라감) 배포된 URL 주소를 얻는 과정도 쉽게 할 수 있다.

4) Heroku CLI를 사용하여 로그인부터 배포까지 간단하게 작업할 수 있다.

### Heroku 실습

1) 아무 터미널에서(git, vscode 상관없음) heroku login
 
```bash
heroku login
```

2) 앱 만들기(이미 존재하는 앱 이름은 사용X)

```bash
heroku create app
```

3) 앱에서 동작할 파이썬 코드 작성

- 먼저 사용하는 라이브러리를 따로 requirements.txt 파일로 정리한다.

```bash
pip freeze > requirements.txt
```

- 파이썬에서 Flask app 작성(flask_app 폴더/ __ init __ .py)

```python
from flask import Flask

#Application Factory 형태 함수.
def create_app():
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return 'Hello everyone', 200
    return app
```

- 그 다음 Flask 실행을 위해 gunicorn을 사용하고 코드를 Procfile에 작성(Procfile이라는 파일을 하나 만든다.)

이 예제에서는 flask_app 안에 application factory 형태로 있는 create_app함수를 앱으로 사용한다. 또한, 웹 어플리케이션이라는 것을 인지시키기 위해 web이라는 키를 사용한다.

```python
web: gunicorn --workers=1 'flask_app:create_app()'
```

4) 상위 폴더를 git 폴더로 만들고, github 주소를 확인한 뒤 그 주소에 원격저장소를 만든다.

현재 상위 폴더를 예를 들어 heroku_test라고 하면, 이 폴더 안에는 flask_app폴더, Procfile, requirements.txt 이 3가지가 무조건 같은 경로에 있어야한다.

![image](https://user-images.githubusercontent.com/97672187/163111044-b18e9b50-0311-4cef-b238-998d164a8a8f.png){: .align-center}

```bash
git init
```

```bash
heroku apps:info
```

![image](https://user-images.githubusercontent.com/97672187/163111319-b6c38ee3-3376-4155-a660-b19cc42a8103.png){: .align-center}

```bash
git remote 위에서 확인한 github 주소
```

5) add, commit, push

아까 만든 폴더, requirements.txt, Procfile을 add,commit하고 main 브랜치에 push 해준다.

```bash
git add ./
git commit -m 'first test'
git push heroku main
```

6) 결과 확인

heroku 서버에 어플리케이션의 정상적으로 작동하는 것을 확인할 수 있다.

![image](https://user-images.githubusercontent.com/97672187/163111844-5d1cb355-2c66-44f4-9566-bc28daf70f20.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/163111947-bc49ea63-18b0-4419-bd77-3fb99075ccc7.png){: .align-center}

7) 서버 종료 및 재시작

종료

```bash
heroku ps:scale web:0
```

재시작

```bash
heroku ps:scale web:1
```

8) 그 외 다른 명령어들

```bash
heroku restart

heroku open  #서버 실행 확인

heroku logs --tail  #서버 실행시 로그보기

heroku ps:stop run.1 #서버 실행 종료

heroku apps  #헤로쿠 서버 실행

heroku ps:scale web=1  #서버 scale up

heroku ps:scale web=0  #서버 scale down

heroku ps #상태확인
```

이렇게 Flask App을 배포할 수 있고, DB를 활용하면 더 다양한 정보를 표현하는 어플리케이션을 만들 수 있을 것이다.

### Metabase
대시보드는 다양한 데이터를 동시에 비교할 수 있게 해 주는 여러 뷰들의 모음이다. 데이터 분석가 관점에서는 BI(Business Intelligence)를 얻기 위해 보고서 형태로 많이 사용된다.

대시보드를 DB에 직접 연결하면 데이터의 변동 사항을 실시간으로 반영할 수 있다. 또한, 비데이터 직군의 사용자라도 다양한 시각화를 통해 데이터로부터 나오는
인사이트를 효과적으로 전달할 수 있다.

여러가지 대시보드 툴이 있지만 Metabase를 활용한 대시보드에 대해 알아보자.

1) Docker를 이용해서 Metabase 설치

metabase라는 container를 만들어서 이미지 파일을 불러온다.

```bash
docker run -d -p 3000:3000 --name metabase metabase/metabase
```

Docker를 보면 이 컨테이너가 활성화되어있는 것을 확인할 수 있다.

2) 필요한 DB를 docker의 이미지 파일 내의 app 폴더로 옮기기

여기서는 local의 sqlite db인 chinook.db를 옮긴다고 해보자.

```bash
docker cp ./ metabase:/app/
```

3) 공식문서를 참고하여 기본 환경설정(메타베이스 이름 설정, DB 경로 설정 등)

3000포트로 연결했기 때문에 127.0.0.1:3000 에 들어가보면 환경설정을 할 수 있다. 설정이 완료되면 metabase로 연결되고 query를 입력 및 결과 저장, 대시보드 작성 등
여러가지 작업을 할 수 있다.

![image](https://user-images.githubusercontent.com/97672187/163121261-a031ec6e-36e7-4f20-81a1-15867437f1f3.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/163121364-e056fd4e-9e17-47de-a1d1-a03d70f564dd.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/163121485-05b20ee5-8fde-4b9d-882c-fb2ac6aaf358.png){: .align-center}

4) metabase를 종료하려면 docker에서 컨테이너를 종료하고, 재시작할 때도 docker에서 시작해서 다시 같은 포트번호로 대시보드 접속하면 된다.

