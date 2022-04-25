---
layout: single
title: "Note313 Database 심화, 트랜잭션, ACID"
category: Section3
toc: true
toc_sticky: true
---

지난시간에 이어서 데이터베이스에 대해 더 심화된 쿼리와 개념들을 학습했다.

### 트랜잭션(Transaction)
트랜잭션은 데이터베이스의 무결성이 보장되는 상태에서 요청된 작업을 완수하기 위한 작업의 기본 단위이다. 데이터베이스의 상태를 변화시키는 작업의 모음이라고 할 수 있다.
보통 데이터베이스의 상태를 변화시키는 INSERT, DELETE, UPDATE 중 한 개 이상과 같이 사용된다.

- Commit

트랜잭션을 실행한 뒤 모든 트랜잭션이 성공적으로 완료됐다는 확정 신호를 보내지 않는다면 데이터베이스의 상태는 변하지 않는다. 이 확정신호를 Commit이라고 한다.

- Rollback

트랜잭션을 실행한 뒤 Commit을 하기 전에 앞으로 변경될 작업에 대한 내용을 취소할 수 있다. 이것을 Rollback이라고 하고, Rollback을 쓰면 Rollback 전에 수행된 모든 Transaction이
취소된다.

```sql
INSERT INTO player VALUES ('id1', 'player1');
INSERT INTO player VALUES ('id2', 'player2');
ROLLBACK;
INSERT INTO player VALUES ('id3', 'player3');
COMMIT;
```

위의 경우는 Rollback을 사용했기 때문에 player3의 정보만 저장된다.

### ACID
ACID는 Atomicity, Consistency, Isolation, Durability의 약자로 하나의 트랜잭션의 안전성을 보장하기 위해 필요한 조건이다.

- Atomicy(원자성)

원자성은 하나의 트랜잭션을 구성하는 작업들은 전부 성공하거나 전부 실패해야 된다는 것이다. 예를 들어 채팅을 보내는 것을 하나의 트랜잭션이라고 하면, 트랜잭션의 작업은
1)메세지를 입력하는 것과 2)메세지를 보내는 작업으로 구성된다. 만약 메세지를 입력은 했으나 보내는 것에 성공을 못했거나, 보내긴 했지만 메세지의 입력된 내용이 가지 않았으면 해당
트랜잭션은 실패로 돌아가야 한다. 여기서 사용되는 개념이 원자성이다. 트랜잭션의 모든 작업이 성공해야 트랜잭션 성공, 하나라도 실패하면 실패가 되는 것. 원래 원자성은 원자처럼 더 이상
쪼개질 수 없는 단위까지 간다는 뜻인데 하나의 트랜잭션을 쪼개서 세부 작업이 성공해도 트랜잭션을 성공이라고 한다면 원자성을 위반하게 되는 것이다. 트랜잭션의 단위를 쪼개서 성공 실패 여부를
파악하면 안 된다.

- Consistency(일관성)

일관성은 트랜잭션 이후의 데이터베이스 상태는 이전과 같이 유효해야 한다는 것이다. 즉, 트랜잭션 이후에도 데이터베이스의 규칙과 제약에 모두 만족하는 것.
id가 기본키, name이 속성이라고 할 때, id 없이 name을 입력한다거나 id는 삭제하지 않고 name만 삭제하는 것은 기존에 id와 name으로 이루어져 있어야 하는 데이터의 상태를
변화시키는 것이므로 일반성을 위반한 것이 된다. name이 NULL값을 가질 수 있지만, 아예 해당 value를 없애버리면 한 행에 두 열이 존재해야 하는데 한 행에 id 열만 존재하기 때문에 위반.

- Isolation(고립성)

고립성은 하나의 트랜잭션이 다른 트랜잭션과 독립되어야 한다는 것이다. 트랜잭션끼리 서로 영향을 주어서는 안 된다. 예를 들어, 단톡방에서 서로 채팅을 칠 때, 상대방이 채팅을 보내는 중이라고 해서
내가 채팅을 못 보내는 상황이 생기면 안 된다. 상대방 채팅에 나의 채팅이 영향을 받으면 안 된다는 뜻이다. 만약 다른 트랜잭션에 의해 트랜잭션이 영향을 받으면 고립성을 위반한 것이다.

- Durability(지속성)

하나의 트랜잭션이 성공적으로 수행되었다면 로그가 남고, 런타임 오류나 시스템 오류로 실패해도 해당 기록이 영구적으로 남아야 한다는 것이다.
만약, 채팅이 성공적으로 보내졌는데 시스템 오류로 데이터베이스에 적재되지 않았다고 해도 채팅창에 채팅이 보여져야 한다. 하지만, 데이터베이스에 적재되지 않았다는 오류가 기록되어
나중에 디버깅 할 수 있도록 해야 한다. 만약 인터넷이 연결되지 않았다거나 하는 오류가 있으면 채팅이 보내지지 않아야 하고, 보내지지 않은 기록을 표시해서 채팅을 보내기 이전의 상태로
돌아가야 한다. 지속성을 통해 한 번 Commit된 데이터는 손실되지 않고 영구적으로 저장된다는 것을 보장할 수 있다.


### SQL 내장함수
- Having: Group에 대해서 조건을 주는 것이기 때문에 무조건 Group By가 있을 때만 사용할 수 있다.

반면 Where은 단일 행에 대한 필터이므로 Group에 필터를 걸 수 없다. 따라서 Where에 집계함수를 사요할 수 없다.(한 번에 하나의 행만 보면서 필터를 하기 때문에 집계가 될 수 없음)

```sql
-- 선수 ID당 평균적으로 15골 이상 넣은 선수들만 표시
-- AVG 뿐만 아니라, sum, max, min, count 등도 쓸 수 있음.
SELECT playerId, AVG(Goal)
FROM players
GROUP BY playerId
HAVING AVG(Goal) > 15
```

- ROW NUMBER() + OVER() : 데이터에 순서를 매길 때 사용한다. Partition 함수는 그룹 안에서 순서를 매길 수 있도록 해준다.

```sql
SELECT teacher_id, student_id, ROW_NUMBER() OVER(ORDER BY student_id) '전체 학생순서' from Teacher t
JOIN Student s ON t.id = s.teacher_id 

SELECT teacher_id, student_id, ROW_NUMBER() OVER(PARTITION BY teacher_id ORDER BY student_id) '선생님별 학생순서' from Teacher t
JOIN Student s ON t.id = s.teacher_id
```


### 쿼리 실행순서
SELECT문이 젤 위에 있다고 먼저 실행되는 것이 아니다. 쿼리의 실행 순서는

1) FROM

2) ON

3) JOIN

4) WHERE

5) GROUP BY

6) CUBE , ROLLUP

7) HAVING

8) SELECT

9) DISTINCT

10) ORDER BY

11) TOP

### CASE 문 사용
파이썬의 if문 처럼 SQL에서는 CASE문을 사용한다.

```sql
SELECT playerID, CASE
      WHEN goal >= 30 THEN 'HIGH'
      WHEN goal >= 20 THEN 'MID'
      ELSE 'LOW'
    END as goal_rank
  FROM players
```

### Sub Query
쿼리 안에 다른 쿼리문을 포함하는 것을 Sub Query라고 한다. 실행되는 쿼리에 중첩으로 위치해 정보를 전달한다.

- JOIN처럼 활용하기

한 경기에서 선수의 이름의 몇번 나왔는지 속성처럼 사용해서 볼 수 있다. GROUP BY를 사용하지 않아도 집계된 결과를 볼 수 있다.

```sql
SELECT players.Name,
  (SELECT COUNT(*) FROM matches WHERE players.playerId = matches.playerId) AS PlayerCount
FROM players
```

위의 결과를 조인으로도 나타낼 수 있다.

```sql
SELECT Name, COUNT(*) FROM players as p
JOIN matches m ON p.playerId = m.playerId
GROUP BY p.playerId
```


- Select에 활용하기

```sql
SELECT playerId, playerId = (SELECT playerId FROM players WHERE playerId > 5)
FROM players
WHERE goals > 15
```


- Where에 활용하기

```sql
-- 다른 테이블에 있는 데이터인데 조인 하지 않고 사용할 수 있음.
-- 부정일 때는 NOT IN 쓰면 됨.
SELECT Name FROM tracks t
WHERE AlbumId IN (SELECT AlbumId FROM albums a WHERE a.Title = 'Unplugged' OR a.Title = 'Outbreak')
```


- FROM에 활용하기

```sql
SELECT * FROM (SELECT playerId FROM players WHERE goals > 15)
```

### 정규화
데이터베이스에서 정규화란 데이터의 무결성을 유지하고, 저장 용량을 줄이기 위해 수행되는 방법이다. 기본 목표는 중복 데이터를 허용하지 않는 것이다. 1~5 정규화, 반정규화, BCNF가 있지만
현업에서는 보통 3정규화까지 사용하기 때문에 3정규화와 BCNF 까지만 알아보자.

- 1 정규화

1 정규화란, 테이블의 로우마다 컬럼이 원자값(하나의 값)을 갖도록 테이블을 분해하는 것이다. 트랜잭션의 원자성의 대상이 트랜잭션이라면, 정규화에서 원자성의 대상은 칼럼이다.
칼럼이 리스트 형식처럼 여러개의 값을 갖고 있으면 제 1정규화를 만족하지 못한다고 하고, 이를 분리 해줌으로써 원자성을 만족시킨다.

예를 들어, 장난감 가게 데이터베이스 테이블에 학부모이름, 학부모의 성별, 구매한 장난감이 적혀있다고 하자. 한 학부모라도 여러개의 장난감을 살 수 있기 때문에 만약 장난감 컬럼에
리스트 형식의 값이 입력되어 있으면 1정규화를 만족하기 위해 하나의 값만 갖도록 행을 늘린다.

![image](https://user-images.githubusercontent.com/97672187/160822355-d9ff9cc8-4650-440d-bdbe-28cdf4d9fa18.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/160822558-ff1bdd96-b3e5-4a71-96b8-6b30276dac03.png){: .align-center}



- 2 정규화

2 정규화란, 1 정규화에서 진행한 테이블에 대해 완전 함수 종속을 만족하도록 테이블을 분해하는 것이다. 완전 함수 종속이라는 것은 어떤 칼럼이 기본키 중에 특정 컬럼에만 종속된 컬럼이 없어야
한다는 것이다.

1 정규화가 만족된 테이블의 기본키는 학부모 이름, 장난감 복합키로 사용한다. 학부모 이름과 장난감을 합쳐야지만 한 행을 구분할 수 있다. 하지만 여기서 성별은 학부모 이름에만
종속되어있다(동명이인이 없다면 이름만 가지고도 성별을 알 수 있는데 장난감을 알았다고 학부모의 성별을 알 순 없다.). 따라서 장난감 때문에 성별이 중복되어 표현되어 있는것은 불필요하기때문에
테이블을 분리해서 중복을 없앤다.

![image](https://user-images.githubusercontent.com/97672187/160823078-22247d22-66c1-4879-a7b3-c4b224222b21.png){: .align-center}

- 3 정규화

3 정규화란, 2 정규화까지 만족시킨 테이블에서 기본키를 제외한 속성들 간의 이행적 종속을 없애도록 테이블을 분해한 것이다. 이행적 종속은 기본키 이외의 다른 컬럼이 그외 다른 컬럼을 결정할 수 없는 것이다.

![image](https://user-images.githubusercontent.com/97672187/160823876-80a49691-f524-4068-bef2-c60d06754c90.png){: .align-center}

위의 테이블에서 기본키는 학부모 id 하나이고, 모든 데이터가 원자값을 갖고 있으므로 1,2차 정규화를 만족시킨다. 하지만, 주소 칼럼만 알면 도/광역시, 시/구 칼럼을 알 수 있다. 기본키 외의
칼럼으로 다른 칼럼의 값을 알 수 있게 된다. 데이터가 굳이 없어도 되는 칼럼까지 섞여있는 것이다. 따라서, 3차 정규화를 만족시키기 위해 2차 정규화 처럼 테이블을 분리시킨다.

![image](https://user-images.githubusercontent.com/97672187/160828954-aed6fe2e-0dfa-4627-932f-404370164b75.png){: .align-center}

- BCNF

수퍼키: 하나 이상의 속성들의 집합으로 이루어진 것.

최소성: 레코드를 식별하기 위해 필요한 최소한의 속성(한 칼럼으로 식별할 수 있는데 굳이 여러 칼럼을 쓰면 최소성을 만족하지 않는 것). 수퍼키는 최소성을 만족하지 않아도 된다. 

BCNF(Boyce and Codd Normal Form)는 3차 정규화를 조금 더 강화한 것이다. 모든 결정자가 후보키가 되도록 테이블을 분해한다. 후보키는 수퍼키 중에서 최소성을 만족하는 키이다. 밑에 테이블에서 학생번호와 특강이름을 기본키라고 하자. 이 기본키에 의해 교수가 결정되기도 하지만, 교수에 의해 특강이름이 결정 되기도 한다(수업은 여러 명의 교수를 가질 수 있는데,
교수는 한 가지 수업만 맡는다고 가정). 교수가 한 가지 수업만 맡기 때문에 교수를 알면 특강 이름을 알 수 있게 된다.
하지만, 기존 테이블에서 교수는 중복된 값을 가지고 있으므로 후보키가 아니다(유일성을 만족하지 못하므로. 유일성은 쉽게 말해 중복이 없는 것)

이 테이블에서의 기본키는 학생번호와 특강이름이지만 특강이름의 값은 교수에 의해 결정되기 때문에 이 테이블의 결정자는 교수가 된다. 
이처럼 후보키가 아닌데 테이블의 결정자가 되는 속성이 존재하기 때문에 BCNF 정규화를 위해 테이블을 분해할 수 있다.

![image](https://user-images.githubusercontent.com/97672187/160827889-ff8c6023-1832-46b5-a175-11fe3b0a34e8.png){: .align-center}

![image](https://user-images.githubusercontent.com/97672187/160829525-4a46b79a-d340-41f9-82ab-9b281482fffb.png){: .align-center}

이미지 출처: https://mangkyu.tistory.com/110

이와 같이 테이블을 분리시켜서 특강신청 테이블에는 학생번호와 교수 모두 중복값이 있어서 결정자가 되지 못하기 때문에 후보키가 필요없고, 특강교수 테이블에는 교수가 테이블의 결정자임과 동시에 최소성, 유일성을 만족하는 후보키가 되므로 BCNF 정규화를 만족하게 된다.
