---
layout: single
title: "Note 321 파이썬 디버깅, OOP, 데코레이터"
toc: true
toc_sticky: true
category: Section3
---

파이썬은 기본적으로 객체 지향 프로그래밍 언어이다. 객체 지향 프로그래밍을 공부하기 위해서는 Class, Instance, Inheritance 등 다양한 개념을 숙지해야 한다.
또한, 디버깅 방법과 데코레이터에 대해서도 정리해보자.

### 디버깅(Debugging)
디버깅은 버그를 찾는 행동이다. 버그를 찾는다는 것은 결국 오류를 찾아 수정하겠다는 뜻인데 만약 코드를 실행하고 에러가 발생하면 코드가 종료되고 그 에러를 수정한 후에 다시 코드를 실행시켜서
에러가 수정이 잘 되었는지 확인을 해야한다. 수정이 안 되었다면 위의 과정을 다시 반복해야한다. 이렇게 할 때마다 코드가 종료된다면 오류를 수정할 때 많은 시간이 걸릴 것이다. 이 문제를 해결하기
위해 사용하는 것이 디버깅이다. 

디버깅을 사용하면 코드 실행을 종료하지 않고, 해당 순간을 잠시 멈추고 에러가 발생한 상황을 탐색할 수 있다. 사람이 손으로 디버깅을 할 수 있지만, 파이썬에서는 pdb 패키지를 사용하면
편리하게 디버깅을 할 수 있다(3.7미만 버전에는 pdb.set_trace(), 3.7이상부터는 breakpoint()함수만 쓰면 됨).

```python
def simple_func(num):
  sum = 0
  breakpoint()
  
  for i in range(1,num + 1):
    breakpoint()
    sum += i
  return sum

simple_fun(4)
```

1) 첫번째 breakpoint() 

for문 전에 breakpoint가 있다. breakpoint 다음에 실행되는 코드를 창에 보여주고 원하는 값을 입력하면 해당 위치에서 내가 입력한 값이 어떤 결과를 갖는지 출력해준다.

![image](https://user-images.githubusercontent.com/97672187/161495736-d0a0bba8-3e52-4082-93db-d7cdbd69248f.png){: .align-center}

코드는 밑에서 함수를 호출하는 것부터 시작된다. 파라미터에 4를 전달했으므로 num은 4가 되고, sum은 초기에 설정한 0이 아직 바뀌지 않아서 그대로 0이다.
i는 아직 정의되지 않았기 때문에 에러가 난다.

2) 두번째 breakpoint()

n을 입력하면 다음줄로 이동하게 된다. l 을 누르면 전체 코드에서 현재 디버깅하고 있는 위치를 표시해준다.

![image](https://user-images.githubusercontent.com/97672187/161498486-2cc6daab-2692-4452-96d3-4c564e03ae46.png){: .align-center}

첨에는 sum이 아직 실행되기 전이니까 0, 한번 돌고나서는 sum이 1로 바껴있는 것을 확인할 수 있다.

![image](https://user-images.githubusercontent.com/97672187/161498971-a0b7a950-b665-4366-add5-97451b0f0623.png){: .align-center}

c를 누르면 다음 breakpoint로 이동한다.

return 함수 직전까지 오면, 모든 for문을 다 돌아서 sum은 10이 나온다.

![image](https://user-images.githubusercontent.com/97672187/161499189-f8770c96-cafb-446e-89be-da800d0293c2.png){: .align-center}

<br>



</br>



### 파라미터(parameter)
파라미터는 함수를 정의할 때 사용되는 변수이다. 함수를 호출할 때 사용되는 변수는 인수(argument)라고 한다.
파라미터는 위치에 따라 영향을 받기 때문에 어느 순으로 받을건지 순서를 지켜서 인수를 넘겨야 한다.

파이썬에서는 인수들이 참조로 함수에 전달된다. 참조라는 것은 객체의 주소값을 의미하고 인수들이 참조로 전달되긴하지만, int,string과 같은 immutable(변경 불가)한 객체들은
값으로 전달이 되고, list, dict와 같이 mutable(변경 가능)한 객체들은 참조값을 전달한다.

mutable은 자신의 주소값은 일정하게 유지한채로, 객체의 값을 바꿀 수 있는 것. immutable은 값을 바꾸려면 주소값도 바꿔야 하는 것이다.

### 인수(Arguments)

- 필수 인수 (required arguments): 위치를 지키며 전달되는 인수

- 키워드 인수 (keyword arguments): 파라미터 이름을 사용해 전달되는 인수

- 기본 인수 (default arguments): 파라미터에서 기본으로 사용되는 값

1) 필수 인수: 순서와 갯수를 지켜야한다.

name과 goal의 순서를 지켜야하고, 갯수도 지켜야한다.

```python
def player_goal(name, goal)
#.
#.
#.
```

2) 키워드 인수: 위치에 상관없이 키워드를 통해 인수를 전달한다.

```python
player_goal(goal = 20, name = 'Messi')
```

3) 기본 인수: 미리 기본 인수를 정의해놓으면, 인수가 넘어오지 않을 때는 기본 인수를 사용한다. 정의할 때 기본값이 없는 파라미터 뒤에 써져야 한다는 조건이 있다.

```python
def player_goal(name, goal = 20)
#.
#.
#.

player_goal('Ronaldo') #기본인수 20을 사용
```

### Object Oriented Porgramming(객체 지향 언어)
파이썬 언어는 객체지향언어를 기반으로 설계되어 있다. 모든 객체지향언어가 그런 것은 아니지만, 파이썬에서는 모든 것이 객치이다.
객체지향언어는 대규모 프로그램의 설계를 위해 만들어진 방법론이다. 프로그래밍 언어 관점에서는 설계 중심의 사고로 똑같은 혹은 비슷한 코드를 반복적으로 작성하지 않기 위해
개발 되었다.

- Class

클래스는 객체를 만들어 내기 위한 설계 도면이다. 클래스는 클래스와 연관되어 있는 변수와 메서드의 집합으로 이루어져있다.

- Object

Object는 소프트웨어 세계에서 구현할 대상으로 클래스에 선언된 모양 그대로 생성된 실체이다. 클래스의 인스턴스라고도 부르고, 객체는 모든 인스턴스를 대표하는 포괄적인
의미를 가진다. OOP의 관점에서 클래스의 타입으로 선언이 되었을 때 '객체'라고 부른다.

- Instance

설계도를 바탕으로 소프트웨어 세계에 구현된 구체적인 실체이다. 위에서 정의한 객체를 소프트웨어에 실체화 하면 이게 인스턴스가 된다. 실체화된 인스턴스는 메모리에 할당된다.

인스턴스는 객체에 포함되고, OOP 관점에서는 객체가 메모리에 할당되어 실제 사용될 때 인스턴스 라고 부른다.

- Object vs Instance

객체도 클래스의 인스턴스라고 부르는 것 만큼 객체와 인스턴스의 관계를 명확하게 구분하기는 어렵다.
간단하게 Class는 설계도, 객체는 설계도로 구현한 모든 대상, 인스턴스는 메모리에 할당되어 실제로 사용되는 객체 라고 생각하자.

- Method vs Function

클래스 안에서 사용되는 함수를 Methods라고 하고, Class 밖에서 사용되는 함수를 Function이라고 한다. 

```python
class Team: #클래스. 설계도
    def __init__(self, player): # 생성자. 클래스를 인스턴스화 시킴. 클래스가 생성될 때 __init__ 함수의 내용이 실행됨.
        self.player = player #self는 자기 자신을 말함.
    
    def name(self): # method
        return self.player

player_name = "Messi"
team = Team("Messi") #객체와 인스턴스화. team은 Team class의 인스턴스(객체). 생성과 동시에 메모리에 할당되므로 객체와 인스턴스가 동시에 적용. 
print(team.name()) # methods. 클래스 내에 있는 함수를 사용. Messi 출력
```
 
### OOP의 사전정의된 메소드
여러 변수가 연결되거나 특별히 관리가 필요하면, __ init __ 변수를 여러번 사용하지 않고 다른 메소드를 사용할 수 있다.

```python
class Team: #클래스. 설계도
    def __init__(self, player, goals = 20):
        self.player = player 
        self.goals = goals
        self.name_goals = self.player + ' ' + str(self.goals)
        
    def name(self): 
        print(self.player)
    

team = Team("Messi")
print(team.name_goals) # Messi 20
team.player = 'Ronaldo'
team.goals = 30
print(team.name_goals) # 여전히 Messi 20. 중간에 이름을 바꿨음에도 초기에 할당되었던 이름이 적용되지 않았다.
# __init__ 함수는 클래스가 생성될 때만 실행되는데 중간에 값을 바꿔도 name_goals 변수는 다시 실행되지 않아서 그대로.
``` 

- @property

위와 같은 문제를 해결하기 위해 사용할 수 있는 것이 @property이다.

```python
class Team: #클래스. 설계도
    def __init__(self, player, goals = 20):
        self.player = player 
        self.goals = goals
        #원래 init함수에 있었던 name_goals를 메서드로 빼내서 사용하자.

    def name(self): 
        return self.player
    
    @property # 데코레이터 사용. property를 적게 되면 해당 메소드를 클래스의 특성(attribute)처럼 접근할 수 있게 된다.
    #클래스내의 다른 특성들과 연관되어 있을 때 사용.
    def name_goals(self):
        return self.player + ' ' + str(self.goals)

team = Team("Messi", 30)
print(team.name_goals) # Messi 30
team.player = 'Ronaldo'
team.goals = 20
print(team.name_goals) # Ronaldo 20.
```

- setter

만약 name_goals에 값을 바로 지정해줘도, 알아서 player와 goals 변수가 할당된다면 굳이 player와 goals에 값을 일일이 할당시켜줄 필요가 없을 것이다.
setter는 위에서처럼 property와 같이 정의된 메소드가 있다면 그 메소드 안의 값을 어떻게 설정할 것인지 정의한다.
값을 어떻게 가져올 것인지에 대한 getter도 존재하지만, @property를 정의할 때 이미 값을 어떻게 가져올지 정했기 때문에 별도로 지정하지 않아도 된다.

```python
class Team: #클래스. 설계도
    def __init__(self, player, goals = 20):
        self.player = player 
        self.goals = goals
        
    def name(self): 
        return self.player
    
    @property # 데코레이터 사용. property를 적게 되면 해당 메소드를 클래스의 특성(attribute)처럼 접근할 수 있게 된다.
    #클래스내의 다른 특성들과 연관되어 있을 때 사용.
    def name_goals(self):
        return self.player + ' ' + str(self.goals)
    
    @name_goals.setter # 직접 접근하기는 싫은데 추가 기능을 변수에 적용하고 싶을 때.
    def name_goals(self, new):
        player, goals = new.split()
        self.player = player
        self.goals = int(goals)


team = Team("Messi", 30)
print(team.player, team.goals)
team.name_goals = 'Ronaldo 20'
print(team.player, team.goals) # Messi와 30골에서 Ronaldo와 20으로 바뀐 것을 알 수 있다.
#만약 setter가 없었으면 값은 'Ronaldo 20' 이라는 text를 name_goals에 할당할 수 없다.
#무엇이 player고 무엇이 goals인지 알 수 없기 때문.
```
 
### _ 과 __ (single vs double underscore) 
밑줄은 보통 클래스 내부 변수에 접근하지 못하게하려는 이유로 사용한다. 굳이 그 변수를 바꿨을 때 사용자가 전혀 이익이 되지 않고, 오히려 난독성이 증가한다면
안 건드는 것이 최선이 될 것이다. single underscore인 _ 는 변수를 숨기는 것을 희망하는 것이고, double인 __ 은 변수를 숨기는 것을 강제한다. 따라서 한줄만 쓰면,
사용자가 희망만 하는 것이니까 클래스 밖에서도 조회가 가능하고, 두 줄을 쓰면 클래스 밖에서 일반적인 방법으로는 조회할 수 없다. 하지만, 
'**인스턴스 _ <클래스 이름> __ <변수 혹은 함수 이름>**' 이런 방법으로 접근하면 결국 접근은 가능하기 때문에 완벽히 접근을 하지 못하는 것은 아니다.

### Python Decorator
데코레이터는 어색하고 반복적인 함수의 표현을 줄이기 위해 제안되었고, 함수 뿐만아니라 클래스, 제너레이터 등의 타입에서도 사용되고 있다.
데코레이터를 사용하면 깔끔하고 간결한 코드를 만들면서 코드의 재사용을 줄일 수 있기 때문에 많이 사용된다.

밑의 함수를 보면 선수정보라는 공통되는 텍스트가 있는데도 각각 다른 함수에서 중복되어 사용되고 있다.
지금은 print를 이용한 간단한 경우이지만, 만약 로직이 더 복잡해진다면 중복으로 인해 계산시간이 늘어날 수 있다.

```python
def player(name):
    print("선수정보")
    print(name)

def age(old):
    print("선수정보")
    print(old)

def goals(num):
    print("선수정보")
    print(num)
```

데코레이터를 사용하여 중복을 최소화 시켜본다면 다음과 같이 표현할 수 있을 것이다.

```python
def player_info(func):
    def info(*args, **kargs): # *args는 바로 값을 찾는것이 아니라 메모리 주소를 찾아서 값을 읽어준다는 의미. # 그냥 'hi'와 같은 string을 입력하면 메모리 주소를 찾아서 읽음.
                              # **kargs는 dictionary 형태로 값을 읽는다. key = value 형식으로 넣어야한다. ex) 그냥 값이 아니라 
                              # width = 100, 이렇게 넣으면 **kargs 형식인 것. width가 key, 100이 value
        print("선수정보")
        func(*args, **kargs)
    return info

@player_info
def player(name):
    print(name)

@player_info
def age(old):
    print(old)

@player_info
def goals(num):
    print(num)

player_info(player('Messi')) # 선수정보 \n Messi
player_info(age('35'))  # 선수정보 \n 35
player_info(goals('30')) # 선수정보 \n 30
```

### 상속(Inheritance)
상속을 사용하면 부모 클래스의 정보를 자식 클래스에서 그대로 활용하거나 필요한 정보를 추가해서 활용할 수 있으므로 불필요한 중복이 사라져서 더 효율적인 코딩이 가능해진다.

```python
class Car:
    def __init__(self, name):
        self._name = name

    @ property
    def name(self):
        return self._name

    def honk(self):
        return "beep"

class Truck(Car): # Car class로부터 상속 받음.
    # super().__init__(가져올 변수)을 하면 부모의 init 그대로 가져옴.
    # 위에서 가져올 변수를 미리 정의해놓고 super().__init__(가져올변수) 하면 됨.
    # 그 후에 더 추가해도 됨. self._다른변수 = ???
    def honk(self):
        return "beep beep"
        #return super().honk() + ", BEEP!" # beep, BEEP!. 이렇게 하면 부모 클래스의 메소드를 불러온 후 거기에서 새로운 값을 추가하게 된다.


    def drive(self):
        return "vroom"

car = Car("Genesis")
print(car.honk()) # beep
truck = Truck("Bongo")
print(truck.name) # truck에는 name 메소드가 없어도, 상속을 받아서 실행시킬 수 없음.
print(truck.honk()) # beep beep. 상속받아도 메소드를 새로 정의하면 자신의 메소드가 실행됨.
print(truck.drive()) # vroom
print(car.drive()) # error. 자식 클래스에 메소드가 있어도, 내가 없으면 실행못함.
```
 
