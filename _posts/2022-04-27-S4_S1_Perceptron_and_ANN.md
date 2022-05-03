---
layout: single
title: "Note 411 퍼셉트론(Perceptron)과 인공신경망(ANN)"
toc: true
toc_sticky: true
category: Section4
---

이번 섹션에서는 해석하긴 어렵지만 머신러닝보다 더 좋은 성능을 낼 수 있는 딥러닝에 대해 배울 것이다. 특히 이번 포스팅에서는 딥러닝을 배우기 위한 기본개념인 퍼센트론, 인공신경망, 활성화
함수 등에 대해 알아보자.

### 퍼셉트론(Perceptron)
뉴런은 자극을 받아 정보를 저장한 후 저장할 수 있는 용량을 넘어섰을 때 다른 뉴런으로 출력값을 내보내는 세포이다. 퍼셉트론은 인간의 몸에 있는 뉴런을 수학적으로 모델링한 인공 신경망에서의 뉴런이다. 즉, 퍼셉트론의 뉴런을 본따서 만들었기 때문에 둘의 역할과 구조는 같다.

퍼셉트론은 신경망을 이루는 가장 기본적인 단위이고, 다수의 신호를 입력 받아서 하나의 신호를 출력하는 **구조** 이다. 단층 퍼셉트론은 입력층과 출력층으로 구성된 퍼셉트론, 다층 퍼셉트론은 입력층, 은닉층(1개 이상), 출력층으로 구성된 퍼셉트론이다.

![image](https://user-images.githubusercontent.com/97672187/165420850-ffcd20cd-ee28-4edd-8c13-6231da0c2886.png){: .align-center}

이미지출처: https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=960125_hds&logNo=221043116238

위의 그림처럼 퍼셉트론은 입력층(x)을 제외하면 가중치-편향 연산(w)과 활성화 함수(y) 크게 두 부분으로 나눌 수 있다. y는 결과값을 의미하지만, 결과값을 도출하기 위해 사용하는 것이
활성화 함수이기 때문에 이 그림에서는 y를 활성화 함수라고 해보자.

먼저, 가중치-편향 연산을 보면 여러개의 입력신호를 받아 입력된 신호가 각각의 가중치와 곱해져서 그 결과를 더해주게 되고, 이를 가중합이라고 한다.

가중합이 계산이 되었다면 활성화 함수가 가중합을 어떻게 변환시켜 출력할지를 결정한다. 쉽게 말하면, 가중합을 적절한 출력값(분류문제에서는 확률, 회귀문제는 원래 선형이니까 굳이
활성화 함수가 없어도 된다)으로 변환 시켜준다.

### 활성화 함수(Activation Function)
여러가지 활성화 함수가 있지만 가장 자주 사용되는 4가지 함수를 알아보자.

1) 계단 함수(Step Function)

계단 함수는 입력값이 임계값을 넘기면 출력값을 1, 아니면 0을 출력하는 함수이다.

![image](https://user-images.githubusercontent.com/97672187/165421654-5d3ab06a-b7bd-4420-a2cc-a0cb6821d5da.png){: .align-center}

이미지출처: https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=leesu52&logNo=90189504569


2) 시그모이드 함수(Sigmoid Function)

신경망의 손실을 줄이기 위해 경사 하강법을 사용해 학습을 하려면 미분 과정이 필요하다. 하지만 계단 함수는 임계값 지점에서의 미분이 불가능하고, 나머지 지점은 모두 미분값(기울기)이
0이 나온다. 따라서 계단함수를 사용하면 학습이 제대로 이루어지지 않기 때문에 시그모이드 함수를 사용함으로써 미분이 가능하도록 한다.

![image](https://user-images.githubusercontent.com/97672187/165478281-a70260d3-3912-4d56-b13e-f2cec6f52fe7.png){: .align-center}

이미지출처: https://muzukphysics.tistory.com/entry/DL-7-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EA%B8%B0%EC%9A%B8%EA%B8%B0%EC%86%8C%EC%8B%A4-%EB%AC%B8%EC%A0%9C-%ED%95%B4%EA%B2%B0-%EB%B0%A9%EB%B2%95-Vanishing-Gradient

위의 그래프를 보면 임계값보다 작은 부분은 차이가 클수록 출력값이 0에 가까워지고, 임계값보다 큰 부분은 차이가 클수록 1에 가까워진다.
직선 형태가 아니기 때문에 모든 지점에서 미분이 가능하고, 미분값도 0이 아니게 된다.

3) ReLU 함수(FeLU Function)

위의 그래프를 보면 알 수 있듯이 시그모이드 함수의 출력값은 0과 1사이, 기울기는 0과 0.25 사이의 값만 표현이 가능하다. 따라서 기울기가 거의 0에 가깝기 때문에 역전파 과정에서 기울기를 곱할 때, 앞단까지 기울기가 잘 전달되지 않는
기울기 소실(Vanishing Gradient)문제가 발생하게 된다. 특히, 입력된 가중합이 음수이면 기울기를 거의 0으로 취급하기 때문에 음수의 가중합을 입력받을 때 소실이 발생할 수 있다.

이 기울기 소실 문제를 해결하기 위해 등장한 것이 ReLU함수이다.
ReLU 함수는 양의 값이 입력되면 그 값을 그대로 출력하고, 음의 값은 모두 0으로 출력한다. 따라서 가중합이 양수이면 그대로 출력되고 기울기는 1, 음수이면 0이 출력되고 0에서의
기울기는 0이다. 

결론적으로, 출력되는 값이 0과 1사이의 범위가 아닌 0부터 입력받은 양수의 가중합까지가 되고, 기울기는 0부터 0.25 사이인 sigmoid에 비해 0 혹은 1의 기울기를 가지므로 연산이 매우 빠르고 기울기가 너무 작아서 발생하는 기울기 손실 문제를 해결할 수 있다. 

가중합이 음수인 데이터를 모두 0으로 출력해서 발생하는 손실문제는 Leaky ReLU를 통해 해결할 수 있다.

![image](https://user-images.githubusercontent.com/97672187/165422719-20835c7b-150a-424a-b0c6-f80a42aa1b84.png){: .align-center}

이미지출처: https://www.v7labs.com/blog/neural-networks-activation-functions

<br>

![image](https://user-images.githubusercontent.com/97672187/165422533-31d70af8-550d-4661-ba1b-1ed8352276b4.png){: .align-center}

이미지출처: https://velog.io/@shonsk0220/ReLU-09-1


4) 소프트맥스 함수(Softmax Function)

소프트맥스 함수는 다중 분류 문제에 적용할 수 있도록 시그모이드 함수를 일반화한 활성화 함수이다. 가중합을 소프트맥스 함수에 통과시키면 해당 입력이 각 클래스일 확률이 얼마나 되는지
확률값을 나타내고, 모든 클래스의 확률값을 더하면 1이 된다. 

![image](https://user-images.githubusercontent.com/97672187/165423247-1f822515-dfc7-4fa3-a74a-4ed37059a9ed.png){: .align-center}

이미지출처: https://www.youtube.com/watch?v=07p67PnYzBI

위의 식에서는 3개의 가중합이 소프트맥스 함수의 정규화 과정을 거쳐서 총 합이 1이 되는 3개의 확률을 만들어낸다. 
따라서 1번 클래스일 확률이 70%, 2번 클래스일 확률이 20%, 3번 클래스일 확률이 10%가 되므로 해당 입력은 1번 클래스로 분류하게 된다.

### 논리 게이트와 퍼셉트론
퍼셉트론의 가장 단순한 형태는 AND, NAND, OR, XOR과 같은 논리 게이트(Logic Gate)이다.

- AND GATE: 입력 신호가 모두 1(T)일 때 1(T)을 출력한다.
- NAND GATE: AND GATE와는 반대로 입력신호가 모두 1일 때 0을 출력하고, 나머지는 모두 1을 출력한다.
- OR GATE: 입력신호 중 하나만 1(T)이면 1(T)를 출력한다.
- XOR GATE: 배타적 논리합(Exclusive-OR)이라고도 불리고, 입력신호가 다르면 1(T)를 출력한다.

XOR GATE는 AND, NAND, OR 게이트를 서로 조합해서 표현할 수 있다.

![image](https://user-images.githubusercontent.com/97672187/165428282-2fc75566-7153-42d4-aec8-fe217c556df0.png){: .align-center}

<br>


<br>

![image](https://user-images.githubusercontent.com/97672187/165428351-89391e12-a400-48e9-a558-4725e9d80f5e.png){: .align-center}

<br>


<br>

![image](https://user-images.githubusercontent.com/97672187/165428397-3082e4c0-1135-4209-9206-e48fb92b2ed1.png){: .align-center}

<br>


<br>

![image](https://user-images.githubusercontent.com/97672187/165428420-9748346f-0907-4395-b097-bb5287613bbd.png){: .align-center}

이미지출처: https://velog.io/@citizenyves/Perceptron-%EB%8B%A4%EC%B8%B5-%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0-XOR-%EA%B2%8C%EC%9D%B4%ED%8A%B8python-%EA%B5%AC%ED%98%84

NAND와 OR 게이트에서 AND 게이트를 적용하면 두 입력이 모두 1일 때만 1을 출력하니까 2번째, 3번째 입력이 1이 되어서 XOR 게이트와 같은 결과를 낼 수 있다.

### 인공 신경망(Artifical Neural Network)
딥러닝은 인공 신경망의 층을 깊게 쌓은 것인데, 여기서 인공 신경망(혹은 Neural-Net)은 실제 신경계를 모사한 계산 모델이다. 밑의 그림과 같이 표현할 수 있고,
딥러닝은 Hidden Layer라고 불리는 은닉층이 2개 이상인(층이 더 깊은) 인공 신경망 모델을 의미한다.

![image](https://user-images.githubusercontent.com/97672187/165429549-f3e86b58-07d1-417e-b589-9389e1ad3f19.png){: .align-center}

이미지출처: https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=apr407&logNo=221237867979


![image](https://user-images.githubusercontent.com/97672187/165428966-20b61ce2-7972-4a42-9f38-fb26b3a7e8b4.png){: .align-center}

이미지출처: https://profailure.tistory.com/18

딥러닝처럼 퍼셉트론을 다층으로 쌓아서 만드는 이유가 무엇일까? 현재 존재하는 많은 문제들 중 선형으로 분류할 수 없는 문제들이 있다. 
예시로 XOR GATE를 들 수가 있는데 위의 그림처럼 XOR GATE 문제를 해결하기 위해서는 2개의 직선이 필요하다. 직선 하나로는 문제를 분류할 수 없기 때문에.

이렇게 1개의 층으로 해결할 수 없던 복잡한 문제를 퍼셉트론을 2개 이상의 층(입력층을 제외하고 층이 2개이상)으로 쌓은 신경망인 다층 퍼셉트론 신경망(Multi-Layer Perceptron, MLP)을 구성해서 풀 수 있다.
단층 퍼셉트론은 입력층과 출력층으로 구성된 퍼셉트론, 다층 퍼셉트론은 입력층, 은닉층(1개 이상), 출력층으로 구성된 퍼셉트론이다.

인공 신경망의 층은 입력층, 은닉층, 출력층으로 나누어져 있다.

- 입력층(Input Layer)

입력층은 데이터셋이 입력되는 층이다. 입력되는 데이터셋의 피처에 따라 입력층 노드의 수가 결정된다. 보통 계산이 수행되지 않고 그냥 값들을 전달하기만 한다. 그렇기 때문에
신경망의 층수(깊이)를 셀 때 입력층은 포함하지 않는다.

- 은닉층(Hidden Layers)

은닉층은 입력층으로부터 입력된 신호가, 가중치 편향과 함께 연산되는 층이다. 일반적으로는 입력층과 출력층 사이에 있는 층을 은닉층이라고 부르고, 은닉층에서 일어나는 계산의
결과를 사용자가 볼 수 없어서 은닉층이라고 한다. 은닉층은 데이터셋의 특성 수와 상관없이 노드 수를 구성할 수 있다.

- 출력층(Output Layer)

출력층은 은닉층의 연산을 모두 마치고 가장 마지막에 있는 층이다. 풀어야하는 문제에 따라서 적절한 활성화 함수를 사용해 출력층을 잘 설계해야 한다.
이진 분류는 시그모이드 함수를 사용하고 출력층의 노드 수는 1이다(이진 분류라서 해당 클래스에 속할 확률만 구하면 다른 클래스에 속할 확률도 구할 수 있게됨). 
다중분류는 소프트맥스 함수를 사용하고 출력층의 노드 수는 타겟의 클래스 수이다.
회귀에서는 활성화 함수를 굳이 지정해주지 않고, 출력층의 노드 수는 출력값의 특성 수와 동일하게 설정한다. 단순히 하나의 수치를 예측하는 문제는 노드의 수를 1로 해주면 된다.

