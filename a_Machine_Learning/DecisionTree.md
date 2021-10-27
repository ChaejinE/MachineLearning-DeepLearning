# Decision Tree Element
![image](https://user-images.githubusercontent.com/69780812/139042078-4304b80d-17a8-4722-b3fe-e39c13ed63b4.png)

- Node, Branch, Leaf
  - Node : 1개의 Attribute(속성) Xi를 판별하는 역할
  - Branch : 각 노드로 부터 나온다. 속성 Xi에 대한 1개의 값
  - Leaf : 최종단에 있다. Xi일 때의 그것에 대응하는 기대값 Y에 해당된다.
  - Attribute이란 Humidity, Outlook, Wind와 같이 적합한 판단에 대한 주요한 판단 기준이며 속성에는 여러 개의 Value가 있을 수 있다. Outlook-> Sunny, Overcast, Rain

# 예제
![image](https://user-images.githubusercontent.com/69780812/139042917-aa3aa7e0-4b26-405f-bb99-bf448fffc0fd.png)

- 타이타닉 호에서의 생존 여부를 나타내는 예제
- 가지에는 또다른 Node or Leaf가 올 수 있다.
- Root Node : 남자인가?
- 어떤 특정 대상이 정해지면 사망과 생존 확률을 예측할 수 있게 되는 것이다.
- 이처럼 굉장히 직관적이고 편리하게 예측 모델을 만들 수 있고 새로운 데이터에 대해 쉽게 판단할 수 있어 많이 사용된다.

# Entropy
- 어떤 속성을 Root Node에 둘 것인지는 매우 중요하다.
  - 특히 속성이 여러개의 경우
- Decision Tree를 구성할 때 목표
  - 더 Compact하게 만드는 것 : Entropy를 이용한다.
  - Entropy : 분포의 순도(purity)를 나타내는 척도다.
  - 데이터 순도가 높을 수록 낮은 Entropy
  - 많이 섞이면 섞일 수록 Entropy 가 커진다.

![image](https://user-images.githubusercontent.com/69780812/139043183-df8cdbc0-43d8-4237-9ce1-e186d8904734.png)

- Entropy를 정의하는 식이다.
- 값을 높게하면, Decision Tree의 구성은 더 Compact 해진다.
- Pi는 특정 값 i가 일어날 확률을 의미한다.
- 3개 Yes, 2개 No인 경우
  - E = -(2/5)log(2/5) - (3/5)log(3/5) = 0.971
- 4개 Yes, 0개 No인 경우
  - E = -(4/4)log(4/4) - (0/4)log(0/4) = 0
- 순도 100%인 경우는 Entropy가 0이된다.
- Yes 2, No 2인 경우는 Entropy가 1이된다.

![image](https://user-images.githubusercontent.com/69780812/139043820-4ab6480b-f597-4887-a172-ef7164af256d.png)

- 이처럼 Entropy는 Class의 분포가 고를 수록 큰 값을 갖고, 특정 값으로 몰려있으면 0이된다.
- Class가 2개인 경우에 대한 데이터 분포에 따른 엔트로피 변화를 보여주는 그림이다.
- 고르면 고를 수록 코딩에 필요한 bit효율이 올라가게 된다.

# Decision Tree 만들기
![image](https://user-images.githubusercontent.com/69780812/139044079-9c025fac-605a-42d1-bcf7-68ef2087155a.png)

- 위와 같은 조합인 경우 Decision Tree는 어떻게 만들까?
- 4개 속성 Temperature, Outlook, Humidity, Windy
  - 골프를 칠 것인지 아닌지를 쉽게 예측해보자.
- 최적의 Decision Tree를 만들려면 각 속성에대한 Entropy를 계산해야 한다.
  - 앞서 살펴본 계산식에 따라 계산하면 Outlook이 최고의 Entropy가 되므로 Root Node로 결정한다.
  - 그 다음 Outlook 속성의 3개값, Sunny, Overcast, Rain을 가지로 사용해 똑같은 방식으로 다음 노드를 결정한다.
- [참고](https://wooono.tistory.com/104)

# Decision Tree Overfitting
![image](https://user-images.githubusercontent.com/69780812/139044750-bc3425c5-9d14-465d-b422-7cc2f6df8b28.png)

- Tree가 커지면 커질 수록 점차 세밀한 분류가 가능해지지만, 학습 데이터에 특화될 가능성이 커진다.

## 피하는 방법 - pruning(가지치기)
![image](https://user-images.githubusercontent.com/69780812/139045108-5b0ab4fc-06f4-4724-b929-075a065b0b8e.png)

- 오캄의 방식처럼 항상 단순한 것이 진리는 아니지만, 학습데이터에 너무 특화되어 있는 일반성을 잃는 것을 경계하기 위해 머신러닝에서 학습 모델을 결정할때 흔히 사용한다.
- Tree를 줄이면 학습 데이터에 대한 Error가 줄어들긴 하지만, Test 데이터에 대한 Error가 줄어들게 된다.
  - Pruning이라한다.
  - 지나치게 세분화 되는 것을 막는다.
- Prepruning과 Postpruning을 적절하게 섞어 사용한다.

# Decision Tree 장점
- 빠르게 구현 가능
- 특징의 유형에 상관없이 잘 동작
- Outlier에 대해 상대적으로 덜 민감
- 튜닝을 위한 파라미터가 적음
- 값이 빠진 경우도 효율적으로 처리 가능
- 해석이 쉬워진다.
