# Batch Normalization
- 딥러닝의 골치 아픈 문제 : Vanishing/Exploding Gradient
  - layer 수가 많아지면 많아질수록 누적되어 나타난다.
  - Sigmoid & hyper-tangent와 같은 비선형 포화함수를 사용하게 되면, 입력의 절대값이 작은 일부 구간을 제외하면 미분값이 0 근처로 가기때문에 역전파를 통한 학습이 어려워지거나 느려지게 된다.
  - ReLu를 Activation Function으로 쓰면서 문제가 완화되었지만 회피이지 본질적인 해결책이 아니라 망이 깊어지면 여전히 문제가 된다.
  - dropout이나 regularization 방법들 역시 본질적인 해결책이 아니므로 여전히 일정 Layer 수를 넘어가면 training을 성공시키는 것을 보장할 수 없다.

## Internal Covariate Shift
- 망이 깊어짐에 따라 이전 layer에서의 작은 파라미터 변화가 증폭되어 뒷단에 큰 영향을 끼치게 될 수 있다.
  - weight initialization, learning rate 등의 hyper-parameters
- 학습하는 도중에 이전 layer의 파라미터 변화로 인해 현재 layer의 입력 분포가 바뀌는 현상을 **Covariate Shift**라고 한다.

![image](https://user-images.githubusercontent.com/69780812/138060088-31d96095-cbf9-4aef-b1d6-6aa1b5ec8e64.png)

- 이는 건축에서 하중에 의해 기둥이 휘어지는 Buckling과 비슷하다.
- c, d 경우 처럼 휘어짐을 방지하는 수단이 필요하게되며 batch normalization, whitening 기법이 그 예다.

## Covariate Shift를 줄이는 방법
- 각 layer로 들어가는 입력을 Whitening 시킨다.
  - whitening : 입력을 평균 0, 분산 1로 바꿔준다는 것이다.
  - 단순히 Whitening만 시키면 parameter를 계산하기 위한 최적화 과정, back-propagation과 무관하게 진행되기 때문에 특정 파라미터가 계속 커지는 상태로 Whitening이 진행될 수 있다.
- 단순하게 Whitening 하는 것 말고 확실하게하는 방법이 batch normalization이다.
- BN은 평균과 분산을 조정하는 과정이 별도의 Process로 있는 것이 아니라 신경망 안에 포함되어 Training 시 평균과 분산을 조정하는 과정 역시 같이 조절 된다는 점이 구별되는 차이점이다.

## Batch Normalization(BN)
- Normalization을 원래는 모든 Training data 전체 집합에 대해 실시하는 것이 좋겠으나 mini-batch SGD 방식을 사용하기 때문에 mini-batch 단위로 BN을 실시하게된다.
- 단, mini-batch 집합의 선정은 가급적이면 **correlation이 적어 mini-batch가 전체 집합을 대표하는 것이라 생각해도 무방하도록**해줘야 한다고 한다.

![image](https://user-images.githubusercontent.com/69780812/138210444-94b5904a-8b25-4b4f-a286-bb723b2fc72b.png)

- 평균과 분산을 구하고, 입력을 정규화시킨다.
  - 정규화 과정에서 평균을 빼주고 분산으로 나눠주게 되면 분포는 -1 ~ 1 범위로 된다.
- BN이 Whitening과 다른 부분은 정규화 후 Scale과 Shift 연산을 위해 gamma, beta가 추가되어있다는 것이다.
  - gamma, beta가 추가됨으로써 정규화 시켰던 부분을 원래대로 돌리는 identity mapping이 가능해진다.
  - 학습을 통해 gamma와 beta를 정할 수 있어 단순하게 정규화만을 할 때 보다 훨씬 강력해진다.

![image](https://user-images.githubusercontent.com/69780812/138210702-8e741f70-bc05-4769-a1ed-76658c89bfe6.png)

- BN은 보통 non-linear activation function 앞쪽에 배치되며 위 그림과 같은 형태가 된다.

![image](https://user-images.githubusercontent.com/69780812/138210823-7edcb91f-ef90-4cbd-92f1-ef75f17d9f2b.png)


- BN은 **신경망에 포함**되므로 Back-propagtion을 통해 학습이 가능하다. 위와 같은 chain rule이 적용된다.

## Training과 Test 시 차이
- Training 시에 각 mini-batch마다 gamma와 beta를 구하고 그 값을 저장해 놓는다.
- Test 시에는 학습 시 mini-batch 마다 구했던 gamma와 beta의 평균을 사용한다.

![image](https://user-images.githubusercontent.com/69780812/138211130-b62d1d52-4d6b-4d82-99e6-5c9b9fae881e.png)

- 유사코드를 봤을 때, mini-batch에서 구한 평균들의 평균을 사용하고 분산은 분산의 평균에 m/(m-1)을 곱해준다.
  - m/(m-1) : unbiased variance에는 "Bassel's correction"을 통해 보정해준다.
  - 이는 학습 전체 데이터에 대한 분산이 아니라 mini-batch 들의 분산을 통해 전체 분산을 추정할 때 통계학적으로 보정을 위해 베셀의 보정값을 곱해주는 방식으로 추정한다.

---
- Batch Normalization은 단순하게 평균과 분산을 구하는 것이 아니라 scale(gamma), shift(beta)를 통한 변환을 통해 훨씬 유용하게 되었다.
- 또한 신경망의 layer 중간에 위치하게되어 학습을 통해 gamma와 beta를 구할 수 있다.
- Covariate shift 문제로 망이 깊어질 경우 학습에 많은 어려움이 있었지만, BN을 통해 Covariate shift 문제를 줄여줬다.
  - 학습의 결과도 좋아지고 빨라졌다.
---
- 앞의 내용들은 Fully-connected network에 적용할 때에 대한 것이었고, Convolutional network에 적용할 때는 그 특성을 고려해줘야한다.
## BN을 Convolutional network에 적용하는 방법
![image](https://user-images.githubusercontent.com/69780812/138211903-cb4bd234-4342-40f7-8568-a73a6f7f9ce3.png)

- g : Activation Function
- W : Weights
- u : input data
- b : bias

![image](https://user-images.githubusercontent.com/69780812/138211937-6fa4273f-ddd6-4e04-8bf5-92937547152e.png)

- BN을 non-linearity function g앞에 적용하고, bias는 shift 항으로 대체될 수 있으므로 무시하면 위와 같이 된다.
- BN transform은 **각각의 activation에 x=Wu에 독립적으로 적용**한다. 또한 학습을 통해 scale과 shift가 결정된다.
- Convolution layer에 적용할 때에는 Convolution 특성을 살린다.
  - Convolution layer는 Shared weight, sliding window 방식을 통해 feature-map의 모든 픽셀에 대해 동일연산을 수행한다.
  - BN을 적용할 때는 mini-batch에 있는 모든 activation, 모든 위치까지 함께 고려해줘야한다.
  - m'(간주되는 mini-batch의 크기) = m(mini-batch_size) x p x q(output featuremap size)
  - 위와 같이 mini-batch가 m x p x q로 커진 것으로 간주하고 평균과 분산을 구한다.
  - scale과 shift는 **fully connected의 경우 처럼 activation 마다 붙는 것이 아니라 feature-map에 대해 scale과 shift 쌍이 학습을 통해 결정**된다.

## MNIST 데이터로 실험

![image](https://user-images.githubusercontent.com/69780812/138212609-c2cc7538-5f55-4188-b6bb-835f8dc8df7a.png)

- Fuuly connected layer로만 구성된 신경망에서 BN을 적용했을 때 효과를 살펴보자.
- (a)를 보면 BN을 적용한 경우 학습 속도가 훨씬 빨랐고 결과도 더 좋았다.
- (b), (c)는 마지막 hidden layer에 있는 특정 뉴런(activation)에서 입력 분포 값을 보여주는 것인데 BN이 적용된 경우 거의 변화가 없이 안정적이었지만, 적용되지 않은 경우에는 값이 흔들리는 것을 확인할 수 있었다.

## Google Inception 구조에 ImageNet 데이터로 실험
- 실제 복잡한 Deep CNN에서도 효과가 있는지 실험했다고한다.
- 기존 Inception 구조에서 BN을 적용하면 기대했던 효과를 거둘 수 없으므로 BN 특성에 맞게 구조를 약간 변경했다고한다.
  - Dropout 제거 -> BN 적용 시 Regularization 효과를 얻을 수 있으므로 dropout layer 제거
  - mini-batch 정할 때 철저하기 섞음 -> correlation이 적을 수록 효과적이므로 중복 데이터 제거
  - L2 weight regularization 값 줄임 -> 그냥 효과적이어서 1/5로 줄였다고함.
  - Learning rate 감소율 증가 -> BN 적용 시 학습 속도가 빨라서 6배 줄였다고함, Learning rate 자체도 줄임
  - Local Response Normalization 제거 -> LRN 빼는 것이 대세라고함
  - Phtometric distortion 감소 -> 학습 시간이 빨라 인위적 변경을 통한 학습 데이터 늘리기 대신 real data 테스트에 집중, 변경을 전혀 시킨 것이아니라 변경의 강도를 줄였다고함)

![image](https://user-images.githubusercontent.com/69780812/138213355-253d02b5-6e34-42d4-9bd1-a86336c03209.png)

- 실험에서 다양한 구조로 테스트했다고 한다.
- Inception : 원래 인셉션 구조
- BN-Baseline : 인셉션과 동일하지만 각 nonlinearity 앞에 BN 배치
- BN-x5 : BN-Baseline에서 초기 학습 진도율을 5배 키워 적용
- BN-x30 : 30배
- BN-x5-Sigmoid : Bn-x5와 동일하지만, nonlinearity에 ReLU 대신 Sigmoid 사용

![image](https://user-images.githubusercontent.com/69780812/138213479-db0ddb66-0ff9-4965-9e1d-cff63942ac4a.png)

- BN-Baseline : 기존 Inception보다 절반 정도의 학습 과정에도 비슷한 수준의 정확도 얻을 수 있었다.
- BN-x5 : 기존보다 14배 빠르게 비슷한 정확도 도달
- BN-x30 : 기존보다 6배 빠른 수준, 최종 학습의 정확도는 74.8%까지 올라가 정확도가 개선되는 것을 확인했다.
---
종합하면
- Batch Normalization을 적용하면 Internal Covariate Shift 문제를 해결할 수 있다 (고질적 문제 해결)
- Vanishing/Exploding gradient 문제 해결책이 생기면서 낮은 학습 진도율이 아니라 높은 학습 진도율을 적용할 수 있어 학습 속도의 향상은 물론 정확도까지 개선할 수 있게 되었다.
