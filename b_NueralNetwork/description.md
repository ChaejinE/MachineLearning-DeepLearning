# Neural Network
- 사람이 학습하는 방식을 비슷하게 구현한 것
## 생물학적 신경망
- Dendrite(수상돌기) : 외부로부터 신경자극을 받는 역할
- Axon(축삭돌기) : 전류와 비슷한 형태로 다른 Nueron으로 신호를 전달하는 역할
- Soma(신경세포체) : 신경세포의 핵을 담당한다. 여러 Neuron으로 부터 전달되는 외부자극에 대한 판정을 하여 다른 뉴런으로 신호를 전달할지 최정 결정한다.
- Synapse(시냅스) : 어떤 Neuron의 Axon 말단과 다음 Nueron의 Dendrite의 연결 부위이다. 다른 Neuron의 Axon으로 부터 받은 신호를 어느 정도 세기(Weight or Strength)로 전달할 것인지를 결정한다.
## 구조
- (Dendrite, 입력) X (Synapse, 가중치) = (Soma, 뉴런의 Activation Function)을 통해 Output, Y를 결정한다.
  - Activation Function의 Threshold와 비교해서 +1, -1을 출력한다.
- ANN(Artificial Neural Network)는 보통 이런 Nueron들을 Multi-layer로 구성하고 Backpropagation 알고리즘을 통해 Sysnapse의 Weight들을 조절해 나가는 과정을 거친다.
  - **이를 Training이라 한다. 반복훈련을 통해 가중치의 최적값이 정해진다.**
# Basic Theory
## Hebbian Rule
- Donald Hebb's Neuron의 Synapse에 기반한 학습 법칙
- 학습이란 Synapse의 Strength를 조정하는 것으로 정의했다.
## Perception
- Activation Function이 추가된 구조
  - 입력의 중요도에 따라 출력이 결정되는 수학적 Model로서 의미가 있다.
- **입력의 중요도는 Weights에 따라 결정된다는 개념이 도입됐다.**
- 초기에는 STEP Function을 사용했다. (Rosanbalt 시절)
- Perception의 문제는 0과 1 같은 극단적인 결과만을 도출한다는 것이었다.
  - Multi Layer 신경망의 경우 좋은 결과를 얻기 어렵다.
  - 또한, Perception 기반의 뉴련은 **Weights or bias의 작은 변화가 Output쪽에 작은 변화를 만들어 내면서 신경망을 학습시킨다**(Backpropagtion의 근본적 개념)는 오늘날의 학습 개념과 부합이 잘 안된다.
- 해결책 : Step Function -> Sigmoid Function
## Sigmoid Function
- 0~1 연속적 출력값이며 Weight나 Bias를 조금 변화시킬 때 출력도 조금 변할 수 있다.
- Sigmoid Function = 1 / (1+e^(-z)) (z : (x1, x2 ..., xn) * (w1, w2, ..., wn) + baias)
- 입력 결정 시 Weights나 bias가 약간 변화하면 편미분했을 때를 보면, 출력이 그에 상응하여 변한다는 것을 알 수 있다.
# Gradient Descent
- Weights, Bias의 작은 변화량에 대한 출력의 변화량은 선형적(Linear) 하다. 이런 선형적 특성으로 Weights, bias를 조금씩 바꾸며 원하는 출력으로 이끌 수 있다.
  - 이를 잘 수행하기위해 Gradient Descent 방법을 도입한다.
- 최적값을 찾아갈 때 흔히 쓰는 방법이다.
- 어느 위치에서 편미분값이 음수가 되는 방향을 계속 선택하면 최적값에 도달할 수 있다.
- 이렇게 KnownInput - KnownOutput을 통해 Weights, bias를 조금씩 바꿔과며 최적의 상태가 되도록 하는 것이 Supervised Learning 이다.
# Backpropagation
- Error의 역전파를 통해 최적의 학습 결과를 찾가는 것이 가능해졌다.
## Cost Function
- Ex) C(w, b) = (1/2n)sum(|y(x)-a|^2)
  - y(x) : target value
  - a : input x에 대한 신경망 실제 출력
- C(w, b) 값은 Error에 대한 값이다. 이를 최소화하고자한다.
## Training based gradient descent
- 뉴런들이 많고, 입력값들이 많을 수록 해를 구하기가 어려워 지는데, 이는 Gradient Descent 방법이 필요한 이유다.
- 어느정도 위치에 바닥이 있는지 정확히 알 수 없지만, 현재 위치에서 **기울기가 가장 큰 방향**으로 내려가면 바닥에 도달할 수 있다.
- Input -> Model -> CostFunction이 최소가 되도록 W,b 반복적 갱신 -> 최소값(최적값)
- Gradient Descent 방법의 단점은 경사가 큰 경우 빠른 속도로 수렴하지만, 거의 바닥에 오면 기울기가 작아져 수렴속도가 현저히 느려진다.
## Backpropagtion 기본 개념
- Weights, bias를 아주 작게 변화시키면 편미분하여 봤을 때, 출력쪽에서 생기는 변화 역시 매우 작은 변화가 생긴다.
  - 작은 변화에서의 관점으로 봤을 때 즉, **작은 구간만 봤을 때 선형적 관계가 있다.**
- Backpropagation은 출력부터 반대 방향으로 순차적으로 편미분을 수행해 가면서 W,b 를 갱신시켜 간다는 뜻에서 만들어 졌다.
---
신경망이 충분히 학습을 못했을 때, Error가 크게 나타나는데, 이는 Data가 한쪽으로 치우치지 않고 Generalization(범용성)을 가져야 학습 결과가 좋다.
## Sigmoid의 좋은 성질과 Delta rule
- Sigmoid는 앞에서의 연속적인 출력값을 통해 W,b의 작은 변화를 통한 출력의 작은 변화를 만들어 낼 수 있었다는 장점 뿐만아니라 **Backpropagation 식을 좀 더 쉽게 풀어쓸 수 있게해준다.**
## Learning Rate
- W+ = W - (LearningRate)\*(편미분값)

