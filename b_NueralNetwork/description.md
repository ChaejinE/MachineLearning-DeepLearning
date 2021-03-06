# Neural Network
- 사람이 학습하는 방식을 비슷하게 구현한 것
## 생물학적 신경망
![image](https://user-images.githubusercontent.com/69780812/137630568-328b06cb-e9d0-4810-832e-76f4215d63f8.png)

- Dendrite(수상돌기) : 외부로부터 신경자극을 받는 역할
- Axon(축삭돌기) : 전류와 비슷한 형태로 다른 Nueron으로 신호를 전달하는 역할
- Soma(신경세포체) : 신경세포의 핵을 담당한다. 여러 Neuron으로 부터 전달되는 외부자극에 대한 판정을 하여 다른 뉴런으로 신호를 전달할지 최정 결정한다.
- Synapse(시냅스) : 어떤 Neuron의 Axon 말단과 다음 Nueron의 Dendrite의 연결 부위이다. 다른 Neuron의 Axon으로 부터 받은 신호를 어느 정도 세기(Weight or Strength)로 전달할 것인지를 결정한다.
## 구조
![image](https://user-images.githubusercontent.com/69780812/137630579-ff06de50-2c57-40cd-9b96-49f9a2d92c2c.png)

- (Dendrite, 입력) X (Synapse, 가중치) = (Soma, 뉴런의 Activation Function)을 통해 Output, Y를 결정한다.
  - Activation Function의 Threshold와 비교해서 +1, -1을 출력한다.
- ANN(Artificial Neural Network)는 보통 이런 Nueron들을 Multi-layer로 구성하고 Backpropagation 알고리즘을 통해 Sysnapse의 Weight들을 조절해 나가는 과정을 거친다.
  - **이를 Training이라 한다. 반복훈련을 통해 가중치의 최적값이 정해진다.**
# Basic Theory
## Hebbian Rule
- Donald Hebb's Neuron의 Synapse에 기반한 학습 법칙
- 학습이란 Synapse의 Strength를 조정하는 것으로 정의했다.
## Perception
![image](https://user-images.githubusercontent.com/69780812/137630469-6e51ca87-1434-472f-9660-01d1f84f319d.png)

- Activation Function이 추가된 구조
  - 입력의 중요도에 따라 출력이 결정되는 수학적 Model로서 의미가 있다.
- **입력의 중요도는 Weights에 따라 결정된다는 개념이 도입됐다.**
- 초기에는 STEP Function을 사용했다. (Rosanbalt 시절)
- Perception의 문제는 0과 1 같은 극단적인 결과만을 도출한다는 것이었다.
  - Multi Layer 신경망의 경우 좋은 결과를 얻기 어렵다.
  - 또한, Perception 기반의 뉴련은 **Weights or bias의 작은 변화가 Output쪽에 작은 변화를 만들어 내면서 신경망을 학습시킨다**(Backpropagtion의 근본적 개념)는 오늘날의 학습 개념과 부합이 잘 안된다.
- 해결책 : Step Function -> Sigmoid Function
## Sigmoid Function
![image](https://user-images.githubusercontent.com/69780812/137630527-09083fea-1b9e-4d60-b935-867b9d4581ad.png)

- 0~1 연속적 출력값이며 Weight나 Bias를 조금 변화시킬 때 출력도 조금 변할 수 있다.
- Sigmoid Function = 1 / (1+e^(-z)) (z : (x1, x2 ..., xn) * (w1, w2, ..., wn) + baias)
- 입력 결정 후 편미분으로 Weights나 bias가 약간 변화시키면, 출력이 그에 상응하여 변한다는 것을 알 수 있다.
# Gradient Descent
- Weights, Bias의 작은 변화량에 대한 출력의 변화량은 선형적(Linear) 하다. 이런 선형적 특성으로 Weights, bias를 조금씩 바꾸며 원하는 출력으로 이끌 수 있다.
  - 이를 잘 수행하기위해 Gradient Descent 방법을 도입한다.
- 최적값을 찾아갈 때 흔히 쓰는 방법이다.

![image](https://user-images.githubusercontent.com/69780812/140022052-6334a27a-057f-4654-8a73-3abe41ee1511.png)

- 어느 위치에서 편미분값이 음수가 되는 방향을 계속 선택하면 최적값에 도달할 수 있다.
  - 이러한 기울기 값들(기울기 벡터)은 해당 weight와 bias의 중요도라고 볼 수 있다. 기울기 벡터는 어느 방향으로 가파른지를 알려주는 데, 딥러닝 학습에서 새롭게 받아들이자면 중요도로도 볼 수 있다는 것이다.
  - 어떤 변수에 대한 기울기 벡터가 \[3 1]이라면 첫번째 변수가 3배 더 중요하다라는 뜻이 된다.
  - 즉, 위 그림에서는 x 라는 변수의 변동이 그 부근에서는 더 큰 영향을 끼치고있다는 의미라고 생각할 수 있다.
- 이렇게 KnownInput - KnownOutput을 통해 Weights, bias를 조금씩 바꿔과며 최적의 상태가 되도록 하는 것이 Supervised Learning 이다.
# Backpropagation
- Error의 역전파를 통해 최적의 학습 결과를 찾가는 것이 가능해졌다.
## Cost Function
![image](https://user-images.githubusercontent.com/69780812/137630597-20c9463d-e25d-4812-b400-d04ba486df0f.png)

- Ex) C(w, b) = (1/2n)sum(|y(x)-a|^2)
  - y(x) : target value
  - a : input x에 대한 신경망 실제 출력
- C(w, b) 값은 Error에 대한 값이다. 이를 최소화하고자한다.

![image](https://user-images.githubusercontent.com/69780812/137630601-2412070e-8bf1-4acf-9657-4c87e527ee28.png)

## Training based gradient descent
![image](https://user-images.githubusercontent.com/69780812/137630665-653f7016-dabe-420d-95a6-17cdeb9831b0.png)

- 뉴런들이 많고, 입력값들이 많을 수록 해를 구하기가 어려워 지는데, 이는 Gradient Descent 방법이 필요한 이유다.
- 어느정도 위치에 바닥이 있는지 정확히 알 수 없지만, 현재 위치에서 **기울기가 가장 큰 방향**으로 내려가면 바닥에 도달할 수 있다.
- Input -> Model -> CostFunction이 최소가 되도록 W,b 반복적 갱신 -> 최소값(최적값)
- Gradient Descent 방법의 단점은 경사가 큰 경우 빠른 속도로 수렴하지만, 거의 바닥에 오면 기울기가 작아져 수렴속도가 현저히 느려진다.
## Backpropagtion 기본 개념
![image](https://user-images.githubusercontent.com/69780812/137630676-9d9a35aa-6cd2-4703-9b99-a527406b1274.png)

![image](https://user-images.githubusercontent.com/69780812/137630686-934353d5-f6fb-4ad4-a6f4-2ecc8bc10213.png)

- Weights, bias를 편미분을 통해 아주 작게 변화시키면 출력쪽에서 생기는 변화 역시 매우 작은 변화가 생긴다.
  - 작은 변화에서의 관점으로 봤을 때 즉, **작은 구간만 봤을 때 [선형적 관계](https://ko.wikipedia.org/wiki/%EC%84%A0%ED%98%95%EC%84%B1)가 있다.**
- Backpropagation은 출력부터 반대 방향으로 순차적으로 편미분을 수행해 가면서 W,b 를 갱신시켜 간다는 뜻에서 만들어 졌다.

## Backpropagtion 수식 정리
![image](https://user-images.githubusercontent.com/69780812/140255482-6106ccd4-6b4f-43f8-9e27-f3e2098828b0.png)

![image](https://user-images.githubusercontent.com/69780812/140255592-3bc537b9-63a1-486d-b0fe-1a31353b1861.png)

---
신경망이 충분히 학습을 못했을 때, Error가 크게 나타나는데, 이는 Data가 한쪽으로 치우치지 않고 Generalization(범용성)을 가져야 학습 결과가 좋다.
## Sigmoid의 좋은 성질과 Delta rule
![image](https://user-images.githubusercontent.com/69780812/137630707-94afc132-79cc-4f1c-9d58-e47a351e0c20.png)

- Sigmoid는 앞에서의 연속적인 출력값을 통해 W,b의 작은 변화를 통한 출력의 작은 변화를 만들어 낼 수 있었다는 장점 뿐만아니라 **Backpropagation 식을 좀 더 쉽게 풀어쓸 수 있게해준다.**
## Learning Rate
![image](https://user-images.githubusercontent.com/69780812/137630716-c8e40ae6-d54a-435f-a6ed-a975d7e36511.png)
- 가중치를 바로 Update하지않고, 학습 속도를 조절해주는 하이퍼 파라미터다.
- W+ = W - (LearningRate)\*(편미분값)

