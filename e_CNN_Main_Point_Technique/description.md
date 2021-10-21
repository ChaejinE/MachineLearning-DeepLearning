# 1. Batch Normalization
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
---

# 2. Dropout
- 머신 러닝의 또다른 문제 : Overfitting
## Overfitting
![image](https://user-images.githubusercontent.com/69780812/138214622-b7c0ee17-6d4a-40c7-b43d-dac82bc7062a.png)

- 학습데이터에 지나치게 집중해서 Test 데이터에서는 결과가 더 나쁘게 나오는 현상
- 가운데가 올바른 추정이라 한다면, 왼쪽은 지나친 단순화를 통해 에러가 많이 발생하는 경우로 underfitting이라하고, 오른쪽은 너무 정확하게 표현한 나머지 학습데이터에 대한 정확도는 좋지만 test에서는 에러가 발생할 수 있는 상황으로 overfitting이라한다.

![image](https://user-images.githubusercontent.com/69780812/138214766-c5de724b-f60b-4e45-98a9-33ecd78ef863.png)

- Overfitting을 표현하는 적절한 그래프로 학습 데이터에 대한 학습 결과는 계속 좋아지는데, Test 데이터를 이용한 결과는 더 이상 개선이 없는 그런 상황이다.

## Overfitting 문제 해결책
1. Data augmentation
2. regularization
- dropout은 일종의 regularization 방법이며 기존 방식들 보다 훨씬 효과적이었다. 하지만 Batch Normalization의 부수적 효과 중 하나가 regularization이 있어서 Dropout 없이도 성능이 충분히 잘 나온다고 주장하고 있어 각 설계자들의 결정에 달려있다.

## Dropout
![image](https://user-images.githubusercontent.com/69780812/138215063-061a72a4-594d-4af3-842d-b460636bb68a.png)

- 네트워크 일부를 생략하는 것이며 네트워크의 일부를 생략하고 학습을 진행하게 되면 생략한 네트워크는 학습에 영향을 끼치지 않게 된다.
- Model combination을 하게 되면 학습의 성능을 개선할 수 있다. 이것이 효과를 얻으려면 서로 다른 학습 데이터를 이용해서 학습하거나 모델이 서로 다른 구조를 가져야한다. 하지만, 복수개의 망을 학습 시키는 것은 매우 힘든 작업이다. 또한, 다양한 모델을 학습 시켰을 지라도 다양한 모델을 실행 시킬 때 연산 시간을 잡아먹기 때문에 빠른 Response time이 요구되는 경우 곤란함이 있다.
- 위 두가지 문제를 해결하기 위해 Dropout이 개발된 것이다.
- Voting효과 : 학습 사이클이 진행되는 동안 무작위로 일부 뉴런을 생략하여 생략되는 뉴런의 조합만큼 지수함수적으로 다양한 모델을 학습시키는 것이나 마찬가지이므로 모델 결합의 효과를 누릴 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138215397-1a79a335-859f-49cf-95dd-ca24449b7f4a.png)

- 실제 실행 시 : 생략된 모델들이 모두 파라미터를 공유하고 있으므로 각각의 뉴런들이 존속할 확률을 각각의 가중치에 곱해주는 형태가 된다.
- 학습 시에는 뉴런은 존속할 확률 p로 학습을 진행하고, 실행 시에는 각각의 넷에 얻어진 가중치에 존속할 확률 p를 곱해주게 된다.

## Dropout 효과
- 왜 regularization 효과가 있는 것인가 ?
  - 학습 데이터에 의해 각각의 net의 weight들이 서로 동조화 되는 현상(co-adaption)이 발생할 수 있다.
  - 무작위로 생략하면서 학습시킴으로써 이런 동조화 현상을 피할 수 있게 된다.

![image](https://user-images.githubusercontent.com/69780812/138215577-a4f8d301-2677-485b-953d-c0d2b66962f7.png)

- ZFNet논문에서 feature visualization을 통해 선명한 feature를 얻게 된다는 것을 확인할 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138215663-dada591b-a0b6-4c7d-9982-30b039156662.png)

- dropout을 하게되면 hideen neuron들의 acitivity(활성도)가 좀더 드문드문, Sparsity 해지는 경향이 생긴다.
  - dropout은 hidden-neuron이 coadaption이 일어나는 것을 막아 결과적으로 활성도를 떨어뜨리는 것을 목표로한다.
- 왼쪽은 히스토그램이 좀 더 넓게 퍼져있는데, 오른쪽을 봤을 때 히스토그램이 좀더 집중되는 것을 알 수 있다.(의도하진 않았다고함)
  - 히스토그램은 특정 값을 같는 element 갯수가 몇개인지를 보여줄 때 유용하게 사용하는 tool이다.
  - 특정 활성도 값을 같는 element가 고르게 퍼져있다는 것은 중구 난방으로 활성화 하고 있다는 것이고, 한쪽으로 집중되고 있다는 것은 특정 활성도 값을 대부분의 뉴런들이 그 값으로 활성화되고있다는 것이기 때문에 dropout의 목표에 부합하여 좋은 side-effect이라는 말인 것 같다.

## Dropout modeling
![image](https://user-images.githubusercontent.com/69780812/138216709-538147c8-506f-4256-830c-52178b38c857.png)

![image](https://user-images.githubusercontent.com/69780812/138216728-8f66f0fa-14fd-4118-8c1e-35238509e3a7.png)

- 보통 표준 네트워크는 위와 같다.

![image](https://user-images.githubusercontent.com/69780812/138216747-61c4e548-065a-4676-9dda-2a9a3474c033.png)

- Dropout을 적용한다는 것은 베르눌리 랜덤 변수 r_i(l)을 곱해주는 것으로 생각할 수 있다.
  - 베르눌리 랜덤 변수 : 유닛의 존재 유/무 두가지 값을 갖는 랜덤 변수
  - 유닛이 존재할 확률이 p라고하면 평균은 p이고 분산은 p(1-p)인 변수이다.
  - 각 유닛에 대해 독립적으로 랜덤변수를 곱해주므로 n개 유닛에대해서 2^n
- r(l)의 값에 따라 네트워크가 줄어드는 thinned 네트워크 y(l)이 되어 가중치 w(l+1)을 곱하면된다.

## with Max-norm Regularization
- Momentum, annealed learning rates 및 L1/L2 weight decay와 같은 기존 regularization을 같이 사용하면 대부분 좋은 결과가 나온다고한다. **그중 max-norm regularization**이 더 좋은 결과를 낸다고한다.

![image](https://user-images.githubusercontent.com/69780812/138217225-4e4defda-7405-46ee-b2c8-cf44f42c9de9.png)

- Max-norm : hidden layer로 들어오는 weight들이 특정 상수 보다 작게 해주는 방법
  - 상수 c는 조율 가능한 하이퍼파라미터다.
  - validation 데이터 집합을 이용해 c를 결정한다.
  - Dropout을 적용하지 않더라도 좋은 regularizer 임이 이미 증명되어 있다고 한다.
- Max-norm과 Dropout을 함께 사용하면 learning rate을 큰 값을 사용하는 것이 가능해져서 학습을 빠른 속도로 진행할 수 있다.
---
![image](https://user-images.githubusercontent.com/69780812/138217546-82c90149-4c29-4604-989b-25612958a9c1.png)

- 다양한 데이터 셋에대해서 dropout의 효과를 실험했다.

## Image Dataset

![image](https://user-images.githubusercontent.com/69780812/138217617-e064b9e1-d779-4db3-8d95-750d03228f84.png)

- DPM : Deep Boltzmann Machine
- 하이퍼 파라미터에 p 고정시키고 실험

![image](https://user-images.githubusercontent.com/69780812/138217689-fe864aeb-22e3-4791-b590-6d233c1bef83.png)

- 결과는 Dropout을 적용시키면 훨씬 성능이 좋고 꾸준히 성능이 개선되는 것을 확인했다.
- 이 외에도 텍스트 데이터 등에서도 효과적이었고, 다양한 데이터와 구조에 drop out을 적용했을 때 모두 성능이 개선되는 것을 확인할 수 있다고 한다.

## Dropout 부가 효과
- Dropout 사용 시 좀 더 선명한 특징(Salient feature)를 끌어 낼 수가 있게된다.
- 뉴런들을 무작위로 생략 시키면서 파라미터들이 co-adaption 되는 것을 막아 좀 더 의미있는 특징들을 더 추출하는 것으로 해석된다.
- 즉, 다른 파라미터와 같이 cost function을 줄여나가다 보면 파라미터의 co-adaption현상이 일어날 수 있는데, dropout을 하게 되면 서로 의지하던 것을 스스로 해줘야 하기 때문에 좀 더 의미있는 Feature를 끄집어낼 수 있게 된다.
- 큰 Activation을 보이는 뉴런의 수가 줄어들게 된다 : Sparse
  - hidden layer에 대한 뉴런들의 acitvation에 대한 히스토그램을 구했을 때, 0에서 큰 peak이 1개 있고, activation이 큰 값을 보이는 뉴런은 몇 개 없다.

## Hyperparameter p가 성능에 미치는 영향
![image](https://user-images.githubusercontent.com/69780812/138218725-25595f64-d15e-407a-b7c1-12fe4c00d801.png)

- 왼쪽은 뉴런 개수 고정을 고정시키고 p만 변화 시켜가며 테스트 진행
- 0.4 ~ 0.8 범위에서는 검사 오차가 거의 일정하게 나오는 것을 확인했다.
- 0.4 이하에서는 Dropout이 많아 underfitting이 일어나는 것으로 추정, 0.8 이상에서는 Dropout이 적어 Overfitting 효과가 나타는 것으로 해석된다.
- 오른쪽 n(뉴런수)xp를 고정하면서 실험 진행
- p가 낮을 때는 n만 고정 시켰을 때보다 오차율이 낮다. 이는 Dropout을 많이 시키고자 하면 뉴런의 개수를 키우는 것이 Underfitting의 영향을 줄일 수 있다는 것을 확인했다. 

## 학습 데이터 양에 따른 Dropout 효과
![image](https://user-images.githubusercontent.com/69780812/138219131-ba88328c-d43f-4f56-b512-32ae247c276c.png)

- 학습 데이터의 양이 극히 적은 경우(100, 500) 더 안좋은 결과가 나오지만, 그외의 경우에는 Dropout이 모두 효과적이라는 것을 확인할 수 있다.
- 어느 순간 늘리다 보면 Dropout의 효과가 줄어드는 것을 확인할 수 있는데, 데이터 양이 많아지면서 overfitting 가능성이 줄어들었기 때문으로 해석할 수 있다.

## Drop Connect
![image](https://user-images.githubusercontent.com/69780812/138219414-028d75f6-1d0e-4994-8db9-5d0eca661809.png)

- connection을 생략하고 노드(뉴런)은 남아있게 하는 것이다.
- dropout이 효과적인 방식임을 받아들이고 이를 모티브로한 새로운 방식이다.

![image](https://user-images.githubusercontent.com/69780812/138219587-e8bc17c7-e78a-4320-89ae-c77b31b4301f.png)

- 기존 베르누이 랜덤 변수를 통한 Dropout 식은 위와 같다.

![image](https://user-images.githubusercontent.com/69780812/138219624-53dec5ef-b39d-43b9-b2c5-9409402a82a4.png)

- DropConnect는 connection을 생략하기 위한 랜덤 변수가 붙었다. 이를 제외하면 dropout과 거의 동일하다.
- DropConnect가 Dropout을 더 일반화 시킨 것으로 이해하면 좋을 것 같다고한다. 노드를 생략하는 것보다 connection만 생략하면 훨씬 가능한 모델이 많이 나올 수 있기 때문이다.

![image](https://user-images.githubusercontent.com/69780812/138219859-b4010d41-c7db-4187-8a79-454fe72163ac.png)

- MNIST데이터로 사용한 Activation Function에 따라 DropConnect방식과 DropOut 방식의 성능이 엇갈린 성능이 나왔다.

![image](https://user-images.githubusercontent.com/69780812/138219918-c3a486a2-acbb-4e47-b77f-34f715b78deb.png)

- CIFAR-10에 대한 실험에서도(ReLU 사용) DropOut, DropCoonect 모두 사용하지 않을 때보다는 좋은 성능을 냈지만, 두 방식의 성능차를 확신할 수준은 아닌 것 같다.
---
- 이론적으로 DropConnect의 자유도나 표현력이 높아 Dropout보다 좋아야할 것 같은데, 약간 좋은 수준이라 논란의 여지가 있다고 한다.
- 하지만, DropOut, DropConnect 둘다 Deep Network에서 효과적이라는 것이 여러 곳에서 입증 되었고, Overfitting을 해결하기 위한 적절한 regularization 방법으로 사용할 수 있다.
---

# 3. Stochastic Pooling
## Pooling
- CNN에서 보통 Pooling이라 불리는 sub-sampling을 이용해 feature map size를 줄이고, 위치나 이동에 좀 더 강인한 성질을 갖는 특징을 추출할 수 있게 된다.
- 그 동안 max-pooling or average-pooling이 많이 쓰였다.
  - average pooling : deep CNN에서는 성능이 그리 좋지 않다고 한다. 활성화 함수 ReLU 특성에 의해 0이 많이 나오게 되면 평균 연산에 의해 강한 자극이 줄어드는 효과, down-scale weighting이 발생한다고 한다. tanh를 사용할 경우 더 나빠질 가능성이 있는데, 그 이유는 강한 양의 자극과 음의 자극에 대한 평균을 취하게 되면 서로 상쇄되는 상황이 발생할 수 있기 때문이다.
  - max pooling : 위와 같은 문제는 없지만 학습 데이터에 overfitting 되기 쉬운 문제가 있다고 한다.
- Stochastic Pooling은 max-pooling의 장점을 유지하면서 overfitting 문제를 해결하기 위한 방법이다.
## Stochastic Pooling
- 단순히 최대 Activation을 선택하거나 모든 Activation의 평균을 구하는 것이 아닌, Dropout과 마찬가지로 확률 p에 따라 적절한 activation을 선택한다.

![image](https://user-images.githubusercontent.com/69780812/138285724-c9d9cef6-9a59-4bfd-9a46-496641dad261.png)

- Pooling Window에 있는 모든 Activation을 합한 후 그 합으로 각각의 Activation을 나눠줘, normalize 확률을 구한다.

![image](https://user-images.githubusercontent.com/69780812/138285782-91dadea3-45e5-43f4-8787-a251187add9b.png)

- 각각의 확률을 구하고 그 영역을 대표하는 Activation은 호가률에 따라 선정하게 되며 식은 위와 같다.
- 이렇게 선택하게 되면 Max Pooling 효과를 그대로 유지하면서 Stochastic component를 이용해 Overfitting을 피할 수 있다고 한다.
- Max-Pooling은 강한 자극만 선택할 뿐이지만, 경우에 따라서는 최대값이 아니더라도 더 중요한 정보를 갖고 있을 수 있기때문에 이를 사용하면 가능해진다고한다.

![image](https://user-images.githubusercontent.com/69780812/138286096-378131af-a5b3-4dc7-99bc-9cfa4da63211.png)

- max-pooling 방식 : 2.4를 선택
- average-pooling 방식 : 4/9 = 0.444 로 의미 없는 작은 값
- stochastic-pooling : 2.4 일수도 1.6일 수도 있다.
- Backpropagation 시 max-pooling 방식과 마찬가지로 선정된 Actiavtion만 남기고 나머지는 전부 0으로 한 후 처리한다.

![image](https://user-images.githubusercontent.com/69780812/138286488-bcae2369-ab61-4e84-a9e3-c28b23e13938.png)

- 3x3 Pooling 윈도우의 Activation이 위와 같다고 가정해본다.

![image](https://user-images.githubusercontent.com/69780812/138286539-a7d2efbc-cd5e-49a3-aa85-ca31ae976c56.png)

- 확률을 구하면 왼쪽 그림과 같으며 크기에 따라 Sorting을 하면 오른쪽과 같은 결과가 나온다.
- Stochastic Pooling을 위해 5번 선택 -> Activation은 1.2가 선택된다.
- 위 예에서 Activation 값이 1.5가 2개가 잇는데, 이 것은 1번이나 2번을 선택하면 나오게된다. 즉, **결과적으로 같은 값을 갖는 것들이 많으면, Stochastic Pooling을 통해 선택될 확률이 증가**하게 된다.

## Probabilistic Weighting
- 학습을 마치고 실제 적용 시에도 Stochastic Pooling을 사용하면 성능이 떨어지는 경향이 있어 실제 Test 시에는 다른 방법을 사용한다.

![image](https://user-images.githubusercontent.com/69780812/138287129-a555275f-db3c-4dd7-b318-0a01b3750f1b.png)

- 각각의 Activation에 확률을 weighting factor로 곱해주고 전체를 더하는 방법을 취한다. 이것을 Probabilistic weighting이라 부른다.
- 각각의 Activation이 다른 Weighting factor를 곱해지기 때문에 Average Pooling과 다르다.
- 이러한 Probabilistic weighting으로 dropout이 test 시 modeling averaging을 통해 성능을 끌어 올리는 것과 비슷한 효과를 얻을 수 있다고 주장한다.
- **학습을 시킬 때는 확률에 기반한 sampling 방식으로 다양한 model을 학습하고, test시에는 probabilistic weighting을 통해 model averaging을 통해 성능을 올린다는 점이 dropout과 개념이 거의 유사하다.**
- Pooling window size를 n, pooling region 개수를 d라하면 가능한 모델의 수는 n^d가 되어 dropout 보다 훨씬 많은 모델 수가 가능하다.
- 실제로 Test 시 stochastic pooling과 probabilistic weighting을 비교한 싫머이 있었는데 probabilistic weighting을 사용하는 것이 더 좋은 결과를 냈다.

![image](https://user-images.githubusercontent.com/69780812/138288018-a6b7a7a6-23fb-4cd6-9bd1-9297a91e741a.png)

- Stochastic Pooling의 성능을 평가하기 위해 CIFAR-10 데이터에대해 여러 Pooling으로 실험해 봤다.
- Train 시 max-pooling이 결과가 좋지만, Test 결과를 보면 Max, average Pooling은 Overfitting이 일어난 것으로 보인다.
- 반면 stochastic Pooling은 training error는 max-pooling보다 높지만 test error는 학습을 계속 시키면서 줄어든다.
- Overfitting 문제에서 max나 average Pooling 보다 자유롭다는 것을 알 수 있다.
- CIFAR-100, SVHN에서도 어느 경우에서나 결과가 가장 좋았으므로 확실히 stochastic pooling이 overfitting을 피할 수 있는 Pooling 방식이라는 것이 입증되었다.

![image](https://user-images.githubusercontent.com/69780812/138288793-6a41851e-9237-4daa-abd3-9f3f08ae99fe.png)

- 위는 MNIST 데이터에 대해 training과 test시 각각 Pooling 방법을 실험한 결과다.
- 학습시 Stochastic Pooling을 사용하고, Test시 Probabilistic Weighting을 사용한 것이 결과가 제일 좋았다.
- Stochastic-100은 실제로 model averaging 효과를 확인하기 위해 stochastic pooling을 이용한 100개 모델을 테스트 시에 만들어 실험한 결과이다. 하지만 모델 수가 커질 수록 연산 시간이 늘어난다.
  - **이는 Test시 Probabilistic weighting을 사용하면 model averaging 효과도 얻을 수 있고, 연산 시간도 매우 짧게 할 수 있다는 것을 의미**한다.
- Dropout 방식이 model averaging으로 Fully connected layer에서 overfitting을 피하는 효과를 얻은 것처럼 pooling 시 model averaging 효과를 얻을 수 있는 stochastic pooling 방식을 공부할 수 있었다.

# 4. Maxout
- Dropout의 효과를 극대화시키기 위해 독특한 형태의 Activation Function을 고안한 것이다.

![image](https://user-images.githubusercontent.com/69780812/138294452-45627aa9-e3df-4de3-83c9-9ea3c5225954.png)

- Maxout hidden layer는 2개의 layer로 구성되어 있다.
  - 녹색 : affine function 수행
  - 파란색 : 최대값을 선택
- 녹색은 전통적인 hidden layer 처럼 activation function이 있는 것이 아니라 단순히 sum(x\*weight)형식이기 때문에 affine function이라고 부른다.
  - 녹색 영역에는 non-linear function이 없다.
- k개의 column이 있는데 이 k개의 column에서 동일 위치에 있는 것들 중 최고 값을 파란 영역에서 취하고 최종적으로 m개의 결과가 나오게 된다.

![image](https://user-images.githubusercontent.com/69780812/138294910-bb03fd5f-867d-4e7b-a009-a171bfc2cd52.png)

- 위 설명을 나타낸 Maxout 수식이라고한다.

## 의미
![image](https://user-images.githubusercontent.com/69780812/138295030-a5fc4b72-2c0e-4b3d-ae35-759f6bcf8077.png)

- 입력이 2개, 출력 1개, k가 3인 Maxout unit을 그린 그림이다.

![image](https://user-images.githubusercontent.com/69780812/138295108-5369394f-c669-4b6f-905d-7b0b19b7d61d.png)

- 3개 유닛을 이용하여 f(x) = x^2를 근사 시킨다로 하면 위와 같은 그림 형태가 나온다.
- f(x)를 3개의 직선으로 근사 시킨다면 위와 같은 그림이 되며 원래 convex한 경우는 각각의 직선 중 가장 큰 값을 선택하면 비교적 비슷한 모양을 얻을 수 있다.
  - 가장 큰 것을 선택한다는 것은 가장큰 activation value를 취한다는 뜻이다.
- k 값이 클수록 구간이 좀 더 세분화 되면서 원래 곡선과 비슷한 형태를 갖게 될 것이다. 그래서 위 그림이 오차가 많은 것으로 보이지만 affine function이 갖는 다양한 표현력을 고려하면 k 값이 크지 않더라도 convex function을 거의 표현할 수 있게 된다.
  - 이런 의미에서 Maxout은 universal approximator라고 생각할 수 있다.
- Maxout은 Dropout을 염두해두고 만든 활성화 함수 이므로 연동 시킴으로써 여러 개의 경로 중 하나를 고를 수 있는 효과를 얻을 수 있다.
- [(http://www.ke.tu-darmstadt.de/lehre/archiv/ws-13-14/seminarML/slides/folien13_Dang.pdf)

![image](https://user-images.githubusercontent.com/69780812/138295692-42d42e67-860f-4389-821a-02486226449e.png)

- 결과적으로 Maxout은 affine 함수 부분과 최대값을 선택하는 부분을 이용해 임의의 convex function을 piecewise linear approximation 하는 것이라고 할 수 있다고 한다. 위는 다양한 함수를 표현할 수 있는 것을 보여주는 예다.

## Maxout의 성능
![image](https://user-images.githubusercontent.com/69780812/138296077-c4df21ae-c129-4d8d-9aa2-baaab93a39d1.png)

- MINIST 데이터에대한 실험결과에서는 stochastic pooling을 써도 성능 개선이 있지만, Maxout을 적용하면 더 좋은 결과를 얻을 수 있음이 밝혀졌다.

![image](https://user-images.githubusercontent.com/69780812/138296214-bf36c9bb-2107-4ed9-a450-9fe5e952351a.png)

- CIFAR-10 데이터셋에서도 data augmentation 까지 같이 사용한 경우 가장 좋은 결과가 나왔다.
- CIFAR-100, SVHN에서도 마찬가지로 좋은 결과였다.

## Model Averaging
- Maxout에 Dropout을 적용하면서 생기는 model averaging 효과로 결과가 좋아진다.
- Dropout은 마치 엄청나게 많은 sub-model을 학습시키는 것과 유사한 효과를 얻고, 실제 적용 시 weight에 dropout 확률 p를 곱해주는 간단한 방식으로 model averaging(esemble효과)를 얻을 수 있었다.
- 수학적으로도 Softmax 함수를 사용하는 경우 p를 곱해주는 것으로 그것도 SOftmax layer가 1 layer인 경우에 한해서 수학적으로 p를 곱해주는 것이 정확하게 averaging 된다는 것이 증명됐다고한다.
  - Sigmoid, tanh 및 ReLU와 같은 함수를 활성함수도 Dropout을 적용하면 효과가있다는 것도 확인되었는데 이는 경험적으로 효과가 있다는 것이지 수학적으로 그런것은 아니다.
- Dropout과 어울리는 활성화 함수 Maxout을 만들어내는데, sigmoid tanh와 같은 함수 대신에 PWL(Piecewise Linear Approximation) 기능을 갖는 활성 함수를 고안하여 Dropout 효과를 극대화 시킨 것으로 보인다고 한다.

![image](https://user-images.githubusercontent.com/69780812/138297850-374c153b-f8fc-4c2e-b62e-56869be9309b.png)

- Model averaging 효과를 확인하기 위해 동일 신경망에 활성함수만 바꿔 실험한 결과인데, Maxout을 사용한 경우가 tanh을 사용한 경우보다 더 좋은 성능을 내었다.
- 결론적으로 tanh 사용하는 것보다 Maxout을 사용하는 것이 model averaging 효과를 극대화 시킬 수 있음을 확인했다.

![image](https://user-images.githubusercontent.com/69780812/138298126-00133414-d4e8-409f-988b-6cd9a9e97798.png)

- 또한 ReLU 함수와의 비교를 위해 layer 수를 증가시키면서 성능을 테스트해본 결과다.
- Maxout을 사용했을 때 완만하게 성능이 안좋아 지는 것이 확인되었다.
- 이는 Overfitting에 있어서 ReLU를 사용하는 것보다 Maxout이 훨씬 효과적이라는 의미가 된다.
- 또한 Qi Wang 논문에서는 경로(pathway)관점에서 해석했는데, 여러 개의 경로 중 하나를 선택할 수 있고, gradient가 back-propagation 되는 방향도 그 경로로만 전파가되어 학습 샘플로부터 얻은 정보를 sparse한 방식으로 encoding이 가능하다는 뜻도 있다.
  - ReLU를 사용할 때도 sparse한 성질이 있지만, 경로 선택에 있어서의 자유가 제한적이다.
  - 이 때문에 ReLU를 사용하는 것이 좀 더 Overfitting될 가능성이 높다고 판단한 것 같다.
  - 이러한 특성 때문에 Sparse한 경로를 통해 Encoding하여 Training하지만 Test 시에는 모든 파라미터를 사용하기 때문에 model averaging 효과를 얻을 수 있다.
