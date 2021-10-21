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
