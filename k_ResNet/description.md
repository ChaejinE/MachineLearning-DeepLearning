# 깊은 망의 문제점
- 더 깊은 망이 학습 데이터 속 존재하는 대표적인 개념을 잘 추출할 수 있어 학습 결과가 좋아진다.
- But, 그 망이 좋은 결과를 낼 수 있도록 학습 시키기가 점점 더 어려워진다.
- Vanishing/Exploding Gradient : CNN 파라미터 Update 시, gradient값이 너무 큰 값이나 작은 값으로 포화되어
더 이상 움직이지 않아 학습의 효과가 없어지거나 학습 속도가 아주 느려지는 문제
  - BatchNormalization
  - Weight Initiailization
  - 위 방법들이 있지만, layer가 일정 개수를 넘어가면 여전히 골치거리라고 한다.
- 더 어려워지는 학습 방법 : 망이 깊어지면, 학습 파라미터의 수가 비례적으로 늘어난다.
  - Overfitting
  - Error가 커지는 상황 발생

# 망이 깊어졌을 경우 결과 - ResNet 팀 실험
![image](https://user-images.githubusercontent.com/69780812/138550541-2c3f99a2-1e87-4e7a-af18-3354a6ee559d.png)

- 20-layer VS 56-layer 비교 실험
- 학습 오차와 테스트 오차 둘다 56-layer가 더 높았다.

# Residual Learning
![image](https://user-images.githubusercontent.com/69780812/138550596-3a8aabf6-4687-40d3-a88e-393f08df6611.png)

- 위 평범한 망은 입력 x를 받아 2개의 wieghted layer를 거쳐 출력 H(x)를 낸다.
- 학습을 통해 최적의 H(x)를 내는 것이 목표다.

![image](https://user-images.githubusercontent.com/69780812/138550619-ef42f81d-0d97-43e1-92a4-186e2bfe8077.png)

- H(x) - x를 얻는 것으로 목표를 수정한다면 ? 출력과 입력의 차를 얻을 수 있도록 학습하게 된다면 ?
- F(x) = H(x) - x 라면 출력은 H(x) = F(x) + x 가 된다.
- 블락은 위 그림처럼 바뀌며 이것이 바로 Residual Learning의 기본 블락이 된다.
- 입력에서 바로 출력으로 연결되는 Shortcut 연결이 생겼고, Shortcut은 파라미터가 없이 바로 연결되는 구조이므로 연산량 관점에서
덧셈이 추가되는 것 외에는 차이가 없다.
  - shortcut 연결을 통한 연산량 증가는 없다.
- H(x) - x 를 얻기 위한 학습을 하게 되며 최적의 경우 F(x) = 0이되어야하므로 학습할 방향이 미리 결정된다.
  - pre-conditioning 구실
- F(x)가 거의 0이 되는 방향으로 학습 하게되면, 입력의 작은 움직임(fluctuation)을 쉽게 검출할 수 있게 된다.
  - 그런 의미에서 F(x)가 작은 움직임, residual을 학습한다는 관점에서 residual learning이라 불리게 된다.
- 몇 개의 layer를 뛰어넘어 입력과 출력이 연결되므로 forward나 backward path가 단순해지는 효과를 얻을 수 있다.
- identity shortcut 연결을 통한 효과
  - 1. 깊은 망도 쉽게 최적화 가능
  - 2. 늘어난 깊이로 인해 정확도 개선
- Residual learning이 고안된 이유 ? -> 어느 일정 정도 이상의 layer 수를 넘어서게 되면, 결과가 나빠지는 문제를 해결하기 위함

# ResNet 팀의 실험
- VGGNet의 설계 철학을 많이 이용
  - 대부분의 Convolution layer는 3x3 kernel을 갖도록 하였다.
  - 복잡도(연산량)을 줄이기 위해 max-pooling, hidden fc, dropout 등을 사용하지 않았다.
  - 1. 출력 feature-map 크기가 같은 경우 해당 모든 layer는 모두 동일한 수의 filter를 갖는다.
  - 2. Feature-map의 크기가 절반으로 작아지는 경우 연산량 균형을 맞추기 위해 filter 수를 두 배로 늘린다.
  - 3. Feature-map의 크기를 줄일 때는 Pooling 대신 Convolution 수행 시 stride를 2로 하는 방식을 취한다.

![image](https://user-images.githubusercontent.com/69780812/138550945-ccf6aff6-4cd3-4b8e-96e1-22f5bf9cab63.png)

- VGGNet 보다 filter수를 줄여 연산량을 20% 미만으로 줄인 Plain Network (34-layer)와 ResNet을 비교했다.
  - ResNet도 비교를 간단히 하기위해 Shortcut connection이 연결되도록 하였다.

![image](https://user-images.githubusercontent.com/69780812/138550954-a5f2e359-6fc8-4d20-a76c-fec68a1762aa.png)

- 또한, 18, 34, 50, 101, 152-layer에 대해 실험을 수행한다.
  - 50, 101, 152는 구조가 조금 다르다.

## 실험 결과
![image](https://user-images.githubusercontent.com/69780812/138550987-f684a4d7-012f-403c-a971-e90113578323.png)

- 18, 34 layer에 대한 Plain, Residual Network의 비교 그림이다.
- Plain의 경우 34가 18-layer보다 결과가 약간 나쁘다.
- Residual network은 34가 18-layer보다 결과가 좋다.

![image](https://user-images.githubusercontent.com/69780812/138551005-82a0e981-9ce7-45a4-ba26-5c67a12940f2.png)

- Top1-error 율에서 비교한 결과를 보더라도 알 수 있다.
- 또한 그 이전 그림을 봤을 때 Residual Net의 수렴속도가 더 빠르다는 점을 알 수 있다.
- 결과적으로 Residual network가 좋은 결과를 내면서도 빠르다는 것을 알 수 있다.

# Deeper Bottleneck Architecture
![image](https://user-images.githubusercontent.com/69780812/138551070-59d96943-d401-484a-b1d1-c7e78838d465.png)

- 50/101/152-layer 에서는 기본 구조를 위와 같이 족므 변경시켰다.
- Bottleneck 구조 ? : 차원을 줄였다가 뒤에서 차원을 늘리는 모습이 병목처럼 보여서 붙여진 구조 이름이라고 한다.
  - 연산 시간을 줄이기 위함이다.
  - 1x1 convolution은 NIN이나 GoogLeNet Inception 구조처럼 차원을 줄이기 위한 목적이다.
  - 마지막은 차원을 늘리지만, 결과적으로 **3x3 conv를 2개 연속으로 연결시킨 구조에 비해 연산량을 절감시킬 수 있다.**

# CIFAR-10 데이터에 대한 실험
- CIFAR-10 dataset : 32x32 크기의 작은 영상 데이터 집합

![image](https://user-images.githubusercontent.com/69780812/138551289-189deb01-f161-405f-a429-b009c477ad7f.png)

- ResNet팀은 CIFAR-10에 대한 검증을 위해 망 구성을 약간 변형시켰다.
  - ImageNet 데이타의 경우 224x224로 크지만, CIFAR-10은 영상의 크기가 작기 때문
  - 동일 layer 수를 갖는 Plain과 Residual Network을 비교하는 실험을 한다.
  - 6n개의 3x3 conv layer 그룹을 사용한다.
  - 각각 2n에 대하여 feature map 크기가 {32, 16, 8}, filter 수는 {16, 32, 64}로 연산량의 균형을 맞춰줬다.
  - 맨 마지막은 global average pooling, 10-way softmax를 배치한다. (6n + 2 -layer)
  - n을 바꿔가며 더 깊은 layer에 어떤 결과가 나타나는지 비교실험을 한다.

![image](https://user-images.githubusercontent.com/69780812/138551311-486ab7bb-9d13-4895-bd4a-2a8965171473.png)
- Residual network에서는 n = 18로 설정하여 110-layer까지 실험했을 때, 56-layer 보다도 성능이 좋았다.

## Layer별 Response에 대한 분석
![image](https://user-images.githubusercontent.com/69780812/138551374-559fb2ad-8822-47e8-8804-cca033820300.png)

- Plain망과 Residual 망에 대해 각각의 layer에서 response에 대한 표준편차를 살펴보자.
  - 아래 그림은 표준편차를 크지별로 다시 sorting한 결과이다.
- BN, ReLU 및 addition(?)을 실행하기 전을 비교한 것이다.
- Residual 망의 response가 Plain 망에 비해 작았고, 표준편차가 작았다. 또한 response가 크게 흔들리지 않은 것을 확인할 수 있다.
  - 망이 깊어졌을 때, Plain 망에서 발생하는 문제가 Residual 망에서는 적게 발생하게 된다.

## 1000 layer 넘었을 때
![image](https://user-images.githubusercontent.com/69780812/138551528-eb726a94-a502-429c-9dbb-8bb0655b830b.png)

- n = 200으로 설정
- 110 layer 경우보다 결과가 약간 나쁘게 나온다.
- 하지만 1000-layer가 넘는 경우에도 최적화에 별 어려움이 없었음을 언급했다.
- 1000-layer 넘었을 때, 결과가 나쁘게 나온 이유 : 망의 깊이에 비해 학습 데이터 양 부족
  - Overfitting된 결과가 나왔을 것으로 추정한다고 한다.

![image](https://user-images.githubusercontent.com/69780812/138551569-7de4d5c5-300c-47dc-95b7-e068db459173.png)

- 2016년, Kaming He가 발표한 논문을 보면, CIFAR-100에서도 비슷한 실험을 했는데 결과는 위와 같다.
- 적당한 수준의 data augmentation만 적용했다고 한다.
- pre-activation을 통해 residual 망의 이슈가 개선되었다는 사실을 확인했다.

# Detection/Localization
![image](https://user-images.githubusercontent.com/69780812/138551630-052a7164-61c4-49d2-bf3d-f2ad60c385d0.png)

- ResNet은 Detection에서도 사용했을 때 효과가 좋았다. 상당한 격차를 보이며 우승했다.
- image detection/localization의 성능을 위해 Faster R-CNN 방법을 적용했다고 한다.
  - Faster RCNN은 real-time object detection을 목표로 하고 있다.

# Residual Network에 대한 고찰
- ResNet 팀은 2015년 대회를 마치고 자신들의 Architecture에 대해 심층적 고민 + 실험을 했다.
- 2016년, Identity Mappings in Deep Residual Networks 논문을 통해 **pre-activation**이라는 개념을 적용한 개선된 Residual Network 구조를 발표하게 된다.

![image](https://user-images.githubusercontent.com/69780812/138596222-ac966826-751a-4add-99ab-f5516a684e1d.png)

- 기존 Residual Network 기본 구조
- x(l+1) = identity 함수(h(x))와 residual 함수 (F(x))에 대해 activation(ReLU)를 한 결과다.
- h(x)를 identity mapping을 위한 short-cut connection이라고 불렀다.
  - 실제로는 identity 함수와 residual 함수의 합을 더한 후 ReLU를 수행하므로 진정한 의미에서의 identity라고 부르기 힘들다.
  - Residual Network 여러개를 연결시킨다고 생각해보면, addition 뒤에 오는 ReLU로 인해 **blocking**이 일어나기 때문에 x(l+1) = x(l) + F(x)의 합으로 표현되지 않고 위 그림의 식처럼 함수 처리 f를 해줘야한다. 합이 아닌 곱의 형태가 된다.

![image](https://user-images.githubusercontent.com/69780812/138596397-3979cf17-403b-4783-af84-4aa40035963b.png)

![image](https://user-images.githubusercontent.com/69780812/138596729-f1178d04-8836-4ca8-9e87-99be6c967e58.png)

- if) f is identity function ? 위 그림이 된다.
- x(l+1) = x(l) + F(x(l)).. 이 수식의 변화는 무엇을 의미하나 ?
  - forward나 backward path를 아주 간단하게 표현할 수 있게 된다.
  - 특정 위치의 출력은 특정 위치에서의 입력과 Residual 함수의 합으로 표현이 가능해진다는 것이다.
  - 전형적 CNN이 성격상 곱으로 forward, backward로 표현되는 것을 비교하면 엄청나게 단순해지는 효과를 얻는다.

![image](https://user-images.githubusercontent.com/69780812/138596768-6b0d5b8e-f0c4-4eda-bf0f-61c0a7d7e398.png)

- 이전과 같이 표시된 수식을 보면, Forward/Backward path 연산이 단순 덧셈의 형태로 간단하게 구현이 가능해 진다.

# Identity skip connection의 중요도
![image](https://user-images.githubusercontent.com/69780812/138596869-e656640a-c5f1-4ec3-b72e-4fb8a580336e.png)

- short-cut connection을 다양한 경우로 바꿔가며 실험을 수행했다.
- (a) : Original ResNet 에서 사용한 구조
- (b) ~ (f) : short-cut connection을 다양하게 변화 시켜가며 실험한 구조
- CIFAR-10 Dataset을 이용하여 실험한 결과 원래의 Identity shortcut의 결과가 가장 좋았다고 한다.
- (a)를 제외한 구조에서는 Multiplication에서 blocking된 형태이므로 앞서 확인한 것처럼 정보가 곧바로 전달되지 않고, 곱셈에 의해 변형되어 최적화에 문제가 생기는 것으로 추정된다고 한다.

# Activation 함수 위치에 따른 성능
![image](https://user-images.githubusercontent.com/69780812/138596984-30083a05-eb0c-482d-8c90-cae0c572a401.png)

- ResNet 팀은 Activation 함수의 위치가 어떤 영향을 주는지 확인을 하기 위해 실험을 수행했다.
- (a) : ResNet의 원래 버전
  - addition 뒤에 ReLU가 오는 형태 (after-add)
- (b) ~ (e) : 다양한 경우 실험한 버전들
- (b) : ResNet 구조에서 BN의 위치를 addition 뒤로 옮긴 경우 이다.

![image](https://user-images.githubusercontent.com/69780812/138597082-43248894-e16d-4179-b84c-ff1c73e56021.png)

- BN을 addition 뒤로 옮긴 결과 BN이 Propagation path를 방해할 수 있다는 것을 확인했다.
  - training, test에서 모두 original 보다 결과가 안좋았다.

- (c) : addtition 뒤쪽 ReLU를 residualNet 쪽으로 옮긴 경우
  - 원래 Residual 함수 범위가 (-inf, inf)인데, non-negative로 바뀌면서 이로 인해 망의 representation이 떨어지고 결과적으로 성능이 나빠지게 된다고 한다.
- (d) : (c)와 마찬가지로 pre-activation 성능을 확인하기 위한 구조다.
  - Pre-activation을 고려하는 이유는 기존의 Residual Net에 있는 ReLU가 진정한 identity mapping 개념을 방해하므로 이것을 움직일 수 있는지 확인하기 위함이다.

![image](https://user-images.githubusercontent.com/69780812/138597249-890fdb2c-3f88-4907-8748-479a39e0d09d.png)

- (a) : ResNet 구조
  - activation, layer의 위치를 조정하면 (b)와 같은 형태를 만들어 낼 수 있다.
- ResNet의 identity mapping 함수를 asymmetric하게 조정하면 actiavtion의 위치를 Residual 쪽으로 옮길 수 있다. 정리하면 (c) 구조
- Activation의 위치가 Weight layer 앞쪽 으로 오면서 pre-activation 구조가 된다.

![image](https://user-images.githubusercontent.com/69780812/138597477-72973d6c-eb7c-4511-b36b-80a442d6c7b7.png)

- Pre-Activation 구조에서 BN이 ReLU 앞으로 간 형태를 **Full Pre-Activation**이라 부른다.
  - 앞에오는 BN으로 Regularization 효과를 얻게 되어 결과가 좋아지게됐다고 한다.
  - 위 실험 결과를 확인하면 Test Error가 origin ResNet보다 좋아졌다는 것을 확인할 수 있다.
  - 하지만, Training에서 iteration의 횟수가 커지게 되면 약간 결과가 나쁘다. 이는 Regularization을 통해 일반화가 더 잘된 것으로 추정된다고 한다.
  - 즉, Original은 학습 데이터에 더 특화된 결과를 보이지만 Generalization 특성은 개선된 ResNet보다 좋지 못하여 Test 결과가 나빠졌다고 볼 수 있다.

# 성능 비교
![image](https://user-images.githubusercontent.com/69780812/138597602-6dfcace5-fc1b-44e4-9bb7-d2bdbb8af009.png)

- Pre-Activation이 Original 보다 성능이 더 개선이 되었다.
---
- short-cut connection의 구조는 간단할 수록 결과가 좋다.
- 수학적 조정을 통해 Activation의 위치를 Residual Net쪽 Weight-layer 앞에 위치시키면 수학적으로 완결된 형태가 된다. -> 성능 개선
- 개선된 구조에서는 Forward/Backward Propagation이 쉬워져 Deep-Network를 어렵지 않게 구현할 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138597862-01956018-6f47-4138-a986-3749ec3212ee.png)

- 훗날 발표되는 Inception-ResNet v2는 굉장히 성능이 좋아졌다.

![image](https://user-images.githubusercontent.com/69780812/138597886-de53c544-ac96-4e4c-82fb-997de4ece2eb.png)

- Inception과 ResNet의 결합은 Inception-v4 보다 빠르게 수렴하기도 했다.