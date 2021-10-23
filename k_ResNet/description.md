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
