# Deeper and deeper
- CNN의 성능 향상의 가장 직접적인 방식 : Network 크기를 늘리는 것
- Network 크기를 늘린다는 것 : layer(depth)를 늘리는 것 뿐만 아니라 각 layer에 있는 unit의 수(width)도 늘리는 것을 의미

![image](https://user-images.githubusercontent.com/69780812/138427624-abecb834-1ac9-4930-b1ee-533995692dc2.png)

- 14년의 대표주자 GoogleNet, VGGnet은 22layer, 19layer로 깊어지게 된다.
- top-5 에러율도 많이 낮아지게 된다.

# Deeper Network.. side effect(부작용)
1. 망이 커지면 커질 수록 free parameter의 수가 증가하며 망이 Overfitting에 빠질 가능성이 높아진다.
   - 즉, 학습 데이터에 특화된 결과로 Test set에 적용하면 만족할만한 결과가 나오지 못할 수 있다.
   - 대량의 데이터에 사람이 일일이 label 달아주는 것도 쉬운일이 아니다.
2. 망이 커지면 연산량이 늘어나게된다.
   - 필터 개수 증가 -> 연산량은 제곱으로 늘어나게 된다.
   - 학습이 잘 못되어 filter의 kernel이 특정한 무리로 쏠리게 되면 기껏 망을 늘렸어도 최적의 결과를 얻지 못할 수 있다.

# Inception 모듈
![image](https://user-images.githubusercontent.com/69780812/138428687-d19d6d9e-cef2-4360-a8d0-9843e7f4f098.png)

- 같은 layer에 서로 다른 크기를 갖는 convolution filter를 적용하여 다른 scale의 feature를 얻을 수 있도록 했다.
- 1x1 convolution으로 적절하게 차원을 줄여 망이 깊어졌을 때, 연산량이 늘어나는 문제를 해결했다.
- 망의 깊이는 훨씬 깊은데 AlexNet보다 free parameter 수는 1/12 수준이고 전체 연산량 숫자도 적다.
- GoogleNet에는 총 9개의 Inceptio 모듈이 적용되어 있다.
- 구글 연구팀은 망을 더 깊게하여 성능 향상을 꾀하면서도 연산량을 증가시키지 않는 CNN 구조를 개발하기 위해 많은 연구를 한 것으로 보인다.

# Network In Network (NIN)
- Inception 모듈 및 전체 구조는 NIN 구조를 더욱 발전 시킨 것이라고 한다.
- NIN은 말그대로 네트워크 속 네크워크를 뜻한다.

![image](https://user-images.githubusercontent.com/69780812/138429687-85f21dc2-c92a-4e56-a60e-5e36766494ef.png)

- NIN 설계자는 Convolution layer가 local receptive field에서 feature를 추출하는 것은 우수하나 filter의 특징이
linear 하여 non-linear한 성질을 갖는 feature를 추출하기에 어려우므로 이 부분을 극복하기 위해 feature map의 개수를 늘려야
한다는 문제에 주목했다.
  - 필터 개수를 늘리면 연산량이 늘어나므로 local receptive field 안에서 좀더 feature를 잘 추출해낼 수 있는 방법을 연구했다.
  - micro neural network를 설계하게 된다.
  - convolution을 수행하기 위한 filter 대신 MLP를 사용하여 feature를 추출하도록 했다고 한다.
- NIN에서는 convolution kernel 대신 MLP를 사용하며 sweeping 한다. sweeping 방식은 CNN과 유사하다.
- MLP 사용시 장점은 convolution kernel보다 non-linear한 성질을 잘 활용할 수 있어 feature 추출 능력이 우수하다는 것이다.
- 망을 깊게 만들기 위해 MLPconv layer를 여러개 쌓아 사용하기 때문에 NIN라는 이름이 붙은 개념이 만들어 졌다.
- GoogLeNet에서도 Inception 모듈을 총 9개 사용하여 개념적으로 NIN과 맥이 닿아 있다고 볼 수 있다고한다. (?)

![image](https://user-images.githubusercontent.com/69780812/138430211-8b6b4c40-ba53-4615-8b03-e0e838f1b508.png)

- 위 그림은 MLPconv layer를 3개 사용한 것이다.
- 또한, 기존 CNN과 다른 점은 최종단의 fully-connected neural network이 없다.
- Fully-connected NN 대신 **Global average Pooling**을 사용했다.
  - 이는 앞에서 효과적으로 feature-vector를 추출했기 때문에 이 벡터들에 대한 Pooling 만으로 충분하다고 주장한다.
  - Average Pooling 만으로 Classifier 역할을 할 수 있고, Overfitting 문제를 회피하며 연산량이 대폭 줄어드는 이점을 취할 수 있다.
  - CNN 최종단의 Fully-connected NN는 free parameter중 90% 수준이므로 많은 파라미터로 overfitting에 빠질 가능성이 높다.

#  1x1 Convolution
- 1x1 Convolution을 하는 결정적 이유 : 차원을 줄인다. -> 연산량을 줄인다. -> 망이 더 깊어질 여지가 생긴다.
- 1x1 convolution을 1-layer fully connected neural network이라하는 이유
  - 1x1 conv가 fully-connected와 동일한 방식이기 때문이다.

![image](https://user-images.githubusercontent.com/69780812/138431067-d637e75e-c2f0-4dd7-b054-63c8d842823f.png)

- 만약 input feature map 개수가 4이고, output feature map 2인 경우를 가정한다.
- 1x1 convolution은 위 그림과 같이 표현할 수 있다.
- 결과적으로 4개의 feature map으로부터 입력을 받아 learned param을 통해 feature map이 2개로 결정된다.
- 즉, 차원이 줄어들게 되며 연산량을 절감하게 된다.
- 또한, nueron에 활성화 함수로 ReLU를 사용하면, 추가로 non-linearity를 얻을 수 있는 이점도 있다.

# Google's Inception
- NIN 구조를 많이 참고했다고 한다.
- Local receptive field에서 더 다양한 feature 추출을 위해 Convolution을 여러개로 병렬적으로 활용하려했다.

![image](https://user-images.githubusercontent.com/69780812/138431627-5ae575e1-ac80-4532-b133-f2ed5525faf1.png)

- 3x3, 5x5는 상대적으로 비싼 연산이 될 수 있으므로 1x1 conv로 차원을 줄여 연산량을 감소시켜 깊은 수준까지 망을 구성할 수 있게 된다.
- NIN 에서는 MLP를 통해 non-linear feature를 얻어내는 데 주목했다.
- MLP는 결국 fully-connected NN 구조이고, 구조적 관점에서도 익숙하지 않다.
- 구글을 기존 CNN 구조에서 크게 벗어나지 않으면서도 효과적으로 Feature를 추출하고자했다.

# GoogLeNet 핵심 철학 및 구조
- 핵심 설계 철학 : 학습능력은 극대화, 넓은 망을 갖는 구조를 설계
- 다양한 Convolution kernel Inception 모듈로 다양한 Scale의 feature를 효과적으로 추출
- 1x1 conv로 연산량 크게 경감
- Inception 모듈로 NIN 구조를 갖는 Deep CNN 구현

![image](https://user-images.githubusercontent.com/69780812/138432313-287ef498-7b47-40bc-b9c5-aca8c9ddf07c.png)

- 빨간색 숫자는 Inception 모듈을 거치면서 만들어지는 feature-map의 숫자이다.

# Auxiliary cliassifier
![image](https://user-images.githubusercontent.com/69780812/138432714-d8653dac-486c-45eb-9855-248faade4c5f.png)

- 이전에 CNN 구조에서 볼 수 없던 독특한 구조다.
- 망이 깊어지면서 생기는 큰 문제중 하나는 **vanishing gradient** 문제다.
  - 학습이 아주 느려지거나 overfitting 된다.
  - Cross Entropy ? : vanishing gradient가 개선은 되지만 본질적 해결이 아니다.
  - ReLU ? : 여러 layer를 거치면서 작은 값이 계속 곱해지면 0 근처로 수렴되면서 망이 깊어질 수록 이 가능성은 커진다.
- GoogleNet은 이 문제를 극복하기 위해 중간 2곳에 Auxiliary classifier를 두었다.
- vanishing gradient 문제를 피하고 수렴을 더 좋게 해주면서 학습결과가 좋아졌다고 한다.
- Deeply supervised nets, Training Deeper Convolutional Networks with Deep SuperVision이 대표적 논문이다.

![image](https://user-images.githubusercontent.com/69780812/138433492-a3557142-ad09-480f-af24-b6666dd98c24.png)

- SuperVision 이라고 부르는 Auxiliary clasifier 를 배치하고, back propagation 시 X4 위치에서는 Auxiliary Classifier와
최종 출력으로 부터 정상적인 back-propagation 결과를 결합한다.
- 이렇게 되면 Auxiliary classifier의 backpropagation 결과가 더해지므로 X4 위치에서 gradient가 작아지는 문제를 피할 수 있다고 한다.
- 몇번의 iteration을 통해 gradient가 어떻게 움직이는지 호가인하고 붙이는 것이 좋다고 한다.

![image](https://user-images.githubusercontent.com/69780812/138433907-5fee6693-6c24-40d2-ad8d-a7452da57bba.png)

- 왼쪽은 실험용 DNN 에서 X4 뒤의 gradient 들이 현저하게 작아지고 있는 것을 보여준다.
- 오른쪽은 Auxiliary classifier 가 없는 경우 0에 근접(파란색)한 반면 빨간색 선은 다시 증가하게 되는 것을 보여준다.
  - 결과적으로 더 안정적인 학습 결과를 얻을 수 있다.
- Auxiliary Cliassifier가 Regularizer와 같은 역할을 한다는 논문도 있다.
  - Rethinking the Inception Architecture for Computer Vision
- 학습이 끝나고 Inference 시에는 Auxiliary Classifier는 제거한다. 학습을 주기위한 도우미 역할일 뿐..

# Factorizing Convolution
- 큰 Filter size 갖는 convolution kernel 인수분해 -> 작은 커널 여러개로 구성된 deep network
  - parameter 수 감소, 망은 깊어질 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138434649-67edf369-c797-496a-bdd0-fd145d5843ab.png)

- 5x5 conv -> 2 layer의 3x3 conv로 구현
- 5x5 conv는 더 넓은 영역에 걸쳐있는 특징을 1번에 추출할 수 있으나 25/9 = 2.78배 비싼 unit이다.

![image](https://user-images.githubusercontent.com/69780812/138434892-31d941a5-4af9-4c7b-b784-086b4acbc143.png)

- 5x5 -> (9 + 9)로 free paramter수를 28% 만큼 절감이 가능하다.
- 7x7 -> 3 layer 3x3 conv -> (9+9+9) => 45% 절감;;

![image](https://user-images.githubusercontent.com/69780812/138435280-66ea2dbe-f81c-445c-a756-c821dc34c85d.png)

- symmetry를 유지하지 않고 row or column 방향으로 인수분해하는 것도 가능하다.
- 3x3 -> 1x3 , 3x1 conv로 인수분해 가능한 것을 보여주는 사진이다.
- (3+3) = 6 => 9(3x3)과 비교하면 33% 절감이 된다.
- n x n => 1xn, nx1으로 분해가 가능하며 n이 클수록 파라미터 절감 효과가 커진다.

![image](https://user-images.githubusercontent.com/69780812/138435506-77026964-b37d-40b5-b4b6-45336ff8a000.png)

- 큰 filter 크기를 갖는 conv kernel을 인수분해로 여러 단의 작은 크기를 갖는 conv로 대체하면 free-param이 줄어들어 연산량의 절감을 가져올 수 있다는 것을
살펴보았다.

# 효과적으로 grid size (해상도) 줄이는 방법
- 대표적인 방식으로는 convolution 수행 시 stride를 1 이상의 값으로 설정하거나 Pooling을 사용하는 것이다.

![image](https://user-images.githubusercontent.com/69780812/138436754-8e4d579e-78a3-4a0e-b1a0-ed3e33934f8d.png)

- 35x35x320 input으로 17x17x640 output을 얻기 위해 어느 쪽이 효과적으로 grid 크기를 줄이는 방식일까?
- 왼쪽 : Pooling으로 size를 절반줄이고 Inception으로 output을 얻는다.
  - 연산량 관점에서만 보면 효율적인 것 같다.
  - 하지만 Pooling 단게에서 feature map에 숨어있는 정보(representatinal concept)가 사라질 가능성이 있어 효율성 관점에서 최적이라고 보기는 어렵다.
- 오른쪽 : Inception 적용 후 Pooling 으로 size를 줄인다.
  - 연산량 관점에서 4배나 더 많은 셈이다.
  - 하지만 숨은 특징을 더 잘 찾아낼 가능성은 높아진다.

![image](https://user-images.githubusercontent.com/69780812/138437111-6c6f0246-1f54-4c98-a385-0c958515cd7a.png)

- Szegedy, Rethinking the inception architecture for computer vision)에서 제시한 구조이다.
- 왼쪽은 Inception module과 비슷하다.
  - Pooling 및 conv layer를 나란히 두고 최종 단에 Stride 2를 적용하여 local feature를 추출하면서 크기가 줄어준며
  - pooling layer를 통해서도 크기를 줄이고 그 결과를 결합한다.
- 오른쪽은 좀더 단순한 방법이다.
  - stride 2를 갖는 Conv를 통해 320개의 featrue-map을 추출하고, pooling layer를 통해 다시 320개를 추출한다.
  - 효율성과 연산량의 절감을 동시에 달성할 수 있게 되었다.

# Inception v2
- Convolution kernel에 대한 인수분해를 통해 망은 더 깊어지게되고 효과적으로 연산량을 더 절감할 수 있게 되었다.

![image](https://user-images.githubusercontent.com/69780812/138438107-7c68d79b-0208-428e-ad86-8d93b6f6f4c8.png)

- 2014년 googleNet의 7x7은 3x3으로 대체되면서 더욱 망이 깊어지게된다.

![image](https://user-images.githubusercontent.com/69780812/138438486-fc87dd70-7986-46ff-867b-9ab2e446409f.png)

- Batch Normalized auxiliary classifer를 적용하면 성능이 더 좋아진다는 것을 확인할 수 있다.
---
- 단순하게 Polling layer을 적용하는 것보다 Conv layer와 같이 나란히 적용하는 것이 효과적이라는 것을 파악했다.
- 또한, conv kernel에 대한 인수분해 방식 적용과 앞 뒤 일부 구조, feature-map 개수 조정으로 성능
