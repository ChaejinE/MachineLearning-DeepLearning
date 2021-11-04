# CNN (Convolutional Neural Network)
- 영상에 기반한 인식 알고리즘에서 좋은 결과를 얻으려면, 사전에 많은 처리 과정을 필요로 하기 때문에 기존 Multi-layered neural network을 바로 적용하는 것은 어려움이 있다.
# 기존 Multi-layered Nueral Network의 문제점
![image](https://user-images.githubusercontent.com/69780812/138057370-c27b455d-76d6-4ad7-a2ca-1f75be78623b.png)

- 필기체 인식을 위해 위 처럼 256개의 입력단과 100개의 hidden layer 및 26개의 출력단으로 구성된다면 이 망에 필요한 가중치와 바이어스는 28,326개가 필요하게 된다.
- 폰트 Size가 커지거나 Hidden layer가 2단 이상이거나 대소문자 구별이나 숫자까지 구별해야한다면 파라미터의 개수가 엄청나게 많아진다.

![image](https://user-images.githubusercontent.com/69780812/138041150-51ec6406-e8d0-46f9-9b38-b871d019ad41.png)

- 기존 신경망은 위 처럼 전체 글자에서 단지 2픽셀값만 달라지거나 2픽셀씩 이동만 하더라도 새로운 학습데이터로 처리를 해줘야하는 문제점이 있다.
- 또한 글자의 크기가 달라지거나 회전되거나 변형(Distortion)이 조금만 생기더라도 새로운 학습데이터를 넣어주지 않으면 좋은 결과를 기대하기 어렵다.

![image](https://user-images.githubusercontent.com/69780812/138041590-d7f479d4-3a65-4ea9-bf8f-306ba79b9dab.png)

- 결과적으로 기존 Multi-layered neural netwokr는 글자의 topology는 고려하지 않고 말 그대로 raw data에 대해 직접적으로 처리하기 때문에 **엄청나게 많은 학습 데이터**를 필요로 하고, 거기에 따른 학습 시간을 대가로 지불해야하는 문제점이 있다.
- 만약 32x32 폰트 크기에 대해 Black/White 패턴을 처리한다해도 2^(32\*32) = 2^1024개의 패턴이 나오고, Gray Sacle에 적용하면,
  256^(32\*32)=256^1024개의 어마어마한 패턴이 나와서 전처리 과정 없이는 불가능하다..
---
- 결과적으로 기존 Fully connected Multi-layered neural network 사용시 3가지 측면에 문제가 발생함을 알 수 있다.
1. 학습 시간(Training time)
2. 망의 크기 (Network size)
3. 변수의 개수 (Number of free parameters)
- MLP는 모든 입력이 위치와 상관없이 동일한 수준의 중요도를 갖는다고 본다.
  - Fully-connected neural network을 구성하게 되면 free parameter의 크기가 엄청나게 커지게 되는 문제가 생기는 것이다.
  - 이것에 대한 해결책으로 신경망 연구자들은 visual cortex와 비슷한 신경망을 만들고 싶어했고, 그 결과가 CNN이다.
---

# Receptive Field
- 수용영역이란 외부 자극이 전체에 영향을 끼치는 것이 아니라 특정 영역에만 영향을 준다는 뜻이다.
- 마찬가지로 영상에서 특정 위치에 있는 픽셀들은 그 주변에 있는 일부 픽셀들과만 Correlation이 높을 뿐이다. 거리가 멀어 지면 멀어질수록 그 영향을 감소하게 된다.
  - 이는 손가락으로 몸의 여러 부분을 찔러봤을 때 느낄 수 있는 범위가 제한적이라는 것을 떠올려보면 같다.
- 위와 유사하게 영상을 해석하여 알고리즘을 수행하고자할 경우 영상 전체 영역에 대해 서로 동일한 연관성으로 처리하기 보다는 특정 범위에 한정에 처리를 한다면 훨씬 효과적이다는 것을 알 수 있다.
- 이는 영상에만 한정할 것이 아니라 Locality를 갖는 모든 신호들에 유사하게 적용할 수 있다고 추론할 수 있다.
- 영상처리 분야에서 Convolution은 주로 Filter 연산에 사용되며 특정 Feature를 추출하기 위한 필터를 구현할 때 Convolution을 사용한다.

# Convolution 연산
![image](https://user-images.githubusercontent.com/69780812/138044779-b10ba872-f1b9-4ea4-973d-45b24ed44bbc.png)

- 노란색 부분 : Convolution이 일어나고 있는 영역
- 빨간색 부분 : Convolution의 kernel
- 노란색 영역의 Mask에 대해 연산을 수행하고, 오른쪽으로 이동시키며 다시 결과를 구한다. 계속 이동 시키면서 연산을 하면 최종적인 결과를 얻을 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138045084-ccb2ab77-5e5d-4de4-8feb-942b146fe699.png)

- 필터의 종류에 따라 각기 다른 특징을 꺼낼 수 있다.

# CNN의 특징
- CNN에 Convolutional이 붙은 이유는 Convolution의 특성을 살린 신경망 연산을 한다는 뜻이다.
## 1. Locality(Local Connectivity)
- Receptive Field와 유사하게 Local 정보를 활용한다.
- 공간적으로 인접한 신호들에 대한 Correlation 관계를 비선형 필터를 적용하여 추출한다.
- 이런 필터를 여러개 적용 -> 다양한 Local Feature를 추출해 낼 수 있게된다.
- Subsampling 과정을 거쳐 영상 크기를 줄이고, Local feature들에 대한 filter 연산을 반복적으로 한다면 점차 global feature를 얻을 수 있게 된다.
  - SubSampling은 작은 Distortion이나 움직임을 제거하는 효과를 얻을 수 있다.
## 2. Shared Weights
- 동일한 계수를 갖는 Filter를 전체 영상에 반복적으로 적용함으로 변수의 수를 획기적으로 줄일 수 있다.
- Topology 변화에 무관한 항상성(Invariance)를 얻을 수 있게 된다.
  - Subsampling 때문이다.

# CNN의 구조
- CNN의 과정은 크게 보면 3단계 과정으로 이루어진다.
1. 특징을 추출하기 위한 단계
2. Topology 변화에 영향을 받지 않도록 해주는 단계
3. 분류기 단계
- CNN 처리 과정은 특징을 추출하기 단계가 내부에 포함되어 있기 때문에 Raw image에 대해 직접 Operation이 가능하고, 기존 알고리즘과 달리 별도의 전처리(Pre-processing) 단계를 필요로 하지 않는다.
- 특징 추출과 Topology Invaraiance를 얻기위해 Filter와 Sub-Sampling을 거친다.
- 이 과정을 여러 번 반복적으로 수행해서 Local Feature로 부터  Global Feature를 얻는다.
- CNN에서 사용하는 Filter 혹은 Convolutional layer는 학습을 통해 최적의 계수를 결정할 수 있게하는 점이 기존의 필터와 다르다.

![image](https://user-images.githubusercontent.com/69780812/138051907-54141f74-de89-4489-b10f-d4846c19b7cf.png)

- CNN에서 Subsampling은 신경 세포와 유사한 방식의 Sub-Sampling 방식을 취한다.
- 신경세포학적으로 보면, 통상적으로 강한 신호만 전달하고 나머지는 무시한다. 이와 비슷하게 CNN에서는 max-pooling 방식의 Sub-Sampling 과정을 거친다.

![image](https://user-images.githubusercontent.com/69780812/138052207-a833965d-b7df-46e7-8fed-2b85b5535594.png)

- max_pooling은 가장 큰 값만 선택
- average_pooling은 윈도우의 평균을 취한다.
- 이동이나 변형 등에 무관한 학습 결과를 보이려면, 좀 더 강하고 Global한 특징을 추출해야하는데, 통상적으로 Convolution + Sub-sampling 과정을 여러번 거치게 되면 좀 더 전체 이미지를 대표할 수 있는 global한 특징을 얻을 수 있게 된다.
- 이렇게 얻어진 특징을 Fully-connected network를 통해 학습 시키면 2차원 영상 정보로 부터 Receptive Field와 강한 신호 선택의 특성을 살려서 Topology 변화에 강인한 인식 능력을 갖게 된다.

![image](https://user-images.githubusercontent.com/69780812/138052672-5dc75b1e-9b1b-41b1-8a6d-11f0462f6573.png)

- feature 추출 -> sub-sampling을 통해 feature map 크기를 줄여주면서 이를 통해 topology invariance도 얻을 수가 있게 된다.
- local feature에 대해 다시 convolution과 sub-sampling을 수행하면서 이 과정을 통해 좀 더 global feature를 얻을 수 있게 된다.
- 여러 단의 convolution + sub-sampling 과정을 거치면, Feature map 크기가 작아지면서 전체를 대표할 수 있는 강인한 특징들만 남게 된다.
- Fully-connected network의 입력으로 연결되면서 기존의 신경망들의 특징처럼 학습을 통해 최적의 인식 결과를 낼 수 있게 된다.

# Convolution Layer
- CNN 알고리즘을 통해 처리하고자 하는 과제에 따라 최종 Convolution kernel의 계수가 달라질 수 있다.
- 동일 과제일지라도 학습에 사용하는 학습 데이터에 따라서 달라질 수 있고, 설정한 hyper-parameter의 값에 따라서도 달라질 수 있다.
- 필터 계수의 값은 기존 신경망과 마찬가지로 Gradient에 기반한 back-propagation에 의해 결정된다.
## 1. Filter 개수
- Feature map의 크기: Convolutional layer의 출력 영상 크기)는 Layer의 depth가 커질수록 작아지므로 일반적으로 영상의 크기가 큰 입력단 근처에 있는 layer는 Filter 개수가 적고, 입력단에서 멀어질수록 Filter 개수는 증가하는 경향이 있다.
- 어떤 원칙으로 Filter 개수를 정하는가 ?
  - convolution layer의 연산 시간을 본다.

![image](https://user-images.githubusercontent.com/69780812/138055152-931be46d-e52d-4f6c-9873-eb99388d3b3d.png)


- 필터의 개수를 정할때 흔히 사용하는 방법은 **각 단에서의 연산 시간/량을 비교적 일정하게 유지하여 시스템의 균형**을 맞춘다.
- 이를 위해 일반적으로 각 layer에서 feature map의 개수와 pixel 수의 곱을 대략적으로 일정하게 유지시킨다.
  - convolution layer에서의 연산 시간이 픽셀 수와 feature map의 개수에 따라 결정되기 떄문이다.
- 2x2 sub-sampling시 convoltion layer를 지나면 pixel 수가 1/4로 줄어들기 때문에 feature map의 개수는 대략 4배 정도로 증가시킨다.
- Feature map의 개수는 가중치와 바이어스 같은 free parameter의 개수를 결정하고, 학습 샘플의 개수 및 수행하고자 하는 과제의 복잡도에따라 결정한다.

## 2. Filter의 형태
- 작은 크기의 입력영상 (32x32, 28x28)에서는 5x5 필터를 주로 사용하고, 큰 크기의 자연 영상을 처리할 때나 혹은 1단계 필터에서는 11x11 or 15x15 같은 큰 크기의 kernel을 갖는 필터를 사용하기도 한다.
- **여러 개의 작은 크기의 필터를 중첩해서 사용하는 것이 좋다.**
  - 작은 필터를 여러개 중첩하면 중간 단계에 있는 **non-linearity를 활용**하여 원하는 특징을 좀 더 돋보이도록 할 수 있다.
  - 연산량도 더 적게 만든다.

## 3. Stride 값
![image](https://user-images.githubusercontent.com/69780812/138056117-a529e695-190e-40c7-b62c-5b04c00d7cfa.png)

- 건널 뛸 픽셀의 개수를 결정한다.
- 2차원 영상 데이터에 대해서는 가로 및 세로 방향으로 Stride에서 정한 수만큼 씩 건너 뛰면서 Convolution 연산을 진행한다.
- **Stride는 입력 영상의 크기가 큰 경우, 연산량을 줄이기 위한 목적으로 입력단과 가까운 쪽에만 적용한다.**
- Stride 1로하고, Pooling 적용하는 방식과 Stride를 크게하는 방식의 차이는 ?
  - Stride를 크게하는 것은 Convolution 연산 수행 후 값을 선택적으로 고르는 기회가 사라지게 된다.
  - **그래서 통상적으로 봤을 때, Stride 1, Pooling을 통해 적절한 Sub-sampling 과정을 거치는 것이 결과가 좋다.**
  - 하지만 큰 영상에 대해 CNN을 적용할 때 연산량을 줄이기 위해 입력 영상 처리 1단계에서 Stride 값을 사용하기도 한다.

## 4. Zero-Padding
- 보통 Convolution 연산을 하게 되면, 경계 처리문제로 인해 출력 영상인 Feature map의 크기가 입력 영상보다 작아지게 된다.
- Zero padding은 작아지는 것을 피하기 위해 입력의 경계면에 0을 추가하는 것을 말한다.
  - 영상 크기를 동일하게 유지할 수 있다.
- 단순하게 영상의 크기를 동일하게 유지하는 장점 이외에도 **경계면의 정보까지 살릴 수가 있어 Zero padding을 지원하지 않는 경우에 비해 Zero-Padding을 지원하는 것이 좀 더 결과가 좋다.**
---
- Feature-map 개수, 필터의 크기, Stride 및 Zero padding은 CNN의 구조를 결정하는 중요한 기본 hyperparameter다.
