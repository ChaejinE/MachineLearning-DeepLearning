# 1. LeNet
- 기존 Fully Connected Neural Network -> Topology 변화에 대응이 어렵다.
- local receptive field, shared weight, sub-sampling의 개념을 결합한 CNN 개념을 개발하게된다.

## LeNet5 구조
![image](https://user-images.githubusercontent.com/69780812/138416323-5f4e2695-ce42-4435-973f-2023d9afa64f.png)

- LeNet5는 MNIST의 28x28 test image를 32x32 영상 중앙에 위치시켜 처리했다고 한다.
- 이전보다 좀 더 큰 영상을 사용해서 영상의 작은 부분, detail에 대한 고려가 훨씬 많아져 더 우수한 성능을 얻을 수 있게됐다고한다.

## 결과
![image](https://user-images.githubusercontent.com/69780812/138417972-f88279a0-6a89-45c3-bf7c-970cf263f67f.png)

- C : Convolution
- S : Sub-Sampling
- F : Fully-Connected layer
- 알파벳 다음에 오는 숫자 : layer 번호
- 단계별 영상을 보면, LeNet-5의 각각의 layer에서 topology 변화나 잡음에 대한 내성이 상당히 강하다는 것을 알 수 있다.
- 그리 복잡하지 않은 망을 이용해서 놀랄만한 성광를 얻을 수 있음을 확인했다.

# 2. AlexNet
- CNN 구조 설계 시 GPU를 사용하는 것을 대세로 만들었다.

## AlexNet 구조
![image](https://user-images.githubusercontent.com/69780812/138418450-5ec873a1-7de3-4c95-bb99-9ac6366554f0.png)

- 총 5개의 Convolution layers, 3개의 fully-connected layers로 구성
- 1000 개 category 분류를 위해 softmax 함수 사용

## 블록도 이해
![image](https://user-images.githubusercontent.com/69780812/138418624-7a3ff776-59bc-413a-b12d-94bacb83cf98.png)

- feature map이 만들어 짐에 따라 중간 영상마다 depth가 달라진다.
- AlexNet은 입력 영상이 227x227x3이라서 첫 convolution에서 비교적 큰 receptive field를 사용한다.
- 또한, stride도 4로 크게했다고한다. (연산량을 줄이기 위함)

![image](https://user-images.githubusercontent.com/69780812/138418928-b2317933-80d0-464b-89a8-5bea51bb1d23.png)

- 위 아래가 각각 다른 GPU로 학습을 시킨 것이다.
- GPU-1에서는 주로 컬러와 상관 없는 정보를 추출하기 위한 kernel 학습
- GPU-2에서는 주로 컬러에 관련된 정보를 추출하기 위한 kernel 학습

![image](https://user-images.githubusercontent.com/69780812/138419047-7eb87536-1e8e-4eca-a6cc-33982820f4fd.png)

## 성능 개선을 위한 고려 (ReLU)
- AlexNet은 성능 향상을 위해 ReLU, overlapped pooling, response normalization, dropout, GPU 2개 병렬 연산을 사용했다.
- 그중 ReLU를 살펴본다.
- 기존 사용하는 sigmoid 함수는 tanh보다 학습 속도가 느리고, tanh 역시 학습 속도가 느린 문제가 있었다고 한다.
  - 망 크기가 엄청 커지는 경우 학습속도에 치명적 영향을 미치는 것이지 작을 때는 비슷하다.

![image](https://user-images.githubusercontent.com/69780812/138419596-605449e0-e856-4b53-b816-3525562eb9a2.png)

- 0에서 미분이 안되는 문제가 있지만 학습속도가 탁월하고 back-propagtion 결과도 매우 단순하다고 하여 요즘 Deep Neural Network 에서는 거의 ReLU를 선호한다.
- 논문에서 실험결과는 signmoid or tanh보다 학습 속도가 6배 정도 빨라진다고 한다.
  - 활성화 함수의 선택이 중요함을 알 수 있다.

## overlapped pooling
- 통상적으로 pooling 시 겹치는 부분이 없게하는 것이 대부분이라고한다.
- AlexNet은 3x3 windwo, Strie 2를 통해 overlapped pooling방식을 사용했다고하는데 이 것이 top-1, top-5 에러율을 각각 0.4%, 0.3% 줄일 수 있었다고 한다.
  - 덤으로 overfitting에 빠질 가능성도 더 줄여준다고 주장한다..(???)

## Local Response Normalization
- sigmoid or tanh 경우 saturation 되는 구간이 있어 Overfitting을 피하기 위해 정규화를 수행한다.
- feature map에서의 결과를 normalization 하면, lateral inhibition: 강한자극이 주변의 약한 자극이 전달되는 것을 막는 효과)를 얻을 수 있어
generalization 관점에서 훨씬 좋아진다고 한다.

![image](https://user-images.githubusercontent.com/69780812/138421072-3259539f-94e2-45cf-b37e-06b559a94eea.png)

- AlexNet에서는 첫번째, 두번째 Convolution을 거친 결과에 대해 ReLU를 수행하고, max pooling을 수행하기 앞서 reponse normalization을ㄹ 수행했다.
- 이릁 통해 top-1, top-5 에러율을 각각 1.4%, 1.2% 개선했다고 한다.

## AlexNet Overfitting 해결책
1. Data Augmentation
- GPU가 이전 이미지를 이용하여 학습하는 동안 CPU에서는 이미지를 늘려 디스크에 저장할 필요가 없도록했다.
- 256x256 크기 원영상에서 224x224 크기 영상을 무작위로 취했다.

![image](https://user-images.githubusercontent.com/69780812/138421855-a066bbad-c972-43fb-933f-1544c3b08e69.png)

- 각 학습 영상으로 부터 RGB 채널의 값을 변화시켰다.
  - 각 학습 이미지의 RGB 픽셀 값에 대한 PCA수행 후 원래 픽셀 값에 더해주는 방식으로 컬러 채널의 값을 바꿔 다양한 영상을 얻음
- 위 방법을 통해 top-1 error에 대해 1% 이상 에러율을 줄일 수 있었다고 한다.

2. Dropout

- Dropout은 성격상 Fully-Connected layer에 대해 행하기 때문에 AlexNet에서는 Fully connected layer의 처음 2개 layer에 대해서만 적용했다.
- dropout rate : 50%

## Why GPU
- 연산량 때문에 GPU 가 필요하다.
  - 수천만장의 영상
- 요즘 Deep CNN은 크게 2개로 구성된다고 한다.
1. 전체 연상량의 90~95%를 차지하지만, freeparameter 의 개수는 5% 정도인 convolution layer
2. 전체 연산량의 5~10%를 차지하지만, free parameter의 개수는 95% 정도인 fully connected layer
- 여기서 convolution layer는 입력 영상의 픽셀의 위치를 옮겨가면서 반복적으로 matrix multiplication이 필요하며, 여러 개의
입력 feature map으로 부터 동일한 paramter를 갖는 Filter 연산을 해야하므로 구조적으로 아주 좋은 병렬적 특징을 갖는다.

# 3. ZFNet
- CNN을 보다 잘 이해하여 최적의 구조인지, 좋은 결과에 대한 근거를 찾을 수 있도록 **Visualizing 기법**을 사용하여 해결하는 시도를 했다.
- ZFNet은 특정 구조를 가리키는 개념이 아니라 CNN을 보다 잘 이해할 수 있는 기법을 가리키는 개념으로 이해하면된다.
- Visualizing 기법은 중간 layer의 activity를 다시 입력 이미지 공간에 mapping 시키는 기법이다.

![image](https://user-images.githubusercontent.com/69780812/138423610-8adb7718-249e-4595-a84e-d19face65640.png)

- 중간 단계 layer는 이전 feature-map으로부터 데이터를 받고 filtering(convolution) 수행 후 현 단계의 feature-map을 만들어 ReLU
를 통과시킨 후 Pooling을 수행한다.
- 이 과정을 역으로 수행하는 것이 visualizing 이다.

![image](https://user-images.githubusercontent.com/69780812/138424053-3257caeb-35aa-49f0-9010-16162fee841b.png)

- MaxPooling을 역으로 돌릴 때, **switch**라고하는 강한 자극의 위치 정보 flag를 통해 강한 자극을 역으로 복원할 수 있다.
- 하지만, 나머지 자극에 대해서는 모르므로 빨간 화살표에서와 같이 큰 자극값만 파악이 가능하다.
- ReLU의 경우 음의 값을 갖는 것들은 이미 정류되었기 때문에 0 이상의 양의 값들만 복웒라 수 있다.
- 위의 문제도 있지만, 연구 결과에 따르면 영향이 미미하다고 한다.
- 이 같은 역과정을 거친것을 **Deconvolution**이라고한다. 정확하게 입력과 같은 상태로 mapping 되지는 못하나 강한 자극이 입력 영상에 어떻게
mapping 되는지를 확인할 수 있다.
  - 덕분에 CNN에 대해 더 잘 이해할 수 있고 어떤 구조가 최적의 구조인지를 좀 더 잘 결정할 수 있다.

## Feature Visualization
![image](https://user-images.githubusercontent.com/69780812/138424764-02cb8faa-b852-4473-937b-f529311b6c80.png)

- Layer1, Layer2를 보여주는 사진이고, 주로 영상의 코너나 edge 혹은 컬러와 같은 **low level feature**를 시각화 한다.

![image](https://user-images.githubusercontent.com/69780812/138425044-b0c2bdb0-c9bc-4355-800a-8b154264e64d.png)

- Layer1/2에 비해 **상위 수준의 향상성(invariance)**를 얻을 수 있거나 **비슷한 외양(texture)**를 갖고 있는 특징을 추출한다.

![image](https://user-images.githubusercontent.com/69780812/138425194-da6b5715-27c9-40d1-830a-c1bc98d57b44.png)

- Layer4에서는 사물이나 개체의 **일부분**을 볼 수 있었고, Layer는 위치나 자세 변화 등 까지 포함한 사물이나 개체의 **전부**를 보여준다.
---
- 이렇게 Visualizing을 수행했을 때, 특정 단계에서 얻어진 feature-map들이 고르게 확보되어 있는지, 특정 feature 쪽으로 쏠려있는지, 
개체의 Pose, Invariance 등을 잘 반영하는지 등등을 파악할 수 있다.
- 결과적으로 학습결과에 대한 양호도, CNN의 구조가 적절했는지 등을 판단하기에 좋다.
- ZFNet은 CNN layer의 시각화를 위해 Deconvolution을 사용했고, max-pooling에 대한 un-pooling 문제 해결을 위해 switch 라는 개념을 적용했다. 
이것이 만능은 아니지만, CNN 구조의 적합성 여부를 판단하는 데 좋은 툴이 될 수 있다는 것은 분명하다.
