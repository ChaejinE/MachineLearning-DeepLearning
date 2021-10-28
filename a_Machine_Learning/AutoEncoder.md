# AutoEncoder
![image](https://user-images.githubusercontent.com/69780812/139225118-33688dbd-322f-4b37-b11b-cd6bb7800cae.png)

- MLP와 비슷한 것 같지만 개념이 완전히 다르다.
- AE는 입력과 출력의 차원이 같다.
- 학습 목표 : 출력을 가능한 한 입력이 근사를 시키는 것이다.
  - 출력을 입력에 근사 시키는게 쓸데 없는 짓 같으나 Constraints가 있어 의미가 있는 것으로 바뀌게 된다.
- Hidden layer의 뉴런 개수가 입력보다 적기 때문에 Input-layer에서 Hidden layer로 넘어가는 과정은 일종의 압축, Encoding이라고하는 것이 필요하다.
  - Hidden layer -> Output layer : Decoding
- Decoding 시, 차원을 줄이는 과정은 실제로 입력에서 의미있는 특징을 뽑아내는 과정이라고 볼 수 있다.
  - AE 학습에서는 매우 중요한 과정
  - 차원을 줄이기 떄문에 주요 응용 분야가 dimensionality reduction이다.
- AE가 MLP와 다른 점은 Unsupervised Learning 이라는 점이다.
- 출력단은 자율학습에서 출력을 입력에 근사시키이 위한 용도였으므로 학습 도우미 역할이 끝나면 의미있는 Hidden layer만 남기고 출력단은 버린다.

# Dimentionality Reduction
- if) Hidden layer neuron > Input layer neuron
  - identity 함수 구현이 매우 쉬워진다. 각각의 파라미터만 연결하고 나머지를 0으로 설정하면 된다.
  - 결과적으로 어떤 중요한 정보도 추출할 수 없다는 것이다.
- 유용한 Feature를 얻어 내려면
  - 뉴런들이 Identity 함수로 연결되는 것을 막는 제한 조건이 필요(입력보다는 낮은 차원으로 연결하는 것이 거의 필수적이라고 보면된다.)
- AE는 입력보다 작은 차원을 갖는 Hidden-layer로 숨어있는 변수들을 발굴할 수 있게 해준다. (**latent variable)**
  - PCA라는 기법도 있지만 선형적인 한계가 있다.
  - AE는 뉴런이 갖고있는 Non-linearity 성질 및 constraints로 인해 훨씬 뛰어난 차원 축소 능력을 갖고 있다.

# Stacked AutoEncoder
![image](https://user-images.githubusercontent.com/69780812/139226918-bc50f9a4-566a-476b-b2b6-f0a06c5cfbbf.png)

- 많은 hidden layer를 쌓는 구조를 살펴본다.
  - 다양한 특징을 끌어낸다.
  - 이를 Stacked AE라고 부른다.

![image](https://user-images.githubusercontent.com/69780812/139227371-37fb32a4-de70-46c2-b363-3149d2eb15ba.png)

- 끊어서 생각하면 결과적으로 기본 AE를 여러개 쌓아놓은 것과 같은 형태다.
- 차원을 계속 줄여나가는 구조가 된다.
- Bottlneck hddien layer에서 가장 압춥된 Feature가 얻어지게 된다.
- 각각의 Hidden layer에서 얻은 Feature를 더 Compact하게 표현한다고 볼 수 있다.
  - 이를 위해 차원을 줄여가며 피라미드 형태를 띄게 된다.

# Greedy Layer-Wise Training
- DNN학습의 문제는 Vanishing/Exploding gradient 문제와 Overfitting이었다.
  - 이 문제를 피할 수 있는 방법중 하나가 Greedy Layer-Wise Training 이다.

![image](https://user-images.githubusercontent.com/69780812/139228468-45e09043-c8cb-4487-934f-d4abb597828e.png)

- 기본적인 AE의 경우이고, Unsupervised로 입력 데이터 X를 학습하고 출력이 다시 X에 근접하도록 파라미터 값을 결정한다.
  - 학습 방법은 Supervised Learning 처럼 오차를 역전파 시키는 방식을 사용한다.
  - AE에서는 출력단에서 입력 X와 실제 출력값의 차를 이용하며 역전파 방식은 거의 동일하다.
  - 학습을 마치고 최종단을 제거하면 Hiddeny Layer의 H는 입력 데이터에 숨어 있는 특징을 추출할 수 있게 된다.

![image](https://user-images.githubusercontent.com/69780812/139228830-73ea896e-a136-40ce-9d1d-e90796ce2b1f.png)

- Greedy Layer-Wise Training은 말그대로 Layer 별로 탐욕스럽게 학습시키는 것이다.
- (a) : 첫번쨰 hidden layer에 대한 학습 진행
  - 두번째 세번째 레이어는 없다고 가정
- (b) : (a)에서 학습시킨 첫번째 layer의 파라미터는 고정
  - 입력은 첫번째 hidden layer의 입력을 사용하는 것이나 마찬가지 상황이 되면 결과적으로 기본 AutoEncoder가 된다.
  - 세번째 레이어는 없다고 가정
- (c) : 첫번째, 두번째 파라미터는 고정
  - 두번째 Hidden layer의 출력을 입력으로 간주한다. 이는 또한 기본 AutoEncoder이므로 학습이 가능해진다.
- 이런방식으로 여러 개의 Hiddenlayer에도 학습이 가능하도록한다.

# Unsupervised Pre-Training
- 신경망 학습시 많은 학습 데이터가 필요한데, label이 달린 학습데이터는 부족한데 비해 label이 없는 학습 데이터는 많은 경우가 있다.
- 이 경우, label이 없는 학습 데이터를 이용해 신경망의 각 layer를 앞서 설명한 방식으로 Pre-training 시킬수 있다.
  - 이것을 Unsupervised Pre-Training이라고한다.
- ReLU, Dropout, maxout, data-aug, Batch Normalization등 다양한 방법이 발표되면서 피곤하게 위 방법을 사용하지 않아 거의 쓰지않게 되었다.
- 몇 년사이 바라보는 시각이 바뀐다는 사실이 흥미롭다.
  - 하지만 방식을 어느 정도 이해하고 있으면 좋을 것 같다.

# Denoising AutoEncoder
- AE는 지금과 같은 딥러닝 학습 기술들이 발표되기 전까지 Pre-Training 용도로 많이 사용되었다.

![image](https://user-images.githubusercontent.com/69780812/139231597-9268fd9b-b6b6-43f6-844c-f92b2a5b00d3.png)

- 기본 AE로 Encoder, Decoder로 구성된다.
  - Encoder : 입력 데이터에 숨어 있는 중요한 특징을 압축된 형태로 추출
  - Decoder : 다시 복원
- 입력 데이터에 약간의 잡음이 있더라도 핵심 본질이 유지가 되면 출력은 어느 정도 원영상을 복원할 수 있다.
- Denoising Auto Encoder는 복원 능력을 더 강화하기 위해 기본적 AE 학습 방법을 조금 변형시킨 것이다.

![image](https://user-images.githubusercontent.com/69780812/139231959-ab7c6cb4-3ce1-4fc8-9ece-c9e5cd06fa41.png)

- 잡음이 없는 원영상에 잡음을 가하여 영상 X^을 만들어 낸다.
- 논문에서는 잡음의 위치에 있는 픽셀값을 모두 0으로 만들었다.
- 기 후 기본 AE에 우너래 영상 X가 아니라 잡음이 있는 영상 X^을 가하고 출력 Z가 잡음이 없는 영상 X^이아니라 X와 가까워 지도록 학습 시킨다.

![image](https://user-images.githubusercontent.com/69780812/139232202-e489c5b2-631f-4c75-9bc5-f99d9a62de5c.png)

- 원영상 X를 잡음이 추가하는 과정 P(X^|X)를 거쳐 X^을 얻어낸다.
  - 해당 위치의 픽셀 값이 0으로 바뀐다.
  - 얻어낸 X^을 입력으로 사용한다.
- 출력 Z가 X에 근접하도록, 즉, X와의 차가 최소화 되도록 학습하면 그것이 Denoising AutoEncoder가 된다.

# Sparse AutoEncoder
## 데이터에 대한 표현: Complete or Overcomplete ?
- 일반적으로 신호는 푸리에나 웨이블렛의 경우처럼 기저함수의 선형적 결합으로 나타낼 수 있다.
- 대부분은 기저함수의 차원은 표현하고자하는 데이터의 차원가같다.
  - 이러한 경우를 Complete하다고한다.

![image](https://user-images.githubusercontent.com/69780812/139233318-bb0bf973-0c1b-4e42-8598-f4aefc805bc2.png)

- 데이터 x를 A라는 기저함수의 집합(Dictionary or Codebook)에 s라고 불리는 벡터를 곱해주는 형태가 되며 차원이 모두 n으롣 동일하다.
- A가 정해지면 s는 x를 표현하는 벡터가 된다.
- Complete한 경우 s와 x의 차원은 동일하며 x를 나타내는 s는 유일하다.

![image](https://user-images.githubusercontent.com/69780812/139233521-fd4091d5-aaf4-46f2-bb1d-908047e82e02.png)

- 신호를 보다 밀도 있게(Compact) 표현하는 방법은 크게 위 2개 방식이있다.
- 표현하고자 하는 데이터를 차원이 최소가 되도록 만드는 방식이있다.
  - 기본 AE 방법

![image](https://user-images.githubusercontent.com/69780812/139233706-e8938829-6034-4ec3-81e2-ad060bb68ea0.png)

- 또다른 방법으로는 Sparse coding이 있다.
- 사용하는 기저함수의 차원:m이 원 데이터의 차원:n보다 크다.
- Sparse coding에서 사용하는 기저함수 처럼 표현하고자 하는 데이터보다 차우너이 큰 경우를 **over-complete**하다고 한다.
- Complete한 기저함수를 사용하면 데이터를 표현할 수 있는 방법이 고유하다.
- OverComplete한 경우 나타낼 수 있는 방법이 하나가 아니라 여러 개가 가능하므로 뭔가 기준을 가지고 많은 것들 중에서 선택을 해줘야한다.
  - 원 데이터의 차원보다 크게하면서 좀더 미로있게 표현이 가능하다는 말이 서로 모순된 것처럼 보인다.
  - 위 수식에 있는 vector s에서 대부분의 계수들을 0으로 한다면(활성화되는 계수들의 숫자를 소수로 제한한다면), 기저함수의 차원은 커지지만 결과적으로 밀도있게 표현할 수 있게 되는 것이다.

## Primary Visual Cortex V1의 성질
![image](https://user-images.githubusercontent.com/69780812/139234178-c61209d2-5bec-4dc7-809c-d3f1097f13ea.png)

- Visual cortex 영역에는 simple cell, comlex cell, hyper-complex cell로 구성된다.
- Simple cell의 receprive field : spatially localized, oriented, bandpass한 성질을 갖는다.
  - 많은 세포로 구성되어 있지만 어떤 특정 순간에는 몇 개의 뉴런만 활성화 되어 기본적으로 Sparse coding과 비슷한 성질을 갖고 있다.
- CNN의 Convolution 적용하는 영역(Receptive Field)도 일부 local만을 필터(convolution)연산에 사용한다.
- FC-layer는 모든 입력을 동일한 중요도로보고 모두 연결하지만, Convolution layer에서는 1픽셀당 1개의 Connection만 가지도록 Sparse connection 형태를 취한다.
- Simple cell은 특정 방향의 엣지들을 검출할수 있도록 방향성을 갖고있다. 사람들이 엣지에 민감한 것도 여기서 기인한다.
- 위 그림에서 고양이의 Simple cell 영역에 전극을 꽂고 막대 형태 조명을 비췄을 때, 특정 방향으로 자극이 크게 나타나는 것을 확인할 수 있다.

## Sparse Coding
- Sparse Coding은 Unsupervised Learning 방법으로 Overcomplete 기저 벡터를 기반으로 데이터를 좀 더 효율적으로 표현하기 위한 용도로 개발되었다.
- 위의 Primary Visual cortex의 동작을 Sparse code를 통해 체계적으로 설명할 수 있음을 보여줬다.

![image](https://user-images.githubusercontent.com/69780812/139235069-b5840d79-c636-40de-b0af-557baecfeaa9.png)

- Dictionary(D) : column 방향으로 atom이라 부르는 기저벡터를 갖고있다.
- 데이터 X : alpha
- alpha는 빨간색에 해당하는 부분만 0이아니어서 결과적으로 해당 위치에 기저벡터를 alpha의 각 원소 만큼 곱해서 더하는 방식으로 표현된다.
  - 0이 아닌 원소의 개수가 3개이기 때문에 이전 입력보다 좀 더 밀도 있게 표현이 가능해진다.

![image](https://user-images.githubusercontent.com/69780812/139237496-9d264b73-fe44-464c-afa0-31ac00d06749.png)

- D와 alpha는 어떻게 구할 수 있는가 ?
- 위와 같은 행렬 형태로 바꾸어서 생각한다.
- 첫번째항목 : reconstruction error를 나타내는 부분
  - AD를 이용해 X를 얼마나 잘 표현할 수 있는지를 나타낸다.
- 두번째항목 : 일종의 penalty
  - 기저 벡터 alpha에서 0이아닌 element 개수를 제한하는 방향으로 penalty 역할을 해준다.
- 결과적으로 위 식을 최소화 하는 방식이 목표다.
- 기본적으로 Matching pursuit(MP), Basic purshuit(BP)라는 방식으로 문제를 풀어낸다.

![image](https://user-images.githubusercontent.com/69780812/139237912-5d6d6c79-4695-4aeb-9787-14ece5d25942.png)

- 자연 이미지의 특정 patch를 이용해 64개의 원소를 갖는 Dictionary를 구한다. (Unsupervised Learning에 해당)
  - 주로 다양한 에지를 표현할 수 있는 벡터들이 구해진 경우
- 실제 Test하는 경우는 기저 벡터중 3개의 원소만 0이 아닌 값으로 Test 샘플을 근사 시킨다.

![image](https://user-images.githubusercontent.com/69780812/139238283-c43be1f9-2a9a-4e19-9121-0e2560cbde34.png)

- 이러한 Sparse coding을 이용하면 data compression이나 잡음 제거 및 컬러 interpolation 등에서 탁월한 효과를 얻을 수 있다.
- 위 그림은 denoising에 sparse coding을 적용한 예이다.

![image](https://user-images.githubusercontent.com/69780812/139238402-8ab5c50b-a28c-45ae-84bc-daf87ce17c89.png)

- inpainting(영상 복원 기술)에 sparse coding을 적용한 경우다.
---
- [Sparse Coding Ref 1](https://www.cs.ubc.ca/~schmidtm/MLRG/sparseCoding.pdf)
- Olshausen & Field
  - "Emergence of simple cell receptive field properties by learning sparse code for natural images"
  - "Sparse coding with an overcomplete set: a strategy emplyes by V1"
- Sparse Coding의 장점은 전체 데이터를 대표할 수 있는 몇개의 작은 활성화된 Code만을 이용해 원래 신호를 복원해낼 수 있다는 것이다.
---

## Auto Encoder & Sparse Coding 비교
![image](https://user-images.githubusercontent.com/69780812/139239544-6999e1b7-1abe-43c7-8946-098f12354e15.png)

- 기본 AE는 통상적으로 Encoding 단계에서 입력보다 더 작은 유닛으로 표현하는 과정을 거치므로 어찌보면 Sparse coding과 비슷하다.
  - Compac coding에 해당하는 부분을을 구현한 예를 AE로 생각할 수 있다.

![image](https://user-images.githubusercontent.com/69780812/139239691-5f1cae43-d58b-4219-994e-a1dca97f0fab.png)

- Sparse Coding의 Cost function은 위와 같은 식으로 한다.
  - 왼쪽 : Reconstruct Error
  - 오른쪽 : Sparsity를 강제하기 위한 일종의 Penalty
- AutoEncoder의 Cost Function에는 Sparsity를 강제하기 위한 항목이 없다.
  - 기본적으로는 Reconstruction error에 해당하는 부분만 있다.
- 자연 영상 데이터에 대해 Sparse coding or Auto Encoder는 비슷한 D(Dictionary or weight matrix)를 보여주지만 일반적으로 AE가 일반화에 쉽고 더 복잡한 문제를 다룰 수 있다.
  - layer 수를 여러 단으로 구성하여 더 복잡한 non-linearity를 표현할 수 있다는 점 및 Cost function을 오차의 제곱이 아닌 다른 함수를 사용할 수 있다는 점도 AE의 장점으로 볼 수 있다.

## Sparse AutoEncoder
![image](https://user-images.githubusercontent.com/69780812/139240153-c1f570e0-95f0-4d42-8c6f-17762f015d83.png)

- Sparse Coding과 AutoEncoder의 장점을 합친 것이다.
- 기본 AE 처럼 Reconstruction Error에 기반한 Weight matrix를 구하는 것이 아니라 Sparsity 조건을 강제하여 Hidden unit에 존재하는 뉴런의 활성호를 제한하는 AE를 Sparse Auto Encoder라 부른다.

## k-Sparse Auto Encoder
- Sparse Coding은 기본적으로 dictionary learning, sparse encoding 2단계로 구성된다.
  - Dictionary Learning : 학습 데이터를 이용해 Dictionary와 Sparse Code vector를 구한다.
  - 위 과정에서 Sparsity 조건을 같이 만족시켜줘야하는데 통상적으로 Convex 함수가 아니므로 Gradient을 통해 접근하게 되면 local minimum에 빠지는 문제가 발생할 수 있다.
- k-Sparse Auto Encoder는 비교적 손쉽게 Sparse Coding을 할 수 있는 방법을 제공했다.
  - Hidden-layer에서 Activation을 최대 k개로 제한하는 방법을 적용해 Sparsity 조건을 적용한다.
  - 크기가 k번째가 되는 뉴런까지는 그 결과를 그대로 사용하고 나머지는 0으로 만들었다.
  - BackPropagation시에도 Activation이 0인 path는 무시되고, 0이 아닌 쪽의 Weight만 수정된다.
  - 이 과정은 Dictionary Learning 과정으로 볼 수 있다.
  - 새롭게 구성된 Dictionary에 의해 다시 학습을 반복적으로 수행하게 되면 새로운 k 개의 뉴런이 활성화 되고, 그것에 기반해 다시 새로운 dictionary의 atom을 구하게 된다.
- 결과적으로 보면 k개의 뉴런만 활성시켜 weight matrix를 학습하는 전형적인 과정이 Sparse Coding의 dictionary 및 Code vector를 학습하는 과정과 동일한 과정이 된다.

![image](https://user-images.githubusercontent.com/69780812/139241062-7540290a-9be3-4ccb-ac86-2815d74de0ae.png)

- suppk(W^TxX + b)가 바로 k개의 Support vector를 구하는 과정이다.
- 3번은 그것을 바탕으로 Weight matrix의 해당 atom을 update 하는 과정이다.
- Sparse Encoding 과정에서는 alpha x k 개인데, 이것은 k만 사용하는 경우보다 1보다 약간 큰 alpha를 사용해 k보다 좀더 많은 활성화된 뉴런을 갖도록 encoding 하는 것이 효율적이기 때문이다.

## k-Sparse AutoEncoder 효율성
- Sparse coding은 기본적으로 ditionary의 성능에 따라 결과가 달라진다.
  - Sparse AE도 weight matrix W를 잘 구해야한다.
- 성능이 좋다는 것
  - Ditionary를 구성하는 Atom 간 유사도가 낮다.
  - 유사도가 낮다는 것을 결과적으로 1개의 atom을 다른 1~2개의 atoms들의 linear 합으로 표현하기 어렵다는 것을 의미한다.
  - 결과적으로 수학적으로 우수한 것은 orthogonality에 근접하다는 이야기가 된다.

![image](https://user-images.githubusercontent.com/69780812/139242157-7f220f39-a064-472a-b792-1125130301b5.png)

- 우수성 비교에서는 atom들 간의 coherence를 체크했다.
  - coherence 결과적으로 내적이 된다.
  - 내적이 작을 수록 coherence가 작아지게되고 결과적으로 우수한 dictionary가 된다.
- k값을 정할 때는 무턱대고 정하는 것이 아니라 k <= (1+u^-1)의 조건을 만족 시켜야 좋은 결과를 얻는다고 한다.

## k 값에 따른 성능 비교
![image](https://user-images.githubusercontent.com/69780812/139242762-91de23d9-198e-4e32-b8a2-1d1e34763c0a.png)

- k값이 너무 큰 경우 : local feature를 얻을 수 있다.
  - pre-training용도에 활용하기 유용
- k값이 너무 작은 경우 : global feature를 얻을 수 있다.
  - Sparsity를 너무 강조하면 입력을 통으로 보는 경향이 생겨 비슷한 것들 조차 다르게 분류한다.
  - 적절한 k 값을 설정해야 Classification에 알맞은 global feature를 추출할 수 있다.

# Convolutional Auto Encoder
- 앞의 AE는 모두 Fully Connected 방식을 사용한 것들이다.
  - 영상에 비례해 파라미터 수가 증가하는 문제 발생
  - local 한 성질을 제대로 이용할 수 없다.
  - Convolution을 적용해보자.
- 데이터 수가 적다 ?
  - Suvervised Learning : Overfitting
  - AutoEncoder와 같은 Unsupervised Learning 방법을 사용해 학습시키는것이 효과적이다.

# DeepPainter
- [DeepPainter](http://elidavid.com/pubs/deeppainter.pdf0) : Painter Classification Using Deep Convolutional Autoencoders

![image](https://user-images.githubusercontent.com/69780812/139243647-134778e2-9246-4fc3-9281-eb348b573061.png)

![image](https://user-images.githubusercontent.com/69780812/139243680-0abbe45f-8eb3-44f7-a159-5b07dc01ea71.png)

- DeepPianter에서는 Unsupervised Learning 방법을 이용해 학습을 시킨다.
- 기존 FC-layer를 분리하고 conv + pooling layer 부분만 Auto Encoder 방식으로 학습시킨다.
- 이후 Supervised Learning 형식으로 Fully connected layer 부분을 Fine tuning 시킨다.
- 앞부분은 Stacked AE의 형태이다.
  - 자세히 보면 FCN(Fully Convolutional Network)과 비슷한 모양이다.

![image](https://user-images.githubusercontent.com/69780812/139244054-9e2c0fee-ce40-4c99-8ca7-a30b1cbef5f8.png)

- Max Pooling -> Decoder Network 구성시 문제가 된다.
  - size가 줄고나면 위치정보를 잃어 decoder 구성시 unpooling을 수행 시 어느 위치에 값을 넣어줄지 알 수가 없게 된다.
- Max-lcocation 정보를 활용하여 진행했다.
  - Pooling 위치를 따로 저장하여 UnPooling 시 위치를 알 수 있도록 했다.
- Unsupervised Learning 시 Feature 학습을 높이기 위해 Denosing AutoEncoder 학습처럼 학습한다.
  - 20% 정도 픽셀을 무작위로 없앰
- CAE(Convolutional AutoEncoder)에 대한 학습을 마치면 Decoder를 버리고 본래 Fully connected layer를 연결해 Classification을 학습시킨다.

![image](https://user-images.githubusercontent.com/69780812/139244420-e41c69f4-f1da-496b-802d-60be463f5ad2.png)

- 결과는 위와같으며 CAE 방식이 다른 방식보다 성능이 더 좋았다.