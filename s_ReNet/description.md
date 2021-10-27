# Overview
- "ReNet: A Recurrent Neural Network Based Alternative to Convolutional Networks", Francesco Visin

# ReNet
![image](https://user-images.githubusercontent.com/69780812/138879891-cee37339-97ca-4a33-8717-ff6369691050.png)

- 이 논문에서는 convolution + pooling layer를 4개의 1차원 RNN-layer를 통해 구현하는 것을 기본 목표로 하고 있다.
- 여기서 4개의 1차원 RNN은 아래에서 위로, 위에서 아래로 왼쪽에서 오른쪽으로 및 오른쪽에서 왼쪽으로 Sliding window를 옮기듯 처리하는 것을 말한다.
- CNN에서 Conv + Pooling을 사용하면 Receptive Field에 해당하는 local 영역의 Feature를 추출할 수 있다.
- 4방향 1-D RNN에서는 전체 이미지에 대한 특성 위치에서의 Activation을 볼 수 있어 좀 더 넓은 영역을 볼 수 있게 된다.

# ReNet의 동작
- 기본동작은 전체 영상의 영역을 패치(window)영역이 겹치지 않도록 나눈다.
  - 겹치게 할 수 도 있지만 연산 시간을 줄이기 위해 한 것 뿐이다.

![image](https://user-images.githubusercontent.com/69780812/138881689-0f9d3379-7bbf-4751-abab-4c8cb2364c28.png)

- RNN에서는 입력(x0 ~ xt)으로 들어가는 값은 각각의 patch 값이며 patch 값을 받아 현재 patch에 대한 activation과 hidden state를 계산한다.

![image](https://user-images.githubusercontent.com/69780812/138881843-15be68bd-2702-4155-9377-6913c411854d.png)

- 먼저 수직 방향으로 2개의 1-D RNN연산(아래에서 위로, 위에서 아래로)를 수행한다. 우의 수식이다.
- f_VFWD : 순방향 수직방향으로 hidden state의 활성함수
- f_VREV : 역방향 수직방향으로 hidden state의 활성함수
- 단순 RNN, LSTM 및 GRU 등을 사용할 수 있다.
- 수직방향 Sweep을 마치고 난 후 얻어지는 결과를 결합하여 임시의 복합 Feature-map을 만든다.
- 이렇게 만들어진 V(i,j)는 전체 입력 Feature-map의 j행(column) 패치들을 고려한 (i,j) 위치에서의 Activation이 된다.
- 이렇게 수직방향으로 Sweeping한 후 얻어진 결과에 대해 다시 수평 방향으로 움직여가며 각 위치에서의 activation h(i,j)를 구한다. 이렇게 구해진 h(i,j)는 전체 영상의 관점에서 patch (i,j)위치에서의 activation이 된다.
- 일반적 CNN을 수행하면 자신의 위치에있는 Receptive Field에 대한 Activation 결과만 사용하지만, RNN을 사용하게 되면 수평 혹은 수직 방향으로의 연결(lateral conncetion)을 통해 전체 이미지를 고려할 수 있기 때문에 영상의 다른 위치에 있는 redundant한 특징을 제거하거나 병합할 수 있게 된다.

![image](https://user-images.githubusercontent.com/69780812/138882948-7bf0aa2f-8636-4c56-86ca-dd55fc23b464.png)

- 이렇게 개발된 4개 1-D layer는 전체적으로 보면 1 RNN layer로 볼 수 있고 여러단으로 쌓게되면 여러단의 Conv+Pooling과 같은 효과를 얻을 수 있다.
- 기존 CNN은 Pooling을 통해 영상의 크기를 줄여가면서 결과적으로 Spatial invariance를 얻는다.
- RNN은 전체 이미지를 보기 때문에 Pooling이 필요없다.
- CNN은 구조적 관점에서 병렬로 수행하기 좋다.
- RNN은 동작이 순차적으로 이루어지므로 병렬 연산에 적합하지는 않다. 하지만, 파라미터 수가 적다.
- RNN도 CNN의 대안이 될 수 있다는 것을 알 수 있는 Netework였다.
