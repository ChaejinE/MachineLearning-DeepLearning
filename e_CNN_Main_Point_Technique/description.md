# Batch Normalization
- 딥러닝의 골치 아픈 문제 : Vanishing/Exploding Gradient
  - layer 수가 많아지면 많아질수록 누적되어 나타난다.
  - Sigmoid & hyper-tangent와 같은 비선형 포화함수를 사용하게 되면, 입력의 절대값이 작은 일부 구간을 제외하면 미분값이 0 근처로 가기때문에 역전파를 통한 학습이 어려워지거나 느려지게 된다.
  - ReLu를 Activation Function으로 쓰면서 문제가 완화되었지만 회피이지 본질적인 해결책이 아니라 망이 깊어지면 여전히 문제가 된다.
  - dropout이나 regularization 방법들 역시 본질적인 해결책이 아니므로 여전히 일정 Layer 수를 넘어가면 training을 성공시키는 것을 보장할 수 없다.

## Internal Covariate Shift
- 망이 깊어짐에 따라 이전 layer에서의 작은 파라미터 변화가 증폭되어 뒷단에 큰 영향을 끼치게 될 수 있다.
  - weight initialization, learning rate 등의 hyper-parameters
- 학습하는 도중에 이전 layer의 파라미터 변화로 인해 현재 layer의 입력 분포가 바뀌는 현상을 **Covairate Shift**라고 한다.

![image](https://user-images.githubusercontent.com/69780812/138060088-31d96095-cbf9-4aef-b1d6-6aa1b5ec8e64.png)

- 이는 건축에서 하중에 의해 기둥이 휘어지는 Buckling과 비슷하다.
- c, d 경우 처럼 휘어짐을 방지하는 수단이 필요하게되며 batch normalization, whitening 기법이 그 예다.

## Covariate Shift를 줄이는 방법
- 각 layer로 들어가는 입력을 Whitening 시킨다.
  - whitening : 입력을 평균 0, 분산 1로 바꿔준다는 것이다.
  - 단순히 Whitening만 시키면 parameter를 계산하기 위한 최적화 과정, back-propagation과 무관하게 진행되기 때문에 특정 파라미터가 계속 커지는 상태로 Whitening이 진행될 수 있다.
- 단순하게 Whitening 하는 것 말고 확실하게하는 방법이 batch normalization이다.
- BN은 평균과 분산을 조정하는 과정이 별도의 Process로 있는 것이 아니라 신경망 안에 포함되어 Training 시 평균과 분산을 조정하는 과정 역시 같이 조절 된다는 점이 구별되는 차이점이다.
- 
