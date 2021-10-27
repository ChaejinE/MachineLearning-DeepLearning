# Overview
- CNN의 conv + pooling은 강인한 local feature를 추출하는데 효과적이다.
- RNN은 전체적인 영상, feature map을 고려할 수 있다.
- CNN과 RNN을 결합하면 좋은 결과가 나올 것이라 예상할 수 있다.
  - 핵심은 **어떻게 결합을 할 것인가**이다.
- ReSeg는 RNN을 별도의 layer 개념으로 사용했지만, RNN을 내부 layer 처럼 사용할 수 있는 개념을 고안하게 된다. (2017)
- ReNet 구조로 부터 영감을 받아 L-RNN 개념을 고안하게 된다.

# L-RNN
- RNN의 가장 큰 장점 : 순환적 성질을 이용하여 입력 영상이나 Feature map의 전체를 볼 수 있다.
  - Receptive Field 제한에서 벗어날 수 있다.
- ReNet 팀은 수평(좌->우, 우->좌), 수직(상->하, 하->상) 방향으로 움직이는 4개의 1-D RNN을 사용해 다양한 스케일에서 Contextual information을 추출할 수 있는 것을 보여줬다.
- ReSeg는 CNN과 RNN의 결합 효과를 입증해다.
## 구조
![image](https://user-images.githubusercontent.com/69780812/139013195-026fa258-dd74-4b79-8ed2-23d60f1af26d.png)

- L-RNN의 기본구조다.
- Convolution을 통해 얻어진 local feature map에 4개의 1-D RNN sub-layer로 구성된 모듈을 통해 연산해주면 공간적인 의존관계(Spatial depedency)를 파악할 수 있게 된다.
- ReNet 구조와 거의 유사하고 좀 더 일반화 시키는 관점엥서 접근한다.
- ReNet은 연산량을 줄이기 위해 2x2 un-overlapping patch를 사용하여 patch 단위로 처리했지만, 여기서는 layer 내부로 들어가므로 입력과 출력이 동일해야한다.
  - patch 단위가 아니라 Pixel 단위로 RNN 연산을 수행한다.
  - 또한, CNN layer에 곧바로 적용 가능하므로 non-linearity를 추가할 수 있게되어 다양한 표현이 가능해지는 장점도 있다.

![image](https://user-images.githubusercontent.com/69780812/139013654-9f296cc2-cd2c-4846-9d9b-1d5c9d818a80.png)

- Within layer 개념의 RNN이므로 CNN의 어느 위치에나 편안하게 위치할 수 있다.
- x : CNN의 결과
- CNN의 결과와 RNN의 결과를 합치는 3가지 방법
  - 1. Forward: RNN의 파라미터를 0으로 하는 경우 RNN의 결과를 사용하지 않고 CNN 결과만 사용
  - 2. Sum : CNN에 결과에 RNN결과 결합 (L-RNN 팀의 Pick)
  - 3. Concatenate : CNN 결과에 RNN의 결과가 추가가 되는 형태 (채널수 증가로 파라미터가 늘어난다.)

# Layer RNN 이해
![image](https://user-images.githubusercontent.com/69780812/139014236-cb1f4dc2-5308-4284-9a10-7fa1ef0933d2.png)

- X(inter) : convolution을 마친 중간 결과
  - 여기에 ReLU와 같은 non-linearity 활성함수가  적용된다.
- 빠른 이해를 위해 1-D 관점(좌->우)를 살펴본다.
- 오른쪽은 L-RNN이 추가된 경우로 CNN 연산을 마친 중간 결과에 RNN 연산 결과를 더하는 형태다.

![image](https://user-images.githubusercontent.com/69780812/139014391-53bb3daf-d77f-4f16-93d5-6ece3ded5b39.png)

- (7) : 일반적은 Convolution 연산 수식
- (9) : RNN을 더해진 것
  - V를 0으로 하면, 일반적 CNN 식과 동일하다는 것을 알 수 있다.
  - L-RNN은 기존 CNN 망의 어느 위치라도 편안하게 추가할 수 있으며 CNN이 갖는 장점과 RNN이 갖는 장점읅 ㅕㄹ합하여 성능을 극대화 시킬 수 있게 되었다.

# L-RNN 성능 평가
- L-RNN 모듈은 위치에 상관없이 삽입이 가능 하므로 Classifaction뿐만 아니라 Semantic Segmentation 망에도 적용 가능하다.

![image](https://user-images.githubusercontent.com/69780812/139015165-91f38884-0431-4617-a1c8-c9ea8746a88d.png)

- FCN에 L-RNN 모듈을 적용한 그림이다.
- FCN은 Fully Connected layer를 1x1 convolution으로 보는 방식을 취해 영상 크기 제한을 받지 않고 픽셀 레벨의 예측이 가능하게 되었다.
- L-RNN을 통해 Feature map 전체를 살펴볼 수 있으므로 마치 CRF를 사용한 것과 같은 효과를 얻을 수 있게 된다.

![image](https://user-images.githubusercontent.com/69780812/139015200-74fd9de7-0368-4200-af3b-779b9248c575.png)

- 실험결과의 표이다.
- L-RNN 모듈을 추가함에 따라 정확도 뿐만아니라 평균 IOU가 크게 개선되었다.
- 결국, RNN의 순환적 성격을 이용해 Feature-map 전부를 고려할 수 있다는 점은 CRF와 일맥상통하다.
---
- CNN과의 결합을 통해 RNN과 CNN 둘다의 장점을 극대화 함으로써 망의 깊이, 파라미터 수, 연산 시간 등의 성능 향상을 얻을 수 있었다.