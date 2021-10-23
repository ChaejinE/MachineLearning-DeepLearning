# Overview
- 2013년 ILSVRC 1위
- fully-connected layer를 1x1 convolution 개념으로 사용하면 고정된 이밎 뿐만아니라 다양한 이미지를 
Sliding Window 방식으로 처리 할 수 있다.
- feature extraction 블락 맨 마지막 단에 오는 max-pooling layer의 전후 처리 방식을 조금만 바꾸면, 기존 CNN 방식보다
dense(조밀하게) feature extraction, localization 및 detection을 수행한다고한다.
  - multi-crop 방식보다 연산량 관점에서 매우 효율적

# Classification
- 특정 대상이 영상 내에 존재하는지 여부를 판단하는 것

# Localization
- bounding box를 통해 물체의 존재하는 영역까지 파악하는 것
- 최대 5개 까지 bounding box를 통해 후보를 내고, ground-truth와 비교하여 50% 이상 영역이 일치하면 맞는 것으로 본다.
- 성능 지표는 error rate이다.

# Detection
- 영상내에 존재하는 Object를 가능한 많이 추정 (없는 경우 0)
- 추정이 틀린 False Positive에 대해서는 감점
- mAP(mean Average Precision)으로 결과를 낸다.

# OverFeat
- Fast, Accurate 라는 2개 model이 있으며 conv/pooling 하나 더 있나 없냐 차이다.

![image](https://user-images.githubusercontent.com/69780812/138491150-a05c8b18-e7a3-4611-8046-6b267f6ec354.png)

- fast model 이며 LRN은 별 효과가 없다고 하여 사용하지 않고, AlexNet 처럼 overlapped pooling을 사용하지 않고,
non-onverlapped pooling 방식을 사용하고 있다.

# Testing - Multi scale dense evaluation
- Multi-crop voting 방식 대신 dense evaluation 방식을 사용
- Multi-crop의 경우 겹치는 곳이 있어도 CNN을 다시 해야했다. dense evaluation으로 이를 해결했다.

![image](https://user-images.githubusercontent.com/69780812/138491850-503d2fbb-5a4e-4975-889e-e0df0401d4f2.png)

- 1차원으로 이해해보자.
- 최종 max pooling layer에 총 20개의 데이터가있고, 3x1 non-overlapped pooling을 한다고 한다.
- Pooling 후에는 양이 1/3으로 줄고, resolution 역시 1/3 준다. -> 위치의 정확도도 그만큼 떨어진다.
- offset을 1단위로 해서 각각 pooling을 진행하고, 그 결과를 Classifier에 적용하여 pooling 이전의 해상도를 유지한다.
  - Pooling 이후 1번만 classifier 연산을 하는 것에 비해 조밀한 검사를 할 수 있다는 것이다.
  - dense evaluation

![image](https://user-images.githubusercontent.com/69780812/138492220-e7d5fd61-fd01-46b4-9d8d-3824bfde3d15.png)

- Accurate model의 그림이다.
- 실제 입력영상이 221x221인것을 감안하면 Layer7의 1 픽셀의 해상도가 36픽셀에 해당하므로 간격이 너무 듬성 듬성하다.
- Layer 6 Pooling 부분을 주목하면 stride가 3x3이므로 해상도는 12픽셀 수준이 되고, Pooling 이전단 기준으로 1픽셀씩
offset을 갖고 pooling 수행 시 Classifier에 들어가는 양은 결과적으로 3x3, 9배만큼 많아지지만 해상려은 그만큼 좋아진다.

- 또한, FC-layer를 1x1 Conv로 해석하게되는데, 이는 FC-layer 앞단의 feature map 크기에 연연해 할 이유가 없어진 것이다.
- 그간, ConvNet 설계자들이 어려워했던 부분은 conv 부분은 영상 크기에 상관없이 적용가능하지만 FC-layer는 Fixed size를 갖고 있어
Sliding window 개념이 아니라 crop을 수행하고 동일한 크기의 feature map이 확보되도록 했었다.

![image](https://user-images.githubusercontent.com/69780812/138493268-fab06ce5-63a8-4075-82b3-b53e4985a981.png)

- 14x14인 경우 최종적으로 5x5를 얻은 후 Classifier를 거쳐 학습한다.
- 16x16을 test 입력으로 사용하게 된다면 FC-layer앞 단에서 6x6 크기의 feature map은 5x5 window 4개가 있는 것으로 볼 수 있고, 
최종적으로 2x2 차원의 출력을 얻을 수 있게 된다.
- 위 개념이 확보되어 큰 이미지의 특정 위치를 무작위로 선택하는 것이 아니라 일정한 Resolution 단위로 선택할 수 있게 된다.
- 또한 영상의 scale이 바뀌더라도 바뀐 Scale에 맞춰 sliding window를 움직이면서 결과를 얻으면 된다.

![image](https://user-images.githubusercontent.com/69780812/138495620-08a48b57-cd4c-46ae-b706-75cd4fed5c8c.png)

- 4개 scale에 대한 dense evaluation을 보여주는 그림이다.
- voting 개념을 호라용해서 대상을 classification 및 localization 시킬 수 있다.
- Localization은 대상의 우치ㅣ나 형태에 맞춰 bounding box까지 학습해줘야 한다.
- 최종단에 있는 Classifier 부분을 Bbox regression network으로 치환해주고, bounding box를 각각의 위치 및 sclae에
맞게 학습 시키면 된다.

![image](https://user-images.githubusercontent.com/69780812/138495854-4b6fbdec-5d43-4a11-9f35-148c22ae106c.png)

- voting에 의해 특정 대상이 검출되고, 대상에 맞춰 bounding box를 찾아주는 것을 보여주는 그림이다.
- crop 처럼 원 학습 영상으로 부터 영상을 잘라낸 후 각각 ConvNet에 적용 시키는 방식이 아니라 큰 영상에 대해 곧바로
ConvNet을 적용 하고 픽셀 간격으로 Sliding Window 하듯이 결과를 끌어낼 수 있어 연산량 관점에서 매우 효율적이다.
- crop과 dense evaluation을 상보적으로 섞어 사용하면 더 성능이 좋아진다.
---
- 근본적으로는 deeper network 잇아의 성능을 얻은 것은 아니지만 ConvNet을 이용해서 localization/detection까지
통합을 시도했다는 점에서 의미가 있고, SPPNet의 Spatial pyramid pooling과 맥이 닿아있다.
