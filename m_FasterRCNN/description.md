# Fast R-CNN
- Fast RCNN은 R-CNN, SPPNet의 Object detection/localization 알고리즘의 정확성과 속도를 개선

![image](https://user-images.githubusercontent.com/69780812/138592973-ea7fd779-68a3-4e91-9204-2aff841e7cbd.png)

- 0.32초로 되어 빨라진 것으로 보이지만 위는 Region Proposal에서의 시간이 고려되지 않은 것이다.
- Region Proposal 시간을 포함하면 이미지 1장 당 2초 정도가 걸린다.
  - Real-Time에서는 부담스러운 처리 속도다.
- Fast RCNN은 후보 영역을 계산하는 과정이 필요하고, 상당한 시간이 걸리는 점은 문제이며, 별도의 과정이기 때문에 진정한 의미에서 1-pass 구현이라고 보기에는 어려움이있다.
- Faster RCNN은 region proposal을 위해 별도의 과정을 거치는 대신 image classifaction, convnet에 region proposal을 위한 특수 용도의 망, RPN을 추가하여 Fast R-CNN의 문제를 해결했다.

## Region Proposal

![image](https://user-images.githubusercontent.com/69780812/138593117-b11c15bb-a9a1-42f9-a7e9-509c1726c4ed.png)

- Input Image -> 적절한 Region Proposal 알고리즘 -> Roi Pooling Layer -> FC layer -> Softmax clsfier, Bbox regressor

![image](https://user-images.githubusercontent.com/69780812/138593252-d8971529-2e6f-4644-9ba2-db40302c585f.png)

- R-CNN부터 Region Propsal의 방식 : **Selective Search**
  - Bottom-up Segmentation을 수행하고 이것들을 다양한 Scale상에서 병합 -> region을 bbox 영역으로 구분
- "What makes for effective detection proposals?"
  - 기존에 발표된 detection 관련 proposal에 대한 비교분석 자료가 있다.
  - 결과만 보면 EdgeBox방식ㄱ이 가장 뛰어나다.
  - Selective Search 방식도 전체적인 평가에서 좋은 편이지만 연산시간 관점에서 보면 좋은 방식이라고 볼 수는 없다.

# RPN(Region Proposal Network)
![image](https://user-images.githubusercontent.com/69780812/138593343-62f51713-d13f-4b0e-8e31-a7f986359c84.png)

- Faster R-CNN의 블락도이다.
- Fast R-CNN과 비슷하지만 RPN이라는 특수항 망이 추가
  - RPN을 통해 Object가 있을만한 영역에 대한 Proposal을 구하고 그 결과를 RoI Pooling layer에 보낸다.
- RPN은 Fast R-CNN에서 사용했던 동일한 ConvNet을 그대로 사용하므로 입력의 크기에 제한이 없다.
- 출력은 각 Proposal에 대해 Objectness score가 붙은 사각형 Object 집합이 된다.
- model의 형태는 Fully-convolution network 형태이다.
- ConvNet 부분의 Feature-map을 입력으로 받아들이고, n x n 크기의 **Sliding window convolution을 수행**하여 256차원 혹은 512 차원의 벡터를 만들어낸다.
- 위 벡터를 다시 물체인지 물체가 아닌지를 나타내는 box classification layer와 후보 영역의 좌표를 만들어내는 box regressor regressor layer에 연결한다.

![image](https://user-images.githubusercontent.com/69780812/138593794-88454962-e310-4a04-a37f-1076aa03d6ae.png)

- k개의 Object 후보에 대해 cls layer에서는 Object인가 아닌가를 나타내는 Score를 구한다.
  - 2k score가 된다.
- reg layer에서는 Object에 대한 (x, y, w, h) 값을 출력하므로 4k 좌표가 된다.
- 각각의 Sliding window에서는 총 k개의 Object 후보를 추천
- sliding window 중심을 기준으로 scale과 aspect ratio를 달리하는 조합이 가능
  - 이 조합을 anchor라고 부른다.
- Faster RCNN 논문에서는 scale 3가지, aspect ratio 3가지를 지원하여 총 9개 조합이 가능
- **Sliding window**방식을 사용하게 되면, anchor와 anchor에 대해 proposal을 계산하는 함수가 **translation-invariant**하게 된다.
  - CNN에서 Sliding Window를 사용하여 Convolution을 했을 때 얻는 효과와 동일하다.
  - 위 성질로 인해 model의 수가 크게 줄어든다.
  - k = 9인경우, (4+2) x 9 차원으로 줄어들게되어 결과적으로 **연산량을 크게 절감**할 수 있다.

# 결과
![image](https://user-images.githubusercontent.com/69780812/138594582-c6f38318-58ef-437a-a6df-1cbcc55c8512.png)

- SS:Selective Search
- EB : EdgeBox
- 위 결과를 보면, RPN + ZF, shared를 봤을 때, region proposal이 300개 정도임에도 불구하고 정확도가 1% 이상 좋다.
- 결과적으로 region proposal을 별도로 수행하는 것보다 RPN을 이용하여 동시에 수행을 하더라도 결과가 더 나쁘거나 하지 않다는 뜻이다.

![image](https://user-images.githubusercontent.com/69780812/138594665-e20d3750-2305-4584-a095-98b3287d3d67.png)

- 연산 속도 관점에서의 결과다.
- SS + Fast R-CNN을 하는 경우 약 2초정도 소요
- RPN + Fast R-CNN을 하는 경우 약 0.2초가 걸린다.
- 결과적으로 초당 5 fps 속도로 Object detection이 가능하므로 Real-Time 영역에 근접하게 되었다.
- 비교적 단순한 망인 ZFNet에서는 약 17fps 까지 속도가 올라간다.
- Faster-RCNN은 ResNet에 적용되어 빛을 발하게 된다.

# NoC (Network on Conv feature map)
- 기존 Faster-RCNN 논문에서는 Feature Extractor로 ZFNet, VGG-16을 이용해 구현했다.
- ResNet은 구조가 다른데 어떻게 구현했는지 알아본다.
- **"Object Detection Networks on Convolutional Feature Maps"라는 논문**을 통해 ResNet 구조를 크게 변화시키지 않고 Faster R-CNN을 구현했다.

![image](https://user-images.githubusercontent.com/69780812/138595111-8fd25231-a9fa-4918-9471-e720f16928a3.png)

- 위 논문에 따르면, Object Detection은 크게 Feature Extraction Part와 Object Classifier 부분으로 나눌 수 있다.
- Feature Extractor 부분은 ConvNet으로 구현하므로 ResNet으로 충분히 구현이 가능하다.
- Region Proposal 및 Roi Pooling 담당하는 부분도 구조적 관점에서 보았을 때 ConvNet으로 구현 가능
- Object Classifier 부분
  - CNN을 통해 얻어진 Feature 기반으로 별도의 망으로 처리하므로 Object classifier부분은 **NoC(Netowrk on Conv feature map)**이라고 명명했다.
  - 논문에서는 NoC 부분을 어떻게 구성해야 정확도가 개선되는지를 밝히고 있다.

![image](https://user-images.githubusercontent.com/69780812/138595430-413bbcd9-b915-4d8e-b353-f51323f1431d.png)

- Classifier 부분
  - 전형적 구현 방법 MLP를 FC-layer 2~3개로 구현
  - 논문에서는 하나의 망으로 간주하고 어떨 때 최적 성능이 나오는지 실험으로 확인했다.
  - 정확도를 개선하려면 기존 FC-layer만 사용하는 것보다 Fully Connected layer 앞에 Conv-layer를 추가해주면 결과가 좋았다고 한다.
  - Multi-Scale에 대응하기 위해 인접한 Sclae에서 Feature map 결과를 선택하는 Maxout 방식을 지원하면 결과가 더 개선되는 것을 확인할 수 있었다.

# Faster R-CNN with ResNet
- ResNet-101 구조를 이용
- NoC 개념 처럼 RoI Pooling 및 그 이후 부분을 ConvNet으로 구현

![image](https://user-images.githubusercontent.com/69780812/138595996-48971f1b-1a1e-429c-8bd4-8a876b50ead5.png)

![image](https://user-images.githubusercontent.com/69780812/138595924-e8b8a25c-6ddf-46a0-a449-b8b8f32e5055.png)

- RPN 이후 과정은 위 그림의 회색영역을 구현해야 한다.
- ResNet-101 Conv1 ~ 4는 Feature를 추출하기 위한 용도로 사용
- ResNet-101의 Conv5 와 Classifier 부분을 최종단의 Classifier로 대체해주면 Object Detection에 활용이 가능하다.

## Object Detection with ResNet 결과
![image](https://user-images.githubusercontent.com/69780812/138596024-289c4548-91a2-4500-a24c-57348c81c590.png)

- VGG-16 보다 ResNet을 사용하는 것이 결과가 좋아지는 것을 확인할 수 있다.
  - 이는 ResNet을 사용하는 것이 더 깊은 망의 효과를 톡톡히 누린 결과라고 볼 수 있다.