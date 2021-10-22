# Object Detection
- 영상 내 특정 대상이 존재하는지 여부 판단 + 대상의 정확한 위치를 파악하고 bounding box 영역으로 구분하는 것까지 수행
- Classification에 비해 훨씬 어렵다.
- R-CNN은 이전 detection 알고리즘에 비해 2배 이상의 성능 향상이 이루어졌다.

# R-CNN(Regions with CNN features)
- 이전 Object detection에 주로 사용되던 방법은 SIFT(Scale Invariant Feature Transform) or HOG(Histogram of Gradient)
에 기반한 알고리즘이었다.
  - 대부분 영상 내 존재하는 gradient 성분을 일정 블락으로 나누고 그 경향성을 이용해 대상을 검출하는 방식을 사용한다.
  - 하지만 low-level feature에 기반하므로 성능상 한계가 존재한다.

![image](https://user-images.githubusercontent.com/69780812/138475408-a252973b-8138-4552-b78b-bd969759f749.png)

- R-CNN은 입력 영상으로부터 약 2000개의 후보 영역을 만든다.
  - Selective search 방법이용
  - 영상 속에 있는 color나 texture 등 단순한 정보뿐만 아니라 영상 속 내재된 계층 구조도 같이 활용한다고 한다.
- Seletive search를 통해 후보 영역 선정 후 AlexNet이 224x224 이미지를 받아들이도록 되어있어 해당 영역을
warping이나 crop을 사용해 224x224로 만들어 AlexNet의 변형 CNN에 인가하여 최종출력에서 해당 영상을 대표할 수 있는
CNN feature vector를 얻어낸다.
- 그 후 linear SVM을 이용해 해당 영역을 분류한다.
- Computer Vision 관련 기술과 CNN 기술을 결합하여 뛰어난 성과를 내게된다.

# R-CNN의 성능 향상 및 학습 방법
![image](https://user-images.githubusercontent.com/69780812/138476697-1f496358-ac8e-43e4-aaee-a02beae0be46.png)

- ILSVRC 데이터로 CNN을 pre-training 한다.
  - PASCAL VOC가 label이 붙은 데이터양이 위 데이터셋보다 상대적으로 적었다고 한다.
- 이 후 warped VOC를 통해 CNN을 fine tuining 한다.
  - ground truth 데이터와 적어도 0.5 IoU 이상되는 region 들만 Positive로 하고 나머지는 negative로 하여 fine tuning을 시행한다.
  - 모든 class에 대해 32개의 positive window와 96개의 background window를 적용하여 128개의 mini-batch로 구성한다.

![image](https://user-images.githubusercontent.com/69780812/138476869-500a6296-8fd5-4e20-ba7f-46a4e66fb388.png)

- R-CNN을 사용하면 이전 방식들 보다 성능이 크게 개선되는 것을 확인할 수 있다.

# R-CNN의 문제점
1. AlexNet 구조를 그대로 사용하여 이미지 크기를 강제로 224x224로 맞추기위해 warping or crop을 사용했다.
   - 이로 인한 이미지 변형이나 crop으로 인한 손실로 성능 저하가 일어날 수 있는 요인이 된다.
2. 2000여개에 이르는 region proposal에 대해 순차적으로 CNN 수행
   - 학습이나 실제 run time이 길다.
3. 사용하는 알고리즘이 특히 region proposal이나 SVM 튜닝 등이 GPU 사용에 적합하지 않다.

# 개선 알고리즘 SPPNet(Spatial Pyramid Pooiling in Deep Convolution Networks for Visual Recognition)
- Kaiming He 연구팀은 AlexNet 중 convolution layer 부분은 sliding window 방식이므로 영상의 크기에 영향을 받지 않지만,
뒷단의 fully connected layer 부분만 영상의 크기에 영향을 받는다는 점이었다.

![image](https://user-images.githubusercontent.com/69780812/138477816-beac875d-5fe9-4bdc-bdb6-c6a9391cce88.png)

- Crop이나 Waping 시 왜곡이나 손실이 발생한다.
- Convolution layer 다음에 Spatial pyramid pooling layer를 두고 이 단계에서 pyramid 연산으로 입력 영상의 크기를
대응하도록 한다.
- SPPNet은 BoW(Bag-of-Words)개념을 사용한다.
  - BoW : 특정 개체를 분류하는데 굵고 강한 특징에 의존하는 대신 작은 여러 개의 특징을 사용하면 개체를 잘 구별할 수 있다는 사실에 기반한다.
  - SPPNet 설계자들은 BoW 개념처럼 여러 단계의 피라미드 레벨에서 오는 자잘한 feature들을 fully-connected layer의 입력으로 사용하고, 피라미드의 출력을 영상의 크기에 관계없이 사전에 미리 정하면 더 이상 영상의 크기에 제한을 받지 않게 된다는 점에 주목했다.
  - Fully-connected가 입력 영상 크기에 제한을 받기 때문이다.

![image](https://user-images.githubusercontent.com/69780812/138478911-6f0e3205-df58-44e2-b966-608a7453d73f.png)

- ZFNet 처럼 신경망의 최종 Convolution layer를 pyramid pooling layer로 변환 시킨다.
- 최종 피라미드 layer 에서는 직전 convolution layer의 결과를 여러 단계의 피라미드로 나눈다.
  - 1. 영상 전체를 커버하도록 1x1 pooling
  - 2. 영상의 4개 영역으로 구분한 2x2 pooling
  - 3. 영상을 9개 영역으로 구분한 3x3 pooling
  - 영상을 Spatial bin이라 불리는 총 M개의 영역으로 나눈다.
  - 이렇게 얻어진 결과들의 각각을 Concatenation 후 fully-connected layer의 입력으로 사용한다.
  - 이 때 feature map 갯수가 k개라면, 앞서 구한 M개의 벡터가 k x M 개의 차원으로 존재하게 되는 셈이된다.
  - [참고하면 좋은자료](https://www.robots.ox.ac.uk/~tvg/publications/talks/fast-rcnn-slides.pdf)

- SPPNet의 위 특성 덕분에 warping/crop 없이 전체 영상에 대해 딱 1번 convolution layer를 거친 후 해당 window에 대해 
SPP를 수행하므로 가장 시간이 오래걸리는 과정을 건너뛸수가 있어 성능이 24 ~ 102배 정도 빠르다.

# GoogLenet Detection
- GoogLeNet에 대한 detection은 자세한 언급이 없다고한다.
- inception 모델을 region classifier로 사용한다.
- region proposal 단계에서는 selective search와 multi-box prediction을 혼합해서 사용했다고 한다.
- super pixel크기를 2배 증가시켜 False positive의 수를 줄였다.
  - 또한, region proposal 수를 절반으로 줄여 mAP가 1% 개선했다고 한다.
  - mAP : Detection 성능 지표로 검사의 Class 별로 정확도를 구하고, 그것을 평균을 취하여 전체적인 정확도를 나타낸 것이다.
- 학습 시간 문제로 bounding-box regression은 적용하지 않았지만 성능은 R-CNN에 비해 훨씬 좋은 결과를 얻었댄다.
