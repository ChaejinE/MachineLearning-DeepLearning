# 1. R-CNN과 SPPNet의 문제점
## R-CNN
1. Training이 여러 단계로 이루어졌다.
- R-CNN은 크게 3단계 과정
  - 1. 2000여개의 후보 영역에 대해 log loss 방식을 사용해 fine tuning
  - 2. 이후 ConvNet 특징을 이용해 SVM에 대한 fitting 작업 진행
  - 3. Bounding box regression에 대한 학습
    - 검출된 객체의 영역을 알맞은 크기의 사각형 영역으로 표시하고 위치까지 파악

2. Training 시간이 길고 대용량 저장 공간 필요
- SVM과 bounding box regressor의 학습을 위해 영상의 후보 영역으로 부터 feature 추출 후 디스크에 저장
- 학습 데이터 5천장에 대해 2.5일 소요
- 저장 공간도 수백 GB 필요

3. 객체 검출 속도가 느리다.
- 실제 검출 시 875MHz로 오버클럭킹된 K40 GPU에서 1장 처리하는 데 47초가 걸린다고 한다.

![image](https://user-images.githubusercontent.com/69780812/138551946-ea38c9ad-3fe3-4582-baa3-d9115733c14e.png)

- 후보 영역에 대하여 모든 연산을 순차적으로 수행하는 구조이므로 느릴 수 밖에 없다.

## SPPNet
![image](https://user-images.githubusercontent.com/69780812/138552018-0308eeb9-1dfc-42ab-9e03-6caba2b57f1d.png)

- 3단계 파이프 라인을 적용하는 것은 동일하다.
- Spatial Pyramid Pooling을 통해 앞단에 있는 Convolution layer에 대해서는 fine tuning 하지 않는다.
  - Deep Network 적용 시 정확도가 문제가 될 수 있다.

# 2. Fast R-CNN
- mAP가 R-CNN이나 SPPNet보다 좋다.
- 학습 시 single-stage로 가능
- 학습의 결과를 모든 망에 있는 모든 layer에 update
- feature caching을 위해 별도 디스크 공간 필요없다.

![image](https://user-images.githubusercontent.com/69780812/138552091-49b2c856-8857-4993-9ee1-44513555a5d2.png)

- Fast R-CNN의 기본 구조다.
- 전체 이미지 및 객체 후보 영역을 한꺼번에 받아들인다.
- Conv와 max-pooling을 통해 이미지 전체를 한번에 처리하고 Feature map을 생성한다.
- 그후 Roi Pooling layer를 통해 feature-map으로 부터 fixed-length feature vector를 추출한다.
  - 이 부분은 SPPNet과 유사하다.
- Fixed length feature vector는 FC-layer에 인가하고, 뒷단에 object class + background를 추정하기 위한 softmax부분과 각각의 위치를 출력하는 bbox regressor가 온다.

## Test
![image](https://user-images.githubusercontent.com/69780812/138552171-19ebc68f-16ae-4f0d-92ed-911a1d2caa45.png)

- test time시, SPPNet과 비슷하게 전체 영상에 대해 ConvNet 연산을 1번 수행 후 그 결과를 공유한다.
- Roi Pooling layer에서 다양한 후보 영역들에 대해 FC layer로 들어갈 수 있도록 크기를 조정해주는 과정을 거친다.
- 이후 Softmax classifier와 Bbox regressor를 동시에 수행하여 multi stage pipline 과정을 순차적으로 수행하는 SPPNet에 비해 빠르다.

## Training
![image](https://user-images.githubusercontent.com/69780812/138552914-d66419e1-0d9f-4a76-8bc2-d80f0b8ca424.png)

- R-CNN, SPPNet과 달리 일반 CNN 처럼 1-stage로 학습 가능
- SPPNet
  - ConvNet 부분은 1단계에서 학습 후 파라미터 고정
  - 마지막 3-layer(FCs, SVM/BboxRegressor)만 학습 가능
- Fast R-CNN
  - ConvNet 부분까지 Error를 역전파 시키면서 학습 가능
  - 정확도 개선 가능
  
## 학습 시간을 줄인 비법
![image](https://user-images.githubusercontent.com/69780812/138553004-c67f7752-0c11-4aee-8c11-d1660ff0878f.png)

- R-CNN, SPPNet은 128개의 mini-batch
  - 서로 다른 이미지로 부터 128개의 RoI를 취한다.
  - 이를 region-wise sampling 방법이라 한다.

![image](https://user-images.githubusercontent.com/69780812/138553891-67fe0a33-5379-4a2e-b472-813275e3811e.png)

- [참고자료](https://www.robots.ox.ac.uk/~tvg/publications/talks/fast-rcnn-slides.pdf)
- R-CNN은 모든 후보 ROI를 224x224로 scaling
  - Fast R-CNN에서는 원 영상을 그대로 사용
  - 이 때문에 경우에 따라 RoI가 receptive field가 매우 클 수 있다. 최악의 경우 전체 이미지
  - 이는 영상 크기에 비례하여 연산량이 증가하여 느려질 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138553260-8de70369-faeb-46a8-b5d5-9b8b475ffeb6.png)

- 위 문제를 피하기 위해 mini-batch 구성 시 hierarchical sampling 방식을 사용했다.
  - 서로 다른 128장 학습 영상에서 RoI를 무작위로 선택하지 않는다.
  - 작은 수의 학습영상으로 부터 128개의 RoI를 정하도록 했다.
  - test time과 마찬가지로 training에도 학습 영상의 결과를 공유할 수 있게 되어 연산 속도가 빨라진다.
  - 다른 이미지에 대한 Conv(feature-map)가 아니라 ROI들이 같은 이미지에서 나온 Conv일 것이기 때문에(Test와 같은 트릭을 쓴 것이다.)