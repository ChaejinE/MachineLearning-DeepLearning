# Overview
- CNN의 local feature 추출 능력과 RNN의 원거리 혹은 전체 영상을 고려한 Feature 추출 능력을 결합
- Semantic Segmentation 성능을 높이자.

# ReSeg 구조
- VGG-16을 기본망으로 하고, 1/8해상도까지만 사용
  - ImageNet을 이용한 PreTraining
- 중간 부분에 ReNet 적용
  - ReNet을 연결해서 이미지 전체를 볼 수 있도록 했다.
  - 개념적으로 보면 CRF를 사용해 좀 더 넓은 범위에서 영상의 Context를 고려하는 것과 일맥 상통한다.
  - 결과적으로 RNN의 구조적 장점을 잘 활용했다.
  - End-to-End 학습이 가능하다.
- 뒷부분 Transposed Convolution 적용

![image](https://user-images.githubusercontent.com/69780812/139011674-e3be3a3d-bfb3-460f-b779-01f22f0c3669.png)

- ReNet을 통해 좀 더 넓은 영역을 볼 수 있게 됐지만 최종 해상에는 2x2 non-overlapping patch 방식이라 해상도가 가로/세로 방향 1/2씩 줄어든다.
- 줄어든 해상도를 원영상 크기 복구를 위해 bilinear interpolation 대신 **Transposed Convolution**을 적용했다.
- Transposed Convolution(Fractionally strided convolution)이란 Stride의 크기가 1보다 작은 경우로 Convolution을 수행하기때문에 결과적으로 보면 원영상의 중간에 0을 끼워 넣고 Convolution을 수행하여 영상의 크기를 크게 만드는 효과를 얻는다.
  - Upsampling이 가능해진다.
  - [참고 Animation](https://github.com/vdumoulin/conv_arithmetic)

# ReSeg 성능
![image](https://user-images.githubusercontent.com/69780812/139012299-2c3a8b98-1958-42c1-8863-368b61c92998.png)

- Global acc : pixel별 정밀도
- Avg IOU : 데이터와 예측이 겹치는 부분을 본다.
- 위 두 척도를 봤을 때 상당히 좋다는 것을 알 수 있다.