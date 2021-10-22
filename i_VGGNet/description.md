# VGGNet Overview
- 2014년에 GoogLeNet에 간소한 차이로 2위를 차지했다.
- 하지만 구조적으로는 VGGNet이 훨씬 간단한 구조로 되어있어 이해가 수비고 변형 시켜가며 테스트하기에 용이하다고한다.

# Neotwork's Depth
![image](https://user-images.githubusercontent.com/69780812/138483696-0739df0e-aa89-45e0-a9fe-52224754b63f.png)

- AlexNet에 비해서는 깊은 신경망 구조이지만, 구조적 측면에서는 AlexNet이나 LeNet에 비해 차이가 없다.

# VGGNet
![image](https://user-images.githubusercontent.com/69780812/138484624-877cad60-9179-4fe5-94fb-d62ca7d95338.png)

- receptive field 크기를 3x3으로 정해서 깊이가 어떤 영향을 주는지 실험했다고 한다.
- Inception 모듈에서 Fatorizing 부분에서 공부했듯이 파라미터 수를 줄여서 깊이를 깊게가져갈 수 있었다.
  - 의도했는지 뽀록이었는지 3x3을 기본으로 하면서 망이 깊은 구조를 만들다 보니 VGGNet은 결과가 좋았다.
- VGGNet의 단점은 간단한 구조임에도 파라미터 양이 엄청 많다는 것이다.

# 특이한 점
1. VGGNet에서 A-LRN은 Local Response Normalization이 적용된 구조인데, 예상과 달리 LRN이 별 효과가 없어서 사용하지 않는다.
2. 1x1이 적용되긴하지만 차원을 줄이는 목적이라기 보다는 차원은 유지하면서 ReLU를 이용해 추가적은 non-linearity를 확보하기 위함이다.
3. Deep Net은 vanishing/exploding gradient 문제로 학습이 어려워질 수 있는데, 나머지 구조를 학습할 때 처음 4 layer,
fully-connected layer를 구조 A의 학습 결과로 초기값을 설정하고 학습시켜 이를 해결했다.
   - GoogLeNet은 Auxiliary classifier로 해결했었다.

# 망 깊게 만들기
- VGGNet팀은 깊은 망의 독특한 구조를 설계하기보다 망의 깊이가 어떤 영향을 끼치는지 확인하기 위해 가장 단순한 구조인 3x3 conv를
겹쳐서 사용하는 방법을 취했다.
- 어느 정도 이상이 되면 성능 개선의 효과가 미미해지는 지점이 있음을 확인했다.

![image](https://user-images.githubusercontent.com/69780812/138485553-04641804-84c3-48c7-b416-72f68ca76711.png)

- 3x3이 2개가 쌓이면 5x5가되고, 3개가 쌓이면 7x7의 효과가 나지만, 파라미터 수는 감소한다. 학습속도를 높일 수 있다는 것을 알고 있다.
- 위 그림은 3x3을 단순히 쌓아가면서 망을 깊게가져가고, 더 깊은 layer의 파라미터 초기화로 vanishing gradient 문제도 해결한다.

# Training
- AlexNet
  - 256x256 -> 224x224 무작위 취하기
  - RGB 컬러를 주성분 부석으로 RGB 데이터 조작하기
  - 모든 이미지를 256x256 single scale만 사용
- GoogLeNet
  - 영상의 가로/세로를 [3/4, 4/3] 범위를 유지하며 원영상의 8% ~ 100% 포함할 수 있도록 다양한 크기의 patch를 학습에 사용
  - photometric distortion을 통해 학습데이터 늘림
- VGGNet
  - training scale : 'S'로 표시
  - single-scale training 과 Multiscaling training을 지원
  - S = 256인경우와 S = 2256, 384 두개의 scale 지원
  - Multi Scale의 경우 Smin(256)과 Smax(512) 범위에서 무작위로 선택
  - 다양한 크기에 대한 대응이 가능하여 정확도가 올라간다. (선택된 scale에서 224x224 영역을 잘라낸다.)
  - Multi-scale 학습은 S = 348로 미리 학습시킨 후 S를 무작위로 선택해가며 fine tuining을 한다고한다.
  - **S를 무작위로 바꿔가며 학습**시킨다고 하여 **scale jittering**이라고 했다고한다.
  - AlexNet 처럼 224x224 무작위 선택, PCA로 RGB 컬로 조작
  - GoogleNet과 VGGNet은 이름과 표현이 조금 다를 뿐이지 multi-scale을 고려했고
  - RGB 컬러 성분 조작 등 적은 학습데이터로 인한 Overfitting 문제에 빠지는 것을 최대한 막으려 노력했다.

# Testing
- 학습 이지미로부터 여러 개의 patch or crop을 이용해 가능한 많은 test 영상을 만들어 낸 후 여러 영상으로부터의 결과를
voting을 통해 가장 기대되는 것을 최종 test 결과로 정한다.
- AlexNet
  - Test 영상 256x256으로 scaling
  - 4코너와 중앙에서 224x224크기로 잘라 5개 영상 제작
  - 좌우 반전하여 총 10장 Test 영상
  - 10장 테스트 영상을 망에 넣고 10개의 결과를 평균하는 방식으로 최종 출력
- GoogLeNet
  - 256x256 뿐만아니라 256, 288, 320, 352로 총 4개 scale로 만들고 각각의 scale로부터 3장의 정사각형 이미지 선택
  - 선택된 이미지로부터 4코너, 센터 2개를 취해 총 6장 224x224 크기 영상
  - 좌우반전 -> 총 4x3x6x2 -> 144개 테스트 영상
  - AlexNet과 마찬가지로 Voting
- VGGNet
  - 'Q' : **test scale**
  - 테스트 영상을 미리 정한 크기의 Q로 크기조절을 한다.
  - training S와 같은 필요 없다.
  - dense evaluataion 개념 적용 (multi crop 방식의 테스트 영상 augmentation 외)
    - 큰 영상에 대해 곧바로 ConvNet을 적용하고 일정한 픽셀 간격(grid)로 마치 sliding window를 적용하듯 결과를 끌어낸다.
    - 연산량 관점에서 매우 효율적이지만 grid 크기 문제로 인해 학습 결과가 약간 떨어질 수 있다.
    - 그러므로 crop, dense evaluation을 상보적으로 섞어 사용하면 성능이 더 좋아진다고 한다.

# VGGNet Result
![image](https://user-images.githubusercontent.com/69780812/138488153-1bf82dca-9f8c-4ee8-a804-81e3c6d402b8.png)

- B구조 에서 3x3을 2개 겹쳐서 사용한 경우와 5x5 1개를 사용한 모델을 만들어 실험했는데 top-1 error에서 7% 정도 2개 쌓는 것이 좋았다고 한다.
  - 3x3이 단순하게 망을 깊게만들고 파라미터 크기를 줄일 뿐만아니라 non-linearity 활성함수를 통해 feature 추출 특성이 더 좋아졌음을 반증한다.

![image](https://user-images.githubusercontent.com/69780812/138488543-84bc5aef-635b-4d5f-9e33-6d02e32b1b8b.png)

- S가 고정인 경우는 \{S-32, S, S+32}로 Q값을 변화 시키며 테스트
- scale jittering 적용한 경우는 \[256, 384, 512]로 테스트 영상의 크기를 정했다.
  - 적용하지 않은 것보다 훠씬 결과가 좋고 single-scale보다 multi-scale이 결과가 좋다는 것을 확읺라 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138489051-48265b9d-feea-4cf0-ae98-062873570bbf.png)

- multi-crop + dense evaluation을 각각 적용했을 때 약간 성능이 좋았다.
---
- 결론적으로 VGGNet은 그 구조가 간단해서 이해나 변형은 쉬우나 파라미터의 수가 엄청나게 많아 학습시간이 오래걸린다.
- 다양한 simulation 발표로 Deep CNN에 대한 이해를 도왔다는 점에서 많은 기여를 한 것 같다.
