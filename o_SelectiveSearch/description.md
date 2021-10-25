# Overview
- Image Segmentation은 Object Recognition이나 Semantic segmentation으로 이어지는 기반 기술이므로 매우 중요하다.
- Object detection을 위해 객체의 후보영역 선정 시 탁월한 성능을 보인 Selective Search 방법을 살펴보자.

# Selective Search ?
- 2013년 detection 분야에서 Overfeat 방식을 누르고 1위 차지
- R-CNN, SPPNet, Fast R-CNN 방식에 후보 영역 추천에 사용
- 사용시 지정한 의미에서 end-to-end 학습을 시키는 것은 불가능해 진다.
- 실시간 적용에도 어려움이있다.

# SS 의 목표
- 검출을 위한 가능한 후보 영역을 알아낼 수 있는 방법을 제공
- Exhaustive search + segmentation 방식을 결합하여 보다 뛰어난 후보 영역을 선정하는 점이 주목할만 하다.
  - Exhaustive serach : 후보가 될만한 모든 영역을 샅샅이 조사하는 방식
    - 후보가의 scale이 일정하지 않고 aspect ratio도 일정하지 않아 모두 찾는 것은 연산시간 관점에서는 수용이 불가능하다.
  - Segmentation 방법 : 영상의 특성을 고려하지 않고 찾는 것이 아니라 영상 데이터의 특성에 기반하는 방식이다. 색상, 모양, 무늬 등 다양한 기준에 따라 Segmentation가능하나 모든 경우에 동일하게 적용할 수 있는 Segmentation 방식을 찾기란 불가능에 가깝다.

![image](https://user-images.githubusercontent.com/69780812/138673666-bb9c93e3-20fe-41ca-94c3-e31de0d8159f.png)

- (a) : 테이블 위에 샐러드 접시, 샐러드 접시에는 집게.. 영상이 본질적으로 계층적이다.
  - 아무것도 없는 나무 테이블만 테이블이라 해야하는지, 테이블 위에 모든 물체까지 포함한 것을 테이블이라할지는 고민거리다.
- (b) : 색으로 고양이를 구별해야하는 경우이다.
- (c) : 카멜레온과 나뭇잎을 색으로 구별하기 어렵지만 무늬로 구별해야하는 경우
- (d) : 자동차의 바퀴가 자동차의 본체와 색깔과 무늬가 다르지만 자동차에 있는 일부이므로 자동차로 고려해줘야한다.

- SS는 Exhustive Search와 같이 무식한 방법으로 후보 영역을 선정하는 대신에 Segmentation 방법이 한계는 있지만 Segmentation에 동원이 가능한 다양한 모든 방법을 활용하여 Seed를 설정하고, 그 Seed에 대해 Exhaustive한 방식으로 찾는 것을 목표로 한다.
  - 논문에서는 이러한 Segmentation 방법을 가이드로한 data-driven SS라고 부른다.

![image](https://user-images.githubusercontent.com/69780812/138674234-b3fdbe7e-2e96-4c64-b226-b3713189736e.png)

- 왼쪽
  - 입력 영상에 대해 Segmentation 실시
  - 이것을 기반으로 후보 영역을 찾기 위한 Seed를 설정
  - 왼쪽 아래 그림 처럼 엄청나게 많은 후보가 만들어진다.
- 오른쪽
  - 적절한 기법을 통해 통합해나간다.
  - 결과적으로 이것을 바탕으로 후보 영역이 통합되면서 갯수가 줄어들게 된다.
- Canny Edge 검출기에서 Non-Maximum Suppression이나 Region growing 같은 기법들을 떠올리면 도움이된다.
- SS의 목표는 3가지로 요약이 가능하다.
  - 1. 영상은 계층적 구조를 가진다. 적절한 알고리즘을 사용해 크기에 상관없이 대상을 찾아낸다.
    - 객체의 크기가 작은 것부터 큰 것까지 모두 포함되도록 계층 구조를 파악할 수 있는 "bottom-up" 그룹화 방법을 사용한다.
  - 2. 컬러, 무늬, 명암 등 다양한 그룹화 기준을 고려한다.
  - 3. 빨라야한다.

# Selective Search 3단계 과정
1. 초기 Sub Segmentation 수행

![image](https://user-images.githubusercontent.com/69780812/138674857-07cab2f1-8780-4f03-8501-768b1327baaf.png)

- 각각의 객체가 1개의 영역에 할당 될 수 있도록 많은 초기영역을 생성

2. 작은 영역을 반복적으로 큰 영역으로 통합

![image](https://user-images.githubusercontent.com/69780812/138675023-565e101f-6d46-4779-bd6a-b5d1dd393c46.png)

- Greedy 알고리즘 사용
- 여러 영역으로 부터 가장 비슷한 영역을 고르고, 이들을 좀 더 큰 영역으로 통합
  - 1개의 영역이 남을 때까지 반복
- 위 그림은 초기에 작고 복잡한 영역들이 유사도에 따라 통합 되어가는 것을 확인할 수 있다.

3. 통합된 영역들을 바탕으로 후보 영역을 만들어 낸다.

![image](https://user-images.githubusercontent.com/69780812/138675238-32477507-254d-4c37-a259-902c55501f36.png)

# Felzenszwalb와 Huttenlocher의 Segmentation 알고리즘
- SS는 Segmentation을 후보 영역을 검출하기 위한 가이드로 사용하기 때문에 후보 영역의 선정이 정말 중요하다.
- 위 알고리즘 방식은 비교적 간단하면서도 그 성능이 뛰어나다.
- "Efficient Graph-based Image Segmentation"
- **인지점 관점에서 의미있는 부분을 모아서 그룹화**, **연산량 관점에서 효율성 증대**라는 2개의 목표를 기반으로 새로운 Segmentation 기법을 개했다.

![image](https://user-images.githubusercontent.com/69780812/138694029-eb5ea3c2-8dca-4387-9ebe-676b6deb6065.png)

- (b) : 사람이 인지하는 방식으로 제대로 Segmentation한 경우
- (a)를 봤을 때 밝기가 조금씩 변해서 (b) 처럼 Segmentation하기란 쉽지 않다. 또한, 바코드 모양을 여러개의 영역으로 구분하지 않고 1개 영역으로 깔끔하게 구별하는 것 역시 기존 Segmentation 알고리즘으로는 쉽지 않다.
- (c) : 사람이 하는 것처럼 의미있는 부분만을 모아서 그룹화를 제대로 시키지 못한 경우
- 사람이 인지하는 방식의 Segmentation을 위해 graph 방식을 사용하였다.
  - 기본적으로 이 방식에는 픽셀들 간의 위치에 기반해 weight를 정하므로 **grid graph 가중치**방식이라고 부른다.

![image](https://user-images.githubusercontent.com/69780812/138694485-27de1d08-a7dd-47b5-aa3d-601348768ebd.png)

- 가중치는 위와 같은 수식으로 결정된다.
- v: vertex(pixel)
- graph는 상하좌우 연결된 픽셀에 대해 만든다.
- E: edge, 픽셀과 픽셀의 관계
- 가중치는 픽셀간의 유사도가 떨어질 수록 큰 값을 갖게 되며 결과적으로 w값이 커지게 되면 영역의 분리가 일어나게 된다.

![image](https://user-images.githubusercontent.com/69780812/138694706-f4a4fec5-caea-4f3a-af13-dda89d88929c.png)

- C1, C2가 있는 경우 영역을 분리할 것인지 통합할 것인지를 판단하는 수식을 사용한다.

![image](https://user-images.githubusercontent.com/69780812/138694792-31b2f36d-c653-428b-a539-1629a0aa9b5d.png)

- Dif(C1, C2) : 두개의 그룹을 연결하는 변의 최소 가중치
- MInt(C1, C2) : C1, C2 그룹에서 최대 가중치 중 작은 것을 선택한 것
- 그룹간의 차가 그룹 내의 차보다 큰 경우는 별개의 그룹으로, 그렇지 않은 경우는 병합을 하는 방식이다.

![image](https://user-images.githubusercontent.com/69780812/138695015-e6fadaf1-a334-4bbb-af0b-3207efef14e6.png)

- 비교적 간단한 알고리즘으로 Segmentation을 수행했음에도 양호한 결과를 얻었다.
- 잔디 부분의 픽셀 변화가 꽤 있음에도 좋은 결과를 얻었다.
- 논문에서는 인접한 픽셀끼리의 공간적 위치관계를 따지는 방법뿐만 아니라 Feature space에서의 인접도를 고려한 방식도 제안했다.
  - **Nearest Neighborhood graph 가중치** 방식이라고 부른다.
  - 적정한 연산 시간 유지를 위해 Feature Space에서 가장 가까운 10개 픽셀에 한해서 Graph를 형성한다.
  - 모든 픽셀을 (x,y,r,g,b)로 정의된 Feature space로 투영하여 사용하며 x, y는 픽셀 좌표, r,g,b는 픽셀의 컬러다.
  - 가중치에 대한 설정은 5개 성분에 대한 **Euclidean distance**를 사용했다.
  - 그룹화 방식은 동일하다.

![image](https://user-images.githubusercontent.com/69780812/138695616-3a5b7f0d-229b-42b5-a637-4f0fdfcc17bc.png)

- Feature Space 방식을 사용하여 Segmentation을 실시한 경우이다.
- 배경에 잡음이 많음에도 같은 영역으로 통합되는 것을 확인할 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138695751-60fdb371-4f9c-4d6f-975c-dbc5d04d31be.png)

- 아주 좋은 결과를 얻을 수 있는 대표적 예이다.
- 왼쪽 사진은 구름이 있음에도 하늘이 거의 같은 영역으로 묶였다.
  - 잔디도 상당한 변화가 있음에도 동일 영역으로 묶였다.
- 오른쪽 사진을 보더라도 중간 밝은 노란색 불이 영역을 나누고 있어 같은 대상으로 처리하기 어렵지만 본 논문의 방식을 사용하면 대체로 좋았다.
---
- Felzeenszwalb의 Segmentation 알고리즘은 결과가 비교적 좋고 연산속도가 매우 빨라 SS 3단계 과정 첫번째 단계에 적용되었다.
- 절대적이거나 완벽한 알고리즘은 아니지만 SS에서 객체 검출을 위한 Guide로서의 관점에서는 적당한 알고리즘인 것 같다.
- 하지만 상당히 많은 영역이 만들어지므로 효율적으로 병합하는 방법이 필요하다. SS에서는 이 척도로 유사도를 사용하게 된다.
---

# Hierarchical Grouping
![image](https://user-images.githubusercontent.com/69780812/138696643-c43bc68e-8514-4870-9f2b-aa5f8a2b0c9e.png)

- 위 Felzenswalb의 segmentation을 사용하더라도 많은 작은 영역들이 만들어진다.
  - 중요한 객체 후보를 찾아내려면 작은영역들을 병합시키는 작업이 필요하다.

![image](https://user-images.githubusercontent.com/69780812/138696787-db5604db-09a3-4c6a-9ff1-12fcb90b96a0.png)

- 영역 병합에는 단순 Greedy 알고리즘을 사용한다.
- 1. Segmentation을 통해 초기 영역 설정
- 2. 인접하는 모든 영역들 간의 유사도 구하기
- 3. 가장 높은 유사도를 갖는 2개 영역 병합
- 4. 병합된 영역에 대해 다시 유사도 구하기
- 5. 새롭게 구해진 영역은 영역 List에 추가
- 전체 영역이 1개의 영역으로 통합될 때까지 반복 수행
- 최종적으로 R-List에 들어있는 영역들에 대한 Bbox를 구한다.

![image](https://user-images.githubusercontent.com/69780812/138697063-a719f503-9bb8-4568-88ee-f265bf403a2e.png)

- 위 과정 수행 후 영역이 통합되고 다양한 Scale에 있는 후보들을 검출 할 수 있게 된다.

# 다양화 전략 (Diversification Strategy)
- 후보 영역 추천의 성능 향상을 위한 다양화 전략
- 다양한 컬러 공간 사용
- Color, texture, size, fill 등 4가지 척도를 적용해 유사도 구한다.

## 다양한 컬러 공간
- 다른 컬러 공간 : 서로 다른 Invariance(항상성)을 보인다.

![image](https://user-images.githubusercontent.com/69780812/138697531-1a850f8c-cfc4-419e-b701-9466c04b65e2.png)

- rgb나 gray 공간을 사용하는 대신 8개의 서로 다른 Color 공간 -> 특징 컬러 공간에서는 검출하지 못했던 후보 영역까지 검출이 가능하게 된다.
- RGB Color Space : 그림자나 광원 세기 변화 등으로 3개 채널이 모두 영향을 받는다.
- HSV Color Space : 색상, 채도는 밝기 변화에 거의 영향 받지 않지만 명도는 영향을 받는다.
- 위 표는 각각의 컬러 공간이 영향 받는 수준을 평가한 것이다.
  - '-' : 영향받는 경우
  - '+' : 영향을 받지 않는 경우
  - '+/-' : 부분적으로 영향받는 경우
  - 1/3, 2/3 : 3개 컬러 채널에서 1개와 2개의 채널이 영향을 받지 않는다는 것을 나타낸다.
- 논문에서는 Sub-Segmentation과 영역을 병합하는 과정에 서로 다른 8개의 Color 공간을 적용했다고 밝힌다.

## 다양한 유사도 검사의 척도
- 유사도 결과는 모두 [0, 1]사이의 값을 갖도록 normalize 시킨다.

![image](https://user-images.githubusercontent.com/69780812/138697971-4056b116-9c36-4cb2-bdb7-bcc3635f412c.png)

- Color Similarity
- 히스토그램을 사용한다.
  - 각 컬러채널에 대해 bin 25
  - 히스토그램은 정규화
  - 각 영역들에 대한 모든 히스토그램을 구한 후 인접한 영역의 히스토그램의 교집합을 구하는 방식

![image](https://user-images.githubusercontent.com/69780812/138698313-aafe3378-acae-43b7-b757-2968ca70e7d9.png)

- Texture Similarity
- Object matching에서 뛰어난 성능을 보이는 SIFT(Scale Invariant Feature Transform)과 비슷한 방식을 사용하여 히스토그램을 구한다.
- 8방향 가우시안 미분값을 구하고 그 것을 bin 10으로 하여 히스토그램을 만든다.
- SIFT는 128차원 descripter vector를 사용하지만, 여기서는 80차원의 descripter vector를 사용한다.
- Color의 경우는 3개 채널이 있으므로 총 240 차원의 vector가 만들어진다.
- 히스토그램은 마찬가지로 normalize 한다.
- 영역간 유사도는 컬러 유사도와 마찬가지 방식을 사용한다.

![image](https://user-images.githubusercontent.com/69780812/138698684-5af7d093-78ab-4a00-89da-ae7e3658bf46.png)

- Size Similarity
- 작은 영역들을 합쳐서 큰 영역을 만들 때, 다른 유사도만 따지면 1개 영역이 다른 영역들을 차레로 병합하면서 영역들의 크기가 차이가 나게 된다.
- 크기 유사도는 작은 영역부터 먼저 합병 되도록 해주는 역할을 한다.
  - 일종의 Guide 역할
- size(im) : 전체 영역에 있는 픽셀 수
- size(r_i), size(r_j) : 유사도를 따지는 영역의 크기, 픽셀 수 이다.
- **영역의 크기가 작을 수록 유사도가 높게** 나오므로 다른 유사도가 모두 비슷한 수준이면 크기가 작은 비슷한 영역부터 먼저 합병된다.

![image](https://user-images.githubusercontent.com/69780812/138699042-6b5f063c-b244-40e0-be37-1ccc8b898107.png)

![image](https://user-images.githubusercontent.com/69780812/138699134-a302b6df-bfb7-4605-9ec9-fd692ef18069.png)

- Fill Similarity
- 2개의 영역을 결합할 때 얼마나 잘 결합이 되는지를 나타낸다.
- BB_ij : r1, r2를 합친 빨간 부분의 Bbox 영역
- BB_ij 크기에서 r1, r2 영역의 크기를 뺐을 때 **작은 값이 나올 수록 2 영역의 결합성(fit)이 좋아지게 된다.**
- **합병을 유도하기 위한 척도**로 보면 된다.
- Fit이 좋을 수록 1에 근접한 값을 얻을 수 있으므로 그 방향으로 합병이 일어나게 된다.

![image](https://user-images.githubusercontent.com/69780812/138699530-42062b5f-145b-4d39-9d75-2589a9ab6a76.png)

- 위의 4개 유사도를 결합시켜서 유사도를 구한다.
- a1~a4는 해당 유사도를 사용할지 말지이다.
---
- SS에서 hierarchical grouping에서 영역 결합 시 무엇을 근거로 하는지를 살펴봤다.
- 다양화를 위해 다양한 컬러 공간을 사용한다.
- 효과적인 영역 병합을 위해 유사도 척도를 사용한다.
- [Selective Search GitHub](https://github.com/belltailjp/selective_search_py)
---

# SS 방식을 사용한 Object Detection 방식
- UvA구조 : SS를 사용하여 후보 영역 추출 뒤 OpponentSIFT 및 RGB-SIFT와 같은 알고리즘으로 Feature 추출 후 디스크립터 생성 후 히스토 그램 방식의 벡터를 SVM을 통해 검출을 수행하는 방식을 사용하고 있다.

![image](https://user-images.githubusercontent.com/69780812/138700899-20fde9a2-9f33-4b0c-b92a-ccb8da0209d7.png)

- 딥러닝 기반의 방식이 아닌 "Feature detection + SVM"이라는 기본 방식을 사용했다.
- 하지만 SS의 성능이 매우 뛰어나 전통적 방식을 사용했음에도 2013년 기준으로는 만족할만한 결과를 얻는다.

![image](https://user-images.githubusercontent.com/69780812/138701511-1651ab47-e409-41c3-9f76-f21b14e338e9.png)

- Uijlings 팀이 Object Detection을 위해 사용한 기본적인 접근법은 SIFT 기반의 Feature descriptor를 사용한 BoW(Bag of words) 모델, 4 레벨의 Spatial pyramid, SVM 분류기 사용이다.

## Bag of Words Model
![image](https://user-images.githubusercontent.com/69780812/138701572-63b87d52-bc4e-4485-ae40-10b1c4a519f4.png)

![image](https://user-images.githubusercontent.com/69780812/138701621-3143bd38-e418-4776-9d74-c84232c0bf91.png)

- 위 사진들의 각 주요 성분들을 나타낸 그림이다.
- 공간적 위치를 특별히 따지지 않더라도, 영상을 구성하는 특정한 성분들의 분포를 이용해 오른쪽 히스토그램으로 나타낼 수가 있게 된다.
- 특성 성분들의 조합이 Object에 따라 확연히 다른 분포를 보인다면 크기, 회전, 조명 등에 영향을 받는 Object 검출은 실제 영상 직접 비교보다는 성분들의 분포만을 이용해  Object 여부를 판별할 수 있게되는 것이다.
- 주요 성분은 영상을 구성하는 Feature라고 볼 수 있다.
  - 이 특징들의 집합이 Codebook이 된다.
  - Codebook에 있는 각각 의 성분들의 크기나 분포가 Object를 분류나 검출 하는 수단이 된다.
  - 이런 방식을 BoW라고 부른다.
  - Computer Vision, 자연 처리 등에 널리 사용된다.
- [BoW 참고](https://cs.nyu.edu/~fergus/teaching/vision_2012/9_BoW.pdf)
- 영상을 구성하는 주요 성분, Codebook은 어떻게 만들어 내는가 ?
  - 많이 사용하는 방식 : SIFT
  - SIFT를 통해 Descriptor를 만들어내 이 것의 분포를 벡터화하면 된다.

## Feature Representation
- 딥러닝이 아닌 전통적인 방식에서는 특징을 잘 추출하고 나타내는 것이 매우 중요하다.
- SVM을 사용하므로 good과 fail을 잘 구별할 수 있도록 양질의 Vector를 추출하는 것이 매우 중요하다.
- SIFT는 속도는 느리지만 local feature를 이용한 패턴 매칭에는 탁월한 성능을 보이는 알고리즘이다.
- SIFT는 기본적으로 Intensity정보의 gradient 방향성에 대한 히스토그램 특성을 구해 이 것을 128차원 Vector로 표시한다.
- 기존은 흑백에서만 사용가능했지만 HSVSIFT, OpponentSiFT 등 많은 SIFT가 나고오고있다.
- "Evaluation of Color Descriptors for Object and Scene Recognition"

## Training
- "BoW + SVM"방식은 2단계 학습 과정을 거친다.
- 경계면 근처의 벡터들을 잘 활용 해줘야 한다.

![image](https://user-images.githubusercontent.com/69780812/138703257-0218a9d7-4d93-4913-a88a-b4af814ec6dd.png)

- 초기 학습
  - ground truth 데이터로 부터 Positive example을 구하고 추천된 후보 영역중 ground truth 데이터와 overlap이 20~50% 수준에 있는 경계 근처의 데이터를 negative example로 정하여 SVM을 학습시킨다.

![image](https://user-images.githubusercontent.com/69780812/138703490-75bf0837-ebd3-43b4-ae99-a15a1c0999cd.png)

- 학습 결과 튜닝
  - 초기 학습을 마친 뒤 SVM을 튜닝 시키는 과정을 거친다.
  - False Posivie를 찾아내고 추가로 Negative Example을 더해서 SVM을 더 정교하게 튜닝하는 과정을 거친다.

## 실험 결과
![image](https://user-images.githubusercontent.com/69780812/138703693-1b4d167f-dc4d-41ea-8bbc-09d90320e760.png)

- ABO(Average Best Overlap)을 척도로 사용했다.
- ground data와 최고 Overlap에 대한 평균을 구한 것이다.

![image](https://user-images.githubusercontent.com/69780812/138703826-f6d56f1c-76c0-46d0-9390-0d4fc9dfd3c9.png)

- 녹색이 ground truth, 빨간색이 알고리즘 결과다.
- ABO 수치가 높을 수록 결과가 좋다.

![image](https://user-images.githubusercontent.com/69780812/138703910-31f219ba-2447-42bd-8530-c3d75ee8f595.png)

- 다양한 전략을 섞어 사용한 경우가 결과가 좋게 나오는 것을 확인할 수 있다.
- Fast는 연산시간에 중점을 두고 Quality는 성능에 중점을 둔것인데, 연산시간이 4배이상 차이 난다.

![image](https://user-images.githubusercontent.com/69780812/138704053-43c1fc2b-bba6-4615-a634-a1fdef1cf9ba.png)

- 앞서 발표된 다른 논문들과 비교헀을 때도 SS가 뛰어난 결과를 보이는 것도 알 수 있다.