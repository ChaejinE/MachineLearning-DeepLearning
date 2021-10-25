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
