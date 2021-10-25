# Overview
- 영상의 중요한 응용분야 중 하나
- 영상에서 의미있는 부분으로 구별해내는 Segmentation
- Thershold나 Edge 기반의 단순 구별이 아니라 영상에서 의미있는 부분으로 구별해내는 기술을 Semantic Segmentation이라고 한다.
- Semantic Segmentatio은 Bbox로 대강의 영역을 표시하는 것이 아닌 정확하게 개체의 경계선까지 추출하여 영상을 의미있는 영역으로 나눠주는 작업을 해주므로 매우 어렵다.

# Gestalt 시지각 이론
- Gestalt : "형태, 모양"을 뜻하는 독일어
- 이 이론에 따르면 개체를 집단화할 수 있는 5개 법칙이 있다.
  - Law of Grouping : 집단화의 법칙

![image](https://user-images.githubusercontent.com/69780812/138641772-e302a18f-6197-4755-a554-701b487915bd.png)


1. Similarity(유사성) : 모양이나 크기 색상 등 유사한 시각 요소들 끼리 그룹을 지어 하나의 패턴으로 보려는 경향, 다른 요인이 동일하다면 유사성에 따라 형태는 집단화

2. Proximity(근접성) : 시공간적으로 서로 가까이 있는 것들을 함께 집단화 해서 보는 경향

3. Commonality(공통성) : 같은 방향으로 움직이거나 같은 영역에 있는 것들을 하나의 단위로 인식, 배열이나 성질이 같은 것들끼리 집단화 되어 보이는 경향

4. Continuity(연속성) : 요소들이 부드럽게 연결될 수 있도록 직선 혹은 곡선으로 서로 묶여 지각되는 경향

5. Closure(통페합) : 기존 지식을 바탕으로 오나성되지 않은 형태를 완성시켜 지각하는 경향

- Gestalt이론은 이런 인지심라헉적인 면에 대한 고찰의 결과이다.

# Image Segmentation
![image](https://user-images.githubusercontent.com/69780812/138641935-cf59215b-a343-4bf4-8d92-3ab1fd993d45.png)

- Segmentation의 목표
  - 영상을 의미적인 면이나 인지적인 관점에서 서로 비슷한 영역으로 영상을 분할하는 것
- Segmentation의 접근 방법에 따라 크게 3가지 방식으로 분류 가능하다.
  - 1. 픽셀 기반
    - thresholding에 기반한 방식으로 histogram을 이용해 픽셀들의 분포를 확인 -> 적절한 thr설정 -> 픽셀 단위 연산을 통해 픽셀 별로 나누는 방식이며 이진화에 많이 사용된다.
    - thresholding 으로 전역(global) or 지역(local)로 적용하는 영역에 따른 구분 가능
    - 적응적(adaptive) or 고정(fixed) 방식으로 경계값을 설정하는 방식에 따른 구별 가능
  - 2. Edge 기반 방법
    - Edge 추출 Filter 등을 사용해 경계 추출 -> non-maximum suppression과 같은 방식을 사용하여 의미있는 Edge와 아닌 Edge를 구별한다.
  - 3. 영역 기반 방법
    - Thresholding or Edge 기반 방식은 의미 있는 영역으로 구별이 쉽지 않다. 특히, 잡음 환경에서 매우 어렵다.
    - 기본적으로 homogeneity(동질성) 기반 이므로 다른 방법보다 의미있는 영역으로 나누는데 적합하다.
    - 하지만, homogeneity 규정 Rule을 어떻게 할 것인가가 관건이 된다.
    - 흔히 Seed라고 부르는 몇 개의 픽셀에서 시작해 영역을 넓혀가는 region growing 방식이 이에 해당된다.
    - region merging, region splitting, split and merge, waterhsed 방식 등도 있다.

## Image Segmentation 과정
![image](https://user-images.githubusercontent.com/69780812/138646679-52c67984-1155-4ccf-8c17-ab3a03fed3a9.png)

- Bottom-up, Top-down 방식이 있다.
- Bottom-up : 비슷한 특징을 갖는 것들끼리 집단화
- Top-down : 같은 객체에 해당하는 것들끼리 집단화

# 1. Pixel 기반의 방법
- 가장 직관적
- 흔히 사용되는 방식이 Thresholding

![image](https://user-images.githubusercontent.com/69780812/138647377-3312e8c2-f7a2-43bf-a76f-cd268403490a.png)

![image](https://user-images.githubusercontent.com/69780812/138647413-6b426a9b-a1db-44df-8a7e-a77b436ea0f5.png)

- 임계값 1개만 적용한 경우이다.

![image](https://user-images.githubusercontent.com/69780812/138647538-e5c58c66-af23-4c11-8849-1d749e81f937.png)

![image](https://user-images.githubusercontent.com/69780812/138647573-a05ceea7-0641-403d-bbb2-999576d14c2e.png)

- 2개 혹은 그 이상의 Threshold가 필요한 경우가 있다.
- 위 예는 threshold를 T1, T2 두개로 두고 그 사이에있는 경우 흰색, 그 외는 검은색으로 표시한다.
- 생각보다 근사하게 경계를 추출할 수 있음을 확인할 수 있다.

## Threshold 설정 방법
![image](https://user-images.githubusercontent.com/69780812/138647722-6564d1bf-adb1-47d6-9a7e-e2d2f9823d4a.png)

- 임계값이 제대로 설정되지 못하면 엉뚱한 결과가 나온다.
- 가장 많이 사용되는 방식이 픽셀값의 누적 분포를 알 수 있는 히스토그램을 사용하는 것이다.

![image](https://user-images.githubusercontent.com/69780812/138647805-a4974f00-4f34-48f4-bc7e-2fc9bf2d5e90.png)

- 위는 다행스럽게도 Peak가 2개이고 그 간격도 상당히 떨어져 있다.
- 이런 경우 대략적으로 2개의 peak 가운데 값으로 잘라주는 방식으로 쉽게 Thresholding이 가능하다.

### Otsu Algorithm
- 하지만 위와 같은 경우는 드물다. 최적의 T를 결정하는 방법은 무엇일까?
- Otsu 알고리즘은 가장 자연스러운 임계값을 설정할 수 있게 해준다.
  - "A threshold selection method from gray-level histograms" 논문
  - 알고리즘이 그리 복잡하지 않으면서도 결과가 상당히 좋다.
  - 직관이 아니라 자동화가 가능하다는 점이 매우 유용하다.

![image](https://user-images.githubusercontent.com/69780812/138648172-0f9d433f-927d-4183-a029-ef4e423b494d.png)

- 쌀의 경계를 추출하는 이미지이다.
- Histogram을 보니 Peak의 경계가 명확하지 않다.

![image](https://user-images.githubusercontent.com/69780812/138648296-466fbc8d-38b5-437d-86cd-3c34f6368f4a.png)

- 임의의 임계값 T 를 기준으로 작은 픽셀의 비율을 q1
- 같거나 큰 픽셀의 비율을 q2
- T를 기준으로 만들어진 2개의 그룹에 대해 평균과 표준편차를 구한다.
- Otsu 알고리즘의 핵심은 2개의 그룹에 대해 **그룹 내 분산을 최소화**하거나 **그룹 간 분산을 최대화**하는 방향으로 T를 정한다는 간단한 아이디어다.

![image](https://user-images.githubusercontent.com/69780812/138648584-ee0dec4a-391b-4068-a07c-d8ccbe6df332.png)

- 6개의 T값이 있다고 하면 T=3인 경우가 Otsu 알고리즘 상으로는 최적의 임계값이 되는 것이다.
- 그룹 간 분산을 구하는 식이 좀 더 쉬워서 그룹간 분산 방식을 더 많이 사용한다고 한다.
  - 그룹 내 분산 최소화와 그룹 간 분산 최대화는 같은 의미를 갖는다.

## Global Or Local
![image](https://user-images.githubusercontent.com/69780812/138648838-7bbb3d39-7480-40db-a5fd-655d3af91328.png)

- 조명, 특정 Noise로 인해 문제가 되는 경우가 많다.
- 왼쪽 그림은 lense shading 문제로 중심은 밝고 주변은 어둡다.
- 그림 전체에 대해 동일한 임계값을 적용하는 경우 중간에 있는 그림과 같이 문제가 생긴다.
- 전체 영역을 여러 개로 나누고 각각의 영역에 대해 적절한 Thresholding을 하는 경우 오른쪽 그림과 같이 좋은 결과를 얻을 수 있다.
  - 이처럼 여러 영역으로 나누고 각각에 대해 다른 Thresholding을 적용하는 것을 **local thresholding**이라고 부른다.
---
- Pixel 기반의 Segmentation 알고리즘은 실제 비전 알고리즘 응용 분약에서 영상의 경계를 추출하거나 Binarization을 위해 흔히 사용된다.
---

# 2. Edge 기반의 방법
- 복잡하지 않으면서도 엣지를 중요한 특징으로 효과가 매우 좋아 많이 사용된다.
- 하지만, 영상에 잡음이 있거나 부분적으로 가려지는 현상 발생 시 성능이 떨어지는 문제가 있다.
- Edge 기반은 Edge 검출(엣지에 있는 픽셀 찾는 과정), Edge 연결(엣지에 있는 픽셀들을 연결하는 과정) 크게 두가지로 이루어진다.
  - Edge 검출 : Gradient, Laplacian, LoG, Canny filter 등
  - Edge 연결 : local processing(Gradient 방향과 크기가 미리 정한 기준 만족하는 경우 인접 픽셀 연결), global processing(Hogh transform)

## Gradient
- Edge : 영상에 존재한 불연속적인 부분
  - 밝기, 컬러, 무늬(Texture) 등
- Edge 검출은 방법이 매우 많고 흔히 공간 필터를 많이 사용한다.

![image](https://user-images.githubusercontent.com/69780812/138649777-ed5c4f8d-aed9-435b-9910-38dcb1b8d543.png)


- Spatial filter(공간필터)
  - Gradient 추출
  - Gradient는 가장 변화의 속도가 빠른 방향을 나타내게 된다.

![image](https://user-images.githubusercontent.com/69780812/138649897-f8993c20-3138-4714-8778-25fd389a0208.png)

- Gradient는 변화의 세기(크기)와 가장 변화 속도가 빠른쪽(방향)을 가지며 위의 식과 같다.

## Edge 검출 Filter(1차, 2차 미분)
![image](https://user-images.githubusercontent.com/69780812/138650067-08651271-1032-4a1f-ba36-07777e826649.png)

- Edge 검출 : 불연속한 부분을 찾아내는 것이므로 흔히 1차, 2차 미분을 이용한다.
- 1차 alqns : Prewitt, Robert, Sobel 등
- 위 그림은 Sobel Filter의 수식이다.

![image](https://user-images.githubusercontent.com/69780812/138650194-dcc9ef58-1376-4a51-9690-5ae1afe3c5de.png)

- 2차 미분을 사용하는 대표적 Spatial Filter : Laplacian이며 수식은 위와 같다.

![image](https://user-images.githubusercontent.com/69780812/138650314-32ae4128-2901-4eff-a87b-cbc6a22339fe.png)

- 잡음이 많은 경우 위 그림처럼 미분을 하더라도 엣지 검출이 어렵다.

![image](https://user-images.githubusercontent.com/69780812/138650403-61c539da-876f-476c-9de2-1a5aa9daa4eb.png)

- 효과적 엣지 검출을 하려면 사전에 Smoothing 필터 등을 통해 잡음의 영향을 줄이는 것이 필요하다.
- 위 그림은 데이터에 대해 가우시안 필터를 적용한 후 엣지 검출을 보여준다.

![image](https://user-images.githubusercontent.com/69780812/138650572-dcc95f6f-7f1d-4e8b-b02e-1af62b048674.png)

- 가우시안 적용으로 잡음 제거 후 2차 미분 필터인 라플라시안 필터를 적용한 것이다.
- 큰 엣지를 비교적 쉽게 찾아 낼 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138650733-50922d96-1df3-4c41-ab4f-ea3f9eb8758a.png)

- 가우시안과 LoG 필터의 커널은 위 그림과 같다.
- LoG는 잡음이 있는 환경에서도 엣지를 잘 찾아 여러 곳에서 많이 쓰인다.
- 가우시안 필터는 sigma 값에 따라 smoothing을 적용하는 범위가 달라지게 되며 SIFT(Scale Invariant Feature Transform)에서는 연산량이 많은 LoG대신 DoG(Difference of Gaussian)를 사용하기 도한다.

![image](https://user-images.githubusercontent.com/69780812/138650930-07170f12-6a09-4a39-8a17-1807290bf07b.png)

- 오른쪽이 5x5 가우시안 필터를 적용한 영상이다.
- 영상의 Detail이 많이 무너진다.

![image](https://user-images.githubusercontent.com/69780812/138651129-98cd52b7-082f-4b32-9243-02b34f6d363d.png)

- 두 그림의 차, Difference를 이용하면 DoG된 영상을 구하면 엣지가 잘 검출되는 것을 확인할 수 있다.

## 검출된 엣지 후처리 (Post-Processing)
- Spatial Filter를 통해 검출된 Edge는 바로 사용하기 부적절하므로 Post Processing이 필요하다.
  - 개체의 경계 부분만 검출되면 좋겠지만 경계가 아닌 부분도 같이 검출된다.
  - 후처리 과정에서 경계에 해당하는 부분에서 검출된 값이 작아 사라지는 경우도 있다.
  - LoG의 장점에도 불구하고 경계 부분이 Smoothing 되면서 사라질 수 있어 정확한 외곽선 검출이 어려울 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138651338-ff1915f2-8489-498b-b675-f0339dce8482.png)

![image](https://user-images.githubusercontent.com/69780812/138651439-89a3387d-2ad2-48a8-bb01-fc6f8f9ad95a.png)

- 아래 그림은 위 그림에 비해 상태가 안좋다.
- 원영상에 작은 Detail이 너무 많아 Gradient를 구하더라도 상당히 어렵다.
  - 아래 그림의 오른쪽은 공간필터를 이용한 Gradient 크기 영상이다.

![image](https://user-images.githubusercontent.com/69780812/138651824-7a02f44c-52d9-4d72-9ed9-cc5e0b81b7a8.png)

- gradient 크기 영상으로 부터 좀 더 의미있는 Edge를 추출하려면 추출된 Edge에 대해 Thresholding을 적용해야 한다.
- Thresholding을 달리했을 때, 왼쪽이 좋아보이지만 뭔가를 얻어 내기에는 쉽지가 않다.

![image](https://user-images.githubusercontent.com/69780812/138651995-5cd5c8cf-611a-4b63-a6e7-00b37a1b6eeb.png)

- 대표적 후처리 과정
  - 1. Thresholding : 일정 크기 이하의 엣지 제거하여 큰 엣지 위주로 정리
  - 2. Thinning : non-maximum suppression 적용하여 굵은 Edge를 얇게 처리
  - 3. Hysteresis thresholding : high와 low 2개 threshold를 사용
    - High 엣지에 연결된 Low엣지는 제거하지 않고 살려둔다.
- 아래 왼쪽 그림은 작은 Threshold를 적용해 많은 작은 엣지가 살아있다.
- 오른쪽은 높은 Threshold로 엣지가 많이 사라졋다.
- 아래는 Hysteresis Thresholding을 적용했을 때이다.
  - 두가지 Threshold 장점을 취할 수 있게 된다.
- 후처리 과정 및 장점들은 Canny Edge Detector에 거의 대부분 적용된다.

### Edge Thining
- 검출한 Edge를 1 Pixel 크기로 얇게 만든다.
- Edge 검출 전 보통 잡음 제거를 위해 median or gaussian smoothing filter를 적용한다.
  - 이 후 Edge 검출을 하면 선폭이 1픽셀 이상이되므로 적절한 방법으로 얇게 만들어 줘야한다. : **Thinning**

![image](https://user-images.githubusercontent.com/69780812/138652952-c98113f4-02b1-4b37-97d7-37482936711c.png)

- 4방향 or 8방향 방법을 주로 사용한다.
- 중심 픽셀의 Edge 값과 주변 픽셀의 Edge 값을 비교하여 중심 픽셀이 가장 큰 값은 경우는 남겨 놓고 그렇지 않은 경우는 제거하는 방식이다.
  - non-maximum suppression이라고도 부른다.

![image](https://user-images.githubusercontent.com/69780812/138653142-8779bc50-0d25-4eed-a113-0fce3fce8318.png)

- 좌상단 부터 3x3 sliding window를 적용시키면서 3x3 윈도 중심값 보다 큰 주변값이 있으면 중심값을 계속 0으로 바꿔가며 진행한 결과이다.

## Canny Edge Detector
- 1986, "A Computational Approach to Edge Detection", J.Canny
- 다른 엣지 검출기에 비해 성능이 뛰어나 가장 많이 사용되는 Edge 검출기 중 하나이다.
- Optimal Edge 검출을 위해 좋아야하는 특성 3가지
  - 1. 잡음 제거 or Noise smoothing(평활화)
  - 2. Edge enhancement (Edge 개선)
  - 3. Edge localization (Edge 위치 파악)

![image](https://user-images.githubusercontent.com/69780812/138653661-7201a4c2-bce7-47a6-8738-91e98e22bfe1.png)

- 위 이상적인 Edge에 백색 가우시안 잡음을 추가해보자.

![image](https://user-images.githubusercontent.com/69780812/138653767-66e00b05-ebd9-402e-894a-3f19fbbf2e7a.png)

- 잡음이 있더라도 우수한 엣지 검출력을 보이려면 ?
  - 실제로 엣지가 위치하고 있는 X=0에서 잡음보다 큰 응답 특성이 나와야한다.
  - 실제로 가장 큰 엣지 값이 나와야한다.
  - Noise Smoothing, Edge Localization

![image](https://user-images.githubusercontent.com/69780812/138653919-bc7d2b30-2807-4804-b640-49d4d9de5434.png)

- X=0 일정 근처에서 단 1개의 최고값만을 갖도록 해줘야 한다.
- 다음 Peak와는 어느 정도 거리 확보가 필요하다.
- Edge enhancement
---
- Canny Edge Detector는 위 3가지 특성을 극대화 하기 위한 목적으로 개발되었다.
- 1. Gaussian Smoothing 후 Gradient 크기와 방향을 구한다.
- 2. Non-maximum suppression을 통해 검출된 엣지를 1 픽셀 크게로 얇게 만든다.
- 3. Edge Linking 및 Hysteresis thresholding을 통해 의미있는 엣지만 남긴다.
### Canny Edge 검출 상세 과정
- 1. 잡음 영향 최소화
  - Gaussian Filter -> 영상을 부드럽게
- 2. Gradient
  - x 및 y 방향 미분

![image](https://user-images.githubusercontent.com/69780812/138654505-234896a7-4034-4313-8744-847aba7755c0.png)

- I : Origin Image
- 원래는 가우시안 필터링 후 미분이지만 가우시안 함수의 Convolution 특성 상 가우시안 함수의 미분을 한 뒤 Convolution해도 상관없다고 한다.

![image](https://user-images.githubusercontent.com/69780812/138654733-478dc3b5-370e-439a-9c78-3497eb6c7f46.png)

- 가우시안 함수를 미분한 수식이다.
- 이 식에서 sigma는 연산을 적용할 커널의 크기를 결정한다.
- sigma: scale, smoothing이 되는 정도를 결정하는 변수
  - 커질수록 잡음 엣지를 더 많이 제거
  - 커질수록 엣지를 더 부드럽게하면서 두껍게 만드는 경향
  - 커질수록 섬세한 Detail들을 제거

![image](https://user-images.githubusercontent.com/69780812/138655027-51b4c05e-424e-4ec1-8142-856d1e6e623e.png)

- Smoothing된 영상에 대해 x, y 미분을 구한 그림이다.

![image](https://user-images.githubusercontent.com/69780812/138655123-58792a9a-fdf3-4542-aa2c-8503c4095bb8.png)

- 이 결과를 통해 Gradient의 크기와 방향을 구한다.
- 구해진 Edge를 Thresholding을 통해 임계값 이상만 보여주면 오른쪽과 같아진다.

![image](https://user-images.githubusercontent.com/69780812/138655395-d088f509-c0aa-49ea-ae48-d9f9a74ed800.png)

- Canny Edge에서 Thinning은 앞서 살핀 것과 약간 다르다.
  - Thinning : Non-Maximum Suppression
- 이 전 과정을 통해 Gradient의 방향을 구했으므로 변화하는 방향을 알게되었다.
  - 변화하는 방향으로 스캔하면서 최대값이 되는 부분을 찾는다.
- 왼쪽은 gradient magnitude 영상이다.
- 오른쪽의 직선은 Gradient의 방향이다.
- 이 직선 방향으로 스캔하면서 가장 큰 값만 남기고 픽셀을 모두 0으로 만든다.
  - 그러면 1픽셀 단위의 얇은 선이 만들어진다.

![image](https://user-images.githubusercontent.com/69780812/138655783-28d3be74-244b-4f23-b554-d8c0239b5963.png)

- Thinning을 적용하면 위와 같이 얇은 선을 얻을 수 있고, 이전과 동일하게 특정 임계값에 대해 Thresholding을 한 결과이다.

![image](https://user-images.githubusercontent.com/69780812/138656019-a0a1dcf7-8e6c-405d-aff5-34efc76f702d.png)

- 위 그림은 Hysteresis thresholding을 사용한 경우이며 적절하게 엣지가 살아있는 것을 확인할 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138656186-cc812972-e11a-47b4-98b0-56f8ff056551.png)

- Canny Edge에서 sigma의 영향에 대해 알아보자.
- scale 변화를 지정하는 파라미터이며 이 변화에 따라 검출되는 엣지의 모양이 달라지는 것을 보여주는 그림이다.
- sigma가 클수록 강한 Edge가 검출되고, 작아지면 섬세한 특징을 보기에 좋다.
- sigma가 커지면 엣지 검출은 잘되지만 엣지의 위치가 Smoothing에 의해 달라질 수 있으므로 엣지 검출과 정확도에 Trade-Off 관계가 있다는 점도 이해해야한다.

## SUSAN Edge Detector
- 1995, Smallest Univalue Segment Assimilating Nucleus의 약어, S.M Smith & J.M Brady
  - "SUSAN - A New Approach to Low level Image Processing"
- low-level 영상 처리에 대한 새로운 접근법 제시
- Edge 검출 및 잡음 제거에도 꽤 괜찮은 성능을 보인다.
- 대부분 Edge 검출 알고리즘은 미분을 사용하지만, SUSAN에서는 사용하지 않는다.
- 또한, 3x3, 5x5 같은 정방향 윈도우 혹은 마스크를 사용하지만 SUSAN에서는 원형 또는 근접한 원형 마스크를 사용한다.
- USAN(Univalue Segment Assimilating Nucelus)를 먼저 이해해야하고, Nucleus는 마스크의 중심에 있는 픽셀값을 의미한다.

![image](https://user-images.githubusercontent.com/69780812/138658431-18e3901d-fe73-4fe9-99f1-69f7fdd8700e.png)

- 원은 마스크 영역을 나타내며 "+" 부호의 위치에 있는 픽셀이 마스크의 Nucleus(중심)이 된다.
- 밝은색 영역과 어두운 색으로 구성된 간단한 위 그림에 5가지 종류의 마스크 형태를 보여준다.
- 'e'는 Nucleus와 주변 픽셀이 모두 같은 경우로 Edge나 Corner가 아니다. 나머지 'a' ~ 'd'는 edge나 corner가 마스크에 포함된 경우이다.
- USAN : 전체 마스크에서 Nucleus와 같은 혹은 비슷한 값을 갖는 면적을 뜻한다.
  - 이 면적을 살펴 평탄한 지역에 있는지, 엣지나 코너 영역에 있는지 파악이 가능하다.

![image](https://user-images.githubusercontent.com/69780812/138658898-7a95fb48-a325-4bfb-b3b6-63b294526bfe.png)

- Nucleus와 같은 부분은 흰색으로, 다른 부분은 검은색으로 표시한 것이다.

![image](https://user-images.githubusercontent.com/69780812/138658997-73d0bf76-2280-40c3-b47a-66f541948125.png)

- Edge나 Corner 검출에는 위와 같은 기본 식이 적용된다.
- r0 : Nucleus
- r : 마스크 영역에 있는 다른 픽셀 위치
- I : 픽셀의 밝기(Intensity)
- t : 유사도를 나타내는 threshold 값
- 중심에 있는 픽셀의 밝기와 임계값 이내로 비슷한 경우에는 1, 그 이상 차이나면 0이되는 비선형 필터이다.
- c : 결과적으로 전체 마스크 내에서 중심과 임계값 범위에 있는 픽셀의 개수
- 흔히 SUSAN 에서는 마스크 반지름 3.4를 많이 사용하며 마스크에 있는 픽셀의 개수는 37개 이다. 최소 마스크의 크기는 3x3이다.

### USAN의 의미
![image](https://user-images.githubusercontent.com/69780812/138659399-4992259c-58f8-4b64-8e16-6a87bd3da953.png)

- "part of original image"를 보면 엣지나 코너가 어떻게 분포 되어있는지 확인할 수 있다.
- 이 원영상에 마스크를 적용해 USAN 값을 구하고, 그것을 3차원 그림에서 표시한 그림이다.
- 균일한 영역 : USAN 값이 크다.
- Edge or Corner : USAN 값이 작다.
- Visibility를 위해 USAN 값을 작은 것을 위로 표시하면 코너의 USAN이 가장 작아 Peak를 보이고, 엣지 부분 역시 값이 낮아 산등성이(ridge)처럼 보인다.

### Edge or Corner 파악
- Canny Edge Detector처럼 Thinning을 이용한다.
  - 엣지 방향을 따라가면서 가장 높은 값을 갖는 위치에 엣지가 있다고 보는 것이다.
- 전체 중 엣지의 방향을 다라 가장 낮은 값을 갖는 USAN을 Edge로 보며 특히 더 낮은 값을 코너로 보면된다.
  - Smallest USAN이라는 뜻에서 SUSAN이 된 이유
- 하지만, USAN은 방향을 구하는 부분이 없다.

![image](https://user-images.githubusercontent.com/69780812/138660207-ef56e573-448d-424f-b1be-3ad13ef5147d.png)

- 이에 대한 해답은 위 그림처럼 USAN의 무게 중심을 구하는 것이다.
- 가장 작은 마스크 3x3을 이용해 USAN을 구해 무게중심을 구한다.('o' 위치)
- 무게 중심의 위치와 Nucleus 위치를 비교하면 엣지의 방향을 파악할 수 있게 된다.
  - (a) : Nucelus의 위치가 무게 중심보다 오른쪽, 이런 경우 엣지가 오른쪽에 있다는 뜻이다. (b)는 반대의 경우다.
  - (c) : 무게 중심의 위치와 Nucleus 위치가 같은데 이런 경우 얇은 엣지가 중심에 있는 경우다.
  - 이렇게 엣지 방향을 파악한 뒤 엣지 방향을 스캔하면서 가장 낮은 USAN의 값을 찾으면 거기에 엣지가 있다.

### SUSAN 엣지 검출기의 엣지 검출 성능
![image](https://user-images.githubusercontent.com/69780812/138660946-104c361f-05b0-4a52-9b61-7bcd732e5041.png)

- 다양한 엣지들에 대해 정확하게 검출이 가능한지 확인한 그림이다.
- 1, 2 pixel 폭을 갖는 ridge profile에 대해 정확한 위치 및 두께까지 검출이 가능한 것을 확인할 수 있다.
- Step Edge : 정확하게 스텝이 생기는 위치 파악
- ramp edge : step 부분을 정호가히 파악
- 다른 두가지 엣지 형체도 문제 없었다.
- roof edge 는 검출이 안된다. 이는 Canny Edge 로도 불가능하며 사람이 볼때도 서서히 값이 바뀌므로 엣지로 인식하지 못한다.
- SUSAN 엣지 검출기는 엣지 검출의 성능이 탁월함을 알 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138661433-3c248de7-b1bf-44c5-9416-c11496aa5684.png)

![image](https://user-images.githubusercontent.com/69780812/138661470-5530b38e-44ad-445d-a2b4-ce549f8d7026.png)

- 왼쪽은 t=10을 적용한 SUSAN Edge Detector 이다.
- 오른쪽은 sigma=0.5를 적용한 Canny Edge Detector 이다.
- Canny Edge Detector는 알고리즘 특성상 코너 부분이 연결이 잘안되고 끊어지는 부분이 있다.
- SUSAN 필터의 경우는 코너쪽에서 휘어지는 현상이 나타난다.

![image](https://user-images.githubusercontent.com/69780812/138661711-515b5070-7785-445f-aeee-3e407598ca6c.png)

- 왼쪽이 SUSAN이고, 오른쪽이 Canny다.
- SUSAN의 경우 동그라미를 정확하게 검출하지만 Canny에서는 방향성 문제로 원 모양이 좀 찌그러진다.
---
- 자연영상의 경우 많은 다른 엣지들이 검출되므로 의미있는 엣지만 남기는 것은 검출기 성능을 넘어서는 부분이다.
- SUSAN 엣지 검출기의 확실한 장점은 미분 연산을 수행하지 않아 속도가 빠르다.
- Canny 역시 sigma를 조절하면 잡음의 영향을 최소화할 수 있지만 엣지의 정확한 위치가 옮겨가는 문제가 있었다.
  - 하지만 SUSAN은 잡음에 강인한 특성이 있다고 한다.(논문)
---

# 3. Region 기반 방법
- Edge 방식은 직관적이라 어렵지 않지만 검출된 많은 엣지중 대체 어떤게 Segmentation에 필요한 것인지 판단하는 것은 호락호락한 문제가 안디ㅏ.
- Thinning, Linking으로 의미있는 엣지만 추출하기도 쉽지 않고, Hough Transform이나 기타 알고리즘들을 사용하더라도 자연 영상에 대해 사람이 느끼는 것처럼 의미있는 boundary만 검출하기는 쉽지 않다.
- 보편적인 방법도 없었다.

![image](https://user-images.githubusercontent.com/69780812/138662466-49ddb23b-881c-460b-b7f9-348c4804b42c.png)

- 위 그림만 보더라도 사람이 구분하는 Segmentation 방법과 엣지 기반의 방식이 많이 차이가 있다.
  - 한계가 있다.

## Region 기반 Segmentation
- 엣지 기반 Seg : 영상에서 차이 difference 부분에 집중
  - outside - in
- Region 기반 Seg : 영상에서 비슷한 속성을 갖는 부분, similarity에 집중
  - inside - out
  - 비슷한 속성 : 밝기(intensity, gray-level), Color, Texture 등
  - 비슷한 속성이라 함은 기본적으로 인지가 가능한 객체의 경계를 다른 것들과 구별짓는 그 무엇이라고 할 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138663029-d7597e2f-f830-4183-adfd-0013fd844a84.png)

- 극단적인 이미지 인데 확대하면 비슷한 밝기로 나비와 육각형 모향이 있다.

![image](https://user-images.githubusercontent.com/69780812/138663117-d44468b9-7828-4247-a54f-ab522085acda.png)

- Canny 엣지와 평균과 표준편차를 이용해 구별하는 것에 대한 결과이다.
- 표준 편차를 객체를 구별할 수 있는 속성으로 이용하면 구별이 되는 것을 확인할 수 있다.
- 객체를 구별할 중요한 속성을 잘 선정하면 엣지 기반 방식에서 거의 불가능한 것을 구별할 수도 있다.

## Region을 정하는 방법
1. Region Growing
2. Region Merging
3. Region Splitting
4. Split and Merge
5. Watershed
etc ..

### Region Growing
- 가장 많이 사용하는 방식
- 기준 픽셀을 정하고 기준 픽셀과 비슷한 속성을 갖는 픽셀로 영역을 호가장해서 더 이상 같은 속성을 갖는 것들이 없으면 확장을 마치는 방식이다.

![image](https://user-images.githubusercontent.com/69780812/138663603-d18a0793-91d5-49d3-824e-b55479dcb10e.png)

- 임의의 픽셀을 seed로 정한다. -> 같은 속성(여기서는 Color)를 갖는 부분으로 확장을 해나간다. -> 최종영역 구별
- Seed(시작픽셀)을 지정하는 방식은 보통 3가지 방식을 사용한다.
  - 1. 사전에 사용자가 Seed 위치를 지정
  - 2. 모든 픽셀을 Seed라고 가정
  - 3. 무작위로 Seed위치를 지정
- Seed 픽셀로 부터 영역을 확장하는 방식도 여러 방식이 사용되고 있다.
  - 1. 원래 Seed 픽셀과 비교
    - 영역 확장 시 원래 Seed 픽셀과 비교하여 일정 범위 이내가 되면 영역을 확장하는 방법
    - 잡음에 민감
    - Seed를 어느 것으로 잡느냐가 결과에 크게 영향을 끼치는 경향이 있다.
  - 2. 확장된 위치의 픽셀과 비교
    - 원래 Seed 위치가 아니라 영역이 커지면 비교할 픽셀의 위치가 커지는 방향에 따라 바뀌는 방식
    - 조금씩 값이 변하는 위치에 있더라도 같은 영역으로 판단이 된다.

    ![image](https://user-images.githubusercontent.com/69780812/138665582-c41109a9-fbcd-4292-8fbd-a11a613c5c27.png)

    - 한쪽으로만 픽셀값의 변화가 생기게 되면 Seed와 멀리 있는 픽셀 값 차이가 많이 나더라도 같은 영역으로 처리될 수 있다. (심각한 drift현상)
    - 위 그림에서는 인접 픽셀만 비교하다보면 전체가 같은 영역이 될 수 있다.
  - 3. 영역의 통계적 특성과 비교
    - 새로운 픽셀이 추가될 때마다 새로운 픽셀까지 고려한 영역의 통계적 특성(평균 등)과 비교하여 새로운 픽셀을 영역에 추가할 것인지 결정
    - 영역 내 포함된 다른 픽셀들이 완충작용
    - 약간의 drift는 있을 수 있지만 안전
    - centroid region growing이라고도 한다.

![image](https://user-images.githubusercontent.com/69780812/138665996-5b6abde0-393f-4174-88c5-62a75a326fd5.png)

- 번개의 가장 밝은 부분을 연결하며 그리고자한다.

![image](https://user-images.githubusercontent.com/69780812/138666051-ee6ab068-27cb-481d-bca1-f2600bd7a61c.png)

- 흑백 영상의 밝기는 0 ~ 255까지 분포 하므로 Seed 값을 255로 설정하면 된다. 그 결과가 위 그림이다.

![image](https://user-images.githubusercontent.com/69780812/138666164-bcc2f8c3-a468-4c88-888d-0ae4011e2808.png)

- 설정된 Seed를 바탕으로 임계값 변경시키면서 region을 확장해 나가면 위와 같은 결과를 얻을 수 있다.
  - 임계값을 어떻게 설정하느냐에 따라 다른 결과를 얻을 수 있음을 확인할 수 있다.

- Region Growing은 처리속도가 빠르고 개념적으로 단순하다.
  - Seed 위치와 영역 확장을 위한 기준 설정을 선택할 수 있다.
  - 동시에 여러 개의 기준을 설정할 수 도 있다.
- 하지만 영상 전체를 보는 것이 아니라 일부만 보는 local method(지역적 방식)이다.
  - 잡음에 민감한 알고리즘이다.
  - Seed 픽셀과 비교하는 방식이 아니면 drift 현상이 발생할 수 있다.

### Region Merging
- merging & splitting은 대표적인 방법
  - 서로 반대 방향으로 움직일 뿐이지 기본 개념은 동일하다고 볼 수 있다.
- 비슷한 속성을 갖는 영역들을 결합시켜 동일한 Label을 달아주는 방식이다.
- region merging은 매 픽셀 단위가 될 수 있다. 일반적으로 심하게 나뉜 영역(over-segmented region)을 시작점으로 한다.
  - 1. 인접 영역을 정한다.
  - 2. 비슷한 속성인지 판단할 통계적인 방법을 적용하여 비교한다.
  - 3. 같은 객체라고 판단되면 합치고 다시 통계를 갱신한다.
  - 4. 더 이상 합칠 것이 없을 때까지 위 과정을 반복한다.
- Region Growing은 Region Merging 방법 중 하나여서 비슷하다고 볼 수 있다.
  - Region Growing은 1개 혹은 적은 수의 Seed를 사용하는 방식이며 픽셀 단위로 판단한다는 점만 차이가 있다.
  - Region Merging은 영역을 기본 단위로 하고 가장 작은 영역은 픽셀이므로 픽셀을 기본 영역으로 볼수도 있다. 그림 전체에 여러 Seed를 사용한다고 볼 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138667394-6cf0ab9a-616f-48c3-a99a-37b3beeefd3f.png)

![image](https://user-images.githubusercontent.com/69780812/138667473-f2d0b4f5-4976-442d-bbcc-58c9216bee58.png)

- 일단 매 픽셀에 각각의 label을 할당하고, 이것을 기반으로하여 밝기의 값의 차가 10 이하이면 동일한 영역으로 본다고 가정하고 merging 알고리즘을 적용하여 분리하면 2번째 그림과 같다.

- 흑백 영상의 밝기 값만 고려해서 merging 적용 했고, 단계적으로 변하는 위 영상의 특성으로 사과와 배경이 깔끔히 분리가 되지는 못하지만 어느 정도 구별이 가능하다는 것을 확인할 수 있다.

- Region merging 방법은 **처리되는 순서에 따라 결과값이 달라진다**는 점을 신경써야한다.

![image](https://user-images.githubusercontent.com/69780812/138667959-05b5e08c-3856-4260-be6e-e228433787f0.png)

- 위 실험보다 밝기 단게를 더 크게하고 분리해보면 결과가 다른 것을 보여주는 그림이다.
- 오른쪽은 상하 반전 시킨 상태에서 merging을 적용하고 다시 상하 반전을 시킨 경우다.

### Region Splitting
![image](https://user-images.githubusercontent.com/69780812/138668178-2b935b84-bf10-4343-9fde-4567922dbe0a.png)

- Merging과 정반대 개념이다.
- 그림 전체와 같은 큰 영역을 속성이 일정 기준 벗어나면 쪼개면서 세분화된 영역으로 나누는 방식이다.
- 보통 위 그림 처럼 4개의 동일한 크기를 갖는 영역으로 나눠서 quad-tree splitting 방식을 많이 사용한다.
- 큰 영역을 먼저 4개 영역으로 나누고, 다시 각 영역을 검토하여 추가로 나눠야할 것인지를 결정한다.
- region splitting 방식에서는 해당 영역에서 variance나 최대값과 최소값의 차와 같은 통계 방식의 일정 기준을 설정하여 그 값이 미리 정한 임계 범위를 초과하게 된 영역을 분할한다.

![image](https://user-images.githubusercontent.com/69780812/138668530-31755472-7997-44b1-b08c-5c648b72e09f.png)

- Region Splitting을 적용한 그림이다.
- 동일하지 않은 속성에 대해 Splitting을 하면 사실은 인접한 부분끼리 비슷한 부분이 나올 수 있음에도 Splitting 방식의 원리가 그렇기 때문에 같은 속성을 갖는 부분도 다른 label이 붙게된다.

### Region Split & Merge 방법
- Splitting 방식 만으로는 원하는 결과를 얻기 어려웠다.
- 동일한 영역을 다시 합쳐주는 것이 필요할 것 같다.

![image](https://user-images.githubusercontent.com/69780812/138668773-fb32869e-944c-4292-bf4b-94a6ea6569ff.png)

- Quad-tree splitting 방식을 사용하면 Over-segmentation이 일어난다. 이것을 같은 특성을 갖는 부분끼리 다시 묶어주면 {1, 3, 233}, {4, 21, 22, 24,231, 232, 234} 두개 영역으로 될 수 있다.
  - 이것을 Split & Merge라고한다.

![image](https://user-images.githubusercontent.com/69780812/138669036-159d61d3-dcc1-4aa2-b71b-bf7b345e4508.png)

- split 적용 후 merge 한 결과이다.
- 의미있는 영역으로 구별됨을 확인할 수 있다.
- Split Merge가 Region Merging 보다 좀 더 속도가 빠르다고 한다.
  - quad-tree splitting을 적용하면 비슷한 영역은 통으로 처리가 되기 때문

### Wartershed Segmentation 방식
- 1979, "Use of Watershed in Contour Detection", Serge Beucher, Christian Lantuej
- Region Rowing, Merging, Splitting과 조금 다른 접근 방식이며 비슷한 속성보다는 지형학에서 사용하는 개념들과 비슷한 방식으로 영역을 나눈다.

![image](https://user-images.githubusercontent.com/69780812/138669522-3b540573-b34b-4f04-849e-20771549bf09.png)

- Wartershed
  - 산등성이나 능선처럼 비가 내리면 양쪽으로 흘러내리는 경계에 해당
- Catchment Basin
  - 물이 흘러 모이는 집수구역에 해당
- Wartershed가 기본적으로 영역을 구분해주는 역할을 하므로 Watershed만 구하면 영상을 Segmentation 할 수 있게된다는 것이다.
- Watershed는 gradient를 이용하여 구한다.
  - 영상에서 gradient를 구한다.
    - 엄밀히 gradient의 크기로 구성된 gradient magnitude 영상을 구한다.
  - 위 영상으로 부터 Watershed를 구한다.

![image](https://user-images.githubusercontent.com/69780812/138669939-68cdf0b9-ebc7-4e03-a73d-7ec591bd6a33.png)

- 맨 왼쪽은 gradient를 눈에 잘띄게 relif(부조) 형태로 보여주는 것이다.
- 두 번째 영상은 gradient 크기 영상이다.
  - 밝을수록 gradient 크기가 크다.
- 세 번째 영상은 Gradient 크기를 이용해 Watershed를 구한 것이다.
  - 의미 없는 부분은 제거됨
- 맨 마지막은 첫번째 영상에 비해 한결 간결해진 결과를 부조형태로 다시 보여준 것이다.

![image](https://user-images.githubusercontent.com/69780812/138670424-5c8700a8-6fde-4a6b-bc05-e7110cbbdf69.png)

- Watershed를 구하는 알고리즘은 다양하다.
- 가장 많이 쓰이는 Flooding 알고리즘을 살펴본다.
  - 기본적으로 각각의 Catchment Basin의 최소값(local minima)에 구멍이 있고, 그 구멍에 물이 차오르기 시작한다고 가정에서 출발한다.
  - 물이 일정 수위에 오르게 되면 서로 다른 2개의 Catchment Basin이 합쳐지는 상황이 발생할 수 있다.
    - 이 때는 물이 합쳐지는 것을 막기 위한 dam을 설치한다.
    - 이 dam이 바로 영역을 구별하는 영역의 경계역할을하게 된다.
    - B는 seg2와 seg3은 애초에 2개의 마커를 할당한 경우로 최종적으로 2개가 영역이 분할된다.

![image](https://user-images.githubusercontent.com/69780812/138670789-51cda565-2edf-4bae-be70-b4653c6ce723.png)


![image](https://user-images.githubusercontent.com/69780812/138670858-3fa07e06-1128-4b22-ae1d-4700197d92ce.png)

- 왼쪽은 Sobel필ㄹ터를 사용하여 gradient magnitude 영상을 구하고 그것으로부터 Watershed 알고리즘을 적용해 Segmentation을 사용하면 Over segmentation된 영상을 얻게 된다.
  - Gradient magnitude 영상이 아주 작은 크기로 localize 되는 형태를 보이기 때문이다.
  - Over segmentation 된 영상에 대해서는 region merging 방법을 사용할 수도 있고 다른 방법을 사용할 수 있다.
  - 영상에 존재하는 detail이나 잡음이 gradient를 구했을 때 local minimum을 만들어 내는 것이 원인이다.
  - 사전에 Smoothing 필터를 적용하거나 잡음 제거에 탁월한 Mophology 연산을 통해 Gradient magnitude  영상을 단조롭게 만드는 것이 한 해결책이 될 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138671368-b006e706-93bc-462a-ba18-42174269dff3.png)

- 자연 영상에 watershed 알고리즘을 적용하면 대부분 Over-segmentation 문제가 발생한다.
- 이 Over-segmentation 문제를 피하려면 Marker를 사용하여 segmentation 될 영역을 지정하는 방식이 있다.
  - Marker 지정은 수동이나 자동으로 가능하다.
  - 보통 최종적으로 Segmentation되는 영역의 수와 Marker의 수가 동일하다.
  - Marker 지정 시 Blob 연산을 많이 사용한다.
  - Mophology 연산을 같이 사용하는 것이 일반적 추세다.
- 위 그림에서 오른쪽이 watershed 알고리즘을 적용해 얻은 over-segmentation image다.

![image](https://user-images.githubusercontent.com/69780812/138671932-b65b79de-6746-40bf-b7e8-a75df794cfaf.png)

- blob(binary large object) 이미지에 marker를 할당하면 왼쪽이 된다.
- marker가 할당된 영역에 대해서만 watershed 알고리즘을 적용하면 오른쪽이된다.
- 기본 Watershed 알고리즘에 Computer Vision 알고리즘을 결합해 Over-Segmentation에 관련된 문제를 줄이는 노력을 해나가고 있다.