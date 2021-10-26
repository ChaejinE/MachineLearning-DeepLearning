# Classfication 기반 망을 Semantic segmentation에 적용할 때 문제점
- Classification or Detection은 기본적으로 대상의 존재 여부에 집중한다. (Object centric)
  - conv + pooling 을 거쳐 영상 속 존재하면서 변화에 영향받지 않는 강인한 개념만을 끄집어내야한다.
  - Detail 보다는 Global한 것에 집중한다.
  - Semantic Segmentation은 픽셀 단위의 조밀한 예측이 필요하므로 Detail 정보를 얻는 데 어려움이 있다.
- FCN 개발자들은 skip layer를 사용해 detail이 줄어드는 문제를 보강했다.
- dilated convolution 팀은 dilated convolution(atrous convolution)을 사용하여 Receptive Field를 확장시키며 detail이 사라지는 것을 커버했다.
- 1/8로 1/32보다 더 높은 해상력까지만 줄이는 방법도 사용했는데, 1/8이더라도 bilinear interpolation을 통해 상대적으로 detail이 살아있겠지만 여전히 정교함이 떨어진다.

# Atrous Convolution
![image](https://user-images.githubusercontent.com/69780812/138867510-5636a296-0f77-4efd-a234-534f013c4f96.png)

- Atrous Convolution이란 Wavelet을 이용한 신호 분석에 사용되던 방식이다.
  - 보다 넓은 Scale을 보기 위해 중간에 hole(zero)를 넣고 Convolution을 수행하는 것을 말한다.
- (b)는 확장 계수 (k)가 2인 경우다. 대응하는 영역의 크기가 커졌음을 알 수 있다.
  - 이처럼 atrous convolution(dilated convolution)을 사용하면 kernel 크기는 동일하게 유지해서 연산량은 동일하지만 Receptive Field의 크기가 커지는 효과를 얻을 수 있다.
- 영상 데이터와 같은 2D에 대해서도 좋은 효과가 있다.

# Atrous Convolution 및 Bilinear interpolation
- DeepLab V2의 뒷단을 FCN이나 Dilated convolution 팀과 마찬가지로 bilinear interpolation을 이용해 원 영상을 복원했다.

![image](https://user-images.githubusercontent.com/69780812/138870899-a69a27b7-2f64-4a7c-9f0f-57becf8f26f2.png)

- bilinear interpolation 만으로는 정확하게 픽셀단위 까지 정교하게 Segmentation을 한다는 것이 불가능 하므로 뒷부분은 CRF(Conditional Random Field)를 이용해 Post-Processing을 수행하도록 했다.
  - 결과적으로 DCNN + CRF 형태이다.

# ASPP(Atrous Spatial Pyramid Pooling)
![image](https://user-images.githubusercontent.com/69780812/138871110-e986b0dd-4b5b-4fe9-ae28-846cee1c3189.png)

- DeepLab v2는 v1과 달리 **Multi-scale에 더 잘 대응하도록** atrous convoltion을 위한 확장 계수를 {6, 12, 18, 24}로 적용하여 그 결과를 취합했다.
- SPPNet 논문에 나오는 Spatial Pyramid Pooling 기법에 영감을 받아 이름을 저렇게 지었다고 한다.
- 확장 계수를 6 ~ 24 까지 변화 시키면서 다양한 receptive field를 볼 수 있게 됐다.

![image](https://user-images.githubusercontent.com/69780812/138871269-1b271715-91d4-41c8-9541-40cc0ecaa80f.png)

- 구글의 인셉션 구조와 비슷하다.
  - 인셉션 구조도 여러 Receptive field의 결과를 같이 볼 수 있게 되어 있다.
- 논문 저자들은 실험에 따르면, 확장 계수를 12로 고정 시키기 보다 ASPP를 지원함으로써 약 1.7% 성능을 개선할 수 있게 되었다.

![image](https://user-images.githubusercontent.com/69780812/138871474-76808259-2689-49a6-acae-ed71653274bb.png)

- 성능을 나타내는 표이다.
- ASPP-S : 좁은 Receptive Field를 대응할 수 있도록 작은 확장계수를 갖는 경우
  - {2, 4, 8 ,12}
- ASPP-L : 넓은 Receptive Field를 볼 수 있게 하는 경우
  - {6, 12, 18, 24}
- 결과는 Scale을 고정 시키는 것보다 Multi-scale을 사용하는 편이 좋다.
- 좁은 Receptive Field 보다는 넓은 Receptive Field를 보는 편이 좋다는 사실을 알 수 있다.

# Fully Connected CRF
![image](https://user-images.githubusercontent.com/69780812/138871898-95db27c3-c785-4ab4-988e-f03e4d1df4b9.png)

- Conditional Random Field를 사용한 후보정 작업을 해주면 좀 더 결과가 좋아진다.

## 왜 CRF가 필요한가 ?
- 기존 skip connection(layer) or dilated convolution을 사용하더라도 분명히 한계를 존재하기에 후처리 과정으로 픽셀 단위 예측의 정확도를 더 높일 수 있다.
---
- 일반적으로 좁은 범위(short-range) CRF는 Segmentation 수행 후 생기는 Segmentation Noise를 없애는 용도로 많이 사용되었다.
- 하지만 기존 DCNN에서 여러 conv+pooling으로 크기가 작아지고 upsampling을 통해 원영상으로 확대하여 충분히 부드러운 상태이므로 기존처럼 Short range CRF를 적용하면 결과가 더 나빠진다.
- 이것에 대한 해결책으로 "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials", Philipp Krahenbuhl의 논문에서 기존 short-range CRF 대신 전체 픽셀을 모두 연결(fully connected)한 CRF 방법이 소개되었다.
  - [참고1](http://swoh.web.engr.illinois.edu/courses/IE598/handout/fall2016_slide15.pdf)

![image](https://user-images.githubusercontent.com/69780812/138872649-338aca26-d798-47d4-bb49-d62a9fbfc2bb.png)

- 기존 short-range CRF는 위 그림 처럼 local connection정보만을 사용한다.
  - detail 정보를 얻을 수 없다.

![image](https://user-images.githubusercontent.com/69780812/138872775-11c3ae29-63b2-488e-8aa4-c75764f21d79.png)

- Fully connected CRF를 사용하게 되면 detail이 살아있는 결과를 얻을 수 있다.
- 위 처럼 MCMC(Markov Chain Monte Carlo)방식을 사용하면 좋은결과를 얻지만 시간이 오래걸려 적용이 불가능하다.
- 하지만, 위에서 소개된 논문에서는 이 것을 0.2초 수준으로 효과적으로 줄일 수 있는 방법을 개발했다.
  - mean field approximation을 적용해 message passing을 사용한 iteration방법 적용
  - 효과적인 Fully connected CRF 수행
- mean field approximation
  - 복잡한 모델 -> 더 간단한 모델을 선택하여 설명하는 방식
  - 수많은 변수 -> 복잡한 관계의 상황 -> 특정 변수와 다른 변수들의 관계의 평균 -> 평균으로 부터 변화(fluctuation) 해석에 용이 -> 평균으로 단순화된 근사된 모델을 사용하면 전체를 조망하기에 좋아진다.

![image](https://user-images.githubusercontent.com/69780812/138873273-2e71de12-d53f-4d7b-b89a-7f2d40583061.png)

- CRF 수식이며 unary term, pairwise term으로 구성된다.
- x : 픽셀위치에 해당하는 픽셀 label
- i, j : 픽셀의 위치
- Unary term : CNN연산을 통해 얻을 수 있다.
- Pairwise term : bi-lateral filter에서 그러듯 픽셀값의 유사도와 위치적 유사도를 함께 고려
- 2개의 가우시안 커널로 구성
  - 각 표준편차들을 통해 scale을 조절할 수 있다.
  - 첫번째 가우시안 커널 : 비슷한 위치 비슷한 컬러를 갖는 픽셀들에 대해 비슷한 label이 붙을 수 있도록 해준다.
  - 두번째 가우시안 커널 : 픽셀의 근접도에 따라 Smooth 수준을 결정한다.
- pi, pj : 픽셀의 position
- Ii, Ij : 픽셀의 intensity(컬러값)
- 위를 고속처리 하기 위해 feature space에서는 Gaussian convolution으로 표현할 수 있게되어 고속 연산이 가능해진다.

![image](https://user-images.githubusercontent.com/69780812/138873817-044209b7-171e-4b0c-9211-28c770295f6e.png)

- DeepLab의 동작 방식이다.
- DCNN으로 1/8 크기의 coarse score-map을 구한다.
- 이것을 bi-linear interpolation으로 원영상 크기로 확대한다.
- 이 결과는 각 픽셀 위치에서의 label에 대한 확률이 된다. 즉, CRF의 unary term에 해당한다.
- 최종적으로 모든 픽셀위치에서 pairwise term까지 고려한 CRF 후보정 작업을 해주면 최종적 출력 결과를 얻을 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138874481-4cbcc801-9acf-4890-a337-84d030a9b1dc.png)

- 확실히 CRF를 수행하면 detail이 상당히 개선될 수 있음을 알 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138875167-9dcca65a-6a7c-4ac1-acde-a7a629cd6bbd.png)

- PASCAL VOC2012 데이터셋에 대한 실험이다.
- ResNet-101에 CRF를 적용하면 평균 IOU rk 79.7% 수준으로 매우 높다는 것을 확인할 수 있다.