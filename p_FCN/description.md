# Overview
- FCN : Fully Convolutional Network 의 약어
- "Fully Convolutional Networks for Semantic Segmentation"
- 별다른 어려운 기법을 사용하지 않았음에도 Semantic Segmentation에서 뛰어난 성능을 보였다.

# Classification
![image](https://user-images.githubusercontent.com/69780812/138846867-72b3a44f-2b72-4cda-aa36-115cd28490a2.png)


- 영상에서 특정 대상이 있는지를 확인하는 기술
- AlexNet, GoogLeNet, VGGNet 등이 CNN + FC -layer로 구성되어진다.
- Fully Connected layer를 거치면 위치나 공간에 관련된 정보는 모두 사라지게 된다.

# Detection
- 특정 대상이 있는지 여부 + 위치 정보
  - Bbox로 대상의 위치 정보까지 포함하게 된다.
- R-CNN, SPPNet, Fast R-CNN, Faster R-CNN 구조

![image](https://user-images.githubusercontent.com/69780812/138847227-251621c6-f30d-44cb-8e9e-c6a45449367d.png)

- 대상의 여부 뿐만 아니라 위치정보를 포함해야하므로 Class 여부를 가리는 softmax부분과 위치 정보를 구하는 bbox regressor로 구성된다.

# Semantic Segmentation
- 단순히 bbox로 검출을 원하는 대상을 나타내지 않고, **픽셀 단위의 예측**을 수행하여 의미있는 대상을 분리해낸다.

![image](https://user-images.githubusercontent.com/69780812/138847398-48607872-2ac1-46cc-a5d8-fc85a4d42ac4.png)

- 영상 속에 무엇이(What) 있는지 확인 하는 것(Semantic) + 어느 위치(Where)에 있는지(location)까지 정확하게 파악해줘야 한다.
- Semantic과 location은 지향하는 바가 다르므로 이것을 조화롭게 해결해야 Semantic segmentation 성능이 올라간다고 한다.

# FCN & Fully convolutional model
- Classifiaction : Classifiaction에서 성능을 검증 받은 좋은 네트워크 등을 이용
  - AlexNet, VGGNet, GoogLeNet 등
  - 뒷단의 Fully Connected layer가 고정된 크기의 입력만 받아들이는 문제가 있다.
  - FC-layer를 거치면 **위치 정보가 사라지는 문제**가 있다. Segmentation에서 위치 정보가 없어지면 불가능 하므로 심각한 문제가 된다.

- FCN 개발자들은 FC-layer가 1x1 convolution으로 볼수 있다는 점에 주목했다.
  - OverFeat 개발자들이 Fully convolutional network 개념을 먼저 적용했지만 classifaction, detection에만 활용했다.

![image](https://user-images.githubusercontent.com/69780812/138848561-9ef79879-5533-4499-8a5d-5c90da39d67e.png)

- FC-layer가 1x1-conv로 간주되면 위치정보가 사라지는 것이 아니라 남게된다.
- 오른쪽 Heat map을 보더라도 고양이에 해당하는 위치 score 값들이 높은 값으로 나오게 된다.
- 모든 Network가 Convolutional Network으로 구성되면 더이상 **입력 이미지의 크기 제한을 받지 않게 된다.**
---
- FC layer -> 1x1 conv로 간주하면 위치 정보(spatial information)을 유지할 수 있다.
- 입력 영상의 제한을 받지 않게 되었다.
- patch 단위로 영상을 처리하는게 아니라 전체 영상을 한번에 처리할 수 있ㅇ서 겹치는 부분에 대한 연산 절감 효과를 얻을 수 있게 되었다.
  - 처리 속도가 빨라지게 된다.
  - R-CNN 속도 개선을 위해 Fast R-CNN or Faster R-CNN이 Conv 부분을 한번만 연산하도록 하여 연산 시간을 절감한 것과 유사하다.
---

# Upsampling (Deconvolution)
![image](https://user-images.githubusercontent.com/69780812/138850049-8f681ce2-12be-4bb1-a339-a63886801682.png)

- 픽셀 단위로 예측을 하려면 줄어든 Feature-map의 결과를 다시 키우는 과정을 거쳐야 한다.
- 1x1 conv를 거치며 얻어진 Score 값을 원 영상의 크기로 확대하는 간단한 방법 : bilinear interpolation
- But, end-to-end 학습 관점에서는 고정된 값을 사용하는 것 보다는 학습을 통해 결정하는 편이 좋다.
  - 경우에 따라 bilinear한 필터를 학습할 수도, non-linear upsampling도 가능하기 때문이다.

![image](https://user-images.githubusercontent.com/69780812/138850480-91869705-b11c-4ea8-b53a-66e3c4abd947.png)

- 단순히 Score를 upsampling 하면 어느 정도 이상의 성능을 기대하기 어려워 FCN 개발자들은 **skip layer**개념을 활용해 성능을 올렸다.
- Skip layer의 기본 개념
  - conv + pooling 과정을 통해 작아진 Feature map이 detail한 부분이 많이 사라지므로 **최종 과정보다 앞선 결과를 사용해 detail을 보강**
  - 여러 단계의 결과를 합쳐주면 정교한 예측이 가능해지게 된다.

![image](https://user-images.githubusercontent.com/69780812/138850810-22a02a87-4b98-45f4-9bf0-5137c574910d.png)

- Stride 32인 경우 자세하게 구별할 수 없지만 Skip layer를 활용한 Stride 8에서는 꽤 정교한 예측이 가능하게 된다.

# 결과
![image](https://user-images.githubusercontent.com/69780812/138851097-8277ee8a-4298-4b3c-8352-2228760a8cbe.png)

- Input Image에 대해 꽤 괜찮은 성능을 얻을 수 있다는 것을 확인할 수 있다.
- FC-layer를 1x1 conv로 간주하는 기본 아이디어를 바탕으로 속도 및 성능을 얻을 수 있었다.

# Classification 모델을 이용한 Semantic Segmentation 구현
![image](https://user-images.githubusercontent.com/69780812/138851664-f85da55f-65fe-440b-b21d-248123149516.png)

- 기존 AlexNet을 간략하게 표현한 Classifcation network이다.

![image](https://user-images.githubusercontent.com/69780812/138851750-cda9b051-4a2e-439c-b5ba-92d07ab7d3e8.png)

- FC-layer를 1x1 conv로 생각하여 재구성할 수 있다.
- 이 덕분에 더이상 입력 이미지 크기 제한을 받지 않는다.
- Semantic Segmentation에서는 픽셀 단위로 조밀하게 예측(Dense Prediction)을 해줘야하는데, 맨 마지막 layer에서는 Feature map의 크기가 매우 작아져 원영상으로 복원하는 작업이 필요하다.
  - Skip Connection 이라는 방법으로 해결

# Dense Prediction
- 여러 단계의 Conv + Pooling을 통해 작아진 Feature-map은 원 영상의 1/32 크기로 줄어든다.
- 픽셀단위로 조밀학 ㅔ예측하려면 다시 원영상 크기로 복원하는 과정을 거쳐야한다.
- 논문에서는 Upsampling을 하는 것을 효과적이라 결론을 내려 이를 사용했다. 
  - 하지만2016년, "Multi-scale context aggregation by dilated convolution", Fisher Yu 논문으로 Dilated conv가 효과적이라는 것을 입증해 딥러닝 프레임웤에서 지원된다.

![image](https://user-images.githubusercontent.com/69780812/138852648-affe1ce7-f6ef-47cd-b668-a12e7ebd9063.png)

- 1/32 크기에서 Feature(score)만 사용하는 것이 아니라 1/16, 1/8에서도 같이 사용하는 방식을 취한다.
- 논문에서는 이것은 **deep jet**이라고 칭했다.
- 이전 layer는 마지막 layer보다 세밀한 Feature를 갖고 있으므로 이것을 합하면 보다 정교한 예측이 가능해진다.

![image](https://user-images.githubusercontent.com/69780812/138852829-4c7c60d0-36de-427f-ab2f-7eccf9b9fa6e.png)

- 1/16, 1/8 크기의 정보를 이용하려면 conv + pool 단계를 거치지 않고 **skip layer or skip connection**이라는 것을 통해 처리한다.

![image](https://user-images.githubusercontent.com/69780812/138853118-e0bacd97-6861-4e10-b3bb-e9667c868f3a.png)

- 1/32에서 32배 만큼 upsample한 결과 : FCN-32s
  - FCN-16s : (pool5의 결과를 2배 upsample한 결과 + pool4의 결과)의 16배 upsample
  - FCN-8s : (중간 결과의 2배 upsample + pool3 결과)의 8배 upsampleQ

![image](https://user-images.githubusercontent.com/69780812/138853678-2efbd28d-2f48-4371-a742-47981e3f9832.png)

- FCN-8s만 따로 표현한 그림이다.
  - 실제 FCN의 결과는 FCN-8s의 결과를 사용한다.

![image](https://user-images.githubusercontent.com/69780812/138854105-cfc86eea-f105-4be7-a510-a0397f0facbc.png)

- skip과 중간 결과를 합치는 과정을 거치게 되면 점점 더 정교한 예측이 가능하게 된다.

![image](https://user-images.githubusercontent.com/69780812/138854245-0b50cafe-8b94-447e-acc1-0494cdacace4.png)

- PSCAL VOC 2011 데이터를 이용한 실험의 결과이다.
- FCN-32s결과와 FCN-8s의 결과는 정밀도가 개선되는 것을 알 수 있었다.

# FCN의 문제점
- Fully Convolutional Network 개념은 다른 많은 연구자들에게 큰 자극이 되었다.
- "Multi-scale context aggregation by dilated convolution", Fisher Yu는  FCN에 대한 분석 및 약간의 구조 변경을 통해 FCN의 성능을 더 끌어올렸다.
  - 일명 **Dilated convolution**이라고 불린다.
- FCN은 Classification용으로 충분히 검증 받은 망을 픽셀 수준의 조밀한 Prediction이 가능한 Segmentic Segmantation에 적용하기 위해 up-sampling 로직과 skip layer 개념을 적용해 떨어지는 해상도를 보강했다.

![image](https://user-images.githubusercontent.com/69780812/138860594-b49d5b48-4eba-40e2-a18a-8082f519a0a0.png)

- FCN에 존재하는 한계나 제한점
  - 사전에 미리 정한 Receptive Field
    - 너무 작은 Object가 무시되거나 엉뚱하게 인식
    - 큰 물체를 여러 개의 작은 물체로 인식하거나 일관되지 않은 결과
  - 여러 단의 Conv + Pooling -> 해상도 감소
    - 줄어든 해상도를 Upsampling 하면서 detail이 사라지거나 과도하게 Smoothing효과에 의해 결과가 아주 정밀하지 못하다.

# FCN 문제점 극복 시도 - New Architecture
- 조밀한 픽셀 단위 예측을 위해 upsampling과 여러 개의 후반부 layer의 conv feature를 합친다.
  - 앞서 살펴본 것과 같은 문제들이 발생

![image](https://user-images.githubusercontent.com/69780812/138860852-41295a8d-e7a2-4dd5-8685-adbad007e666.png)

- Convolutional Network에 대칭되는 Deconvolutional Network를 추가
  - Upsampling의 해상도 문제를 해결하고자 함.
  - 기본망은 VGG-16 기반

![image](https://user-images.githubusercontent.com/69780812/138861046-6d72d8e2-e1a0-4ab0-b72d-554c56907c48.png)

- Deconvolution 개념은 ZFNet 개발자들이 내부 layer의 feature를 visualization하는 작업에서 사용한 max-pooling 위치를 기억하여 정 위치를 찾아가는 **switch variable** 개념을 비슷하게 사용
- 단순한 bilinear interpolation을 통한 upsampling이 아니라 **uppooling + deconvolution**에 기반해 훨씬 정교한 복원이 가능해진다.
  - Deconvolution layer의 filter 계수 역시 학습을 통해 결정
  - 계층 구조를 갖는 Deconv layer를 통해 다양한 Scale의 Detail 정보를 살릴 수 있게 됐다.

![image](https://user-images.githubusercontent.com/69780812/138861449-6b240128-39ae-48cd-9665-b5f072f39959.png)

- (b)는 마지막 14x14 deconv layer의 activation
- (c) : 28x28 unpooling layer의 activation

![image](https://user-images.githubusercontent.com/69780812/138861536-bdefc726-9299-45ff-88c9-9b3767c2ef90.png)

- (d)는 마지막 28x28 deconv layer activation
- (e)는 56x56 unpooling layer에서의 activation
- 좀 더 세밀해 지는 것을 확인할 수 있다.

- 논문 저자들은 단순한 deconvolution이나 upsampling을 사용하는 대신 **coarse-ti-fine deconvolution**망을 구성하여 정교한 예측이 가능함을 확인할 수 있다고 주장한다.
- Unpooling -> 가장 강한 Activation을 보이는 위치를 정확하게 복원 -> 특정 개체애 특화된 구조를 얻어낸다.(exampling-specific)
- Deconvolution -> 개체 class에 특화된 구조를 추출할 수 있다. (class-specific)

![image](https://user-images.githubusercontent.com/69780812/138862044-404c5e07-7daa-4b23-aed9-e9bf0cad8bca.png)

- FNC과 Deconvolution network 방법을 비교한 결과이다.
- 확실히 좀 더 정밀한 Segmentation이 가능하다는 것을 확인할 수 있다.

# FCN 문제점 극복 시도 - 2 단계 Training 방법, Batch Normalization
- VGG-16망을 2개 쌓아놓은 것이나 마찬가지다.
  - 망의 깊이가 깊어지면 Overfitting 발생
- 모든 Convolutional layer및 Deconvolutional layer의 출력에 Batch Normalization을 적용했다.
  - 학습에서 절대적 영향을 미쳤다고 한다.
- 효율적 학습을 위해 2단계의 학습법을 사용
  - 1. 1단계는 먼저 쉬운 이미지를 이용하여 학습
    - 가급적 학습 시킬 대상이 영상의 가운데 위치
    - 크기도 대체적으로 변화가 작도록 설정
    - Pretraining 용도로 사용
    - data augmentation을 거친 20만장 이미지 사용
  - 2. 2단계는 270만장 이미지를 사용
    - 좀 더 다양한 경우에 대응할 수 있도록 다양한 크기 및 위치에 대응이 가능할 수 있도록했다.

# 결과 검토 및 개선
![image](https://user-images.githubusercontent.com/69780812/138862625-d902459a-1089-4f52-9721-6888045a5eaf.png)

- 구조적 특성으로 Deconvolutional Network은 좀 더 정밀한 Segmentation에서 좋은 특징을 보인다.
- FCN은 전체적인 형태를 추출하는 것에 적합하다.
  - 둘을 섞어 쓰는 것이 더 좋은 결과를 보인다.
- EDconvNet : FCN + Deconvolutional Network

![image](https://user-images.githubusercontent.com/69780812/138862922-969f56ee-0977-4213-942b-73dc4dddca2e.png)

- Deconvolutional network이 FCN보다 나쁜 경우의 그림이다.
- 확실히 FCN은 전체적 모양에 관심이 있다면, DeconvNet은 Detail에 집중한다는 것을 확인할 수 있다.
  - 이런 경우 여러 방법을 섞어 쓰는 것이 효율적임을 알 수 있다.

![image](https://user-images.githubusercontent.com/69780812/138863122-12815edd-ad3f-4d79-ace7-9c04316d49a4.png)

- FCN, DeconvNet 모두 결과가 좋지 않은 경우다.
- 모델 결합을 하면 결과가 개선되는 경우를 보여주는 예다.
- FCN보다 정교한 예측을 하는 DeconvNet이지만 좋지 못한 결과가 나오기도 하기 때문에 뭐가 절대적인 방법이라고는 할 수 없을 것같다.

# Dilated Convolution
- FCN 개발자들은 Dilated conv 대신 Skip layer와 upsampling 개념을 사용했다.
- Dilated Convolution의 개념은 "wavelet decomposition"알고리즘에서 atrous algorithm이라는 이름으로 사용된다.

![image](https://user-images.githubusercontent.com/69780812/138863744-b7aa1cf9-6cbf-4da5-93c8-22d2136d8c0a.png)

- 기본적인 Convolution과 유사하지만 빨간 점의 위치에 있는 픽셀들만 이용하여 Convolution을 수행하는 것이다.
- 해상도 손실없이 Receptive Field 크기를 확장할 수 있다.
  - 빨간 점의 위치만 계수가 존재하고 나머지는 모두 0
- (a) : 1-dilated convolution
  - convolution과 동일
- (b) : 2-dilated convolution
  - 이렇게 되면 Receptive Field의 크기가 7x7 영역으로 커지는 셈이 된다.
- (c) : 4-dilated convolution
  - Receptive Field 크기가 15x15로 커지게 된다.
- 큰 Receptive Field는 파라미터의 개수가 많아야 하지만 Dilated Convolution을 사용하면 Receptive Field는 커지지만 파라미터 개수는 늘어나지않아 연산량 관점에서 탁월한 효과를 얻는다.
  - 위 그림은 Receptive Field가 커지지만 연산량 부담은 3x3 filter에 불과하다.

## 장점
- Receptive Field가 커진다.
  - 다양한 Sclae에서의 정보를 끄집어내려면 넓은 Receptive Field가 필요하다.
  - Deilated conv를 사용하면 별 어려움없이 가능해진다.
- Dilation 게수를 조정하면 다양한 Scale에 대해 대응이 가능해진다.

![image](https://user-images.githubusercontent.com/69780812/138864452-ce2a810c-cde7-4a9b-9032-3727da95e2cb.png)

- 기존 CNN 망에서는 Receptive Field 확장을 위해 Pooling layer로 크기를 줄이고 Convolution을 수행하는 방식을 취했다.
- 아래 그림은 Dilated conv를 통해 얻은 결과다.

# Front-end 모듈
![image](https://user-images.githubusercontent.com/69780812/138864839-24a8bd2e-131f-49c2-938a-4be10be2ef57.png)

- Fisher Yu팀은 Pool4, Pool5를 제거하여 좀더 해상도를 높였다.
  - 영상의 크기가 1/8 수준으로만 작아져서 Upsampling을 통해 원영상을 크기로 크게 만들더라도 상당한 Detail이 살아있다.
- conv5, conv6에 일반적 Conv를 사용하는 대신 conv5에는 2-dilatedconv, conv6에는 4-dilated conv를 적용했다.
- 결과적으로 Skip layer도 없고 망도 더 간단해 져서 연산 측면에서 훨씬 가벼워졌다.
- Front-end만 수정해도 이전 결과들 보다 정밀도가 더 좋아졌다고 한다.

# Context 모듈
- 다중 Scale의 Conext를 잘 추출하기 위한 모듈을 개발

![image](https://user-images.githubusercontent.com/69780812/138865724-12203d21-59c0-4325-9b26-e361f4d84764.png)

- Context 모듈 구성을 나타내는 표이다.
- Basic Type : Feature map개수가 동일
- Large Type : Feature map개수가 늘었다가 최종단만 Feature map개수를 원래대로와 같게 해줬다.
- Front-end 모듈 뒤에 Context모듈을 배치했다.
  - 어떤 망이든 적용할 수 있도록 설계했다고 한다.
- C : Feature map 개수
- Dilation : Dilated conv의 확장 계수
- 뒷단으로 갈수록 Receptive Field가 커진다.
- Front-end 모듈만 적용해도 개선이 있었으나 Context 모듈을 추가하면 추가적 성능 개선이 있었다고 한다.
  - CRF-RNN까지 적용하면 더 좋아졌다고 한다.
---
- Dilated convoltion을 통해 망의 복잡도를 높이지 않으면서도 Receptive Field를 넓게 볼 수 있었다.
  - 다양한 Scale에 대응이 가능하게 했다.
- 위 개념을 활용한 Context 모듈도 만들어 성능 개선을 꾀했다.
---
