---
# Training Data for Supervised Learning
- 통계에서 좋은 결과를 위해서는 ?
  - 표본조사 시 전체를 대표할 수 있는 많은 Samples
  - 위 Sample들로 부터 전체를 잘 설명할 수 있는 Model
- 전체를 대표할 수 없는 치우친 Sample을 경계해야한다.
---
# Overfitting
- 모델이 제한된 Sample들에 너무 특화되어 New Samples에 대한 예측 결과가 안좋거나 학습 효과가나타나지 않는 경우

![image](https://user-images.githubusercontent.com/69780812/137900421-eb9aee7e-95eb-40d1-abae-b3ed7ffe261f.png)

- (a) : 단순하게 추정하는 경우로 얼핏 보기에도 오류가 많다.
- (b) : 주어진 점들의 특성을 잘 나타내고, 약간의 오차가 있지만 새로운 샘플이 들어올 때는 좋은 결과가 나올 수 있다.
- (c) : 모든 점들을 그대로 살려 오차가 없이 추정했다. 이 경우 Training Data에 대해서 최적의 결과를 나타낼 수 있지만, New Samples이 주어지는 경우 결과가 엉터리 일 수 있다.
## Solution
- 정답은 없다. (c)가 올바른 추정일 수 있고, (b)가 올바른 추정일 수 있다.
- Sample의 수를 늘리거나 Training Data 양을 늘리는 것이 좋다. 하지만 이는 많은 비용과 노력이 필요하며 이어서 공부한다.
- (b) or (c) 선택의 상황
  - 오캄의 저서 -> 오캄의 면도날 -> 설명이 더 복잡한 이론은 배제한다. 간단한걸 선택해라.
  - 즉, 오캄에 이론을 따른다면 (b)가 맞겠다.

# Regularization
- Training Data 양을 늘린다. -> 문제는 많은 시간적 비용이 발생한다. 또한 추가 데이터 확보가 어려울 수 있다.
- Regularization은 일종의 penalty 조건에 해당된다. 대체로 복잡한 쪽보다는 간단한 쪽으로 선택을 유도한다.
- 기계학습 또는 통계적 추론 시 Cost Function 값이 작아지는 쪽으로 진행한다. 단순히 작아지는 쪽으로 진행하는 경우, **특정 가중치 값들이 커지면서 오히려 결과를 나쁘게하는 경우**도 있다.

![image](https://user-images.githubusercontent.com/69780812/137901433-6e35cdf8-133e-4de3-a291-87ed608849e7.png)
- Regularzation을 통해 더 좋은 학습 결과를 가져오는 경우의 그림이다.

## 수학적 표현
### 1. L2 Regularization
![image](https://user-images.githubusercontent.com/69780812/137901523-131b9add-b731-45cb-920d-214b166062dc.png)

- C0 : Orginal Cost Function
- n : Training Data Num
- lambda : regularzation parameter
- w : weights
- 학습의 방향이 단순하게 C0가 작아지는 방향으로만 진행되는 것이 아니라 **W 값들 역시 최소가 되는 방향**으로 진행하게 된다.

![image](https://user-images.githubusercontent.com/69780812/137901761-24793cfa-2b8f-46c1-a347-21b9b80c618f.png)

- Cost Function을 W에 대해서 편미분 한다면 위 그림과 같이되며 최종적으로 W는 위와 같이 Update 되는 것이다.
- (1 - learning_rate * lambda / n)의 형태의 항목 덕분에 W가 작아지는 방향으로 진행된다.
  - 이를 **weight decay**라고 부른다.
  - 특정 가중치가 비이상적으로 커지고 그것이 학습의 효과에 큰 영향을 끼치는 것을 방지한다.
  - w가 작아지도록 학습한다 -> local noise, outlier의 영향을 적게 받도록 하고 싶은 것이다. -> 결과적으로 일반화에 적합한 특성을 갖게 만드는 것이라 볼 수 있다.
  - Outlier는 통계나 추정을 왜곡 시킬 정도로 크게 벗어나있는 Data이고, Local noise는 그 정도는 아니라고한다. 정도의 차이에 따라 부르는 용어가 다르다. ([참고1](http://kt.ijs.si/petra_kralj/IPS_DM_1516/NoiseAndOutlierDetection.pdf), [참고2](https://www.mathworks.com/matlabcentral/answers/54798-what-is-the-difference-between-noise-and-outlier))
### 2. L1 Regularization
![image](https://user-images.githubusercontent.com/69780812/137906761-c6ce271b-b382-4662-979b-55b49a5a238c.png)

![image](https://user-images.githubusercontent.com/69780812/137906790-5c4e5ed5-f3db-4559-9442-aab61163536a.png)

- 결과적으로 Weight 값 자체를 줄이는 것이 아니라 W의 부호에 따라 상수 값을 빼주는 방식으로 Regularization을 수행한다.
- L1과 L2의 차이
  - 수식을 보면, L1은 통상적으로 상수 값을 빼주므로 작은 가중치 들은 거의 0으로 수렴되어 몇 개의 중요한 가중치만 남는다.
  - 그러므로 몇 개의 의미있는 값을 끄집어 내고 싶은 경우에는 L1 Regularization이 효과적이다.
  - Sparse model에 적합하다. Gradient Based Learning에 적용할 때는 주의가 필요하다고한다.

# Training Data를 효율적으로 늘리는 방법
## 1. Affine Transform
![image](https://user-images.githubusercontent.com/69780812/137908101-ea986dbd-f1ca-4102-9e9b-32b80f4e66e2.png)

- Rotation, Shearing, Translation, Scaling
## 2. Elastic Distortion을 이용한 지능적 훈련 데이터 생성
![image](https://user-images.githubusercontent.com/69780812/137908341-dda8ae9a-5611-4ba3-a62d-0bf19845a871.png)

- 다양한 방향으로 displacement vector 생성 -> 이를 이용한 더 복잡한 형태의 훈련 Data 생성

## 3. 기타
- 음성 인식의 경우 잡음 없는 상태에서 녹음 후 잡음과의 합성을 통해 훈련 데이터 집합을 만들 수 있다.
- 학습에서 좋은 결과를 얻으려면 대표성을 갖는(Orthogonal)한 벡터 집합을 얻는 것이 중요하다.
- 또한, 환경을 잘 이해하고 있다면 적용 환경에 맞게 지능적인 방법으로 훈련 데이터를 획득하는 것 역시 효율적이다.

# Dropout
- Regularization이 Cost Function에 Penalty 함수를 추가하여 penalty를 통한 조작으로 결과를 얻는 방식이라면, Dropout은 망 자체를 변화시키는 방식으로 근본적으로 차이가 있다.

![image](https://user-images.githubusercontent.com/69780812/137908801-5c4350ab-c499-4365-9c0b-6b6cb2e31507.png)

- Network의 입력 layer나 hidden layer의 일부 뉴련을 생략(dropout)하고 줄어든 신경망을 통해 학습을 수행한다.
- 일정한 mini-batch 구간 동안 생략된 망에 대한 학습을 마치면, 다시 무작위로 다른 뉴런들을 생략하면서 반복적으로 학습을 수행한다.
## 1. Voting 효과
- 일정한 mini-batch 구간 동안 줄어든 network를 이용해 학습을 하게 되면 그 망은 그 망 나름대로 Overfitting
- 다른 mini-batch 구간 동안 다른 망에 대해 학습하게 되면 그 망에 대해 다시 일정 정도 overfitting 된다.
- 이런 과정을 무작위로 반복한다면, **voting에 의한 평균효과**를 얻을 수 있어 regularization과 비슷한 효과를 얻을 수 있게 된다.

## 2. Co-adaptation을 피하는 효과
- 특정 뉴런의 Bias or Weights가 큰 값을 갖게 되면 그것의 영향이 커지면서 다른 뉴런들의 학습 속도가 느려지거나 학습이 제대로 진행이 되지 못하는 경우가 있다.
- dropout을 하면, 결과적으로 **어떤 뉴런의 가중치나 바이어스가 특정 뉴런의 영향을 받지 않기 때문에** 결과적으로 co-adaptation이 되는 것을 피할 수 있다.
- 특정 학습 데이터나 자료에 영향을 받지 않는 Robust(강건한) 망을 구성할 수 있게 되는 것이다.
- [DropOut논문](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

# 학습 속도 저하 현상의 원인
- 신경망 학습 속도의 문제점은 **학습 속도 저하**이다.

## 1. Sigmoid 함수의 미분 특성
- Cost Function으로 MSE(Mean Square Error) 방식을 사용하면 큰 에러가 발생할 수록 학습이 잘되지 않는다.
  - 이는 Signmoid 함수를 사용하는 경우 그 특성과 결합하면서 문제가 생기기 때문이다.

![image](https://user-images.githubusercontent.com/69780812/137911297-129bbc36-bdd8-4b25-9c8f-be03e3571cef.png)

- w : 뉴런의 가중치
- b : bias
- Activation Function : Sigmoid
- z = w\*x + b
- a = sigmoid(z)

![image](https://user-images.githubusercontent.com/69780812/137910599-37473a11-59d1-429e-a682-c3c87ad5e5db.png)

- Error(y-a) 크고, back-propagation 시키면 학습속도가 빨라져야할 것 같지만 그렇지 못하다.

![image](https://user-images.githubusercontent.com/69780812/137911023-6630e8e2-82d6-439e-8105-fe291a18dd44.png)

- cost function C를 가중치와 바이어스에 대해 편미분을 수행하면 위와 같다.
- 가중치와 바이어스의 편미분값에는 Sigmoid함수의 미분 값을 곱하는 부분이 있다. 문제의 주범이다.
- Sigmoid 함수에 대해 미분을 취하면 0일때 최대 값을 갖고, 0으로 부터 멀어질수록 미분값이 0에 수렴하는 작은 값으로 간다.
  - 즉, 바이어스나 가중치의 갱신값이 아주 작은 값을 곱해주는 형태가 되기 때문에 (a-y)항이 크더라도 Sigmoid의 미분값이 아주 작은 값으로 되기 때문에 학습 속도 저하가 일어난다.

## 2. Gradient descent 특성
- Error(a-y)가 작아지면 a-y가 다시 거의 0으로 수렴하기 때문에 편미분식을 봤을 때, 결과적으로 바이어스와 가중치의 갱신값이 작아지게 된다.

![image](https://user-images.githubusercontent.com/69780812/137915429-7a8b656b-647a-4c49-b0e8-8a5defceaf58.png)

- Error(a-y)가 0 근처에 오면 학습의 속도가 현저하게 저하되며 학습을 더 시키더라도 학습의 결과가 그렇게 좋아지지 않는 현상이 발생하게 된다.
- 학습 시간도 결과적으로는 비용이므로 결과가 더 이상 개선되지 않는 경우는 적절한 곳에서 Stop해준다.

# Cross Entropy Cost Function
- 오차가 클수록 학습속도가 빨라야할 것 같은데 Sigmoid 함수의 특성과 결합하면 학습속도가 느려지는 경우가 있었다.

![image](https://user-images.githubusercontent.com/69780812/137915917-574a6fad-4dc6-4a6a-a152-fb52a7653be2.png)

- 이 형태에서 Sigmoid의 편미분항을 없앨 수 있다면 원래 기대했듯이 **학습속도가 오차의 크기에 비례**하게 된다.

## 신경망에서의 Cross Entropy
- Information Theory에서 Entropy는 확률분포의 무작위성(randomness)를 설명하는 용도로 사용한다.

![image](https://user-images.githubusercontent.com/69780812/137916192-fd84288d-5131-4f5f-b038-ae45c8813ea9.png)

- 확률분포 P를 갖는 Random 변수 X를 표현하기 위한 최소 비트 수를 나타낸다.
- Cross Entropy는 2개의 확률 분포의 차이를 나타내는 용도로 정의가 되었다.
- 두 개의 확률 분포가 얼마나 가까운지 혹은 먼지를 나타내며 2개의 확률 분포 p와 m에 대한 CE는 아래와 같이 나타낸다.

![image](https://user-images.githubusercontent.com/69780812/137916523-a26133c8-726f-4199-9ef7-4358691ebc31.png)

- 위 식에서 p와 m이 같다면 Entropy 식과 같아진다. 즉, **차이가 클수록 큰 값이 나오고 두개가 같아질 때 최소값이 나온다*.**
- Cost Function 처럼 기대값과 **실제 연산값의 차이가 클수록 큰 결과**가 나오고, **항상 양**이므로 Cost Function으로 사용이 가능하다.

![image](https://user-images.githubusercontent.com/69780812/137917583-6db59b85-e7a1-48b6-a0c3-81cc05d2e8f4.png)

- p : 랜덤 변수 X의 실제 분포
- m1, m2 -> 추정한 경우의 분포
- H(p) = 1.86이고, H(p, m1) = 2, H(p, m2) = 2.02이다. 결과만 봤을 때 m1이 더 좋은 추정이 된다. 

## Cross Entropy, 정말로 MSE 보다 좋은가 ?
![image](https://user-images.githubusercontent.com/69780812/137918106-0d056928-cc10-4171-8c5b-d21555881fcb.png)

- y : target value
- a : 실제 망에서 출력된 값
- n : 훈련 데이터 개수

![image](https://user-images.githubusercontent.com/69780812/137918192-a2052404-df95-4472-a652-6c29d73ec132.png)

- Weights와 Bias에 대한 편미분을 구한 식이다. [Cross Entropy 편미분 참고](http://solarisailab.com/archives/2237)
- 처음 기대했던 것처럼 기대값과 실제 출력의 차에 비례하는 결과를 얻을 수 있게 됐다.
- 결과적으로 CE Cost Function을 사용하여 학습을 수행하면 **훨씬 빠른 속도로 학습이 진행**된다.
  - 학습 속도는 Cross Entropy가 빠르지만 학습 결과는 비슷한 측면이 있다고한다.

# Softmax (Activation Function)
- 일반적으로 신경망에서 Activation Function을 주로 Sigmoid 함수를 사용한다.
- 단순하게 Step Function이나 Linear Function을 사용할 때보다 훨씬 많은 일을 할 수 있기 때문이다.
  - 입력단, hidden layer, 출력단에서 사용된다.
- 실제 신경망의 응요에서 출력단에 Softmax 함수를 더 많이 사용하고, 특히 분류기 관련 Application은 Softmax를 쓰면 더 좋다고한다.

## 정의
![image](https://user-images.githubusercontent.com/69780812/137921430-a1b42e15-5f39-4121-b0b7-e9e2b70cd9d7.png)

- K-dim vector z를 (0,1) 범위를 갖는 sigmoid function으로 치환시킨 것과 같다.
- 확률 분포와 같다는 것을 할 수 있으며 Logistic regression 이라고도 불린다.
- Z_k에 대해 편미분 시 j=k 일때는 양수이고, j!=k 이면 음수가 된다.
  - 즉, Z_k를 증가시키면 해당 뉴런의 출력값은 증가하고, 다른 뉴런의 출력값은 감소하게 되는 성질을 갖게 된다.
## Sigmoid와의 차이
- Sigmoid를 사용하면 해당 뉴런으로 들어오는 입력들과 바이어스에 의해 출력이 결정된다.
- 하지만, Softmax는 해당 뉴런으로 들어오는 입력들과 바이어스에 의해 출력이 결정되는 구조는 비슷하지만, **다른 뉴런의 출력값과 상대적인 비교**를 통해 최종 출력값이 결정된다는 점이 다르다.
  - **non-locality**라고 부른다 한다.
- Softmax의 수식을 살펴보면 모둔 뉴런의 출력값들을 합하면 1이 되기 떄문에 평균화(normalization)효과를 얻는다. 이는 분류 확률과 개념이 딱 맞아 떨어진다.
- 그러므로 분류를 목적으로하는 신경망의 최종단에 Softmax를 쓰면, 해당 값이 나올 **확률**을 알 수 있으며 가장 큰 값이 나오는 쪽으로 분류되는 것을 알 수 있다.
- 또한, 특정 항목의 변화가 최종 출력에 어떻게 영향을 끼치는 지도 바로 파악 가능한 효과도 덤으로 얻는다.

# Hyperparameters
- 신경망에서의 Hyperparameter란 신경망 학습을 통해서 튜닝 또는 최적화 해야하는 주 변수가 아니라, 학습 진도율이나 일반화 변수처럼 사람들이 선험적 지식으로 설정하거나 외부 모델 메커니즘을 통해 자동으로 설정되는 변수를 말한다.
  - meta-parameters or free-parameters라고도 불린다.
- 1. Learning Rate
  - gradient의 방향으로 얼마나 빠르게 이동할 것인지를 결정한다.
  - 학습 진도율이 너무 작으면 학습의 속도가 너무 느리게되고, 반대로 너무 크면 학습의 결과가 수렴이 안되고 진동하게 될 수 있다.
  - 사람마다 또는 학습하는 분야마다 최적의 학습 진도율이 다르듯이 학습 진도율을 학습하고자 하는 대상이나 망에따라 조절해야한다.
- 2. Cost Function
- 3. Regularization parameter
  - 일반화 변수 lambda는 weight decay의 속도를 조절하기 위한 용도로 사용할 수 있다.
- 4. Mini-batch Size
  - Mini batch의 크기가 큰 경우 병렬연산 구조를 사용할 때 효과적일 수 있고, 크기가 작으면 더 많은 update를 할 수 있다.
- 5. Training 반복 횟수
  - 학습의 조기 종료를 결정하는 변수가 된다.
  - 조기에 학습을 종료하는 것은 Overfitting을 방지할 때 중요하게 사용된다.
- 6. Hidden unit 개수
  - Hidden layer의 개수가 많아질수록 특정 훈련 데이터에 더 최적화 시킬 수 있다.
  - 또한 모든 hidden layer의 뉴런 개수를 동일하게 유지하는 것이 같은 hidden layer의 개수에 뉴런의 개수를 가변적으로 하는 것 보다 효과적이다.
  - 첫번째 hidden layer에 있는 뉴런의 개수가 input layer에 있는 뉴런의 개수보다 큰 것이 효과적인 경우가 많다.
- 7. 가중치 초기화(Weight initialization)
  - bias는 일반적으로 0으로 초기화가 많이 된다.
  - 가중치의 경우 초기화가 학습 결과에 큰 영향을 끼치므로 주의가 필요하다.
  - 가중치는 보통 무작위로 초기화가 되며 범위는 \[-r, r]범위를 가진다. r은 뉴런의 개수 제곱의 역수가 된다.
  - 입력 뉴런의 개수가 6이라면 \[-1/36, 1/36] 범위 내에서 무작위로 설정한다.
- 이 외에도 많은 hyperparameter들이 있으며 최적의 결과를 내는 정립된 이론은 없다.

# Hyperparameter Optimization
- 신경망을 통한 학습이란 결과적으로 Cost Fucntion을 최소화 시키는 Weight와 Bias 값을 찾아내는 것이다.
  - 하지만 Overfitting 문제에 빠지지 않고 기대했던 수준의 결과를 얻으려면 hyperparameter에 대한 최적화 작업이 선행되야한다.
- 하지만 Golden Rule은 없고, 설계자의 직관에 의존해야한다. 이는 큰 골칫거리다.
## 1. Manual Search
- Learning Rate에 대한 최적화 작업을 수행한다고 하면, 먼저 임의 값을 대입하여 결과를 살핀 후 그 값으로 일정 정도 떨어진 값을 다시 적용하여 결과가 움직이는 방향으로 추정해보고 반복해야한다.
- 어떤 탐색이론을 사용하느냐에 따라 시간이나 질이 달라질 수 있다.

## 2. Grid Search
- 큰 틀에서 보면 Manual Search와 별다를게 없고, 개념적으로도 비슷하다.
- 단, Grid Search의 경우 선험적 지식을 활용하여 문제를 분석하고, hyperparameter의 범위를 정한다.
- 그 범위 안에서 일정한 간격으로 점을 정하고 그 점들에 대해 1개씩 차례로 실험을 해보면서 최적의 값을 찾은 후 다시 Best로 추정되는 점을 기준으로 세분화하여 최적값을 찾는다.
- Manual Search나 Grid Search를 할 때는 결과를 판정하기 위한 Validation Set이 필요하다.

## 3. Random Search
- 선험적 지식을 이용해 hyperparamter의 범위를 한정한다.
- 그 이후 일정한 간격으로 탐색하는 대신 무작위로 최적값을 찾는 작업을 진행한다.
- 일정한 시간 안에 결과를 내야하는 경우 Random Search를 사용하는게 좋은 방법일 수 있다.

## 4. Baysian Optimization
- Bayesian Optimization의 기본 원리가 Prior Knowlege를 활용하는데 있으므로 현재까지 실험 결과를 바탕으로 통계적 모델을 만들고, 그것을 바탕으로 다음 탐색을 해야할 방향을 효과적으로 정하자는 것이 이 방법의 핵심이다.
- 다른 방식들에 비해 좀 더 짧은 시간에 최적값을 찾아내는 경향이 있다.
---
역시 최적의 hyperparameter를 찾는 것도 그리 쉽지 않다.
---
