# Machine Learning Definition
- **Field of study that gives computers the ability to learn without bieng explicitly programmed**
  - 가능한 모든 변수를 프로그래머가 정의하지 않더라도 Data 학습을 통해 최적의 판단이나 예측을 가능하게 해주는 연구 분야
# Machine Learning의 분류
- Machine Learning : Data Minig, Pattern 인식으로도 불린다.
1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning
# Supervised Learning
- 이미 주어진 Input, 이미 어떤 결과가 나올지 알고 있는 Output.
- 위 Input과 Output의 관계를 이용해 Data를 해석할 수 있는 Model을 만들고 Prediction하는 것이 Supervised Learning이다.
- (inputs) Known Data + (Output) Knwon Response = Supervised Learnign Model(Function)
  - Many Trainig Data + Generlized Traning Data = Good Performance Model
## 학습 순서
1. 학습에 사용할 Trainig data 선정
   - Training Data는 모델의 성패에 아주 지대한 영향을 미친다.
2. Training data 수집
3. Input의 Feature를 표현할 방식 선정
   - 일반적으로 Vector 형태로 한다.
   - 차원의 저주(Curse of dimentionality)에 주의한다.-> 너무 차원이 크면 결과가 안좋음
4. 학습 알고리즘 결정
   - Artificial Network, Boosting, Bayesian statics, Decision Tree etc..)
5. Trainig data 학습
6. Preidction 평가
   - 보통 Training data와 다른 Test data를 사용한다.
# Unsupervised Learning
- Input들을 선정 -> Decision making(입력에 대한 의사결정) -> Predicting Future input Model
  - 위 과정들을 통한 Input에 대한 결정할 수 있는 프레임워크를 만들 수 있다.
- 대표적으로 Clustering, Dimentinality reduction이 있다.
  - Dimentinality reduction : PCA, ICA, Non-negative matrix factorization, SVD etc..
- 어떠한 보상, 출력에대한 정보가 없기 때문에 굉장히 어려운 학습이라고 할 수 있다.
# Reinforcement Learning
- 학습자가 훈련 시 잘하면 Reward를 받고, 못하면 Punishment를 받는다.
  - 훈련자가 원하는 방향으로 학습이 가능하다.
  - Supervised Learning과 다르게 '잘했다'라는 상황이 여러가지라는 것에 유의한다.
- Reinforcement라는 것은 **Reward와 Punishment를 통해 그 행위 또는 반대 행위를 강화시킨다는 것이다."**
- 시행착오를 거치면서 보상을 최대화하는 것을 배운다.
- Decision making through trial and error, Delayed reward
  - 이 후 상황과 같은 '결과'로서의 보상에도 영향을 받는다. 이것은 다른 학습과 구별하게 만드는 특징들이기도 하다.
