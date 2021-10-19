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
  
