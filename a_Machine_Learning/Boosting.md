# Boosting
- 초보자가 경주마의 승률 예측을 위해 고수들에게 조언을 구한다면 경주마를 고를 수 있는 Golden rule은 없다는 말을 듣게 될 것이다.
- 하지만 나름대로의 규칙(weak rule)을 알려줄 거다.
  - 여러 전문가들로 부터 들은 조언에 따라 선택하면 다양한 경우의 수가 발생하겠지만 경우의 수가 겹치는 쪽으로 선택의 범위를 좁히면 승률이 올라간다.
- 전문가로 부터 조언을 듣고 의견이 분분한데 경주마 선정 규칙을 정해야한다.. 다양한 조언을 어떻게 통합해서 적용해야할까?
  - 이 때 쓸 수 있는 방법이 Boosting이다.
- Boosting이란 무작위로 선택하기 보다 약간 가능성 높은 규칙들을 결합시켜 보다 정확한 예측 모델을 만들어 내는 것을 말한다.
  - Weak한 것들을 여러개 결합 시켜 Strong Model을 만들어 낸다는 의미이다.

## 예시 - 스팸 여부 판단
- 링크만 있는 경우 -> 스팸
- 도메인 주소 확실 -> 노 스팸
- 보낸 사람 확실 -> 노 스팸
- 스팸 여부를 판단할 기준을 열거했지만 확실히 구분할 규칙들은 아니다.
  - Weak learner 라고한다.
- Strong Learner로 만드는 방법은 ?
  - 평균/가중 평균을 사용하는 방법
  - 가장 많은 의견을 사용하는 방법 (Vote)

## 학습 방법
![image](https://user-images.githubusercontent.com/69780812/139027721-92f26e1f-2e43-4a2b-bdd9-c316fdb61d7d.png)

- Weak learner 정의 : 오차율이 50% 이하인 학습 규칙
- 머신 러닝 알고리즘을 적용해 서로 다른 분포를 갖게해준다.
- 매번 기본 러닝 알고리즘을 적용할 때마다 새로운 Weak learner를 만들어 이 과정을 반복적으로 수행한다.
- 반복 수행 후 Weak Learner를 Boosting 알고리즘으로 묶어 Strong learner를 만든다.
- 학습 하면서 에러가 발생하면 그 에러에 좀 더 집중하기 위해 Error에 대한 Weighting을 올려 에러를 잘 처리하도록 Weak learner 학습을 해나간다.

![image](https://user-images.githubusercontent.com/69780812/139028129-9056e045-dffb-4c67-93b2-48936ec171ea.png)

- 모든 것들이 결합된 최종 결과는 위 식과 같이 표현된다.
  - alpha : 가중치
- Boosting은 새로운 learner(classifier)를 학습 할 때 마다 이전 결과를 참조하는 방식이다.
  - bagging(boostrap aggregation)과 다른점
- 최종적으로 weak learner로 부터 출력을 결합해 더 좋은 예측율을 갖는 Strong learner를 만든다.