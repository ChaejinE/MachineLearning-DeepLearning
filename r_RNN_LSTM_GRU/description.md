# RNN(Recurrent Nueral Network)
- recurrent : 순환적인
  - 신호가 한쪽 방향으로 흘러가는 것이 아니라 순환 구조를 갖는다.
  - 기존 신경망들은 신호의 흐름이 입력에서 출력으로만 즉, 순방향으로만 전개되는 것들이 대부분이었다. 실제 신경망은 이렇지 않다.

- 내부에 과거의 상태를 저장하는 메모리를 갖고있어 순차적인 문제나 맥락을 파악하는 경우나 시간에 대한 의존성을 갖고있는 문제 해결에 적합하다.

![image](https://user-images.githubusercontent.com/69780812/138876338-9e9039f7-d958-4c2a-945c-f81dbd53afb3.png)

- RNN과 Feed Forawrd 신경망의 차이를 보여주는 그림이다.

- [RNN 설명 추천](http://aikorea.org/blog/rnn-tutorial-1/)
- [RNN 설명 추천2](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## RNN 문제점
![image](https://user-images.githubusercontent.com/69780812/138876830-46902b98-c23f-4035-b18d-67939106251c.png)

- 순환 구조를 풀어서 생각하면 이해하기 쉽다.
- RNN의 결과는 Cell에 저장된 이전 State와 입력에 의해서 결정되는 구조다.
  - RNN의 학습 방식은 BPTT(Back-Propagation Thorugh Time)방식을 취한다.
  - 펼쳐놓은 상태에서 back-propagtion을 시키게 된다. 여기서 Feed-forward와 달리 파라미터를 공유한다는 점에서 차이가 있다.
  - 시간 domain에서 펼쳤을 뿐이고 원래 망에서는 순환 구조라서 동일한 파라미터를 사용한다.
- RNN을 사용하여 장시간 데이터 의존도가 있는 문제가 있다.
  - long term 메모리가 필요하다.
  - 현재 상태가 계속 이전 상태와 연관이있다.
  - BPTT를 통한 연산에서 chain rule에 의해 연결의 길이가 매우 길어지게된다. 결과적으로 신경망에서 생기는 Vanishing/Exploding gradient 문제로 인해 학습하기가 어려워진다.
- 결과적으로 순수 RNN 만으로 장시간에 걸친 의존도 문제에 대해 풀기가 어려워진다.

# LSTM
![image](https://user-images.githubusercontent.com/69780812/138877505-0eb9a337-6bd8-46cd-9864-7699dbd3c434.png)

- RNN으로 잘 해결될 수 없는 **장기 메모리가 필요한 문제를 해결**하기 위해 개발되었다.
- 위 그림은 전통적인 RNN 구조다.

![image](https://user-images.githubusercontent.com/69780812/138877656-51e9f977-0e09-4778-a337-3470752fe9f3.png)

- LSTM의 구조이며 RNN 구조보다 좀 더 복잡하게 이루어져있다.
- 장기간 메모리 역할을 수행하는 Cell state와 연결 강도를 조절하는 3개의 Gate(forget, input, output)으로 구성된다.
  - Cell State는 위쪽으로 수평으로만 흐르는 라인이다.
- Cell State는 Gate 조절을 통해 **이전 State가 현재 State로 끼치는 영향을 조절**할 수 있다. 또한, **현재 입력과 연관된 정보를 추가**할 수도 있으며 **다시 출력에 끼치는 영향의 수준을 정할 수 있게 된다.**
- 결과적으로 장기간 메모리가 필요한 문제가 해결이 가능하게 되었고 ㅁ낳은 분야에서 성공적으로 활용이 된다.

# GRU(Gated Recurrent Unit)
- LSTM의 구조를 변경한 LSTM류 구조다.
- "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", 조경현에서 처음 소개됐다.
  - "Empirical Evaluation of Gated Recurrent Nueral Networks on Sequence Modeling", 정준영
  - 위 tanh 기반의 RNN, LSTM, GRU 비교 내용을 보는 것이 더 좋다.

![image](https://user-images.githubusercontent.com/69780812/138878935-5971b189-0865-4cdc-ab24-0d83aaa8b9cd.png)

- LSTM과 GRU 구조이다.
- GRU 구조가 LSTM에 비해 더 간결하다.
- LSTM과 마찬가지로 Gate를 이용해 정보의 양을 조절하는 것은 동일하다.
  - Gate 제어 방식에서 차이가 있다.
- GRU는 update, reset 2개의 gate가 있다.

![image](https://user-images.githubusercontent.com/69780812/138879221-f355c789-bc9e-4522-9a76-81c397086bfc.png)

- 시간 t에서 GRU의 Activation은 과거의 activation h(t-1)과 후보 activation h(t)와의 interpolation을 통해 결정난다.
  - 여기서 interpolation 비율은 update gate z를 통해 결정된다.
- 결과적으로 GRU에서는 LSTM의 Foget과 Input gate를 결합하여 1개의 Update gate를 만든 것이다.
- 또한, 별도의 Cell State와 Hidden state를 Hidden State로 묶은 셈이 되었다.