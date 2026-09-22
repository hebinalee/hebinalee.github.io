---
layout: post
title: "편향-분산 트레이드오프, 그리고 딥러닝에서 깨지는 지점"
date: 2026-09-14 09:00:00 +0900
tags: [machine-learning, ml-fundamentals, bias-variance, regularization]
section: ml-concepts
---

학습 데이터에서는 잘 맞는데 실제 데이터에서는 성능이 뚝 떨어지는 모델, 반대로 학습 데이터에서조차 제대로 맞히지 못하는 모델. 둘 다 "성능이 안 나온다"는 같은 증상이지만 원인도 처방도 정반대입니다. 이 둘을 구분하는 틀이 편향-분산 분해입니다.

### 오차를 세 조각으로 나누기

지도학습 모델의 기대 예측 오차는 세 항으로 분해됩니다.

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \sigma^2$$

- **편향(Bias)**: 모델이 표현할 수 있는 함수의 한계 때문에 생기는 오차. 실제 관계가 곡선인데 직선으로만 맞추려 하면 아무리 데이터를 많이 줘도 좁혀지지 않습니다.
- **분산(Variance)**: 학습 데이터가 조금 달라졌을 때 모델이 얼마나 흔들리는가. 훈련셋의 우연한 특성까지 학습하면 데이터가 바뀔 때마다 예측이 크게 달라집니다.
- **줄일 수 없는 오차(Irreducible error, σ²)**: 데이터 자체의 노이즈. 모델을 아무리 잘 만들어도 제거할 수 없는 부분입니다.

과소적합(underfitting)은 편향이 지배적인 상태, 과적합(overfitting)은 분산이 지배적인 상태입니다. 선형회귀로 복잡한 관계를 맞추려는 게 전자, 깊이 제한 없는 결정트리가 훈련셋을 통째로 외워버리는 게 후자의 전형입니다.

### 왜 트레이드오프인가

모델 복잡도를 올리면 표현력이 커지니 편향은 줄어듭니다. 대신 데이터의 우연한 패턴까지 따라갈 여지가 생기므로 분산은 커집니다. 이 둘의 합이 최소가 되는 지점이 있고, 그래서 복잡도 대비 테스트 오차를 그리면 U자 곡선이 나온다 — 이것이 교과서적인 설명입니다.

정규화(regularization)는 이 곡선 위에서 위치를 옮기는 도구입니다. L2 정규화(Ridge)는 가중치가 커지는 걸 억제해서 모델이 데이터에 과하게 반응하지 못하게 만들고, 드롭아웃이나 조기 종료(early stopping)도 결국 같은 일을 합니다. **분산을 줄이는 대신 편향을 조금 감수하는 거래**인 셈입니다. 정규화 강도를 무한정 올리면 모델이 거의 상수 함수에 가까워지면서 이번엔 편향 때문에 성능이 나빠지는 것도 같은 이유입니다.

### 코드로 보는 U자 곡선

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, 80).reshape(-1, 1)
y = np.sin(X).ravel() + rng.normal(0, 0.3, 80)   # 노이즈 포함 (= irreducible error)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

for degree in [1, 3, 5, 9, 15]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression()).fit(X_tr, y_tr)
    train_mse = mean_squared_error(y_tr, model.predict(X_tr))
    test_mse = mean_squared_error(y_te, model.predict(X_te))
    print(f"차수 {degree:2d} | train {train_mse:.3f} | test {test_mse:.3f}")

# 차수가 낮을 때는 train/test 둘 다 나쁨 (편향 지배)
# 차수가 올라가면 train은 계속 좋아지지만 test는 어느 순간부터 나빠짐 (분산 지배)
```

train 오차와 test 오차가 벌어지기 시작하는 지점이 분산이 커지는 구간입니다. 두 오차가 모두 높으면 모델을 키우거나 피처를 추가해야 하고, train은 낮은데 test만 높으면 데이터를 늘리거나 정규화를 강화해야 합니다. 진단이 달라지면 처방도 달라집니다.

### 그런데 딥러닝에서는 이 그림이 그대로 맞지 않는다

고전적 설명대로라면 파라미터가 데이터 수보다 훨씬 많은 모델은 분산 때문에 망가져야 합니다. 그런데 현대의 대형 신경망은 훈련 데이터를 완전히 외울 수 있을 만큼 과매개변수화(over-parameterized)되어 있는데도 일반화 성능이 좋습니다.

이 모순을 설명하는 것이 **이중 하강(double descent)** 현상입니다. 모델 크기를 키우면 테스트 오차가 처음엔 U자 곡선을 따라 내려갔다 올라가지만, 훈련 데이터를 완전히 보간(interpolation)할 수 있는 지점을 지나 더 키우면 **테스트 오차가 다시 감소**합니다. 즉 교과서의 U자 곡선은 전체 그림의 왼쪽 절반에 해당하고, 그 오른쪽에 또 하나의 하강 구간이 존재한다는 것입니다.

이게 편향-분산 분해가 틀렸다는 뜻은 아닙니다. 분해 자체는 수학적으로 항상 성립합니다. 다만 "복잡도를 올리면 분산이 단조 증가한다"는 고전적 가정이 과매개변수화 영역에서는 성립하지 않는다는 것이고, 최근 연구들은 딥러닝에서 편향과 분산이 서로 상충하기보다 정렬(align)되는 양상도 보고하고 있습니다. 대형 모델을 다루면서 "파라미터를 더 키우면 과적합되지 않을까"라는 직관이 자꾸 빗나가는 이유가 여기에 있었습니다.

### 정리하면

편향과 분산은 성능 문제를 진단하는 언어입니다. train/test 오차가 어떻게 벌어지는지를 보고 지금 문제가 표현력 부족인지 과적합인지를 먼저 구분해야, 모델을 키울지 정규화를 걸지가 정해집니다. 다만 그 위에 얹혀 있던 "복잡할수록 과적합"이라는 경험칙은 딥러닝 스케일에서는 더 이상 그대로 통하지 않습니다. 분해는 유지하되 곡선의 모양에 대한 가정은 업데이트해야 하는 셈입니다.

---

**참고한 자료**

- [Reconciling modern machine-learning practice and the classical bias–variance trade-off (PNAS, Belkin et al.)](https://www.pnas.org/doi/10.1073/pnas.1903070116)
- [Double Descent — MLU-Explain](https://mlu-explain.github.io/double-descent/)
- [It's an Alignment, Not a Trade-off: Revisiting Bias and Variance in Deep Models (arXiv)](https://arxiv.org/pdf/2310.09250)
