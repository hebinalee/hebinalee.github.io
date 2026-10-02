---
layout: post
title: "L1과 L2 정규화는 무엇이 다른가: 왜 L1만 가중치를 0으로 만드는지"
date: 2026-10-02 09:00:00 +0900
tags: [machine-learning, ml-fundamentals, regularization, lasso, ridge]
section: ml-concepts
---

[편향-분산 글]({% assign bv = site.notes | where_exp: "doc", "doc.path contains 'bias-variance' " | first %}{{ bv.url | relative_url }})에서 정규화를 "분산을 줄이는 대가로 편향을 조금 감수하는 거래"라고만 쓰고 넘어갔습니다. 그런데 실제로 코드를 쓸 때 먼저 마주치는 질문은 더 구체적입니다. L1이냐 L2냐, 그리고 L1은 왜 가중치를 아예 0으로 만드는가.

### 페널티 항의 차이

손실 함수에 붙는 항이 다릅니다.

$$L_{\text{ridge}} = \text{Loss} + \lambda \sum_j w_j^2 \qquad L_{\text{lasso}} = \text{Loss} + \lambda \sum_j |w_j|$$

L2(Ridge)는 가중치의 제곱합, L1(Lasso)은 절댓값의 합입니다. 둘 다 "가중치가 커지면 벌점을 준다"는 같은 목적인데, 결과는 질적으로 다릅니다. **L2는 모든 가중치를 작게 만들지만 0으로는 보내지 않고, L1은 일부 가중치를 정확히 0으로 만듭니다.** 그래서 Lasso는 그 자체로 피처 선택기처럼 동작합니다.

### 왜 L1만 0을 만드는가 (1): 기울기로 보기

페널티를 가중치로 미분해보면 차이가 선명합니다.

- L2: $\frac{d}{dw}w^2 = 2w$ → w가 0에 가까워지면 **밀어내는 힘도 0에 가까워집니다.** 그래서 계속 작아지기만 하고 0에 도달하지 못합니다.
- L1: $\frac{d}{dw}|w| = \text{sign}(w)$ → w가 0.001이든 10이든 **같은 크기의 힘이 작용합니다.** 데이터가 그 가중치를 붙잡아둘 이유가 충분하지 않으면 0까지 밀려가고, 거기서 멈춥니다.

"작게 만드는 힘"과 "0으로 보내는 힘"의 차이가 여기서 갈립니다.

### 왜 L1만 0을 만드는가 (2): 기하학으로 보기

같은 내용을 제약 영역으로 볼 수도 있습니다. 정규화는 "가중치 벡터의 크기가 일정 이하"라는 제약 안에서 손실을 최소화하는 문제로 바꿔 쓸 수 있는데, 그 제약 영역의 모양이 다릅니다. L2는 **원(구)**, L1은 **다이아몬드(마름모)**입니다.

손실 함수의 등고선은 타원으로 퍼져 나가면서 이 제약 영역과 처음 닿는 점에서 해가 결정됩니다. 다이아몬드에는 **축 위에 뾰족한 모서리**가 있고, 타원이 모서리에 닿으면 그 좌표값은 정확히 0이 됩니다. 반면 원에는 모서리가 없으니 어디서 닿아도 모든 좌표가 작지만 0은 아닌 값이 됩니다. L1의 희소성은 페널티 함수의 모양에서 나오는 기하학적 결과인 셈입니다.

### 상관된 피처가 있을 때의 차이

실무에서 더 중요한 차이는 이쪽입니다. 비슷한 정보를 담은 피처가 여러 개 있을 때,

- **Ridge**는 그 그룹에 가중치를 **나눠 줍니다.** 각자 조금씩 기여하게 됩니다.
- **Lasso**는 그중 **하나만 남기고 나머지를 0으로** 만듭니다. 모델은 깔끔해지지만, 거의 동일한 피처들 중 어느 것이 살아남는지는 다소 임의적입니다. 데이터가 조금 바뀌면 선택된 피처가 바뀔 수도 있습니다.

피처들이 실제로 관련된 것을 측정하고 있어서 함께 남기는 게 타당하다면 Ridge, 피처 수를 줄이는 게 목적이라면 Lasso가 맞습니다. 둘을 섞은 **ElasticNet**은 희소성을 유도하면서도 상관 그룹을 함께 살리는 중간 성질을 가집니다.

### 베이즈 관점에서 보면

[베이즈 정리 글]({% assign bayes = site.notes | where_exp: "doc", "doc.path contains 'conditional-probability-bayes'" | first %}{{ bayes.url | relative_url }})의 사전 확률 개념이 여기서도 나옵니다. 정규화는 가중치에 대한 사전분포를 가정하고 사후확률을 최대화하는 것으로 해석할 수 있습니다. L2는 가중치에 **가우시안 사전분포**를, L1은 0에서 날카롭게 뾰족한 **라플라스 사전분포**를 가정한 것에 대응합니다. "가중치는 대체로 0 근처일 것"이라는 믿음의 모양이 다르고, 라플라스가 0에 더 강한 확률 질량을 주기 때문에 해도 0이 많아집니다.

### 코드로 비교하기

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.datasets import make_regression

# 20개 피처 중 실제로 유효한 건 5개
X, y, true_coef = make_regression(n_samples=200, n_features=20, n_informative=5,
                                  noise=10, coef=True, random_state=0)

for name, model in [("Ridge   ", Ridge(alpha=1.0)),
                    ("Lasso   ", Lasso(alpha=1.0)),
                    ("ElasticNet", ElasticNet(alpha=1.0, l1_ratio=0.5))]:
    model.fit(X, y)
    zeros = np.sum(np.abs(model.coef_) < 1e-8)
    print(f"{name} | 0이 된 계수: {zeros:2d}/20 | 최대 계수: {np.abs(model.coef_).max():.1f}")

# Ridge:      0이 된 계수 0개  (전부 살아 있고 크기만 작아짐)
# Lasso:      0이 된 계수 다수 (유효 피처 쪽만 남김)
# ElasticNet: 그 사이
```

### 딥러닝에서는

신경망에서 흔히 쓰는 **weight decay가 사실상 L2 정규화**입니다. L1은 희소한 가중치를 원하는 경우(모델 압축, 프루닝 전처리 등)에 선택적으로 쓰이고, 일반적인 학습에서는 L2 + 드롭아웃 + 조기 종료 조합이 더 흔합니다. 파라미터가 수억 개인 모델에서 "피처 선택"이라는 L1의 장점이 큰 의미를 갖기 어렵다는 점도 이유입니다.

### 정리하면

L2는 가중치를 **고르게 줄이고**, L1은 일부를 **잘라냅니다.** 미분으로 보면 0 근처에서 작용하는 힘의 크기가 다르고, 기하학으로 보면 제약 영역에 모서리가 있는지 없는지의 차이입니다. 선택 기준은 단순합니다. 해석 가능한 적은 피처가 필요하면 L1, 상관된 피처들을 함께 살리고 안정적인 계수를 원하면 L2, 둘 다 어느 정도 필요하면 ElasticNet.

---

**참고한 자료**

- [L1 vs L2 Regularization: The Geometric Intuition Behind Each (MetricGate)](https://metricgate.com/blogs/regularization-l1-vs-l2-intuition/)
- [L1 vs L2 Regularization: Intuition, Geometry, and Bayesian View](https://medium.com/@iambeingferoz/l1-vs-l2-regularization-intuition-geometry-and-bayesian-view-da18a2b74de1)
- [Understanding L1 and L2 regularization (Weights & Biases)](https://wandb.ai/mostafaibrahim17/ml-articles/reports/Understanding-L1-and-L2-regularization-techniques-for-optimized-model-training--Vmlldzo3NzYwNTM5)
