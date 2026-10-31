---
layout: post
title: "불균형 데이터, SMOTE보다 먼저 봐야 할 것"
date: 2026-10-31 09:00:00 +0900
tags: [machine-learning, ml-fundamentals, imbalanced-data, calibration]
section: ml-concepts
---

클래스 비율이 99:1일 때 무엇부터 할까요. 반사적으로 떠오르는 건 SMOTE 같은 오버샘플링인데, 여러 데이터셋에 걸친 비교 연구들이 가장 일관되게 효과를 보고하는 건 더 단순한 쪽입니다. 임계값을 조정하는 것.

### 먼저: 불균형 자체가 문제인가

비율이 치우쳤다는 사실만으로는 문제가 아닐 수 있습니다. 실제로 문제가 되는 건 보통 셋 중 하나입니다.

1. **소수 클래스의 절대 개수가 적다** — 비율이 아니라 개수입니다. 양성이 1%여도 10만 건이면 1,000건이고, 학습이 가능합니다. 반대로 비율이 같아도 전체가 500건이면 양성 5건으로는 아무것도 못 합니다.
2. **오류 비용이 비대칭이다** — 위양성과 위음성의 대가가 다른데 모델은 그걸 모릅니다.
3. **평가 지표를 잘못 골랐다** — [분류 지표 글]({% assign met = site.notes | where_exp: "doc", "doc.path contains 'classification-metrics'" | first %}{{ met.url | relative_url }})에서 다룬 문제입니다. 정확도를 보고 있으면 불균형이 문제처럼 보이지만, 실은 지표가 문제인 경우가 많습니다.

2번과 3번은 데이터를 건드려서 풀 문제가 아닙니다. 이 구분을 먼저 해야 리샘플링이 필요한 상황인지 알 수 있습니다.

### 1순위: 임계값 조정

모델은 확률을 출력하고, 0.5에서 자르는 건 관례일 뿐 수학적 근거가 없습니다. 비용이 비대칭이면 최적 임계값도 0.5가 아닙니다. 위음성이 위양성보다 10배 비싸다면 더 낮은 임계값에서 잘라야 합니다.

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

def best_threshold_by_cost(y_true, proba, cost_fp=1.0, cost_fn=10.0):
    """비용이 최소가 되는 임계값 찾기"""
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        costs.append(fp * cost_fp + fn * cost_fn)
    return thresholds[int(np.argmin(costs))], min(costs)

# 검증 세트에서 정하고, 테스트 세트에는 고정해서 적용한다
t_opt, _ = best_threshold_by_cost(y_val, model.predict_proba(X_val)[:, 1])
y_pred_test = (model.predict_proba(X_test)[:, 1] >= t_opt).astype(int)
```

데이터도 손실 함수도 건드리지 않고, 학습된 모델의 출력을 어디서 자를지만 바꿉니다. 비용 구조가 바뀌면 재학습 없이 임계값만 다시 정하면 된다는 점도 운영상 장점입니다. 임계값은 **검증 세트에서 정하고 테스트에는 고정**해야 하는데, 테스트 성능을 보면서 임계값을 고르면 그건 누수입니다.

### 2순위: 클래스 가중치

손실 함수에서 소수 클래스의 오류에 더 큰 벌점을 주는 방식입니다. scikit-learn은 `class_weight="balanced"`, 부스팅 계열은 `scale_pos_weight`로 제공합니다. 데이터를 복제하거나 합성하지 않으므로 학습 시간이 늘지 않고, 구현도 한 줄입니다.

### 3순위: 리샘플링(SMOTE 등)

SMOTE는 소수 클래스 샘플들 사이를 보간해서 합성 샘플을 만듭니다. 여전히 쓸 만한 경우가 있습니다. 소수 클래스가 공간상에 희박하게 흩어져 있을 때, 모델이 약한 분류기일 때, 그리고 확률이 아니라 하드 레이블만 쓸 수 있는 상황일 때입니다.

반대로 위험한 경우도 분명합니다. 클래스가 겹쳐 있거나 노이즈가 있는 데이터에서는 **보간이 애매한 지점에 가짜 샘플을 만들어** 경계를 흐립니다. 그리고 최신 그래디언트 부스팅 모델에 임계값 조정을 함께 쓰면, 확률 기반 지표에서 SMOTE의 추가 이득이 거의 사라진다는 보고도 있습니다.

주의할 점이 하나 더 있습니다. [데이터 누수 글]({% assign leak = site.notes | where_exp: "doc", "doc.path contains 'data-leakage'" | first %}{{ leak.url | relative_url }})에서 다룬 전처리 누수의 교과서적 사례가 리샘플링입니다. 교차검증 전에 전체 데이터를 리샘플링하면, 합성된 샘플의 원본이 검증 fold에 들어가 점수가 부풀려집니다. **리샘플러는 반드시 fold 안에서만** fit되어야 합니다.

```python
from imblearn.pipeline import Pipeline       # sklearn이 아니라 imblearn의 Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

pipe = Pipeline([("smote", SMOTE()),          # fold의 학습 부분에만 적용됨
                 ("clf", HistGradientBoostingClassifier())])
scores = cross_val_score(pipe, X, y, cv=5, scoring="average_precision")
```

### 공통 부작용: 확률이 망가진다

이게 가장 자주 간과되는 부분입니다. **클래스 가중치와 리샘플링은 둘 다 예측 확률을 왜곡합니다.** 모델이 소수 클래스를 과대추정하도록 밀어붙이기 때문입니다. 한 벤치마크에서는 SMOTE가 확률 신뢰도를 눈에 띄게 떨어뜨리고, 클래스 가중치가 그보다 더 나빴다고 보고합니다.

하드 레이블만 쓴다면 큰 문제가 아닐 수 있습니다. 하지만 출력 확률을 **실제 위험도로 해석해서 쓴다면** — 기대 손실 계산, 우선순위 큐, 가격 책정 등 — 왜곡된 확률은 그대로 잘못된 의사결정이 됩니다. 이 경우 `CalibratedClassifierCV`(Platt scaling, isotonic regression) 같은 방법으로 재보정을 거쳐야 합니다.

임계값 조정이 1순위로 꼽히는 이유도 여기에 있습니다. 모델이 추정한 확률분포를 건드리지 않고, 그 위에서 결정 규칙만 바꾸기 때문입니다.

### 정리하면

순서는 이렇게 잡는 게 합리적으로 보입니다. 먼저 지표를 올바르게 고르고(PR-AUC 등), 비용 구조를 명시해서 임계값을 정하고, 그래도 부족하면 클래스 가중치를 쓰고, 소수 클래스가 희박한 특수한 경우에만 리샘플링을 고려하는 것. 그리고 가중치나 리샘플링을 썼다면 출력 확률을 그대로 믿지 않는 것. "불균형을 없애는 것"이 목표가 아니라 "비대칭한 비용에 맞는 결정을 내리는 것"이 목표라는 게 이 주제의 핵심인 것 같습니다.

---

**참고한 자료**

- [Balancing the Scales: A Comprehensive Study on Tackling Class Imbalance in Binary Classification (arXiv)](https://arxiv.org/pdf/2409.19751)
- [When Single-Dataset Conclusions Fail: A 45-Task Study of Threshold Tuning and Resampling for Imbalanced Classification (arXiv)](https://arxiv.org/pdf/2608.16147)
- [Class Imbalance in Machine Learning: Why It's Rarely the Problem (Train in Data)](https://blog.trainindata.com/class-imbalance-in-machine-learning/)
- [Handling Class Imbalance in ML: Better Alternatives to SMOTE (Analytics Vidhya)](https://www.analyticsvidhya.com/blog/2026/07/class-imbalance-ml/)
