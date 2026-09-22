---
layout: post
title: "정확도 99% 모델의 함정: 분류 지표 제대로 고르기"
date: 2026-09-20 09:00:00 +0900
tags: [machine-learning, ml-fundamentals, metrics, evaluation]
section: ml-concepts
---

정확도 99%짜리 모델을 만드는 가장 쉬운 방법은, 전체의 1%만 양성인 데이터에서 무조건 "음성"이라고 답하게 하는 것입니다. 아무것도 학습하지 않은 모델이 그럴듯한 숫자를 받아가는 이 상황이, 지표를 잘못 고르면 어떤 일이 벌어지는지 가장 압축적으로 보여줍니다.

### 혼동행렬에서 출발하기

이진 분류의 결과는 네 칸으로 정리됩니다. 참양성(TP), 위양성(FP), 참음성(TN), 위음성(FN). 주요 지표는 전부 이 네 값의 조합입니다.

- **정밀도(Precision) = TP / (TP + FP)**: 양성이라고 예측한 것 중 실제 양성의 비율. "이 알람을 믿어도 되는가"
- **재현율(Recall) = TP / (TP + FN)**: 실제 양성 중 모델이 찾아낸 비율. "놓친 게 얼마나 되는가"
- **F1 = 정밀도와 재현율의 조화평균**: 둘의 균형을 하나의 숫자로

정확도가 위험한 이유는 분모에 TN이 들어가기 때문입니다. 음성이 압도적으로 많으면 TN이 숫자를 지배해버려서, 정작 관심 있는 소수 클래스의 성능이 묻힙니다. 반대로 F1은 TN을 아예 쓰지 않습니다.

### 정밀도와 재현율은 임계값 하나로 맞바꿔진다

모델이 출력하는 건 보통 0과 1 사이의 확률이고, 어디서 자를지는 우리가 정합니다. 임계값을 낮추면 더 많은 걸 양성으로 잡아내니 재현율이 올라가지만 위양성도 함께 늘어 정밀도가 떨어집니다. 올리면 반대가 됩니다.

그래서 "정밀도와 재현율 중 뭐가 중요한가"는 사실 모델의 문제가 아니라 **위양성과 위음성 중 어느 쪽이 더 비싼가**의 문제입니다. 스팸 필터라면 중요한 메일이 스팸함에 들어가는 위양성이 더 치명적이니 정밀도 쪽에, 질병 선별 검사라면 환자를 놓치는 위음성이 더 치명적이니 재현율 쪽에 무게를 둡니다. 이 판단을 먼저 하지 않으면 어떤 지표를 봐도 결론이 안 납니다.

### ROC-AUC와 PR-AUC 중 무엇을 볼 것인가

임계값 하나에 묶이지 않고 모델 자체를 평가하려면 곡선 아래 면적을 씁니다. 문제는 두 곡선이 불균형 데이터에서 아주 다르게 반응한다는 점입니다.

ROC 곡선은 재현율(TPR)과 위양성률(FPR = FP / (FP + TN))의 관계를 그립니다. 여기서 FPR의 분모에 거대한 음성 집합(TN)이 들어갑니다. 음성이 99%인 데이터에서는 위양성이 꽤 늘어나도 FPR은 조금밖에 안 움직여서, ROC-AUC가 실제 체감 성능보다 낙관적으로 보일 수 있습니다.

PR 곡선은 정밀도와 재현율만 사용하므로 TN이 개입하지 않습니다. 소수 클래스를 얼마나 정확하게 잡아내는지에 집중하기 때문에, 양성이 희귀하고 그 양성을 찾는 게 목적인 문제(이상 탐지, 사기 거래 탐지 등)에서는 PR-AUC가 더 정직한 신호를 줍니다. 다만 ROC-AUC가 불균형 상황에서도 유효하다는 반론도 있어서, 둘 중 하나만 맹신하기보다 둘을 같이 보고 차이가 벌어지면 왜 벌어지는지 확인하는 편이 안전합니다.

그리고 여기서 앞서 다룬 [베이즈 정리]({% assign bayes = site.notes | where_exp: "doc", "doc.path contains 'conditional-probability-bayes'" | first %}{{ bayes.url | relative_url }})가 다시 등장합니다. 정밀도는 결국 P(실제 양성 \| 양성 예측)이고, 이 값은 기저율에 따라 크게 달라집니다. 같은 모델이라도 양성 비율이 낮은 환경에 배포하면 정밀도는 떨어집니다. 모델이 나빠진 게 아니라 사전 확률이 달라진 겁니다.

### 코드로 확인하기

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)

# 양성 1%인 불균형 데이터
X, y = make_classification(n_samples=20000, n_features=20, weights=[0.99, 0.01],
                           n_informative=5, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          stratify=y, random_state=0)

model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
proba = model.predict_proba(X_te)[:, 1]
pred = (proba >= 0.5).astype(int)

print(f"전부 음성이라 답할 때 정확도: {(y_te == 0).mean():.3f}")
print(f"모델 정확도:   {accuracy_score(y_te, pred):.3f}")
print(f"정밀도:        {precision_score(y_te, pred):.3f}")
print(f"재현율:        {recall_score(y_te, pred):.3f}")
print(f"F1:            {f1_score(y_te, pred):.3f}")
print(f"ROC-AUC:       {roc_auc_score(y_te, proba):.3f}")
print(f"PR-AUC:        {average_precision_score(y_te, proba):.3f}")
```

실행해보면 정확도는 아무 학습도 안 한 기준선과 거의 차이가 없고, ROC-AUC는 높게 나오는 반면 PR-AUC는 훨씬 낮게 나옵니다. 같은 모델, 같은 예측인데 어떤 숫자를 보고하느냐에 따라 "훌륭한 모델"도 "쓸 수 없는 모델"도 될 수 있다는 뜻입니다.

### 정리하면

지표를 고르는 일은 계산이 아니라 문제 정의에 가깝습니다. 클래스가 불균형한가, 위양성과 위음성 중 어느 쪽 비용이 큰가, 배포 환경의 양성 비율이 학습 데이터와 같은가. 이 세 가지를 먼저 정하고 나면 어떤 지표를 주 지표로 삼을지는 거의 따라옵니다. 순서를 바꿔서 지표부터 고르면, 보고하기 좋은 숫자를 찾는 일이 되기 쉽습니다.

---

**참고한 자료**

- [ROC AUC vs Precision-Recall for Imbalanced Data (Machine Learning Mastery)](https://machinelearningmastery.com/roc-auc-vs-precision-recall-for-imbalanced-data/)
- [Understanding F1 Score, Accuracy, ROC-AUC & PR-AUC Metrics (Deepchecks)](https://deepchecks.com/f1-score-accuracy-roc-auc-and-pr-auc-metrics-for-models/)
- [The receiver operating characteristic curve accurately assesses imbalanced datasets (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2666389924001090)
