---
layout: post
title: "데이터 누수: 검증 점수가 좋아지는데도 문제인 이유"
date: 2026-09-22 09:00:00 +0900
tags: [machine-learning, ml-fundamentals, data-leakage, cross-validation]
section: ml-concepts
---

대부분의 버그는 에러를 내거나 성능을 떨어뜨리기 때문에 금방 드러납니다. 데이터 누수(data leakage)는 반대입니다. 검증 점수가 **올라갑니다**. 눈에 보이는 신호가 전부 "잘 되고 있다"는 방향을 가리키기 때문에, 배포하고 나서야 문제가 드러나는 경우가 많습니다.

### 누수란 무엇인가

데이터 누수는 실제 예측 시점에는 알 수 없는 정보가 학습이나 검증 과정에 흘러들어간 상태를 말합니다. 정의의 핵심은 "부정확한 정보가 들어갔다"가 아니라 **"그 시점에 가질 수 없는 정보가 들어갔다"**는 것입니다. 정보 자체는 정확하기 때문에 모델은 그것을 성실하게 학습하고, 검증 점수는 정직하게 올라갑니다. 다만 그 점수가 배포 환경에서 재현되지 않을 뿐입니다.

### 유형 1: 타겟 누수

예측하려는 값의 정보가 피처에 섞여 들어간 경우입니다. 이탈 예측 모델에 "이탈 사유 코드" 같은 컬럼이 들어가 있거나, 대출 부도 예측에 "연체 후 조치 여부"가 들어가 있는 식입니다. 예측하는 순간에는 존재하지 않는 값인데, 과거 데이터에는 기록되어 있어서 자연스럽게 딸려 들어옵니다.

이 유형은 단일 피처의 중요도가 비정상적으로 높거나, 검증 점수가 상식적인 수준을 넘어설 때 의심해볼 수 있습니다. "이 컬럼의 값은 예측을 실행하는 시점에 실제로 알 수 있는가"를 피처마다 물어보는 게 가장 확실한 점검 방법입니다.

### 유형 2: 전처리 누수

더 흔하고 더 잡기 어려운 쪽입니다. 스케일링, 결측치 대치, 피처 선택, 오버샘플링, 타겟 인코딩 같은 전처리를 **분할 전에 전체 데이터로 수행**하면, 검증 세트의 통계가 학습 과정에 스며듭니다.

```python
# 흔히 저지르는 형태
scaler = StandardScaler().fit(X)          # 전체 데이터의 평균·분산 사용
X_scaled = scaler.transform(X)
X_tr, X_te = train_test_split(X_scaled)   # 이미 늦음
```

여기서 `scaler`가 본 평균과 분산에는 테스트 세트의 정보가 들어 있습니다. 효과가 미미해 보일 수 있지만, 결측치 대치나 타겟 인코딩처럼 레이블이 개입하는 전처리에서는 점수 차이가 크게 벌어집니다.

교차검증에서는 이 문제가 한 겹 더 꼬입니다. 교차검증 전에 전처리를 끝내두면, fold를 나누는 의미 자체가 사라집니다. **전처리는 각 fold 안에서 학습 데이터만 보고 fit되어야 합니다.** scikit-learn의 `Pipeline`이 존재하는 이유가 정확히 이것입니다.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 파이프라인으로 묶으면 각 fold의 학습 부분에서만 fit 됨
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scores = cross_val_score(pipe, X, y, cv=5, scoring="average_precision")
```

### 유형 3: 시계열에서의 미래 정보 참조

시간 순서가 있는 데이터에 일반적인 k-fold를 적용하면, 미래 데이터로 학습하고 과거 데이터를 맞히는 fold가 생깁니다. 현실에서는 불가능한 설정이므로 검증 점수가 낙관적으로 나옵니다. 시계열 예측에서 검증 방식에 따라 오차가 20% 이상 차이 난다는 보고도 있습니다.

해결책은 시간 기준으로 자르는 것입니다. 특정 시점 이전을 학습, 이후를 검증으로 쓰거나 `TimeSeriesSplit`처럼 순서를 보존하는 분할을 씁니다. 피처를 만들 때도 마찬가지로, 이동평균이나 집계 피처가 예측 시점 이후의 값을 포함하지 않는지 확인해야 합니다.

### 어떻게 알아챌까

누수는 에러를 내지 않으므로 능동적으로 찾아야 합니다. 실무적으로 유용한 신호는 이런 것들입니다.

- 검증 점수가 문제 난이도에 비해 지나치게 높다
- 특정 피처 하나를 빼면 성능이 급격히 무너진다
- 검증 점수와 실제 운영 지표의 격차가 크고 일관되게 벌어진다
- fold마다 점수 편차가 이상하게 작거나, 반대로 설명되지 않게 튄다

특히 첫 번째는 그냥 넘기기 쉬운데, "생각보다 잘 나왔다"는 결과는 축하할 일이 아니라 점검해볼 일에 가깝습니다.

### 정리하면

앞선 [분류 지표 글]({% assign metrics = site.notes | where_exp: "doc", "doc.path contains 'classification-metrics'" | first %}{{ metrics.url | relative_url }})이 "무엇을 측정할 것인가"의 문제였다면, 누수는 "그 측정이 애초에 믿을 만한가"의 문제입니다. 지표를 아무리 정교하게 골라도 검증 설계가 오염되어 있으면 그 숫자는 의미가 없습니다. 분할을 먼저 하고, 전처리는 파이프라인 안에서, 시간이 있는 데이터는 시간 순서대로. 규칙 자체는 단순한데 지키기 쉽지 않다는 게 이 주제의 특징인 것 같습니다.

---

**참고한 자료**

- [Leakage (machine learning) — Wikipedia](https://en.wikipedia.org/wiki/Leakage_(machine_learning))
- [Data Leakage in Machine Learning: Why You Must Split Before Preprocessing](https://pub.towardsai.net/data-leakage-in-machine-learning-why-you-must-split-before-preprocessing-3ddc3dcde4e9)
- [Hidden Leaks in Time Series Forecasting (arXiv)](https://arxiv.org/html/2512.06932v1)
