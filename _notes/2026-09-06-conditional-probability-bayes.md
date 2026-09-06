---
layout: post
title: "조건부 확률과 베이즈 정리: P(A|B)와 P(B|A)는 다르다"
date: 2026-09-06 09:00:00 +0900
tags: [probability, statistics, bayes-theorem]
---

확률·통계를 다시 훑어보면서 가장 먼저 헷갈렸던 부분이 이거였습니다. P(A|B)와 P(B|A)는 분명 다른 값인데, 말로 풀어놓으면 은근히 같은 것처럼 읽힙니다. "이 병에 걸렸을 때 양성이 나올 확률"과 "양성이 나왔을 때 이 병에 걸렸을 확률"은 전혀 다른 질문인데도요.

### 조건부 확률: B가 일어났다는 전제 위에서

조건부 확률 P(A|B)는 "B가 일어났다는 조건 하에서 A가 일어날 확률"입니다. 정의는 다음과 같습니다.

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

여기서 중요한 건 분모가 전체 표본 공간이 아니라 **B로 좁혀진 세계**라는 점입니다. B가 일어난 경우들만 모아놓고, 그 안에서 A가 차지하는 비율을 보는 것이 조건부 확률입니다.

### 베이즈 정리: 조건을 뒤집는 공식

베이즈 정리는 P(B|A)를 알 때 P(A|B)를 구하는 방법입니다.

$$P(A|B) = \frac{P(B|A) \, P(A)}{P(B)}$$

말로 풀면 이렇습니다. 어떤 가설 A에 대한 사전 확률 P(A)가 있고, 새로운 증거 B를 관찰했을 때, 이 증거를 반영해서 P(A)를 P(A|B)로 업데이트하는 것이 베이즈 정리입니다. P(A)는 **사전 확률(prior)**, P(A|B)는 **사후 확률(posterior)**, P(B|A)는 **가능도(likelihood)**라고 부릅니다.

### 왜 이 둘을 자주 헷갈릴까: 검사 예시

가장 유명한 예시가 의료 검사입니다. 어떤 병의 유병률이 1%이고, 검사의 정확도가 90%(양성/음성 모두)라고 해봅시다. "검사 정확도 90%"라는 말만 들으면 양성이 나왔을 때 병에 걸렸을 확률도 90%에 가까울 거라 생각하기 쉽습니다. 하지만 실제로 계산해보면 전혀 다릅니다.

1,000명 중 실제 환자는 10명(1%)이고, 나머지 990명은 건강합니다.

- 환자 10명 중 90%가 양성 → 참양성(true positive) 9명
- 건강한 990명 중 10%가 잘못 양성 → 위양성(false positive) 99명

양성 판정을 받은 사람은 총 108명(9+99)인데, 그중 실제 환자는 9명뿐입니다. 즉 **양성이 나왔을 때 실제로 병에 걸렸을 확률은 90%가 아니라 약 8.3%(9/108)**입니다. 검사의 민감도(sensitivity, P(양성|환자))가 90%로 높아도, 유병률(base rate)이 낮으면 양성 예측도(precision, P(환자|양성))는 크게 떨어질 수 있습니다. 이 사전 확률을 무시하고 판단하는 오류를 **기저율 무시(base rate fallacy)**라고 부릅니다.

### 코드로 확인하기

```python
def bayes_posterior(prior, sensitivity, false_positive_rate):
    """P(질병|양성)을 계산"""
    p_disease = prior
    p_positive_given_disease = sensitivity
    p_positive_given_healthy = false_positive_rate

    # P(양성) = P(양성|환자)P(환자) + P(양성|건강)P(건강)
    p_positive = (
        p_positive_given_disease * p_disease
        + p_positive_given_healthy * (1 - p_disease)
    )

    return (p_positive_given_disease * p_disease) / p_positive


posterior = bayes_posterior(prior=0.01, sensitivity=0.9, false_positive_rate=0.1)
print(f"{posterior:.1%}")  # 8.3%
```

유병률(prior)을 0.01에서 0.3으로 바꿔보면 사후 확률이 극적으로 올라가는 걸 확인할 수 있습니다. 결국 "증거가 얼마나 강한가"만큼 "애초에 사전 확률이 얼마였는가"가 결과를 좌우합니다.

### ML에서는 어디에 쓰이나

이 원리는 스팸 필터에도 그대로 적용됩니다. 어떤 단어가 포함된 메일 중 실제로 스팸인 비율(P(스팸|단어))을 구하려면, 그 단어가 스팸에서 나올 확률(P(단어|스팸))과 전체 메일 중 스팸의 비율(P(스팸), prior)을 같이 봐야 합니다. 특정 단어가 스팸에 자주 등장한다는 사실만으로 그 메일이 스팸이라고 단정하면, 유병률을 무시한 것과 같은 실수를 하게 됩니다. 나이브 베이즈(Naive Bayes) 분류기는 이 계산을 각 단어(피처)에 대해 독립이라고 가정하고 확장한 것입니다.

### 정리하면

P(A|B)와 P(B|A)를 같은 것으로 취급하면 안 됩니다. 베이즈 정리는 이 둘을 사전 확률로 연결해주는 다리이고, 사전 확률(기저율)을 무시하면 아무리 정확도 높은 검사나 모델도 실제 의미와는 동떨어진 숫자를 내놓을 수 있습니다. 이후 가설검정을 다룰 때도 이 기저율 감각이 계속 등장할 것 같습니다.

---

**참고한 자료**

- [Bayes' Theorem in email spam filtering (Cornell Networks Course Blog)](https://blogs.cornell.edu/info2040/2018/10/27/bayes-theorem-in-email-spam-filtering/)
- [Base rate fallacy — Wikipedia](https://en.wikipedia.org/wiki/Base_rate_fallacy)
- [Bayes' Theorem — Formula, the Medical-Test Example & Base Rates (Cogn-IQ Encyclopedia)](https://www.cogn-iq.org/learn/theory/bayes-theorem/)
