---
layout: post
title: "95% 신뢰구간의 95%는 무엇에 대한 숫자인가, 그리고 부트스트랩"
date: 2026-10-04 09:00:00 +0900
tags: [probability, statistics, confidence-interval, bootstrap]
section: probability
---

95% 신뢰구간으로 [3.2, 4.8]을 얻었다고 해봅시다. "참값이 이 구간 안에 있을 확률이 95%"라고 읽으면 틀립니다. 그런데 이게 왜 틀렸는지 설명하려면 생각보다 말이 길어지고, 그러다 보니 실무에서는 대충 그렇게 읽고 넘어가게 됩니다. 95%가 정확히 무엇에 붙는 숫자인지 정리해봤습니다.

### 95%는 구간이 아니라 절차에 붙는다

빈도주의 관점에서 모수(참값)는 고정된 상수입니다. 우리가 모르는 값일 뿐, 확률변수가 아닙니다. 랜덤한 쪽은 **표본**이고, 따라서 표본에서 계산한 **구간**입니다.

그래서 올바른 해석은 이렇습니다. **같은 방식으로 표본을 뽑아 구간을 만드는 절차를 아주 많이 반복하면, 그렇게 만들어진 구간들 중 약 95%가 참값을 포함한다.** 지금 내 손에 있는 [3.2, 4.8] 하나는 참값을 포함하거나 포함하지 않습니다. 그 확률은 1 또는 0이고, 다만 우리가 어느 쪽인지 모릅니다.

"참값이 이 구간에 있을 확률"을 말하고 싶다면 그건 다른 개념입니다. 모수에 확률을 부여하려면 사전분포가 필요하고, 그렇게 얻는 건 신뢰구간이 아니라 **신용구간(credible interval)**입니다. [베이즈 정리 글]({% assign bayes = site.notes | where_exp: "doc", "doc.path contains 'conditional-probability-bayes'" | first %}{{ bayes.url | relative_url }})에서 본 P(데이터|가설)과 P(가설|데이터)의 구분이 여기서도 똑같이 반복됩니다. 신뢰구간은 전자 쪽 세계의 도구입니다.

### 가설검정과의 관계

[가설검정 글]({% assign ht = site.notes | where_exp: "doc", "doc.path contains 'hypothesis-testing' " | first %}{{ ht.url | relative_url }})과 연결되는 지점이 있습니다. 두 집단의 차이에 대한 95% 신뢰구간이 0을 포함하지 않으면, 유의수준 5%의 양측 검정에서 유의하다는 결론과 (보통) 일치합니다. 같은 정보를 다르게 보여주는 셈입니다.

다만 신뢰구간이 더 유용한 경우가 많습니다. p-value는 "차이가 있는가"에만 답하는데, 신뢰구간은 **차이가 어느 정도 크기인지, 그 불확실성이 얼마나 넓은지**를 같이 보여줍니다. "유의했다"보다 "전환율 차이가 0.3%p에서 2.1%p 사이로 추정된다"가 의사결정에 훨씬 유용합니다.

### 부트스트랩: 분포 가정 없이 구간 만들기

전통적인 신뢰구간 공식은 표본평균이 정규분포를 따른다는 가정(혹은 [중심극한정리]({% assign dist = site.notes | where_exp: "doc", "doc.path contains 'common-probability-distributions'" | first %}{{ dist.url | relative_url }}))에 기대고 있습니다. 문제는 평균이 아닌 통계량입니다. 중앙값, 90분위수, 두 모델의 AUC 차이, 지니계수 같은 값들은 표본분포를 수식으로 적기가 어렵거나 불가능합니다.

부트스트랩은 여기서 접근을 바꿉니다. 모집단에서 다시 표본을 뽑을 수 없으니, **내가 가진 표본을 모집단처럼 취급해서 복원추출로 재표집**합니다. 그렇게 만든 수천 개의 가상 표본에서 매번 통계량을 계산하면, 그 통계량의 분포를 경험적으로 얻게 됩니다. 이 분포의 2.5 분위수와 97.5 분위수를 취하면 95% 신뢰구간이 됩니다.

```python
import numpy as np

rng = np.random.default_rng(0)
data = rng.lognormal(mean=1.0, sigma=0.8, size=200)   # 비대칭 분포

def bootstrap_ci(x, stat=np.median, n_boot=10000, alpha=0.05):
    n = len(x)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)   # 복원추출이 핵심
        stats[i] = stat(sample)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi

print("중앙값:", np.median(data).round(3))
print("95% 부트스트랩 CI:", np.round(bootstrap_ci(data), 3))
```

표본평균뿐 아니라 어떤 통계량에도 같은 코드를 쓸 수 있다는 게 부트스트랩의 실용적인 장점입니다.

### 95%가 정말 95%인지 확인해보기

앞서 "절차를 반복하면 95%가 참값을 포함한다"고 했는데, 이건 시뮬레이션으로 직접 확인할 수 있습니다.

```python
true_median = np.exp(1.0)        # lognormal의 이론적 중앙값
hits = 0
trials = 500

for _ in range(trials):
    sample = rng.lognormal(mean=1.0, sigma=0.8, size=200)
    lo, hi = bootstrap_ci(sample, n_boot=1000)
    if lo <= true_median <= hi:
        hits += 1

print(f"참값을 포함한 구간의 비율: {hits / trials:.1%}")   # 95% 근처
```

여기서 드러나는 게 신뢰구간의 정체입니다. 95%는 **구간들의 집합이 가지는 장기적 성질**이고, 개별 구간 하나에 대해 말해주는 건 없습니다.

### 주의할 점

부트스트랩이 만능은 아닙니다. 작은 표본에서 단순 분위수(percentile) 부트스트랩은 t-구간보다 덜 정확하고, 표본이 커지면 반대로 더 정확해지는 경향이 보고됩니다. 더 근본적인 제약은 **원래 표본이 모집단을 제대로 대표하지 못하면 부트스트랩도 그 편향을 그대로 복제한다**는 점입니다. 재표집은 표본 안의 정보를 최대한 활용하는 기법이고, 표본에 없는 정보를 만들어내지는 않습니다.

### 정리하면

신뢰구간의 95%는 내 구간에 대한 확신도가 아니라 절차의 적중률입니다. 그리고 수식을 적기 어려운 통계량이라면 부트스트랩으로 같은 성질을 가진 구간을 만들 수 있습니다. 실무에서는 "유의했다/아니다"로 끊는 대신 구간을 함께 보고하는 습관이 결국 더 많은 정보를 전달하는 것 같습니다. 폭이 넓다면 결론을 내리기에 데이터가 부족하다는 뜻이기도 하니까요.

---

**참고한 자료**

- [Chapter 8: Bootstrapping and Confidence Intervals — ModernDive](https://moderndive.com/8-confidence-intervals.html)
- [Confidence Intervals by Bootstrapping Approach: A Significance Review](https://www.researchgate.net/publication/368809659_Confidence_Intervals_by_Bootstrapping_Approach_A_Significance_Review)
- [Bootstrap — Mathematical Tools for Neuroscience (Princeton, lecture notes)](https://pillowlab.princeton.edu/teaching/mathtools16/slides/lec21_Bootstrap.pdf)
