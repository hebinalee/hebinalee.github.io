---
layout: post
title: "가설검정과 A/B 테스트의 함정: p-value는 무엇이 아닌가"
date: 2026-09-11 09:00:00 +0900
tags: [probability, statistics, hypothesis-testing, ab-test]
---

"p-value가 0.03이니까 이 실험안이 더 좋을 확률이 97%네요." 회의에서 이런 문장이 나오면 넘어가기 쉽지만, 여기엔 틀린 부분이 있습니다. p-value는 가설이 맞을 확률이 아닙니다. 이 오해가 왜 생기는지, 그리고 실제 A/B 테스트에서 어떤 방식으로 결과를 망가뜨리는지 정리해봤습니다.

{% assign bayes_post = site.notes | where_exp: "doc", "doc.path contains 'conditional-probability-bayes'" | first %}

### p-value의 정확한 정의

p-value는 **귀무가설이 참이라고 가정했을 때, 관측된 것만큼 혹은 그보다 더 극단적인 결과가 나올 확률**입니다. 수식으로 쓰면 P(데이터 | 귀무가설)입니다.

우리가 실제로 알고 싶은 건 P(귀무가설 | 데이터), 즉 "이 데이터를 봤을 때 귀무가설이 틀렸을 확률"입니다. 앞선 [베이즈 정리 글]({{ bayes_post.url | relative_url }})에서 다룬 P(A|B)와 P(B|A)의 혼동이 그대로 반복되는 지점입니다. 이 둘을 연결하려면 사전 확률이 필요한데, p-value는 그걸 전혀 포함하지 않습니다. 그래서 p-value만 보고 "가설이 맞을 확률"을 말할 수 없습니다.

### 1종 오류, 2종 오류, 그리고 검정력

- **1종 오류(Type I error, α)**: 실제로는 차이가 없는데 있다고 판단 (거짓 양성)
- **2종 오류(Type II error, β)**: 실제로는 차이가 있는데 없다고 판단 (거짓 음성)
- **검정력(Power, 1-β)**: 실제로 존재하는 차이를 제대로 잡아낼 확률

유의수준 α를 0.05로 잡는다는 건 "차이가 없는데도 있다고 잘못 말할 확률을 5%까지 허용한다"는 뜻입니다. 그런데 이 5%라는 보장은 **정해진 시점에 단 한 번 검정한다는 전제** 위에서만 성립합니다. 실무의 두 가지 함정이 전부 이 전제를 깨뜨립니다.

### 함정 1: 중간에 계속 들여다보기 (Peeking)

실험을 돌려놓고 매일 대시보드를 확인하다가 "어, 지금 유의하게 나왔네, 여기서 끝내자"라고 판단하는 경우입니다. 이게 왜 문제일까요. 중간에 볼 때마다 암묵적으로 별도의 검정을 한 번씩 하는 셈이 되고, 각 검정마다 5%의 거짓 양성 기회가 생깁니다. α=0.05에서 10번 들여다보면 실제 거짓 양성률은 20%를 넘어갑니다.

더 고약한 건 우연히 유의해지는 순간이 언젠가는 오기 때문에, 효과가 전혀 없는 실험도 충분히 오래 지켜보면 "성공"으로 끝낼 수 있다는 점입니다. 해결책은 단순합니다. 실험 전에 필요한 표본 크기나 기간을 미리 계산해두고 그때까지 결정을 미루는 것. 중간 확인이 꼭 필요하다면 순차 검정(sequential testing)처럼 조기 종료를 전제로 설계된 방법을 써야 합니다.

### 함정 2: 여러 개를 동시에 검정하기 (Multiple Testing)

변형 A/B/C/D를 동시에 돌리거나, 하나의 실험에서 전환율·체류시간·재방문율·매출 등 지표 10개를 같이 본다면, 검정 횟수가 그만큼 늘어납니다. 각 검정이 독립이라고 가정하면, 10번 검정했을 때 적어도 하나가 우연히 유의하게 나올 확률은 1 - 0.95^10 ≈ 40%입니다. 실제로는 아무 효과가 없어도 "뭔가 하나는 유의하게" 나오는 게 자연스러운 상황인 겁니다.

가장 간단한 보정은 **본페로니 교정(Bonferroni correction)**으로, 유의수준을 검정 횟수로 나눕니다(α/m). 10개를 검정하면 0.05 대신 0.005를 기준으로 쓰는 거죠. 다만 이 방법은 꽤 보수적이라, 검정 수가 많아지면 검정력이 크게 떨어지고 2종 오류(실제 효과를 놓치는 것)가 늘어납니다. 1종 오류를 줄이는 대가를 2종 오류로 치르는 트레이드오프입니다. 그래서 실무에서는 애초에 **주요 지표(primary metric) 하나를 미리 정해두고** 나머지는 탐색적 지표로 구분하는 방식을 함께 씁니다.

### 코드로 확인하기: 들여다볼수록 올라가는 거짓 양성률

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

def simulate(peeks, n_total=2000, n_sim=5000, alpha=0.05):
    """A와 B가 실제로 '완전히 동일'한데도 유의하다고 판단하는 비율"""
    false_positives = 0
    check_points = np.linspace(n_total // peeks, n_total, peeks).astype(int)

    for _ in range(n_sim):
        a = rng.normal(0, 1, n_total)
        b = rng.normal(0, 1, n_total)  # 효과 없음: 동일한 분포
        for n in check_points:
            _, p = stats.ttest_ind(a[:n], b[:n])
            if p < alpha:
                false_positives += 1
                break  # 유의하게 나온 시점에 실험 종료

    return false_positives / n_sim

for peeks in [1, 5, 10, 20]:
    print(f"{peeks:2d}번 확인 → 거짓 양성률 {simulate(peeks):.1%}")
# 1번 확인 → 약 5%  (의도한 수준)
# 10번 확인 → 20% 이상
```

효과가 전혀 없는 두 집단을 비교하는 시뮬레이션인데, 확인 횟수만 늘려도 거짓 양성률이 의도한 5%에서 크게 벗어납니다. 데이터를 조작한 것도 아니고 검정 방법을 바꾼 것도 아니라, "언제 멈출지를 결과를 보고 정했다"는 것만으로 생기는 왜곡입니다.

### 정리하면

p-value는 "귀무가설이 참일 때 이런 데이터가 나올 확률"이고, 우리가 듣고 싶어 하는 "이 가설이 맞을 확률"이 아닙니다. 그리고 α=0.05라는 보장은 한 번만 검정할 때 성립하는 약속이라, 중간에 들여다보거나 여러 지표를 동시에 보는 순간 깨집니다. 결국 통계적으로 올바른 A/B 테스트의 상당 부분은 계산이 아니라 **실험을 시작하기 전에 무엇을 언제 어떻게 판단할지 미리 정해두는 설계**의 문제인 것 같습니다.

---

**참고한 자료**

- [A/B Testing: Avoiding the Peeking Problem (GoPractice)](https://gopractice.io/data/peeking-problem/)
- [Where Experimentation goes wrong (GrowthBook Docs)](https://docs.growthbook.io/using/experimentation-problems)
- [How to use Bonferroni correction for multiple hypothesis testing (Statsig)](https://www.statsig.com/perspectives/bonferroni-correction-multiple-testing)
- [Addressing Common Misuses and Pitfalls of P values in Biomedical Research (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9354639/)
