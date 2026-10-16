---
layout: post
title: "심슨의 역설: 전체와 부분이 반대 결론을 가리킬 때"
date: 2026-10-16 09:00:00 +0900
tags: [probability, statistics, causal-inference, confounding]
section: probability
---

치료법 A가 전체 성공률에서 B를 앞섭니다. 그런데 작은 결석 환자만 보면 B가 더 높고, 큰 결석 환자만 봐도 B가 더 높습니다. 둘 다 계산 착오가 아니고, 둘 다 사실입니다. 1986년 신장결석 치료 데이터에서 실제로 관측된 결과이고, 이런 역전 현상을 심슨의 역설(Simpson's paradox)이라고 부릅니다.

### 왜 역전되는가

원인은 **교란변수(confounder)가 그룹에 불균등하게 분포**하기 때문입니다. 신장결석 사례에서는 결석 크기가 그 변수였습니다. 큰 결석(어려운 케이스)은 주로 침습적 수술로 보냈고, 작은 결석(쉬운 케이스)은 덜 침습적인 시술로 보냈습니다. 그래서 수술은 어려운 환자를 많이 떠안은 탓에 전체 성공률이 낮아 보였고, 각 난이도 안에서 비교하면 더 우수했습니다.

숫자로 보면 구조가 단순합니다.

```python
# 치료법별 성공/전체 (결석 크기로 층화)
data = {
    "수술(A)":   {"작은 결석": (81, 87),   "큰 결석": (192, 263)},
    "시술(B)":   {"작은 결석": (234, 270), "큰 결석": (55, 80)},
}

for name, groups in data.items():
    tot_s = sum(s for s, n in groups.values())
    tot_n = sum(n for s, n in groups.values())
    rates = " | ".join(f"{g}: {s/n:.1%}" for g, (s, n) in groups.items())
    print(f"{name} → {rates} || 전체: {tot_s/tot_n:.1%}")

# 수술(A) → 작은 결석: 93.1% | 큰 결석: 73.0% || 전체: 78.0%
# 시술(B) → 작은 결석: 86.7% | 큰 결석: 68.8% || 전체: 82.6%
```

각 그룹에서는 A가 이기는데 합치면 B가 이깁니다. 가중평균을 낼 때 **각 치료법이 어려운 케이스를 얼마나 떠안았는지**가 결과를 뒤집습니다.

### 같은 구조의 다른 사례들

- **버클리 대학원 입학(1973)**: 전체 합격률은 남성이 높아 성차별로 보였지만, 학과별로 쪼개면 차별 근거가 사라졌습니다. 남성은 합격률이 높은 학과에, 여성은 경쟁이 치열한 학과에 더 많이 지원했던 것이 원인이었습니다.
- **코로나 치명률, 이탈리아 vs 중국**: 전체 치명률은 이탈리아가 높았지만, **모든 연령대에서는** 중국이 더 높았습니다. 이탈리아의 고령 인구 비중이 높았기 때문입니다.
- **입원 환자 중 접종자 비율**: 입원자의 60%가 접종자라는 수치만 보면 백신이 무용해 보입니다. 연령으로 층화하면 중증 예방 효과는 매우 높게 나옵니다. 고령층은 접종률도 높고 중증 위험도 높기 때문입니다. [베이즈 정리 글]({% assign bayes = site.notes | where_exp: "doc", "doc.path contains 'conditional-probability-bayes'" | first %}{{ bayes.url | relative_url }})에서 본 기저율 문제가 여기서도 반복됩니다.

공통점은 하나입니다. 비교하려는 두 집단의 **구성이 다르다**는 것.

### 그래서 항상 층화하면 되는가

여기가 이 주제의 진짜 어려운 부분입니다. "전체는 틀리고 쪼갠 게 맞다"는 규칙은 성립하지 않습니다.

어떤 변수로 쪼개느냐에 따라 역설이 사라지기도, 새로 생기기도, 다시 뒤집히기도 합니다(noncollapsibility). 그리고 **인과 경로 위에 있는 변수를 통제하면 오히려 효과를 지워버립니다.** 어떤 약이 혈압을 낮춰서 심장질환을 줄인다면, 혈압을 통제한 채 약의 효과를 보면 효과가 사라진 것처럼 나옵니다. 혈압은 교란변수가 아니라 약이 작동하는 경로(매개변수)이기 때문입니다.

즉 어떤 변수로 층화해야 하는지는 **데이터가 알려주지 않습니다.** 변수들 사이의 인과 구조에 대한 가정이 먼저 있어야 결정할 수 있습니다. 상관관계만 가지고는 어느 쪽 숫자를 믿어야 하는지 판단할 수 없다는 게, 상관관계와 인과관계가 다르다는 말의 실질적인 의미라고 생각합니다.

### 실험이 하는 일

이 문제를 가장 깔끔하게 끊는 방법이 무작위 배정입니다. 치료법을 환자 상태와 무관하게 랜덤으로 할당하면, 결석 크기 같은 변수가 두 그룹에 균등하게 섞입니다. 알려진 교란변수뿐 아니라 **우리가 측정하지 않은 변수까지** 평균적으로 균등해진다는 게 무작위화의 핵심 가치입니다.

[A/B 테스트 글]({% assign ht = site.notes | where_exp: "doc", "doc.path contains 'hypothesis-testing'" | first %}{{ ht.url | relative_url }})에서 실험 설계를 미리 정해두라고 썼는데, 그 설계가 보호하는 게 바로 이 지점입니다. 반대로 관측 데이터로 결론을 내려야 한다면, 어떤 변수가 교란변수이고 어떤 변수가 경로인지 명시적으로 가정하는 작업을 피할 수 없습니다.

### 정리하면

전체 수치와 부분 수치가 충돌할 때 자동으로 정답인 쪽은 없습니다. 먼저 물어야 할 것은 "비교하는 두 집단의 구성이 왜 다른가"이고, 그다음은 "내가 쪼개려는 변수가 원인 쪽인가 결과 쪽인가"입니다. 대시보드에서 숫자가 뒤집히는 걸 봤을 때 계산을 다시 확인하기보다 이 두 질문을 먼저 던지는 편이 빠른 것 같습니다.

---

**참고한 자료**

- [Simpson's Paradox (Stanford Encyclopedia of Philosophy)](https://plato.stanford.edu/entries/paradox-simpson/)
- [Simpson's Paradox Explained — Statistics By Jim](https://statisticsbyjim.com/basics/simpsons-paradox/)
- [Simpson's Paradox — Data Science Discovery (Illinois)](https://discovery.cs.illinois.edu/learn/Basics-of-Data-Science-with-Python/Simpsons-Paradox/)
- [Causal inference—so much more than statistics (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5841844/)
