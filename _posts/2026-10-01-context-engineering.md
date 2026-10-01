---
layout: post
title: "컨텍스트 윈도우가 커져도 컨텍스트 엔지니어링이 필요한 이유"
date: 2026-10-01 09:00:00 +0900
tags: [llm, agent, context-engineering, compaction]
section: agent
---

컨텍스트 윈도우가 충분히 커지면 "무엇을 넣을지 고민하는 일"은 사라질 거라고 생각했습니다. 다 넣으면 되니까요. 그런데 실제로는 반대 방향으로 갔습니다. 넣을 수 있는 양이 늘어난 만큼 넣게 되고, 그러면서 품질이 떨어지는 패턴이 반복적으로 관측됩니다. Agent에게 매 호출마다 무엇을 보여줄지 설계하는 일, 즉 컨텍스트 엔지니어링이 별도 주제가 된 이유입니다.

### 토큰을 더 넣으면 왜 나빠지는가

두 가지 현상이 겹칩니다.

하나는 **lost in the middle**입니다. 모델의 정확도는 관련 정보가 컨텍스트의 앞이나 뒤에 있을 때 가장 높고, 중간에 묻히면 측정 가능한 수준으로 떨어집니다. 긴 컨텍스트를 전제로 만들어진 모델에서도 이 경향이 남아 있습니다.

다른 하나는 실무에서 **context rot**이라 부르는 현상입니다. Agent가 오래 돌수록 이전 스텝의 도구 호출 결과, 실패한 시도, 중간 추론이 계속 쌓입니다. 토큰 수는 늘지만 그중 지금 판단에 필요한 비율은 줄어듭니다. 신호 대비 잡음이 나빠지면서 모델의 선택이 점점 나빠집니다.

[Agent 메모리 글]({% assign mem = site.posts | where_exp: "doc", "doc.path contains 'agent-memory'" | first %}{{ mem.url | relative_url }})에서 "무엇을 기억할 것인가"를 다뤘다면, 컨텍스트 엔지니어링은 **"기억한 것 중 이번 호출에 무엇을 꺼낼 것인가"**의 문제입니다. 저장과 주입은 별개의 설계 대상입니다.

### 전략 1: 압축(compaction), 다만 조건부로

긴 대화나 긴 실행 이력을 요약해서 줄이는 방식입니다. 한동안 "길어지면 압축한다"가 기본값처럼 쓰였는데, 2026년 들어 논의가 바뀌었습니다. 요지는 **압축을 상시 하지 말고 조건이 충족될 때만 하라**는 것입니다.

조건은 보통 세 가지로 정리됩니다. 윈도우에 물리적으로 안 들어갈 때, 캐시된 입력 토큰 단가가 요약 비용보다 비싸질 때, 그리고 품질 저하가 실제로 측정될 때. 압축을 자주 하면 프롬프트 캐시가 매번 깨져서 비용이 오히려 올라가고, 더 중요하게는 **요약 과정에서 무엇이 사라졌는지 통제하기 어렵습니다.** 장기 실행 Agent에서 압축이 반복되며 초기에 걸어둔 제약이나 안전 조건이 조용히 유실되는 현상(governance decay)을 지적하는 연구도 나와 있습니다. 요약은 손실 압축이고, 무엇을 잃을지는 요약하는 모델이 정합니다.

### 전략 2: 서브에이전트 컨텍스트 격리

지금 가장 널리 쓰이는 패턴입니다. 작업을 서브에이전트에 맡기고, **서브에이전트는 자기만의 컨텍스트 윈도우에서 탐색한 뒤 최종 요약만 부모에게 돌려줍니다.** 탐색 과정의 도구 호출과 시행착오는 부모 컨텍스트에 아예 들어오지 않습니다. 서브에이전트가 수만 토큰을 쓰면서 조사하고 1~2천 토큰 요약만 반환하는 구성이 대표적인 예입니다.

압축과 비교하면 차이가 분명합니다. 압축은 이미 더러워진 컨텍스트를 사후에 정리하는 것이고, 격리는 **애초에 더러워지지 않게 경계를 긋는 것**입니다. [멀티 에이전트 글]({% assign multi = site.posts | where_exp: "doc", "doc.path contains 'multi-agent-orchestration'" | first %}{{ multi.url | relative_url }})에서 조정 오버헤드 때문에 멀티 에이전트를 신중하게 쓰라고 썼는데, 컨텍스트 격리는 그 오버헤드를 지불할 만한 가장 설득력 있는 이유 중 하나입니다. 작업을 나누기 위해서가 아니라 컨텍스트를 나누기 위해 에이전트를 쪼개는 셈입니다.

덧붙여 격리는 통제 문제이기도 합니다. 서브에이전트가 동료의 컨텍스트를 전부 보면, 자신이 설계되지 않은 정보를 근거로 판단하게 됩니다.

### 전략 3: 배치와 캐시를 의식하기

같은 정보라도 어디에 놓는지가 영향을 줍니다. 핵심 지시와 현재 과업은 앞이나 뒤에 두고, 길고 덜 중요한 참고 자료를 중간에 배치합니다. 그리고 프롬프트 캐싱을 쓴다면 **앞부분을 안정적으로 유지**해야 합니다. 시스템 프롬프트나 도구 정의처럼 고정된 내용을 앞에, 매 호출마다 바뀌는 내용을 뒤에 두는 순서가 캐시 적중률을 좌우합니다.

### 코드로 보면: 컨텍스트 예산 관리

```python
MAX_CONTEXT_TOKENS = 100_000
RESERVED_FOR_OUTPUT = 8_000

def build_context(system_prompt, tools, task, history, retrieved_docs):
    budget = MAX_CONTEXT_TOKENS - RESERVED_FOR_OUTPUT

    # 1) 고정 영역은 앞에 (캐시 적중을 위해 순서를 바꾸지 않는다)
    fixed = [system_prompt, tools_spec(tools)]
    budget -= count_tokens(fixed)

    # 2) 현재 과업은 반드시 포함 (끝부분에 다시 한 번 상기)
    budget -= count_tokens(task)

    # 3) 남은 예산을 최근 이력에 우선 배분, 오래된 도구 결과는 요약으로 대체
    kept, summarized = [], []
    for step in reversed(history):
        cost = count_tokens(step)
        if cost <= budget:
            kept.append(step); budget -= cost
        else:
            summarized.append(step)

    parts = fixed
    if summarized:
        parts.append(summarize(reversed(summarized)))   # 손실을 감수하는 구간
    parts += list(reversed(kept))
    parts.append(select_top_k(retrieved_docs, budget))  # 검색 결과는 남은 만큼만
    parts.append(task)                                  # 과업을 마지막에 한 번 더
    return parts
```

중요한 건 함수의 세부가 아니라, **컨텍스트가 "되는 대로 쌓이는 것"이 아니라 예산을 가진 설계 대상이 된다는 점**입니다. 어떤 항목이 밀려나는지, 무엇이 요약으로 대체되는지가 코드에 명시되어 있어야 나중에 품질 문제가 생겼을 때 추적할 수 있습니다.

### 정리하면

컨텍스트 윈도우 크기는 제약을 느슨하게 만들었을 뿐, 설계 책임을 없애주지는 않았습니다. 지금 기준으로 정리하면 이렇습니다. 격리로 애초에 덜 쌓이게 하고, 압축은 조건이 충족될 때만, 배치는 캐시와 주의 분포를 의식해서. 결국 "모델에게 더 많이 알려주는 일"과 "모델이 더 잘 판단하게 하는 일"이 같은 방향이 아니라는 게 이 주제의 핵심인 것 같습니다.

---

**참고한 자료**

- [Context Engineering in 2026: Why We Stopped Compacting Our Agent's Context](https://www.louisbouchard.ai/context-engineering-2026/)
- [Sub-agent context isolation: the fix for context rot](https://usewire.io/blog/sub-agent-context-isolation-fixes-context-rot/)
- [Governance Decay: How Context Compaction Silently Erases Safety Constraints in Long-Horizon LLM Agents (arXiv)](https://arxiv.org/pdf/2606.22528)
- [Context Engineering: A Practical Guide for AI Agents (Sourcegraph)](https://sourcegraph.com/blog/context-engineering)
