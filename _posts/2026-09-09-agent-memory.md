---
layout: post
title: "Agent는 무엇을, 얼마나 오래 기억해야 할까"
date: 2026-09-09 09:00:00 +0900
tags: [llm, agent, memory, context]
---

Agent가 몇 턴 전에 사용자가 말한 조건을 잊어버리거나, 지난 세션에 이미 실패했던 방법을 또 시도하는 걸 보면, 아무리 도구를 잘 쓰고 루프가 정교해도 신뢰하기 어렵습니다. 이번엔 Agent가 무엇을 기억하고, 무엇을 잊어야 하는지를 정리해봤습니다.

### 단기 메모리: 지금 이 루프 안에서의 상태

단기(작업) 메모리는 지금 진행 중인 루프 안에 있는 것들입니다. 현재 목표, 세운 계획, 최근 도구 실행 결과, 지금 지켜야 할 제약 조건. 이건 컨텍스트 윈도우 안에 그대로 들어갑니다.

여기서 중요한 원칙은 **작게, 언제든 덮어쓸 수 있게 유지하는 것**입니다. 단기 메모리가 계속 쌓이기만 하면 두 가지 문제가 생깁니다. 오래된 세부사항 때문에 판단이 흐트러지고(drift), 이미 쓸모없어진 내용에 토큰을 계속 낭비하게 됩니다. 지난 평가 글에서 다룬 "스텝이 길어질수록 신뢰도가 떨어진다"는 문제도 실은 상당 부분 여기서 옵니다.

가장 흔한 대응은 오래된 메시지를 요약하거나 잘라내는(truncation) 것인데, 요약은 과하게 압축하면 정보 손실이, 덜 압축하면 여전히 토큰을 많이 먹는 딜레마가 있습니다.

### 장기 메모리: 세션을 넘어서는 기억

장기 메모리는 지금 세션이 끝나도 남아야 하는 것들입니다. 2026년 현재 업계에서 대체로 수렴한 분류는 세 갈래입니다.

- **에피소드 기억(episodic)**: 언제 무엇을 했는가 — 세션 로그, 의사결정 기록, 과거 디버깅 흔적
- **의미 기억(semantic)**: 사용자 선호나 도메인 지식처럼, 시간과 무관하게 유효한 사실
- **절차 기억(procedural)**: 어떻게 하는지에 대한 노하우 — 반복되는 작업의 절차나 규칙

이 세 가지는 성격이 달라서 저장 방식도 다릅니다. 에피소드 기억은 시간순 로그에 가까워 벡터 검색으로 최근 관련 상호작용을 빠르게 찾는 데 적합하고, 의미/절차 기억은 구조화된 저장소(그래프, 키-값 스토어)와 더 잘 맞는 경우가 많습니다. 그래서 실무에서는 벡터 스토어 하나로 다 해결하기보다, **벡터 + 그래프/구조화 저장소 + 단기 에피소드 버퍼를 함께 쓰는 하이브리드** 구성이 표준에 가깝습니다.

### 실무 패턴: 컨텍스트는 RAM, 외부 저장소는 디스크

코딩 어시스턴트나 고객 지원 봇 같은 실제 production agent에서 가장 흔히 보이는 구조는 이렇습니다. **작업 메모리는 컨텍스트 윈도우 안에, 장기 기록은 외부 벡터/구조화 저장소에 두고, 매 스텝마다 관련된 기록만 검색해서 컨텍스트에 주입**합니다. RAG 글에서 다룬 검색 파이프라인이 여기서도 그대로 재사용됩니다 — 다만 검색 대상이 문서가 아니라 "과거의 나"라는 점이 다를 뿐입니다.

이걸 운영체제에 비유하면 이해가 쉽습니다. 컨텍스트 윈도우는 RAM처럼 빠르지만 용량이 제한적이고, 외부 저장소는 디스크처럼 용량은 넉넉하지만 접근에 한 단계가 더 필요합니다. Letta 같은 프레임워크는 이 비유를 그대로 구현해서, 필요한 정보를 컨텍스트(RAM)와 외부 저장소(디스크) 사이로 지능적으로 옮기는 가상 컨텍스트 관리 계층을 둡니다.

### 코드로 보면: 토큰 예산을 넘으면 오래된 대화를 요약으로 대체

```python
MAX_CONTEXT_TOKENS = 8000
KEEP_RECENT_TURNS = 4  # 최근 N턴은 요약하지 않고 그대로 유지

def compact_if_needed(messages, long_term_store):
    if count_tokens(messages) < MAX_CONTEXT_TOKENS:
        return messages

    recent = messages[-KEEP_RECENT_TURNS:]
    to_summarize = messages[:-KEEP_RECENT_TURNS]

    # 잘라낼 내용은 버리지 않고 장기 메모리(벡터 스토어)에 먼저 적재
    long_term_store.add(to_summarize)

    summary = summarize(to_summarize)  # LLM 호출로 압축
    return [{"role": "system", "content": f"이전 대화 요약: {summary}"}] + recent

def build_context(user_query, messages, long_term_store):
    messages = compact_if_needed(messages, long_term_store)
    relevant_memories = long_term_store.search(user_query, top_k=5)
    return inject_memories(messages, relevant_memories)
```

핵심은 두 가지입니다. 컨텍스트에서 밀려나는 내용을 그냥 버리지 않고 장기 저장소로 옮겨서 나중에 검색 가능하게 남겨두는 것, 그리고 매 요청마다 지금 질문과 관련 있는 기억만 다시 불러오는 것. 압축(compaction)과 검색(retrieval)이 한 쌍으로 움직이는 구조입니다.

### 정리하면

Agent 메모리는 결국 컨텍스트 윈도우라는 한정된 예산을 어떻게 쓸지의 문제입니다. 지금 당장 필요한 건 작업 메모리에 가볍게 유지하고, 나중에 필요할 수 있는 건 버리지 말고 장기 저장소로 옮기고, 필요한 순간에만 다시 불러오는 것. 서빙 글에서 다룬 KV 캐시가 "한 요청 안에서" 메모리를 관리하는 문제였다면, Agent 메모리는 "여러 세션에 걸쳐" 같은 문제를 푸는 것에 가깝다는 걸 이번에 느꼈습니다.

---

**참고한 자료**

- [Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers](https://arxiv.org/html/2603.07670v1)
- [Agent Memory Architectures: Vector vs Graph vs Episodic](https://www.digitalapplied.com/blog/agent-memory-architectures-vector-graph-episodic)
- [The 6 Best AI Agent Memory Frameworks You Should Try in 2026](https://machinelearningmastery.com/the-6-best-ai-agent-memory-frameworks-you-should-try-in-2026/)
