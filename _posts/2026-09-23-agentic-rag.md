---
layout: post
title: "Agentic RAG는 항상 더 나은 선택일까"
date: 2026-09-23 09:00:00 +0900
tags: [llm, rag, agent, agentic-rag]
---

Agentic RAG가 이름부터 더 발전된 형태처럼 들려서 당연히 더 낫다고 생각했는데, 최근 연구들을 찾아보니 이야기가 좀 다릅니다. RAG 글에서 다룬 검색-생성 파이프라인과, Agent 글들에서 다룬 "스스로 판단하는 루프"가 만나는 지점이 Agentic RAG인데, 이 둘을 합친다고 항상 이득인 건 아니라는 게 이번 글의 결론입니다.

### 기존 RAG의 한계: 고정된 파이프라인

지난 RAG 글에서 다룬 파이프라인은 청킹 → 임베딩 → 검색 → 리랭킹 → 생성으로, 매 요청마다 정확히 같은 순서를 거칩니다. 문제는 검색 결과가 부실할 때입니다. 검색된 청크가 질문과 별로 관련이 없어도, 파이프라인은 이를 그대로 프롬프트에 넣고 생성 단계로 넘어갑니다. 전략을 바꿀 방법이 없습니다.

### Agentic RAG: 검색 자체를 판단의 대상으로

Agentic RAG는 고정된 파이프라인 대신, **에이전트가 "지금 검색이 필요한가", "이 결과로 충분한가", "다시 검색해야 하는가", "어떤 도구나 소스를 써야 하는가"를 매 단계 판단**하게 만드는 방식입니다. 대표적인 구현이 **Evaluator 에이전트**를 두는 방식인데, 검색된 청크의 관련성을 점수로 매겨서 일정 기준(예: 70%) 미만이면 다른 소스로 재검색하거나 웹 검색으로 전환하는 corrective 메커니즘을 넣습니다. 지난 Agent 멀티스텝 글에서 다룬 루프 구조가, 이번엔 "검색"이라는 도구 하나에 적용된 셈입니다.

복잡한 질문은 하위 질문으로 쪼개서(query decomposition) 순차적으로 검색하는 **multi-hop retrieval**도 Agentic RAG의 핵심 기법입니다. "A 회사의 이번 분기 매출이 작년 대비 얼마나 늘었나" 같은 질문은 "이번 분기 매출은?"과 "작년 같은 분기 매출은?"으로 나눠 각각 검색한 다음 계산하는 식입니다.

### 그런데, 최근 연구는 다른 이야기를 한다

여기서 반전이 있습니다. 최근 비교 연구에서 나온 결과가 꽤 의외입니다.

- **고정 하이브리드 검색이 규칙 기반 적응형 라우팅보다 일관되게 낫다**: adaptive routing이 오히려 거의 모든 멀티홉 하위 질문에 등장하는 고유명사 때문에 BM25 쪽으로 과도하게 쏠리는 문제가 있었습니다.
- **검색 반복 2번이 5번의 이득 중 95%를 이미 확보한다**: 반복을 늘릴수록 얻는 게 급격히 줄어듭니다.
- **고정된 로컬 모델 예산 안에서는, 단순하고 고정된 선택이 적응형 버전과 경쟁하거나 오히려 더 낫다**: 이득 대부분은 "짧은 검색 루프를 한 번 도는 것" 자체에서 오지, 정교한 적응형 라우팅이나 반복 횟수를 늘리는 데서 오지 않습니다.

즉, 무작정 판단 로직을 정교하게 만드는 것보다, 검색을 한두 번 더 반복하게 해주는 것만으로 이미 대부분의 이득을 챙길 수 있다는 뜻입니다.

### 코드로 보면: 최소한의 corrective 루프

```python
MAX_RETRIEVAL_ROUNDS = 2  # 2회 반복이 이득의 95%를 차지한다는 연구 결과 반영

def agentic_retrieve(query, retriever, relevance_threshold=0.7):
    for round in range(MAX_RETRIEVAL_ROUNDS):
        chunks = retriever.search(query)  # 하이브리드 검색 + 리랭킹 (지난 RAG 글 참고)
        relevance = evaluate_relevance(query, chunks)  # LLM 또는 리랭커 점수

        if relevance >= relevance_threshold:
            return chunks

        # 관련성이 낮으면 쿼리를 재작성해서 한 번 더 시도
        query = rewrite_query(query, chunks)

    return chunks  # 마지막 라운드 결과라도 반환
```

핵심은 판단 로직을 복잡하게 쌓기보다, "관련성이 낮으면 쿼리를 재작성해서 한 번 더 검색한다"는 단순한 규칙 하나와, 반복 횟수에 상한을 두는 것입니다. 지난 멀티스텝 Agent 글의 `MAX_STEPS`와 같은 이유로 여기도 상한이 필요합니다.

### 정리하면

2026년 기준 업계의 시각은 "Agentic RAG가 늘 더 낫다"가 아니라 **"Agentic RAG는 에스컬레이션(escalation)"**이라는 쪽에 가깝습니다. 대부분의 질문은 여전히 고정된 하이브리드 RAG로 충분히 처리되고, 단순 검색으로 부족하다는 게 확인될 때만 판단 계층을 얹는 게 맞습니다. 이번 시리즈에서 반복해서 확인한 원칙 — 워크플로우로 충분하면 워크플로우로, Agent가 필요할 때만 Agent로, 단일 Agent로 충분하면 굳이 여러 개로 나누지 않는다 — 가 검색에도 그대로 적용된다는 걸 다시 한번 확인했습니다.

---

**참고한 자료**

- [Traditional RAG vs. Agentic RAG — NVIDIA Technical Blog](https://developer.nvidia.com/blog/traditional-rag-vs-agentic-rag-why-ai-agents-need-dynamic-knowledge-to-get-smarter/)
- [Agent-Orchestrated Adaptive RAG: A Comparative Study on Structured and Multi-Hop Retrieval](https://arxiv.org/abs/2606.05658v1)
- [Agentic RAG: What it is, how it works, and when to use it — Neo4j](https://neo4j.com/blog/agentic-ai/what-is-agentic-rag/)
