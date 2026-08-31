---
layout: post
title: "RAG: 모델을 다시 학습시키지 않고 지식을 넣는 법"
date: 2026-07-29 09:00:00 +0900
tags: [llm, rag, retrieval]
---

파인튜닝 글에서 다룬 LoRA/QLoRA는 결국 모델 파라미터를 학습시켜서 지식을 "모델 안"에 넣는 방식입니다. 그런데 최신 뉴스나 매일 바뀌는 사내 문서처럼 계속 갱신되는 지식을, 바뀔 때마다 다시 학습시키는 건 현실적이지 않습니다. 이럴 때 쓰는 게 RAG(Retrieval-Augmented Generation)입니다. 모델은 그대로 두고, 답변에 필요한 지식을 그때그때 검색해서 프롬프트에 넣어주는 방식입니다.

### 기본 파이프라인

RAG의 기본 흐름은 생각보다 단순합니다.

1. 문서를 작은 단위(청크)로 나눈다
2. 각 청크를 임베딩(벡터)으로 변환해서 벡터DB에 저장한다
3. 질문이 들어오면 질문도 임베딩으로 바꿔서, 벡터DB에서 가장 가까운 청크들을 찾는다
4. 찾은 청크를 프롬프트에 붙여서 모델에게 "이 내용을 참고해서 답해줘"라고 요청한다

문제는 이 4단계 각각에서 디테일이 결과 품질을 크게 좌우한다는 점입니다.

### 청크를 어떻게 나누느냐가 생각보다 중요하다

임베딩 모델이나 벡터DB보다도, **청크를 어떻게 나누는지가 검색 품질에 더 큰 영향을 준다**는 게 여러 벤치마크에서 공통적으로 나오는 결론입니다. 일반적인 기본값으로는 재귀적으로 512토큰 단위, 10~20% 정도 겹치게(overlap) 나누는 방식이 무난한 출발점으로 꼽힙니다. 문서 종류에 따라서는 페이지 단위 청킹이 더 안정적인 성능을 보이는 경우도 있어서, 정답이 하나로 정해져 있다기보다는 데이터 특성에 맞춰 실험해봐야 하는 영역입니다.

### 임베딩 모델은 리더보드 점수만 보고 고르면 안 된다

임베딩 모델을 고를 때 MTEB 같은 공개 리더보드 점수를 참고하게 되는데, 이 점수가 내 도메인에서의 성능을 보장하지는 않습니다. 실제로 특정 도메인 데이터로 비교했을 때, 리더보드 순위와 무관하게 특정 모델이 다른 모델보다 recall이 10포인트 이상 차이 나는 사례들이 보고되고 있습니다. 결국 내 데이터로 직접 벤치마크해보는 과정을 건너뛸 수 없습니다.

### 검색 품질을 가장 크게 올리는 두 가지: 하이브리드 검색과 리랭킹

기본적인 벡터 유사도 검색(dense retrieval)만으로는 놓치는 게 있습니다. 키워드가 정확히 일치해야 하는 경우(제품 코드, 고유명사 등)는 오히려 BM25 같은 키워드 기반 검색이 더 잘 잡아냅니다. 그래서 **하이브리드 검색**(dense + BM25를 Reciprocal Rank Fusion으로 결합)이 사실상 기본 구성으로 자리잡았습니다.

여기에 더해 **리랭킹(reranking)**을 얹으면 품질이 한 번 더 올라갑니다. 하이브리드 검색으로 넉넉하게 상위 100개 정도를 빠르게 뽑아둔 다음, cross-encoder 기반 리랭커(Cohere Rerank, BGE-Reranker 등)로 그 100개를 다시 정밀하게 채점해서 상위 5~10개만 추려 모델에 넘기는 방식입니다. 이 두 가지(하이브리드 검색 + 리랭킹)만 추가해도 naive RAG 대비 정밀도가 큰 폭으로 개선된다고 알려져 있습니다.

### 코드로 보면

```python
from chromadb import Client
from sentence_transformers import CrossEncoder

# 1) 청크 임베딩 저장 (dense)
collection = Client().create_collection("docs")
collection.add(documents=chunks, embeddings=embed(chunks), ids=chunk_ids)

# 2) 하이브리드 검색: dense + BM25 결과를 함께 모음
dense_hits = collection.query(query_embeddings=[embed([query])[0]], n_results=100)
bm25_hits = bm25_index.search(query, top_k=100)
candidates = reciprocal_rank_fusion(dense_hits, bm25_hits)

# 3) 리랭킹: 후보 100개를 cross-encoder로 재채점
reranker = CrossEncoder("BAAI/bge-reranker-base")
scores = reranker.predict([(query, c.text) for c in candidates])
top_chunks = [c for c, _ in sorted(zip(candidates, scores), key=lambda x: -x[1])[:8]]

# 4) 프롬프트에 삽입
context = "\n\n".join(c.text for c in top_chunks)
prompt = f"다음 내용을 참고해서 질문에 답해줘.\n\n{context}\n\n질문: {query}"
```

### 평가는 검색과 생성을 따로 봐야 한다

RAG가 잘 작동하는지 확인할 때, "답변이 그럴듯한가"만 보면 안 됩니다. 검색 단계(원하는 청크를 제대로 찾아왔는가, recall@k)와 생성 단계(찾아온 내용에 실제로 근거해서 답했는가, faithfulness)를 분리해서 각각 평가해야 합니다. 검색이 엉망인데 생성 모델이 그럴듯하게 답을 지어내면, 겉으로는 그럴듯해 보여도 실제로는 틀린 정보를 자신 있게 말하는 상황이 됩니다. 아무리 강력한 모델도 애초에 잘못된 근거를 받으면 이 문제를 스스로 해결할 수 없습니다.

### 정리하면

파인튜닝이 지식을 모델 파라미터에 새겨 넣는 방식이라면, RAG는 지식을 모델 바깥에 두고 필요할 때마다 찾아서 보여주는 방식입니다. 자주 바뀌는 정보, 출처를 명확히 밝혀야 하는 정보에는 RAG가 훨씬 실용적입니다. 다만 파이프라인의 각 단계(청킹, 임베딩, 검색, 리랭킹)마다 품질을 좌우하는 선택지가 많아서, "일단 벡터DB에 넣고 검색하면 되겠지"라는 생각보다 훨씬 세심한 튜닝이 필요하다는 걸 이번에 확인했습니다.

---

**참고한 자료**

- [RAG Techniques Compared: A Practical Guide (2026)](https://blog.starmorph.com/blog/rag-techniques-compared-best-practices-guide)
- [Best Chunking Strategies for RAG (and LLMs) in 2026](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)
- [End-to-End RAG Workflow — Databricks](https://www.databricks.com/blog/rag-workflow)
