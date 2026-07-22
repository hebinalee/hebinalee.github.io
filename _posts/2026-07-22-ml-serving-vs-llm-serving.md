---
layout: post
title: "ML 서빙과 LLM 서빙은 무엇이 다를까"
date: 2026-07-22 09:00:00 +0900
tags: [llm, serving, ml-engineering]
---

지난 5년 동안 모델을 서빙하는 일은 늘 익숙한 영역이었습니다. 이미지 분류 모델이든 추천 모델이든, 서빙 관점에서 신경 쓰는 것은 대체로 비슷했습니다. 배치를 얼마나 크게 묶을지, 응답 시간을 어떻게 줄일지, GPU를 얼마나 놀리지 않을지. 그런데 LLM을 서빙하면서는 이 익숙한 감각이 잘 통하지 않는다는 걸 느꼈습니다. 같은 "모델 서빙"이라는 이름 아래 있지만, 실제로 마주치는 문제는 상당히 다릅니다.

이번 글에서는 그 차이를 정리해보려 합니다. ML과 AI(LLM)의 차이를 개념적으로 설명하기보다는, 실무에서 서빙 엔지니어링이 어떻게 달라지는지를 기준으로 풀어보겠습니다.

### 기존 ML 서빙: compute-bound 문제

전통적인 ML 서빙(이미지 분류, 추천 모델, 정형 데이터 기반 예측 모델 등)은 대체로 **compute-bound** 문제입니다. 입력 하나에 대해 한 번의 forward pass만 수행하면 출력이 나오고, 지연시간은 그 한 번의 연산 시간으로 거의 결정됩니다. 그래서 TorchServe, TensorFlow Serving, Triton Inference Server 같은 프레임워크는 주로 다음을 최적화합니다.

- 여러 요청을 하나의 배치로 묶어 GPU 연산을 효율적으로 채우는 것 (static batching)
- 모델 자체를 가볍게 만드는 것 (프루닝, 양자화, 지식 증류)
- 고정된 입력 크기에서 처리량을 최대화하는 것

배치 크기와 지연시간 사이의 트레이드오프는 있지만, 구조 자체는 비교적 단순합니다. 요청이 들어오면 배치를 채우고, 한 번 연산하고, 결과를 반환합니다.

### LLM 서빙: memory-bound 문제로 바뀐다

LLM 서빙이 근본적으로 다른 이유는, 이 문제가 **memory-bound**로 바뀌기 때문입니다. 그리고 그 핵심에는 **KV 캐시**가 있습니다.

LLM은 토큰을 한 번에 하나씩, 순차적으로(autoregressive) 생성합니다. 매 토큰을 생성할 때마다 이전 토큰들의 attention 계산 결과(Key, Value)를 다시 쓰기 위해 캐시에 저장해두는데, 이 KV 캐시는 시퀀스가 길어질수록 계속 커집니다. 문제는 이 캐시 크기를 요청 시점에는 정확히 알 수 없다는 점입니다. 그래서 기존 방식대로 최대 길이만큼 미리 연속된(contiguous) 메모리를 할당해두면, 실제로 다 쓰이지 않는 공간이 생기고(internal fragmentation), 요청마다 크기가 제각각이라 메모리가 조각나는(external fragmentation) 문제도 생깁니다.

이 문제를 해결하기 위해 나온 것이 **PagedAttention**입니다(UC 버클리 Sky Computing Lab이 제안, vLLM의 핵심 아이디어). 운영체제의 가상 메모리 페이징처럼, KV 캐시를 고정 크기 블록(보통 16개 토큰 단위) 단위로 쪼개서 필요할 때마다 할당합니다. 연속된 메모리를 미리 확보해둘 필요가 없어지니 낭비가 크게 줄어듭니다.

### 배칭 전략도 완전히 달라진다

기존 static batching은 배치에 포함된 요청들이 다 끝날 때까지 기다렸다가 다음 배치를 시작합니다. 그런데 LLM은 요청마다 생성하는 토큰 수(응답 길이)가 천차만별이라, 짧은 응답이 이미 끝났는데도 긴 응답 하나 때문에 배치 전체가 GPU를 붙잡고 있는 비효율이 생깁니다.

그래서 등장한 것이 **continuous batching**(iteration-level scheduling)입니다. 배치가 통째로 끝나길 기다리지 않고, 토큰을 하나 생성하는 매 스텝마다 끝난 요청은 배치에서 빼고 대기 중인 새 요청을 그 자리에 채워 넣습니다. GPU가 노는 시간이 크게 줄어드는 대신, 스케줄러 자체가 훨씬 복잡해집니다.

지연시간을 재는 방식도 달라집니다. 기존 모델은 "요청 하나 처리하는 데 걸리는 시간" 하나만 보면 됐지만, LLM은 최소한 두 가지를 따로 봐야 합니다.

- **TTFT (Time To First Token)**: 첫 토큰이 나올 때까지 걸리는 시간
- **TPOT / ITL (Time Per Output Token / Inter-Token Latency)**: 이후 토큰이 하나씩 나오는 간격

사용자 입장에서 체감하는 "느리다"는 이 두 지표 중 어느 쪽이 나쁜가에 따라 원인이 완전히 다릅니다.

### 모델이 너무 크다 — 양자화가 선택이 아니라 기본값이 된다

기존에도 양자화는 있었지만, LLM에서는 사실상 기본 옵션에 가까워졌습니다. 모델 자체가 수십~수백억 파라미터인 경우가 흔하다 보니, 그대로 올리면 GPU 메모리가 감당이 안 되는 경우가 많습니다. 현재는 **AWQ(Activation-aware Weight Quantization)**가 INT4 양자화의 사실상 표준처럼 쓰이고 있고(GPTQ 대비 품질과 처리량 모두에서 우세한 경우가 많다는 평가), 이 외에도 GGUF, EXL2, FP8(H100/H200급 하드웨어 네이티브 지원), NF4 등 상황에 맞는 포맷을 고르게 됩니다. 주요 모델들이 Hugging Face에 이미 양자화된 체크포인트를 배포하는 것도 이제는 자연스러운 일이 됐습니다.

### 그래서 서빙 엔진 자체가 따로 발전했다

이런 이유들 때문에, LLM 서빙은 아예 전용 엔진들이 자리를 잡았습니다.

- **vLLM**: PagedAttention과 continuous batching을 중심으로 GPU 활용률과 동시 처리량을 극대화하는 데 초점
- **TensorRT-LLM**: NVIDIA 하드웨어에 맞춘 저수준 최적화가 강점이지만, 모델마다 컴파일 과정이 필요해 배포 유연성은 상대적으로 떨어짐
- **SGLang**: 프롬프트 구조와 생성 과정 자체를 최적화하는 방향(structured generation)으로 접근해, 특정 워크로드에서는 vLLM보다 높은 처리량을 보이기도 함
- **TGI(Text Generation Inference)**: Hugging Face 생태계와의 통합에 강점

기존 ML 서빙에서는 프레임워크 선택이 상대적으로 덜 중요한 문제였다면(대부분 비슷한 배칭 로직), LLM 서빙에서는 어떤 엔진을 쓰느냐가 그 자체로 아키텍처 결정에 가깝습니다.

### 정리하면

기존 ML 서빙에서 익숙했던 감각 — 배치를 채우고, 모델을 가볍게 만들고, GPU를 놀리지 않는 것 — 은 LLM 서빙에서도 여전히 유효한 목표입니다. 다만 문제의 성격이 compute-bound에서 memory-bound로 바뀌면서, 그 목표를 이루는 방법이 완전히 새로워졌습니다. KV 캐시, continuous batching, 양자화, 전용 서빙 엔진까지 — 5년간 쌓아온 서빙 감각이 무용해진 게 아니라, 그 감각을 새로운 문제에 다시 적용하는 과정이라고 느꼈습니다.

다음 글에서는 실제로 오픈소스 모델을 파인튜닝해본 과정(LoRA/QLoRA)을 정리해보려 합니다.

---

**참고한 자료**

- [vLLM 공식 문서](https://docs.vllm.ai/en/latest/)
- [PagedAttention vs Continuous Batching vs vLLM vs SGLang — A Practical Breakdown](https://python.plainenglish.io/pagedattention-vs-continuous-batching-vs-vllm-vs-sglang-a-practical-breakdown-4c19cc9e21c0)
- [Best LLM Inference Engines (2026): vLLM, SGLang & TensorRT-LLM 비교](https://www.yottalabs.ai/post/best-llm-inference-engines-in-2026-vllm-tensorrt-llm-tgi-and-sglang-compared)
