---
layout: post
title: "vLLM으로 LLM 띄우기: 플래그 세 개가 처리량을 결정한다"
date: 2026-10-13 09:00:00 +0900
tags: [llm, serving, vllm, inference, infra]
section: serving
---

[첫 글]({% assign first = site.posts | where_exp: "doc", "doc.path contains 'ml-serving-vs-llm-serving'" | first %}{{ first.url | relative_url }})에서 PagedAttention과 continuous batching이 왜 필요한지 정리했는데, 정작 vLLM을 처음 띄울 때 막힌 건 다른 데였습니다. `--gpu-memory-utilization`을 얼마로 둬야 하는지, `--max-num-seqs`는 왜 저렇게 큰 기본값인지 같은 것들. 개념과 플래그 사이의 거리를 메우는 글입니다.

### 일단 띄우기

vLLM은 OpenAI 호환 API 서버를 바로 제공합니다.

```bash
vllm serve Qwen/Qwen3-8B \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
resp = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "안녕"}],
)
```

기존 코드에서 `base_url`만 바꾸면 되기 때문에, [모델 선택 글]({% assign sel = site.posts | where_exp: "doc", "doc.path contains 'model-selection-guide'" | first %}{{ sel.url | relative_url }})에서 이야기한 "API로 쓰다가 자체 호스팅으로 전환" 시나리오의 전환 비용이 꽤 낮습니다. 문제는 띄우는 게 아니라 **같은 GPU에서 처리량을 얼마나 뽑아내는가**입니다.

### 메모리 모델: GPU에는 가중치만 올라가지 않는다

vLLM은 시작할 때 GPU 메모리를 선점해서 두 덩어리로 씁니다. 모델 가중치, 그리고 남은 공간 전부를 **KV 캐시 풀**로 잡습니다. 이 KV 캐시 풀의 크기가 동시에 처리할 수 있는 요청 수의 상한을 결정합니다. 그래서 세 플래그가 서로 묶여 있습니다.

**`--gpu-memory-utilization`** (기본 0.90): 전체 GPU 메모리 중 몇 퍼센트를 쓸지. 0.95까지 올리면 KV 캐시가 늘어 동시성과 처리량이 올라가고, 0.85로 내리면 OOM 여유가 생깁니다. 체감상 0.85로 내리면 처리량 5% 정도를 내주고 안정성을 사는 거래입니다. 프로덕션에서 간헐적 OOM으로 죽는 것보다는 5%가 싸기 때문에, 트래픽 패턴이 아직 파악되지 않았다면 낮게 시작하는 쪽이 낫습니다.

**`--max-model-len`**: 최대 컨텍스트 길이. 이 값이 요청 하나가 점유할 수 있는 KV 캐시의 상한을 정합니다. 모델이 128K를 지원한다고 그대로 열어두면, 긴 요청 몇 개가 캐시를 독점해서 동시성이 급격히 떨어집니다. **실제 트래픽의 길이 분포를 보고 필요한 만큼만** 잡는 게 중요합니다.

**`--max-num-seqs`**: 한 배치에서 동시에 처리할 시퀀스 수. V1 엔진 기본값은 1024로, V0의 256에서 크게 올랐습니다. 올리면 처리량이 오르지만 큐 대기가 길어져 개별 요청의 지연이 늘어나고, 메모리도 더 씁니다. 큰 모델에서 메모리가 빡빡하면 256이나 128로 내립니다.

세 값은 독립적으로 튜닝할 수 없습니다. 메모리 총량 안에서 "요청당 길이 × 동시 요청 수"가 KV 캐시에 들어가야 하기 때문입니다.

### V1 엔진이 바꾼 것

지금 기준으로 새로 배포한다면 V1 엔진이 기본 선택입니다. 스케줄러가 단순해지면서 몇 가지가 달라졌습니다.

- **거의 공짜가 된 prefix caching**: 공통 접두부의 KV 캐시를 재사용합니다. 오버헤드가 거의 없어져서 기본으로 켜둘 만합니다.
- **chunked prefill**: 긴 프롬프트의 프리필을 쪼개서 처리합니다.
- **prefill/decode 스케줄링 분리**: 긴 프리필 작업이 이미 진행 중인 요청들의 디코드 스텝을 블로킹하지 않습니다. 긴 프롬프트 요청 하나가 들어오면 다른 사용자들의 토큰 생성이 멈추던 문제가 완화됩니다.

여기서 [컨텍스트 엔지니어링 글]({% assign ctx = site.posts | where_exp: "doc", "doc.path contains 'context-engineering'" | first %}{{ ctx.url | relative_url }})의 조언이 그대로 성능으로 이어집니다. 시스템 프롬프트와 도구 정의를 앞에 고정하고 매번 바뀌는 내용을 뒤에 두라고 썼는데, 그렇게 하면 prefix caching이 공통 접두부를 재사용합니다. 프롬프트 구조를 어떻게 짜느냐가 TTFT와 GPU 사용량에 직접 반영되는 셈입니다.

### 튜닝은 측정부터

플래그를 감으로 정하는 대신 순서가 있습니다.

```bash
# 동시성을 바꿔가며 베이스라인 측정
vllm bench serve \
  --model Qwen/Qwen3-8B \
  --num-prompts 500 \
  --max-concurrency 32 \
  --random-input-len 1024 --random-output-len 256
```

측정해야 할 값은 세 개입니다. 처리량(tok/s), **TTFT**(첫 토큰까지의 시간), 그리고 요청 전체 지연. 동시성을 올려가며 그려보면 처리량이 평탄해지기 시작하는 지점이 나오고, 그 근처부터 지연이 급격히 나빠집니다. 그 변곡점이 `--max-num-seqs`를 정하는 기준점입니다. 채팅처럼 TTFT가 중요한 서비스는 변곡점보다 보수적으로, 배치 처리라면 처리량이 포화되는 쪽으로 잡습니다.

### 운영에서 챙길 것

- **도커 이미지 태그를 고정**합니다. vLLM은 변화가 빠르고, 기본값이 버전 사이에 바뀝니다(`--max-num-seqs`가 256에서 1024로 바뀐 것처럼).
- 텐서 병렬(`--tensor-parallel-size`)로 여러 GPU를 묶을 때는 **GPU가 모두 동일 모델**이어야 합니다. 섞이면 느린 쪽에 맞춰집니다.
- 멀티 노드 확장은 수익이 체감합니다. 노드를 늘릴 때 이득의 절반 이상이 첫 번째 추가 노드에서 나오는 양상이 보고됩니다. 노드를 늘리기 전에 단일 노드 튜닝이 끝났는지 먼저 확인하는 게 순서입니다.
- 추측 디코딩(speculative decoding)은 V1의 통합 스케줄러에서 쓸 수 있는데, 효과가 워크로드에 많이 의존합니다. 켜기 전후를 같은 벤치마크로 비교하지 않으면 득실을 알 수 없습니다.

### 정리하면

vLLM에서 처리량은 사실상 KV 캐시를 어떻게 배분하느냐의 문제입니다. `--gpu-memory-utilization`으로 캐시 총량을 정하고, `--max-model-len`으로 요청당 점유 상한을, `--max-num-seqs`로 동시 처리 개수를 정하는 구조입니다. 그리고 이 숫자들은 서비스의 길이 분포와 지연 요구사항에 따라 달라지니, 결국 남의 설정값을 가져오는 것보다 자기 트래픽으로 한 번 측정해보는 게 빠릅니다.

---

**참고한 자료**

- [vLLM V1 User Guide (공식 문서)](https://docs.vllm.ai/en/stable/usage/v1_guide/)
- [Practical strategies for vLLM performance tuning (Red Hat Developers)](https://developers.redhat.com/articles/2026/03/03/practical-strategies-vllm-performance-tuning)
- [vLLM Production Deployment: Complete 2026 Guide (SitePoint)](https://www.sitepoint.com/vllm-production-deployment-guide-2026/)
- [vLLM Multi-Node Benchmark 2026: Throughput, Latency & VRAM Scaling](https://markaicode.com/benchmarks/vllm-multi-node-benchmark/)
