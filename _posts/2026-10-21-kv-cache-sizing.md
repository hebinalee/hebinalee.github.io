---
layout: post
title: "KV 캐시 용량을 손으로 계산해보기: 이 GPU에 몇 명이 들어가는가"
date: 2026-10-21 09:00:00 +0900
tags: [llm, serving, kv-cache, gqa, quantization]
section: serving
---

[지난 글]({% assign vllm = site.posts | where_exp: "doc", "doc.path contains 'vllm-deployment'" | first %}{{ vllm.url | relative_url }})에서 `--max-model-len`을 실제 트래픽 길이에 맞춰 잡으라고 썼는데, 그 문장은 사실 절반만 말한 셈입니다. 길이를 줄이면 동시 처리량이 얼마나 늘어나는지, 이 GPU로 몇 명을 받을 수 있는지는 계산해볼 수 있는 값입니다. 공식 자체는 곱셈 몇 번입니다.

### 토큰 하나가 차지하는 메모리

KV 캐시는 레이어마다 Key와 Value를 저장합니다. 그래서 토큰 하나당 메모리는 이렇게 됩니다.

$$\text{bytes/token} = 2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{bytes per element}$$

앞의 2는 K와 V 두 개라는 뜻입니다. 여기서 주의할 점은 **쿼리 헤드 수가 아니라 KV 헤드 수**가 들어간다는 것입니다. GQA(Grouped-Query Attention)를 쓰는 모델은 이 둘이 다릅니다.

Llama 3 8B로 넣어보면 (32 레이어, KV 헤드 8개, head_dim 128, bf16 = 2바이트),

```
레이어당: 2 × 8 × 128 × 2 = 4,096 bytes
토큰당:   4,096 × 32 = 131,072 bytes = 128 KiB
```

토큰 하나에 128 KiB입니다. 그러면 8K 컨텍스트를 꽉 채운 요청 하나는 **정확히 1 GiB**를 씁니다.

### 그래서 몇 명이 들어가는가

80GB GPU에 이 모델을 올린다고 해봅시다.

```
--gpu-memory-utilization 0.90  →  vLLM이 쓰는 총량 72 GB
모델 가중치 (8B × 2바이트)      →  16 GB
KV 캐시로 남는 공간             →  약 56 GB
```

8K를 꽉 채운 요청이 1 GiB니까 **동시 56개**가 상한입니다. 여기서 `--max-model-len`을 4K로 줄이면 요청당 0.5 GiB가 되어 상한이 두 배가 됩니다. 길이 설정이 동시성에 선형으로 작용한다는 게 숫자로 보이는 부분입니다.

물론 실제 트래픽은 모든 요청이 최대 길이를 쓰지 않습니다. PagedAttention이 블록 단위로 필요한 만큼만 할당하기 때문에, 짧은 요청이 많으면 훨씬 많은 요청을 동시에 처리할 수 있습니다. 위 계산은 **최악의 경우에 대한 상한**으로 보는 게 맞습니다.

### GQA가 왜 표준이 되었나

만약 Llama 3 8B가 GQA 없이 MHA(쿼리 헤드와 KV 헤드 수가 같음, 32개)였다면 어떻게 될까요.

```
2 × 32 × 128 × 2 × 32 레이어 = 512 KiB / token  (4배)
8K 요청 하나당 4 GiB  →  동시 14개
```

같은 GPU, 같은 파라미터 수인데 동시 처리량이 4분의 1로 줄어듭니다. KV 헤드를 8개로 묶은 것만으로 처리량이 4배가 되는 셈입니다. MQA(KV 헤드 1개)는 더 극단적으로 줄이지만 품질 손실이 커서, 그 사이 타협점인 GQA가 사실상 기본값이 되었습니다. [첫 글]({% assign first = site.posts | where_exp: "doc", "doc.path contains 'ml-serving-vs-llm-serving'" | first %}{{ first.url | relative_url }})에서 LLM 서빙이 memory-bound 문제라고 썼는데, 어텐션 구조 선택이 곧 서빙 비용 결정이라는 게 이 계산에서 드러납니다.

DeepSeek 계열의 **MLA(Multi-head Latent Attention)**는 접근이 다릅니다. K와 V를 그대로 저장하는 대신 저차원 잠재 벡터로 압축해서 보관합니다. 잠재 차원 512 + RoPE 차원 64 구성이라면 레이어당 (512+64) × 2 = 1,152 바이트, 60 레이어면 토큰당 약 68 KiB입니다. 모델 전체 크기는 훨씬 큰데도 토큰당 KV는 8B 모델보다 작습니다.

다만 이 수치들을 비교할 때 **단위 기준을 확인**해야 합니다. 어떤 자료는 레이어당 값을, 어떤 자료는 전체 합을 적어두고 "약 몇 KB"라고만 씁니다. 블로그의 "98% 절감" 같은 숫자도 비교 대상이 무엇인지에 따라 달라지니, 모델 config의 `num_key_value_heads`, `head_dim`, `num_hidden_layers`(MLA는 `kv_lora_rank`, `qk_rope_head_dim`)를 직접 넣어보는 게 확실합니다.

### KV 캐시 양자화

공식에서 가장 직접적으로 건드릴 수 있는 항이 바이트 수입니다. bf16(2바이트)을 FP8(1바이트)로 바꾸면 KV 캐시가 절반이 되고, 동시 처리량은 그만큼 늘어납니다. INT4는 이론상 0.5바이트지만 채널별 스케일 같은 부가 데이터가 붙어서 실제 절감률은 계산보다 낮습니다.

```python
def kv_bytes_per_token(cfg, dtype_bytes=2):
    """HF config 기준 (GQA/MHA)"""
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    return 2 * cfg["num_hidden_layers"] * n_kv * head_dim * dtype_bytes

def max_concurrent(gpu_gb, util, weight_gb, ctx_len, bytes_per_token):
    kv_budget = gpu_gb * util - weight_gb
    per_request_gb = bytes_per_token * ctx_len / 1024**3
    return int(kv_budget / per_request_gb)

llama3_8b = {"num_hidden_layers": 32, "num_attention_heads": 32,
             "num_key_value_heads": 8, "head_dim": 128}

bpt = kv_bytes_per_token(llama3_8b)
print(bpt // 1024, "KiB/token")                                  # 128
print(max_concurrent(80, 0.90, 16, 8192, bpt), "개 (bf16)")       # 56
print(max_concurrent(80, 0.90, 16, 8192, bpt // 2), "개 (fp8)")   # 112
```

### 계산을 하한으로 봐야 하는 이유

실제 서빙 스택에는 공식에 없는 소비가 있습니다. 페이징 블록의 내부 여유 공간, 활성값(activation) 버퍼, 양자화 스케일 텐서, 구현에 따라 추가되는 로터리/인덱서 버퍼 같은 것들입니다. 그래서 위 계산으로 나온 숫자를 그대로 설정값으로 쓰면 OOM에 가까워집니다. 여유를 두고 시작한 뒤 벤치마크로 맞춰가는 순서가 안전합니다.

### 정리하면

KV 캐시 용량은 레이어 수, KV 헤드 수, head_dim, dtype의 곱이고, 거기에 컨텍스트 길이와 동시 요청 수가 곱해집니다. 이 식을 한 번 세워두면 "컨텍스트를 두 배로 열면 동시성이 반으로 줄어든다", "FP8로 바꾸면 두 배를 받는다", "GQA 모델을 고르는 것이 GPU를 두 배 쓰는 것과 맞먹는다" 같은 판단을 설정값을 건드려보기 전에 할 수 있습니다. [모델 선택 글]({% assign sel = site.posts | where_exp: "doc", "doc.path contains 'model-selection-guide'" | first %}{{ sel.url | relative_url }})에서 비용을 따질 때 파라미터 수만 봤다면, config의 KV 헤드 수를 같이 보는 게 실제 서빙 비용에는 더 가까운 지표입니다.

---

**참고한 자료**

- [KV Cache Calculations — LLM Architecture Gallery (Sebastian Raschka)](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)
- [GQA vs MHA: What It Actually Saves in KV Cache Memory](https://www.spheron.network/blog/gqa-vs-mha-kv-cache-memory-savings/)
- [Multi-Head Latent Attention (MLA): Cut KV Cache and Serve More Users Per GPU (2026 Guide)](https://www.spheron.network/blog/multi-head-latent-attention-mla-gpu-cloud/)
- [KV Cache Memory: The Real Cost of Long-Context Inference](https://intuitionlabs.ai/articles/kv-cache-memory-long-context-inference-cost)
