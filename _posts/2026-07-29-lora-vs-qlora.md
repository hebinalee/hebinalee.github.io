---
layout: post
title: "LoRA와 QLoRA로 직접 파인튜닝해본 기록"
date: 2026-07-29 09:00:00 +0900
tags: [llm, fine-tuning, lora, qlora]
---

지난 글 마지막에 예고했던 대로, 이번엔 직접 오픈소스 모델을 파인튜닝해보면서 LoRA와 QLoRA를 비교해봤습니다. 서빙 글이 개념과 비교 위주였다면, 이번 글은 실습 로그에 가깝습니다.

### 왜 전체 파인튜닝은 부담스러운가

모델의 모든 파라미터를 업데이트하는 full fine-tuning은 파라미터 자체의 메모리뿐 아니라, 그래디언트와 옵티마이저 상태(Adam 기준 파라미터당 2개)까지 GPU에 올려야 합니다. 대략적으로 파라미터 하나당 16바이트 안팎(가중치 + 그래디언트 + 옵티마이저 상태, 혼합 정밀도 기준)이 필요하다고 보면, 8B 모델만 해도 학습에 100GB 이상의 메모리가 쉽게 필요해집니다. 개인이 GPU 한두 장으로 시도하기엔 애초에 문턱이 높습니다.

### LoRA: 원본은 고정하고, 작은 행렬만 학습한다

LoRA(Low-Rank Adaptation)는 원본 가중치 행렬 `W`는 그대로 얼려두고, 그 옆에 저랭크(low-rank) 행렬 `A`, `B` 두 개를 붙여서 `W + BA` 형태로 만든 다음, `A`와 `B`만 학습하는 방식입니다. `A`, `B`의 랭크(rank, `r`)를 원본 차원보다 훨씬 작게 잡기 때문에, 학습해야 할 파라미터 수 자체가 원본 대비 극히 일부로 줄어듭니다.

실무적으로는 랭크(`r`)와 알파(`alpha`) 두 값을 정하는 게 핵심입니다.

- **r (rank)**: 스타일/포맷 정도의 가벼운 조정이면 8, 어느 정도 도메인 지식을 주입하려면 16~32, 지식을 깊게 새로 넣으려면 64까지도 씁니다. 7B~13B급 모델 기준으로는 16을 기본값으로 잡고 시작하는 게 무난합니다.
- **alpha**: 보통 alpha = r 또는 alpha = 2r로 잡습니다. `alpha / r` 비율이 어댑터 출력에 곱해지는 스케일링 계수 역할을 합니다.

LoRA는 원본 가중치를 16비트로 그대로 들고 있어야 하기 때문에, 모델 자체는 여전히 꽤 큰 메모리를 차지합니다.

### QLoRA: 원본을 4비트로 압축하고 그 위에 LoRA를 얹는다

QLoRA는 여기서 한 걸음 더 나갑니다. 원본 모델을 **4비트(NF4, Normal Float 4-bit)**로 양자화해서 고정해두고, 그 위에 LoRA 어댑터만 fp16으로 얹어서 학습합니다. 핵심 기법은 세 가지입니다.

- **NF4 양자화**: 신경망 가중치가 대체로 정규분포에 가깝다는 점에 착안해서, 일반적인 균등 4비트 양자화보다 정보 손실이 적도록 설계된 양자화 방식
- **Double Quantization**: 양자화에 쓰인 스케일 상수 자체도 다시 양자화해서, 파라미터당 약 0.5비트를 추가로 절약
- **Paged Optimizer**: 그래디언트 체크포인팅 중 순간적으로 메모리가 튀는 구간을 CPU-GPU 페이징으로 흡수해서 OOM을 방지

이 조합 덕분에 QLoRA 원 논문 기준으로 65B급 모델도 단일 48GB GPU에서 파인튜닝이 가능하다고 보고되었고, 품질도 16비트 LoRA 대비 벤치마크에서 큰 차이가 나지 않는 것으로 알려져 있습니다.

### 실습: Qwen 계열 소형 모델로 비교해보기

작은 오픈소스 모델(Qwen2.5-7B-Instruct 기준)에 소규모 instruction 데이터셋 일부를 얹어서 LoRA와 QLoRA 각각으로 파인튜닝을 돌려봤습니다. 대략적인 구성은 이렇습니다.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# QLoRA: 4bit로 로드
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16",
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,  # LoRA만 쓸 땐 이 줄을 빼고 16bit로 로드
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```

체감한 차이는 대략 이렇습니다.

- **메모리**: QLoRA 쪽이 확실히 여유가 있었습니다. 같은 배치 크기에서 LoRA는 원본 가중치를 16비트로 들고 있어야 해서 VRAM 압박이 훨씬 컸고, QLoRA는 그 부담이 크게 줄었습니다.
- **속도**: 스텝당 속도는 QLoRA가 양자화/역양자화 연산 오버헤드 때문에 약간 더 느렸지만, 애초에 더 큰 배치를 돌릴 수 있어서 전체적으로는 상쇄되는 편이었습니다.
- **품질**: 같은 데이터, 같은 스텝 수 기준으로 결과물의 체감 품질 차이는 크지 않았습니다. 이 부분은 QLoRA 논문에서 보고한 결과와도 방향이 비슷합니다.

결국 "GPU가 넉넉하지 않다면 QLoRA, 이미 여유가 있고 약간의 품질/속도 이점이 아쉽지 않다면 LoRA"로 정리했습니다. 개인 환경에서는 QLoRA가 기본값이 되는 이유를 실습해보고 나서야 체감했습니다.

### 정리하면

Full fine-tuning의 메모리 부담을 LoRA가 "학습 파라미터 수"로 줄이고, QLoRA는 거기에 "원본 모델 자체의 메모리"까지 양자화로 줄여줍니다. 랭크와 알파 같은 하이퍼파라미터는 정답이 있다기보다, 원하는 조정의 크기(스타일 조정인지, 지식 주입인지)에 맞춰 고르는 감각이 필요하다는 걸 느꼈습니다.

다음 글에서는 이렇게 파인튜닝한(혹은 기존) 모델에 간단한 tool use 하나를 붙여서, 아주 작은 Agent를 직접 만들어보려 합니다.

---

**참고한 자료**

- [QLoRA: Efficient Finetuning of Quantized LLMs (원 논문)](https://arxiv.org/pdf/2305.14314)
- [QLoRA Explained: NF4 Quantization, Double Quantization & Paged Optimizers](https://www.maximofn.com/en/qlora/)
- [LoRA fine-tuning Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
