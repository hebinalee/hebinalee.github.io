---
layout: post
title: "도구를 여러 개 쓰는 멀티스텝 Agent 만들어보기"
date: 2026-08-12 09:00:00 +0900
tags: [llm, agent, tool-use]
---

"테슬라 시가총액이 애플의 몇 배인지 알려줘"라는 질문에 답하려면, 검색을 두 번 하고 계산을 한 번 더 해야 합니다. 도구가 하나뿐이었던 지난 글의 계산기 Agent로는 부족한 상황이라, 이번엔 도구를 두 개(검색 + 계산기)로 늘려서 여러 단계를 거치는 예제를 만들어봤습니다.

### 달라지는 것: 한 턴에 여러 도구, 여러 턴에 걸친 흐름

지난 글의 루프는 "도구 하나 호출 → 결과 확인 → 끝"이 사실상 전부였습니다. 도구가 늘어나면 두 가지가 새로 생깁니다.

- **한 턴에 여러 도구를 동시에 호출하는 경우**: "테슬라랑 애플 시가총액 각각 찾아줘"처럼 서로 의존하지 않는 요청은, 모델이 한 번의 응답에 검색 도구 호출을 두 개 동시에(병렬로) 담아 보낼 수 있습니다.
- **여러 턴에 걸쳐 순차적으로 이어지는 경우**: 계산기 도구는 두 검색 결과가 모두 나온 다음에야 호출할 수 있으므로, 이 부분은 턴을 넘어가며 순서대로 진행됩니다.

즉 루프 구조 자체는 지난 글과 같지만, **한 번의 응답에 tool_use 블록이 여러 개 들어올 수 있다는 점**과 **그 블록 수만큼 tool_result를 정확히 매칭해서 돌려줘야 한다는 점**이 새로 추가됩니다.

### 코드로 보면

```python
tools = [
    {
        "name": "web_search",
        "description": "웹에서 정보를 검색한다.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": "사칙연산 계산을 수행한다.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]

def run_tool(name, tool_input):
    if name == "web_search":
        return web_search(tool_input["query"])
    if name == "calculator":
        return str(eval(tool_input["expression"]))  # 실습용 예시
    raise ValueError(f"unknown tool: {name}")

messages = [{"role": "user", "content": "테슬라 시가총액이 애플의 몇 배인지 알려줘"}]

MAX_STEPS = 6  # 무한 루프 방지용 상한

for step in range(MAX_STEPS):
    response = client.messages.create(model="...", tools=tools, messages=messages)

    if response.stop_reason != "tool_use":
        print(response.content)
        break

    # 한 턴에 tool_use 블록이 여러 개 올 수 있으므로 전부 순회
    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
    tool_results = []
    for block in tool_use_blocks:
        result = run_tool(block.name, block.input)
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result,
        })

    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": tool_results})
else:
    print("최대 스텝 수를 넘었습니다. 루프를 강제 종료합니다.")
```

지난 글과 다른 점은 딱 두 가지입니다. `tool_use_blocks`를 리스트로 받아서 전부 실행한다는 것, 그리고 `MAX_STEPS`로 상한을 걸어둔다는 것. 이 상한이 없으면 모델이 도구 호출을 계속 반복하는 경우 루프가 끝나지 않을 수 있는데, 실제 서비스에서는 꼭 필요한 안전장치입니다.

### 병렬로 부르면 뭐가 좋아지나

검색 두 번을 순서대로 부르면 각각 300ms라고 해도 600ms가 걸리지만, 동시에 부르면 300ms로 끝납니다. 도구 호출이 늘어날수록 이 차이는 커지기 때문에, 서로 의존하지 않는 호출은 병렬로 묶어서 보내는 게 응답 속도에 꽤 크게 기여합니다. OpenAI, Anthropic, Google 모두 한 턴에 여러 도구 호출을 담아 응답하는 것을 지원하는 이유이기도 합니다.

### 그런데, 이게 꼭 "Agent"여야 할까

여기서 한 가지 짚고 넘어가고 싶은 점이 있습니다. 지금 예제처럼 "검색 두 번 하고 계산 한 번 한다"는 흐름이 사실 매번 정해져 있다면, 모델에게 매 턴 무엇을 할지 스스로 판단하게 맡기는 것보다, 그냥 코드로 순서를 고정해버리는 게(prompt chaining) 더 예측 가능하고 저렴합니다. Anthropic이 정리한 가이드에서도 같은 이야기를 합니다 — 작업의 흐름이 이미 정해져 있다면 워크플로우로, 모델이 상황에 따라 다음 행동을 동적으로 판단해야 하는 경우에만 Agent 루프로 가라는 것입니다. 지금 예제는 사실 워크플로우로도 충분히 풀리는 문제이지만, 도구가 여러 개일 때 루프가 어떻게 동작하는지 보여주기 위해 일부러 Agent 형태로 만들어봤습니다.

### 정리하면

도구가 하나에서 여러 개로 늘어나면 루프 자체가 복잡해지는 게 아니라, "한 턴에 여러 tool_use를 처리하는 법"과 "무한 루프를 막는 상한"이 추가로 필요해질 뿐입니다. 그리고 정작 중요한 질문은 "Agent를 얼마나 정교하게 만들 것인가"가 아니라 "이 작업이 정말 매 순간 동적 판단이 필요한가"였습니다. 항상 Agent가 정답은 아니라는 걸 이번에 체감했습니다.

---

**참고한 자료**

- [Building Effective AI Agents — Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- [Why Parallel Tool Calling Matters for LLM Agents](https://www.codeant.ai/blogs/parallel-tool-calling)
- [Tool-Augmented LLM Agents: Production Architecture Patterns](https://zylos.ai/research/2026-04-16-tool-augmented-llm-agents-production-architecture/)
