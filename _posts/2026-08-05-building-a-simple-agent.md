---
layout: post
title: "가장 작은 형태의 Agent 직접 만들어보기"
date: 2026-08-05 09:00:00 +0900
tags: [llm, agent, tool-use, react]
---

이번엔 파인튜닝한 모델 대신 기존 모델에 tool use 하나를 붙여서 아주 작은 Agent를 만들어봤습니다. 거창한 프레임워크 없이, "Agent"라고 부를 수 있는 가장 작은 단위가 뭔지부터 확인해보고 싶었습니다.

### Agent와 챗봇의 차이는 결국 "루프"

LLM에 그냥 질문하고 답을 받는 건 챗봇입니다. 여기에 **"모델이 스스로 도구를 쓸지 말지 판단하고, 도구 실행 결과를 다시 보고 다음 행동을 결정하는 루프"**가 더해지면 Agent가 됩니다. 이 루프를 구조화한 대표적인 패턴이 **ReAct**(Reasoning + Acting)입니다.

ReAct 루프는 단순하게 세 단계를 반복합니다.

1. **Thought**: 지금 상황에서 뭘 해야 할지 모델이 판단
2. **Action**: 필요하면 도구를 호출 (필요 없으면 바로 최종 답변)
3. **Observation**: 도구 실행 결과를 모델에게 다시 보여줌

이 세 단계를 "더 이상 도구가 필요 없다"고 모델이 판단할 때까지 반복하는 게 전부입니다.

### Tool use는 결국 구조화된 함수 호출

요즘 LLM API(OpenAI, Anthropic 등)는 대부분 **tool use / function calling**을 기본으로 지원합니다. 사용 가능한 도구 목록(이름, 설명, 입력 스키마)을 요청에 같이 넘기면, 모델이 자연어 대신 "이 도구를, 이 인자로 호출하고 싶다"는 구조화된 JSON을 응답으로 돌려줍니다. 실제 실행은 모델이 하는 게 아니라, 그 결과를 받아서 **내가 직접 실행**하고, 실행 결과를 다시 대화에 추가해주는 방식입니다. 모델은 절대 도구를 직접 실행하지 않는다는 점이 포인트입니다.

### 최소 구현: 계산기 도구 하나만 있는 Agent

프레임워크 없이 루프만 직접 짜보면 구조가 명확하게 보입니다. 의사코드에 가깝게 정리하면 이렇습니다.

```python
tools = [
    {
        "name": "calculator",
        "description": "사칙연산 계산을 수행한다.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    }
]

def calculator(expression: str) -> str:
    return str(eval(expression))  # 실습용 예시. 실제로는 안전한 파서를 써야 함

messages = [{"role": "user", "content": "127 곱하기 39는 얼마야?"}]

while True:
    response = client.messages.create(
        model="...",
        tools=tools,
        messages=messages,
    )

    if response.stop_reason != "tool_use":
        print(response.content)  # 최종 답변
        break

    tool_call = next(b for b in response.content if b.type == "tool_use")
    result = calculator(**tool_call.input)

    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_call.id,
            "content": result,
        }],
    })
```

핵심은 `while` 루프 하나뿐입니다. 모델이 도구가 더 필요 없다고 판단하면(`stop_reason != "tool_use"`) 루프를 빠져나오고, 그렇지 않으면 도구를 실행한 결과를 대화에 이어붙여서 다시 모델에게 넘깁니다. 이 정도만 있어도 "Agent"라고 부를 수 있는 최소 구조는 갖춘 셈입니다.

### 도구가 여러 개, 여러 서비스로 늘어나면: MCP

도구가 하나뿐일 땐 위 코드로 충분하지만, 실무에서는 Slack, 데이터베이스, 사내 API 등 도구가 빠르게 늘어납니다. 도구마다 스키마를 직접 정의하고 연결하는 게 반복 작업이 되다 보니, 이 부분을 표준화한 게 **MCP(Model Context Protocol)**입니다. Anthropic이 2024년 말 공개했고, 2026년 현재는 OpenAI, Google DeepMind 등 주요 업체가 함께 채택하면서 사실상 업계 표준처럼 자리잡았습니다. 도구를 직접 스키마부터 짜는 대신, 이미 공개된 MCP 서버(GitHub, Slack, DB 커넥터 등 수천 개가 공개되어 있습니다)에 연결하기만 하면 되는 경우가 많아졌습니다.

### 프레임워크는 언제 필요할까

지금처럼 도구가 하나고 루프가 단순하면 프레임워크 없이 직접 짜는 게 오히려 이해하기 쉽습니다. 다만 도구가 여러 개로 늘어나고, 상태를 여러 단계에 걸쳐 관리해야 하고, 병렬로 여러 하위 작업을 나눠 처리해야 하는 시점부터는 LangGraph 같은 프레임워크가 이 복잡도를 대신 관리해주는 게 낫습니다. 처음부터 프레임워크로 시작하면 이 루프 구조 자체를 체감하기 어려워서, 이번엔 일부러 직접 짜보는 쪽을 택했습니다.

### 정리하면

Agent는 결국 "모델 + 판단-실행-관찰 루프 + 도구"입니다. 복잡한 멀티 에이전트 구조나 프레임워크 이전에, 이 최소 단위를 직접 손으로 짜보니 그동안 막연하게 느껴졌던 "Agent"라는 말이 훨씬 구체적으로 다가왔습니다. ML 서빙, 파인튜닝을 거쳐 여기까지 오니, 결국 Agent도 "모델을 실행하고 결과를 다루는 엔지니어링 문제"라는 점에서는 크게 다르지 않다는 생각이 듭니다.

다음 글에서는 도구를 하나 더 늘려서, 여러 단계를 거치는 조금 더 현실적인 예제를 다뤄보려 합니다.

---

**참고한 자료**

- [ReAct Prompting — Prompt Engineering Guide](https://www.promptingguide.ai/techniques/react)
- [Implement a simple ReAct Agent using OpenAI function calling](https://peterroelants.github.io/posts/react-openai-function-calling/)
- [The Complete Guide to Model Context Protocol (MCP) in 2026](https://essamamdani.com/blog/mcp-complete-guide-2026)
