---
layout: post
title: "Agent가 이상하게 동작할 때 어디를 봐야 하는가: 관측성과 실패 복구"
date: 2026-10-06 09:00:00 +0900
tags: [llm, agent, observability, tracing, reliability]
section: eval-safety
---

일반적인 서비스에서 장애가 나면 로그에 스택 트레이스가 남습니다. Agent는 그렇지 않습니다. 열 번의 도구 호출 중 세 번째에서 잘못된 판단을 하고도 끝까지 정상적으로 완주하고, 응답은 그럴듯한 문장으로 돌아옵니다. 예외가 발생하지 않았으니 모니터링 대시보드는 초록색입니다. 이게 Agent 운영이 기존 백엔드 운영과 가장 다른 지점이라고 생각합니다.

### 무엇을 기록해야 하는가

Agent 실행은 중첩된 구조입니다. 하나의 요청 안에 모델 호출, 도구 호출, 검색, 서브에이전트 실행이 트리 형태로 들어갑니다. 그래서 **계층적 span(스팬)으로 남기는 추적(tracing)**이 기본 단위가 됩니다. 사실상의 표준은 OpenTelemetry이고, Langfuse·Phoenix·MLflow Tracing 같은 도구들이 이 위에서 LLM에 특화된 뷰를 제공합니다.

실제로 디버깅이 가능하려면 span마다 아래 정보가 필요합니다.

- **어느 단계인가**: 스텝 번호, 부모 span, 서브에이전트 식별자
- **무엇을 넣고 무엇을 받았는가**: 프롬프트, 도구 호출 인자, 도구 응답(길면 요약과 원본 참조)
- **비용과 지연**: 입력/출력 토큰, 캐시 적중 여부, 단계별 소요 시간
- **판정 결과**: 가드레일이 차단했는지, 스키마 검증이 통과했는지, 재시도 몇 번째인지

여기서 토큰과 지연시간을 단계별로 쪼개 기록하는 게 중요합니다. "전체 응답이 느리다"는 사실만으로는 손을 쓸 수 없고, 어느 도구가 p99에서 튀는지 보여야 조치가 가능합니다.

### 조용한 실패를 어떻게 잡는가

예외를 던지지 않는 실패가 Agent의 주된 실패 양상입니다. 추적 데이터에서 비교적 싸게 뽑을 수 있는 신호들이 있습니다.

- **같은 도구를 같은 인자로 반복 호출** → 모델이 결과를 이해하지 못하고 맴도는 상태
- **스텝 수가 비정상적으로 많음** → 목표에 수렴하지 못하는 궤적
- **도구 오류율 급증** → 외부 의존성 쪽 문제가 Agent 품질로 번지는 경우
- **중간에 멈춘 궤적(stalled trajectory)** → 타임아웃 없이 대기하는 호출

지표 수준에서는 두 가지를 같이 보는 게 유용합니다. 작업이 끝까지 성공한 비율(Agent Success Fraction)과, **도구 실패를 만났을 때 Agent가 복구에 성공한 비율(Error Recovery Rate)**입니다. 전자만 보면 "실패했다"는 사실은 알 수 있지만, 실패가 모델의 판단 문제인지 복구 로직의 문제인지 구분되지 않습니다.

[평가 글]({% assign eval = site.posts | where_exp: "doc", "doc.path contains 'llm-agent-evaluation'" | first %}{{ eval.url | relative_url }})에서 궤적 평가를 다뤘는데, 그것과 관측성의 차이는 시점입니다. 평가는 배포 전에 정해진 데이터셋으로 판정하고, 관측성은 배포 후에 실제 트래픽에서 벌어지는 일을 본다는 점입니다. 둘은 대체 관계가 아니라, 관측성에서 모은 실패 사례가 평가 데이터셋으로 들어가는 순환 구조가 되어야 합니다.

### 복구는 코드가 해야 한다

최근 도구들이 탐지와 대시보드에는 강하지만 **자동 복구는 여전히 애플리케이션 몫**입니다. 실무에서 쓰는 패턴은 단순한 편입니다.

```python
MAX_STEPS = 12
MAX_TOOL_RETRIES = 2

def run_agent(task, tracer):
    with tracer.start_as_current_span("agent.run") as root:
        root.set_attribute("task", task)
        seen_calls = {}

        for step in range(MAX_STEPS):
            with tracer.start_as_current_span(f"agent.step.{step}") as span:
                action = model.decide(context)
                if action.is_final:
                    return action.output

                # 1) 동일 호출 반복 차단 (맴돌이 방지)
                key = (action.tool, json.dumps(action.args, sort_keys=True))
                seen_calls[key] = seen_calls.get(key, 0) + 1
                span.set_attribute("tool.name", action.tool)
                span.set_attribute("tool.repeat_count", seen_calls[key])
                if seen_calls[key] > 2:
                    context.append(system_note(
                        "같은 도구를 같은 인자로 반복 호출했습니다. 다른 접근을 시도하세요."))
                    continue

                # 2) 도구 실패는 재시도하되, 실패 사실을 모델에게 알린다
                for attempt in range(MAX_TOOL_RETRIES + 1):
                    try:
                        result = call_tool(action, timeout=10)
                        break
                    except ToolError as e:
                        span.record_exception(e)
                        span.set_attribute("tool.retry", attempt)
                        if attempt == MAX_TOOL_RETRIES:
                            result = error_message_for_model(e)  # 숨기지 않고 전달

                context.append(result)

        # 3) 스텝 예산 소진은 '성공'이 아니라 명시적 실패로 기록
        root.set_attribute("agent.outcome", "max_steps_exceeded")
        raise AgentBudgetExceeded(task)
```

세 가지가 포인트입니다. 반복 호출을 코드가 감지해서 모델에게 알려주는 것, 도구 실패를 삼키지 않고 모델이 볼 수 있는 형태로 전달하는 것, 그리고 **스텝 예산을 다 쓴 경우를 성공으로 처리하지 않는 것**입니다. 마지막이 의외로 자주 놓치는 부분인데, 예산 초과를 조용히 마지막 응답으로 반환하면 그게 바로 앞에서 말한 "대시보드는 초록색인데 결과는 틀린" 상황이 됩니다.

### 비용도 관측 대상이다

Agent는 실패할 때 비용을 가장 많이 씁니다. 수렴하지 못하는 궤적이 스텝 예산을 전부 소진하면서 토큰을 태우기 때문입니다. 멀티 에이전트 구조에서는 이 낭비가 병렬로 일어납니다. 그래서 작업당 토큰 소비 분포를 보고, 꼬리 쪽(상위 1%)에 어떤 작업들이 모여 있는지 확인하는 게 비용 관리의 출발점입니다. 실패 탐지와 비용 절감이 같은 지표를 보는 일이 되는 셈입니다.

### 정리하면

Agent 운영에서 가장 위험한 상태는 에러가 나는 상태가 아니라 **에러 없이 틀리는 상태**입니다. 그래서 단계별 추적을 남기고, 맴돌이·스텝 폭증·도구 실패율 같은 간접 신호로 조용한 실패를 찾고, 복구 로직은 모델의 선의에 기대지 않고 코드로 강제해야 합니다. 관측성 도구를 붙이는 것 자체는 하루면 되지만, "무엇이 실패인지"를 정의하는 작업은 결국 서비스마다 직접 해야 하는 일인 것 같습니다.

---

**참고한 자료**

- [When Agentic Executions Fail: Detecting and Localizing Runtime Faults from Telemetry (arXiv)](https://arxiv.org/pdf/2608.14680)
- [Early Diagnosis of Wasted Computation in Multi-Agent LLM Systems via Failure-Aware Observability (arXiv)](https://arxiv.org/pdf/2606.01365)
- [12 Open-Source LLM Observability Tools to Know in 2026 (Turing Post)](https://www.turingpost.com/p/llm-observability)
- [OpenTelemetry Traces Your LLM. It Does Not Fix It.](https://dev.to/anilatambharii/opentelemetry-traces-your-llm-it-does-not-fix-it-2hl8)
