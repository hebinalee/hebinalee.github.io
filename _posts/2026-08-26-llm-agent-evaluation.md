---
layout: post
title: "LLM/Agent를 어떻게 평가할 것인가"
date: 2026-08-26 09:00:00 +0900
tags: [llm, agent, rag, evaluation]
---

지난 몇 편에 걸쳐 두 가지 질문을 계속 미뤄왔습니다. RAG 글에서는 "검색과 생성을 분리해서 평가해야 한다"고 짚고 넘어갔고, Agent 글에서는 "이게 잘 작동하는지 어떻게 확인하나"라는 질문을 남겨뒀습니다. 이번 글에서 이 두 질문을 한 번에 다뤄보려 합니다.

### 출력을 사람이 일일이 볼 수는 없다: LLM-as-judge

모델 출력이 많아지면 사람이 매번 채점할 수 없습니다. 그래서 다른 LLM에게 "이 답변이 얼마나 좋은지 평가해줘"라고 시키는 **LLM-as-judge**가 사실상 표준이 됐습니다. 생각보다 신뢰도가 높습니다. 코드 번역 품질 평가에서 LLM 판정이 사람 평가와 81.3% 상관관계를 보인 반면, 전통적인 자동 지표(ChrF++)는 34.2%에 그쳤다는 보고도 있습니다. 비용도 사람 평가 대비 수백~수천 배 저렴합니다.

다만 맹신하면 안 됩니다. 판정 순서에 따라 결과가 달라지는 위치 편향(position bias), 모델이 특정 패턴의 출력을 무조건 선호하는 편향 같은 함정이 있어서, "LLM 판정으로 사람 평가를 대체한다"가 아니라 "대량 평가는 LLM 판정으로 하고, 애매하거나 낮은 점수가 나온 케이스만 사람이 다시 본다"는 조합이 실무에서 권장되는 방식입니다.

### Agent는 최종 답변만 보면 안 된다: 트레이스 평가

Agent는 한 번의 출력이 아니라 여러 단계를 거칩니다. 그래서 평가도 세 층위로 나눠서 봐야 합니다.

- **최종 답변(final-answer)**: 마지막 메시지만 채점
- **트레이스(trajectory)**: 추론 과정, 도구 호출, 상태 전이까지 이어지는 전체 경로를 채점
- **턴 단위(per-turn)**: 실서비스에서 각 턴이 의미상 적절했는지 채점

특히 트레이스 평가가 중요한 이유는, **도구 호출을 전부 정확하게 했더라도 최종적으로 작업에 실패할 수 있기 때문**입니다. 반대로 최종 답변만 보면, 과정에서 불필요한 도구를 여러 번 호출했거나 같은 실수를 반복한 비효율은 전혀 드러나지 않습니다. 그래서 최소한 다음 지표들을 같이 봐야 합니다.

- **Task completion**: 최종 상태 기준으로 실제 작업을 완료했는가
- **Tool-call accuracy**: 올바른 도구를, 올바른 인자로 호출했는가
- **Step/loop count**: 불필요하게 스텝을 반복하지 않았는가 (지난 글의 `MAX_STEPS`가 여기서 다시 등장합니다)
- **Trajectory match**: 기대한 경로와 실제 경로가 비슷한가
- **Cost/latency**: 스텝이 늘어난 만큼 비용과 지연시간도 늘어남
- **Groundedness**: RAG를 함께 쓰는 경우, 답변이 검색한 근거에 실제로 기반했는가

### 스텝당 95%의 정확도가 왜 무서운가

여기서 체감이 확 오는 숫자가 하나 있습니다. 한 단계에서 95% 정확한 Agent라도, 20단계를 거치는 작업이라면 전체 성공률은 0.95의 20제곱, 즉 **약 36%**로 떨어집니다. 스텝 하나하나는 꽤 믿을 만해 보여도, 단계가 길어질수록 전체 신뢰도는 생각보다 훨씬 빠르게 무너진다는 뜻입니다. 멀티스텝 Agent를 설계할 때 스텝 수 자체를 줄이는 게 왜 중요한 설계 목표인지, 이 계산 하나로 납득이 됐습니다.

### 코드로 보면: 트레이스를 LLM judge로 채점하기

```python
JUDGE_PROMPT = """
다음은 Agent가 작업을 수행한 전체 트레이스입니다.

[사용자 요청]
{user_request}

[트레이스: 각 단계의 thought / action / observation]
{trajectory}

[최종 답변]
{final_answer}

아래 기준으로 각각 1~5점을 매기고 이유를 설명해줘.
1. task_completion: 사용자 요청을 실제로 해결했는가
2. tool_call_accuracy: 도구 호출이 정확했는가 (불필요한 호출 포함해서 감점)
3. groundedness: 답변이 도구 결과에 실제로 근거했는가

JSON으로만 답해줘: {{"task_completion": n, "tool_call_accuracy": n, "groundedness": n, "reason": "..."}}
"""

def evaluate_trajectory(user_request, trajectory, final_answer):
    result = judge_client.messages.create(
        model="...",
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                user_request=user_request,
                trajectory=trajectory,
                final_answer=final_answer,
            ),
        }],
    )
    scores = json.loads(result.content[0].text)
    # 낮은 점수는 사람 검토 큐로
    if min(scores["task_completion"], scores["tool_call_accuracy"], scores["groundedness"]) <= 2:
        flag_for_human_review(user_request, trajectory, final_answer, scores)
    return scores
```

핵심은 채점 자체를 자동화하되, 점수가 낮게 나온 케이스만 사람이 다시 보게 큐에 쌓아두는 구조입니다. 전수 검토는 불가능하지만, 전수 자동 평가 후 이상 케이스만 걸러내는 건 충분히 가능합니다.

### 정리하면

평가는 한 번 만들고 끝나는 게 아니라, 데이터와 프롬프트가 계속 바뀌는 만큼 지속적으로 돌려야 하는 파이프라인입니다. RAG의 faithfulness든 Agent의 trajectory든, "그럴듯해 보인다"는 인상이 아니라 구조화된 기준으로 채점하는 습관을 들이는 게, 지금까지 다룬 서빙·파인튜닝·Agent·RAG를 실제로 신뢰할 수 있는 시스템으로 만드는 마지막 조각이라는 생각이 듭니다.

---

**참고한 자료**

- [LLM-as-a-Judge in 2026: Top evaluation techniques and best practices](https://deepeval.com/blog/llm-as-a-judge)
- [LLM Agent Evaluation Metrics in 2026: Tool Calling, Task Completion, Reasoning, and Trace-Based Evals](https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide)
- [AI Agent Evaluation (2026): Metrics, Frameworks, and Production Failures](https://www.morphllm.com/ai-agent-evaluation)
