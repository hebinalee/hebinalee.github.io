---
layout: post
title: "도구를 붙일 때마다 연동 코드를 새로 짜지 않으려면: MCP"
date: 2026-09-27 09:00:00 +0900
tags: [llm, agent, mcp, tool-use, protocol]
section: agent
---

이 시리즈에서 Agent에 도구를 붙이는 이야기를 여러 번 했는데, 그때마다 빠져 있던 게 있습니다. 도구 하나를 추가할 때마다 스키마를 정의하고, 호출 결과를 파싱하고, 인증을 붙이는 코드를 매번 새로 짠다는 것. 도구가 N개이고 이걸 쓰는 애플리케이션이 M개면 M×N개의 연동이 필요해집니다. MCP(Model Context Protocol)는 이 지점을 표준화하려는 시도입니다.

### M×N을 M+N으로

MCP는 호스트(Host) — 클라이언트(Client) — 서버(Server)의 3계층 구조입니다. 도구를 제공하는 쪽이 MCP 서버를 한 번 구현하면, MCP를 지원하는 모든 호스트 애플리케이션에서 그 도구를 쓸 수 있습니다. 반대로 호스트는 MCP 클라이언트를 한 번 구현하면 공개된 모든 MCP 서버에 붙을 수 있습니다. M×N개의 개별 연동이 M+N개의 구현으로 줄어듭니다.

통신은 JSON-RPC 2.0 위에서 이뤄지고, 표준 전송 방식은 로컬 프로세스용 stdio와 원격용 Streamable HTTP입니다. 초기에 쓰이던 HTTP+SSE는 현재 deprecated 상태입니다.

서버가 제공할 수 있는 것은 세 가지입니다.

- **Tools**: 모델이 호출할 수 있는 함수. 우리가 function calling으로 다루던 그것
- **Resources**: 모델이 읽을 수 있는 데이터(파일, DB 레코드 등)
- **Prompts**: 재사용 가능한 프롬프트 템플릿

반대로 클라이언트가 서버에 제공하는 기능도 있습니다. 서버가 추론이 필요할 때 클라이언트의 LLM을 빌려 쓰는 **sampling**, 사용자에게 직접 입력을 요청하는 **elicitation** 같은 것들입니다. 도구가 단순한 함수 호출을 넘어 "중간에 사용자에게 물어보는" 흐름까지 표현할 수 있다는 뜻입니다.

### 서버 하나 만들어보기

Python SDK 기준으로는 기존 함수에 데코레이터를 붙이는 수준입니다.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("internal-metrics")

@mcp.tool()
def get_daily_active_users(date: str) -> int:
    """지정한 날짜(YYYY-MM-DD)의 DAU를 반환합니다."""
    return query_warehouse("SELECT count(*) ... WHERE dt = %s", date)

@mcp.resource("schema://tables")
def list_tables() -> str:
    """조회 가능한 테이블 목록."""
    return "\n".join(fetch_table_names())

if __name__ == "__main__":
    mcp.run()
```

함수의 docstring과 타입 힌트가 그대로 도구 설명과 스키마가 됩니다. 이전에 직접 작성하던 JSON 스키마를 손으로 관리하지 않아도 되는 부분이 실무에서는 생각보다 큽니다.

### 2026-07-28 스펙: 상태를 버렸다

올해 7월 스펙에서 꽤 큰 변화가 있었습니다. 핵심은 **stateless 전환**입니다.

기존에는 클라이언트가 연결 시 `initialize` 핸드셰이크를 하고 `Mcp-Session-Id`로 세션을 유지했습니다. 문제는 그 세션이 핸드셰이크를 처리한 특정 서버 인스턴스에 고정된다는 점이었습니다. 수평 확장을 하려면 sticky session을 걸거나 세션 저장소를 따로 두어야 했고, 서버를 재배포하면 연결이 끊겼습니다.

새 스펙은 핸드셰이크와 세션 ID를 모두 제거했습니다. 대신 모든 요청이 자기 자신을 설명합니다. 프로토콜 버전, 클라이언트 식별자, 역량(capabilities)이 각 요청의 `_meta`에 담겨 오고, 역량을 미리 알고 싶은 클라이언트는 `server/discover`를 호출하면 됩니다(선택 사항). 결과적으로 **어떤 요청이든 아무 인스턴스에나 도착해도 처리 가능**해져서, 평범한 라운드로빈 로드밸런서 뒤에 서버를 늘어놓을 수 있게 됐습니다. 메서드와 도구 이름이 `Mcp-Method`, `Mcp-Name` HTTP 헤더로 전달되기 때문에 게이트웨이가 본문을 파싱하지 않고도 라우팅과 인가를 할 수 있다는 점도 운영 관점에서 의미가 큽니다.

양방향 스트림을 계속 열어둬야 했던 sampling과 elicitation은 **MRTR(Multi Round-Trip Request)** 패턴으로 바뀌었습니다. 서버가 추가 입력이 필요하면 `resultType: "input_required"`와 함께 요청을 돌려주고, 클라이언트가 응답을 붙여서 원래 요청을 다시 보내는 방식입니다. 상태를 서버가 들고 있지 않아도 여러 번 주고받는 흐름을 표현할 수 있게 한 절충입니다.

### 그래서 지금 도입해야 할까

현재 Anthropic, OpenAI, Google, Microsoft, GitHub 등이 MCP를 지원하고, TypeScript·Python·C#·Java·Swift SDK가 나와 있으며 공개된 서버도 500개를 넘었습니다. 사내 데이터나 툴을 여러 AI 클라이언트에서 쓰게 될 것 같다면, 각 클라이언트마다 연동을 짜는 것보다 MCP 서버 하나를 두는 쪽이 유리한 시점이라고 봅니다.

다만 [프롬프트 인젝션 글]({% assign sec = site.posts | where_exp: "doc", "doc.path contains 'prompt-injection-guardrails'" | first %}{{ sec.url | relative_url }})에서 다룬 문제는 그대로, 아니 더 커집니다. MCP 서버가 반환하는 데이터도 결국 **Agent가 외부에서 읽어들이는 콘텐츠**이고, 서드파티 서버를 붙이는 순간 그 서버의 도구 설명과 응답이 모두 공격 표면이 됩니다. 표준화는 연동 비용을 줄여주지만 신뢰 문제를 대신 풀어주지는 않습니다. 도구 호출을 독립적으로 검증하고 민감한 작업에 승인 게이트를 두는 원칙은 MCP를 쓰든 직접 짜든 동일하게 필요합니다.

### 정리하면

MCP는 새로운 능력을 주는 기술이라기보다, 이미 하고 있던 일의 인터페이스를 통일하는 규약에 가깝습니다. 그래서 "MCP로 무엇을 할 수 있나"보다 "지금 도구 연동에 들어가는 반복 비용이 얼마인가"를 기준으로 판단하는 게 맞는 것 같습니다. 올해 stateless로 넘어오면서 프로덕션에서 확장하기 어렵던 부분이 상당히 정리됐다는 점도, 도입을 미뤄왔다면 다시 볼 만한 이유입니다.

---

**참고한 자료**

- [The 2026-07-28 Specification — Model Context Protocol Blog](https://blog.modelcontextprotocol.io/posts/2026-07-28/)
- [MCP went stateless: What changed in the 2026-07-28 spec (WorkOS)](https://workos.com/blog/mcp-stateless-spec-2026-07-28)
- [What's new in the MCP 2026-07-28 specification (Appwrite)](https://appwrite.io/blog/post/mcp-goes-stateless-in-the-2026-07-28-specification)
- [Model Context Protocol — Wikipedia](https://en.wikipedia.org/wiki/Model_Context_Protocol)
