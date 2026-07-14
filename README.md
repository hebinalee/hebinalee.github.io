# From Models to Agents

backpropagent의 기술 블로그. Jekyll + GitHub Pages 기반.

## 배포 방법

1. GitHub에서 새 저장소를 만든다.
   - 저장소 이름을 `hebinalee.github.io`로 하면 바로 `https://hebinalee.github.io`에서 열린다.
   - 다른 이름(예: `blog`)으로 만들면 `https://hebinalee.github.io/blog`에서 열리고, `_config.yml`의 `baseurl`을 `/blog`로 바꿔야 한다.

2. 이 폴더 내용을 저장소 루트에 push한다.

   ```bash
   cd from-models-to-agents
   git init
   git remote add origin https://github.com/hebinalee/hebinalee.github.io.git
   git add .
   git commit -m "Initial blog scaffold"
   git branch -M main
   git push -u origin main
   ```

3. GitHub 저장소 Settings → Pages에서 Source를 `main` 브랜치, `/ (root)`로 설정한다.
   (저장소 이름이 `hebinalee.github.io`면 보통 자동으로 활성화된다.)

4. 1~2분 후 `https://hebinalee.github.io`에서 확인.

## 로컬에서 미리보기 (선택)

Ruby가 설치되어 있다면:

```bash
bundle install
bundle exec jekyll serve
```

`http://localhost:4000`에서 확인 가능.

## 새 글 쓰는 법

`_posts/` 폴더에 `YYYY-MM-DD-제목.md` 형식으로 파일을 추가한다. 예:

```
_posts/2026-07-20-llm-serving-tips.md
```

파일 맨 위에 아래 형식의 front matter를 넣는다.

```yaml
---
layout: post
title: "글 제목"
date: 2026-07-20 09:00:00 +0900
categories: [llm]
tags: [llm, serving]
---
```

## 구조

```
from-models-to-agents/
├── _config.yml       # 블로그 설정 (제목, 설명, 소셜 링크 등)
├── _posts/           # 발행된 글
├── about.markdown    # 소개 페이지
├── index.markdown    # 홈 (자동으로 글 목록 표시)
├── Gemfile           # GitHub Pages 지원 gem 목록
└── README.md
```
