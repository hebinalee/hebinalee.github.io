---
layout: page
title: LLM/Agent
permalink: /llm-agent/
---

LLM과 Agent를 중심으로 정리한 글 모음입니다.

<ul class="post-list">
  {% assign sorted_posts = site.posts | sort: "date" | reverse %}
  {% for post in sorted_posts %}
  <li>
    <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>
    <h3>
      <a class="post-link" href="{{ post.url | relative_url }}">
        {{ post.title | escape }}
      </a>
    </h3>
    {% if site.show_excerpts %}{{ post.excerpt }}{% endif %}
  </li>
  {% endfor %}
</ul>
