---
layout: page
title: LLM/Agent
permalink: /llm-agent/
---

LLM과 Agent를 중심으로 정리한 글 모음입니다.

{% assign board = site.data.nav | where: "collection", "posts" | first %}
{% for sec in board.sections %}
  {% assign docs = site.posts | where: "section", sec.key | sort: "date" | reverse %}
  {% if docs.size > 0 %}
<h2 id="{{ sec.key }}" class="section-heading">{{ sec.label }}</h2>
<ul class="post-list">
  {% for post in docs %}
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
  {% endif %}
{% endfor %}
