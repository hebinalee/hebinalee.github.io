---
layout: page
title: 기초 다지기
permalink: /notes/
---

확률·통계, ML 기본기를 다시 훑어보며 정리한 기록입니다. LLM/Agent 시리즈와는 별도로 이곳에 모아둡니다.

{% assign board = site.data.nav | where: "collection", "notes" | first %}
{% for sec in board.sections %}
  {% assign docs = site.notes | where: "section", sec.key | sort: "date" | reverse %}
  {% if docs.size > 0 %}
<h2 id="{{ sec.key }}" class="section-heading">{{ sec.label }}</h2>
<ul class="post-list">
  {% for doc in docs %}
  <li>
    <span class="post-meta">{{ doc.date | date: "%b %-d, %Y" }}</span>
    <h3>
      <a class="post-link" href="{{ doc.url | relative_url }}">
        {{ doc.title | escape }}
      </a>
    </h3>
    {% if site.show_excerpts %}{{ doc.excerpt }}{% endif %}
  </li>
  {% endfor %}
</ul>
  {% endif %}
{% endfor %}
