---
layout: page
title: ML 기초
permalink: /notes/
---

확률·통계, ML 기본기를 다시 훑어보며 정리한 기록입니다. LLM/Agent 시리즈와는 별도로 이곳에 모아둡니다.

<ul class="post-list">
  {% assign sorted_docs = site.notes | sort: "date" | reverse %}
  {% for doc in sorted_docs %}
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
