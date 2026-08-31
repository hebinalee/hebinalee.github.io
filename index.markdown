---
layout: default
---

## LLM/Agent 시리즈

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

## 공부 노트

<ul class="post-list">
  {% assign sorted_notes = site.notes | sort: "date" | reverse %}
  {% for doc in sorted_notes %}
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

<p class="rss-subscribe"><a href="{{ "/notes/" | relative_url }}">공부 노트 전체 보기 →</a></p>
