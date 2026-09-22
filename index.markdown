---
layout: default
---

<div class="two-col-sections">
  <div class="col" markdown="1">

## LLM/Agent 시리즈

<ul class="post-list">
  {% assign recent_posts = site.posts | sort: "date" | reverse %}
  {% for post in recent_posts limit: 5 %}
  <li>
    <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>
    <h3>
      <a class="post-link" href="{{ post.url | relative_url }}">
        {{ post.title | escape }}
      </a>
    </h3>
  </li>
  {% endfor %}
</ul>

<p class="rss-subscribe"><a href="{{ "/llm-agent/" | relative_url }}">전체 보기 →</a></p>

  </div>
  <div class="col" markdown="1">

## 기초 다지기

<ul class="post-list">
  {% assign recent_notes = site.notes | sort: "date" | reverse %}
  {% for doc in recent_notes limit: 5 %}
  <li>
    <span class="post-meta">{{ doc.date | date: "%b %-d, %Y" }}</span>
    <h3>
      <a class="post-link" href="{{ doc.url | relative_url }}">
        {{ doc.title | escape }}
      </a>
    </h3>
  </li>
  {% endfor %}
</ul>

<p class="rss-subscribe"><a href="{{ "/notes/" | relative_url }}">전체 보기 →</a></p>

  </div>
</div>
