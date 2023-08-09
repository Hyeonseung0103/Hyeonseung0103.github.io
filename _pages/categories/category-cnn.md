---
title: "AlexNet, VGG,GoogLeNet"
layout: archive
permalink: categories/cnn
author_profile: true
sidebar_main: true
---

{% assign posts = site.categories.cnn %}
{% for post in posts %} {% include archive-single2.html type=page.entries_layout %} {% endfor %}
