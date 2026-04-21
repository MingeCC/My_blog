+++
title = "Writing With Hugo"
date = 2026-04-20T20:10:00+08:00
draft = false
description = "简单记录 Hugo 博客的写作工作流。"
tags = ["workflow", "markdown"]
categories = ["Writing"]
toc = true
+++

这是一篇示例文章，用来确认列表、单篇页面和标签系统都能正常工作。

## 新建文章

推荐直接使用 Hugo 自带命令：

```bash
hugo new content posts/my-new-post.md
```

## 本地预览

在项目根目录启动开发服务器：

```bash
hugo server -D
```

默认访问地址通常是 `http://localhost:1313/`。

## 发布构建

生成静态文件：

```bash
hugo
```

构建结果会输出到 `public/` 目录。
