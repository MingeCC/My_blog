+++
title = "Writing With Hugo"
date = 2026-04-20T20:10:00+08:00
draft = false
description = "从 0 到 GitHub Pages 上线：我的 Hugo 博客完整搭建与写作流程。"
tags = ["workflow", "markdown", "github-pages", "deployment"]
categories = ["Writing"]
toc = true
+++

这篇不再是示例文，而是我这次把博客从 0 搭到可在线访问的完整复盘。  
目标很简单：本地写 Markdown，`git push` 后自动发布到 GitHub Pages。

## 1. 环境准备：先让 Hugo 可用

我在 Windows 环境里做了两件事：

- 安装 Hugo（建议 `extended` 版本）
- 把 Hugo 可执行文件目录加到环境变量 `PATH`

验证安装是否成功：

```bash
hugo version
```

只要能看到版本号，后续命令就都能跑。

## 2. 初始化项目：创建站点骨架

初始化命令：

```bash
hugo new site blog_web
cd blog_web
git init
```

核心目录含义：

- `content/`：文章与页面内容
- `layouts/`：模板（主题不满足时自己改）
- `static/`：静态资源（头像、图片、图标）
- `public/`：构建产物（无需手动编辑）

## 3. 站点基础配置：把访问路径一次配对

我最终用的是 GitHub 项目页（不是用户主页），所以 `baseURL` 必须带仓库名：

```toml
baseURL = "https://mingecc.github.io/My_blog/"
languageCode = "zh-cn"
title = "Mingyi's Blog"
```

这一步很关键。  
如果你用 `username.github.io/repo` 这种形式，`baseURL` 不带 `/repo/` 往往会导致样式或链接错位。

## 4. 页面结构：先把博客“可用性”搭齐

我先把最常用页面补齐，再做美化：

- 首页（Hero 区 + 作者信息）
- 文章列表页（`/posts`）
- 标签页（`/tags`）
- 关于页（`/about`）

导航菜单配置（`hugo.toml`）：

```toml
[[menus.main]]
name = "首页"
pageRef = "/"

[[menus.main]]
name = "文章"
pageRef = "/posts"

[[menus.main]]
name = "标签"
pageRef = "/tags"

[[menus.main]]
name = "关于"
url = "/about/"
```

这样“阅读文章”“了解我”按钮就能分别跳到文章页和关于页，不会变成无效按钮。

## 5. 内容写作：从命令开始，不手搓路径

新建文章推荐统一用 Hugo 命令：

```bash
hugo new content/posts/my-new-post.md
```

我当前文章元信息大致是：

- `title`：文章标题
- `date`：发布时间
- `draft`：是否草稿
- `tags` / `categories`：归档与检索
- `toc = true`：自动目录

写作上我现在遵循一个小规则：  
每篇文章都至少有“背景、步骤、结果、问题与解决”四段，后期检索会非常省心。

## 6. 本地预览：改一点，看一点

开发预览命令：

```bash
hugo server -D
```

- 本地地址通常是 `http://localhost:1313/`
- 我这个项目页场景里，常用访问路径是 `http://localhost:1313/My_blog/`

发布前我会再跑一次生产构建确认无报错：

```bash
hugo --gc --minify
```

## 7. 上 GitHub：仓库、分支、远程

标准流程：

```bash
git add .
git commit -m "init blog"
git branch -M main
git remote add origin https://github.com/MingeCC/My_blog.git
git push -u origin main
```

我当时遇到的典型报错是：

`Invalid username or token. Password authentication is not supported for Git operations.`

解决方式是改用 PAT（Personal Access Token）而不是账号密码。

## 8. GitHub Pages 自动部署：用 Actions 而不是旧式分支发布

仓库里我用的是 `.github/workflows/hugo.yaml`，触发条件是 push 到 `main`：

```yaml
on:
  push:
    branches: [main]
```

构建核心命令：

```yaml
hugo \
  --gc \
  --minify \
  --baseURL "${{ steps.pages.outputs.base_url }}/"
```

同时在仓库设置里确认：

- `Settings -> Pages -> Source = GitHub Actions`

如果没开对，可能会出现 `Get Pages site failed. Not Found` 这类错误。

## 9. 我这次踩过的坑（按优先级）

### 9.1 推送认证失败
原因：还在用密码推送。  
处理：生成 PAT 并用于 HTTPS 推送。

### 9.2 Actions 能跑但 Pages 不发布
原因：Pages 没切到 `GitHub Actions`。  
处理：在仓库 Pages 设置里切换来源后重跑 workflow。

### 9.3 首页按钮不跳转
原因：按钮链接指向错误路径或未绑定路由。  
处理：把“阅读文章”指向 `/posts/`，“了解我”指向 `/about/`。

## 10. 日常更新流程（现在我就这么发文）

每次发文我只做这几步：

1. `hugo new content/posts/xxx.md`
2. 写完后本地检查：`hugo server -D`
3. 构建校验：`hugo --gc --minify`
4. 提交并推送：`git add . && git commit -m "new post" && git push`
5. 等 GitHub Actions 完成，在线站点自动更新

---

到这里，这个博客已经从“能打开”变成“可持续更新”。  
下一步我会继续做两件事：优化文章模板（封面、目录、阅读时间）和补充更完整的 About 页面。
