# Hugo Personal Blog

这个目录已经初始化为一个 Hugo 个人博客。

## 常用命令

```bash
hugo new content posts/my-new-post.md
hugo server -D
hugo
```

## GitHub Pages

这个项目已经包含 GitHub Pages 的工作流文件：

`/.github/workflows/hugo.yaml`

推到 GitHub 后，按下面步骤启用：

1. 在 GitHub 创建仓库并推送当前项目
2. 打开仓库 `Settings > Pages`
3. 把 `Source` 设为 `GitHub Actions`
4. 推送到默认分支后，GitHub 会自动构建并发布

如果这是项目站点，最终地址通常会是：

`https://<用户名>.github.io/<仓库名>/`

如果这是用户站点，仓库名需要是：

`<用户名>.github.io`

## 主要目录

- `content/posts/`: 文章内容
- `layouts/`: 页面模板
- `static/css/main.css`: 样式
- `hugo.toml`: 站点配置

## 初始化后建议修改

1. 把 `hugo.toml` 里的站点标题、作者、描述改成你的信息
2. 修改 `content/about.md`
3. 删除或替换示例文章
4. 把 `baseURL` 改成你的正式域名
