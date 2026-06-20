# arXiv 网站提交完整指南

## 第一步：发送邮件（可选但推荐）

邮件草稿已保存在 `/tmp/soma_arxiv_announcement_email.md`

**发送建议**：
- 发送给领域内研究者寻求反馈
- 邮件主题：`Preprint: Soma - Signal Field Native Architecture for Efficient LLM Inference`
- 附件：main.tex 或 PDF（如有）

---

## 第二步：arXiv 网站提交

### 2.1 准备工作

**编译PDF**（因为本机没有LaTeX编译器）：

1. 打开 https://overleaf.com
2. 注册/登录账号
3. 点击 "New Project" → "Upload Project"
4. 上传 `arxiv_submission/main.tex` 和 `arxiv_submission/references.bib`
5. 点击 "Recompile"，确认PDF生成成功
6. 下载PDF（用于预览确认）
7. 下载ZIP（包含LaTeX源文件，用于arXiv提交）

**或者**直接用命令行编译（如果装了MacTeX）：
```bash
cd arxiv_submission/
pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex
```

### 2.2 提交页面填写

访问 https://arxiv.org/submit

| 字段 | 填写内容 |
|------|---------|
| **Primary Category** | `cs.LG` (Machine Learning) |
| **Secondary Category** | `cs.CL` (Computation and Language) |
| **Title** | Soma: A Signal Field Native Architecture for Efficient Long-Sequence Processing |
| **Authors** | Dalin Jia |
| **Abstract** | （从main.tex复制，已写好） |
| **Comments** | Preliminary results, simulation-based experiments included. 16 pages, 16 references. |
| **Journal Reference** | （留空） |
| **MSC Class** | （留空） |
| **ACM Class** | （留空） |
| **License** | 默认即可（arXiv非独占许可） |

### 2.3 上传文件

1. 上传 `arxiv_submission.zip`（包含main.tex + references.bib + 代码）
2. 或上传编译好的PDF（如果arXiv允许直接上传PDF）

### 2.4 确认提交

- 检查所有元数据
- 确认作者顺序和信息
- 点击 "Submit"
- 等待审核通知（通常1-2个工作日）

### 2.5 审核通过后

- 获得arXiv ID（格式：arXiv:2606.xxxxx）
- 邮件通知到注册邮箱
- 可在 https://arxiv.org/abs/[ID] 查看
- 可更新GitHub仓库链接

---

## 常见问题

**Q: 独立研究者可以提交吗？**
A: 可以。arXiv完全接受无机构作者。

**Q: 模拟数据会被拒吗？**
A: 不会，只要明确标注。我们的论文已将所有模拟数据标为"simulated"。

**Q: 可以修改已提交的论文吗？**
A: 可以，提交v2更新（保留v1记录）。

**Q: 需要付费吗？**
A: arXiv提交免费。

---

**最后更新**: 2026-06-16
