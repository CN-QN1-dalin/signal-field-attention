# arXiv 提交指南

## 1. 提交准备

### 提交材料
- `arxiv_submission/arxiv_submission.zip` — 包含 main.tex + references.bib + 代码源文件
- `arxiv_submission/main.html` — HTML预览（无LaTeX编译器时的替代）

### 元数据填写

| 字段 | 值 |
|------|-----|
| **Title** | Soma: A Signal Field Native Architecture for Efficient Long-Sequence Processing |
| **Authors** | Dalin Jia (Dalin Jia) |
| **Affiliation** | Independent Researcher |
| **Email** | 362118251@qq.com |
| **Categories** | cs.LG (Machine Learning) [primary], cs.CL (Computation and Language) |
| **License** | Not required for paper; code uses MIT |
| **Abstract** | (已写入main.tex) |
| **Comments** | Preliminary results, simulation-based experiments included |
| **Journal-ref** | (leave blank — preprint) |

## 2. 提交步骤

1. 访问 https://arxiv.org/signup （如果还没有账号）
2. 登录，点击 "Submit" → "Start new submission"
3. 选择类别: cs.LG 或 cs.CL
4. 上传 `arxiv_submission.zip`
5. 填写元数据（见上表）
6. 确认摘要、作者信息
7. 选择许可: 默认可（arXiv非独占许可）
8. 提交，等待审核（通常1-2工作日）

## 3. 注意事项

### arXiv 规则
- 单文件最大 50MB（我们的zip约45KB，完全合规）
- 允许预印本（preprint），不需要已发表
- 作者可以是个人，不需要机构
- 需要至少一个确认邮箱

### 我们的特殊情况
- **独立研究者**: arXiv完全接受无机构作者
- **模拟数据**: 论文中已明确标注所有模拟数据，符合学术伦理
- **免责声明**: 论文末尾包含预印本免责声明
- **无机构**: 在submission form中affiliation填"Independent Researcher"或"Independent"

### 如果PDF生成失败
由于系统缺少LaTeX编译器，提交方式有两种：

**方式A（推荐）**: 使用Overleaf在线编译
1. 访问 https://overleaf.com
2. 创建新项目，上传 `main.tex` 和 `references.bib`
3. 在Overleaf上编译生成PDF
4. 下载PDF上传到arXiv（arXiv接受单独PDF上传）

**方式B**: 上传LaTeX源文件
- arXiv接受纯LaTeX源文件上传，会自动编译
- 上传 `arxiv_submission.zip` 即可

## 4. 提交后

提交后会获得arXiv ID（如 2606.xxxxx），格式:
- `arXiv:2606.xxxxx`
- 可在 https://arxiv.org/abs/[ID] 查看

首次提交建议检查:
- [ ] PDF渲染是否正常（公式、表格）
- [ ] 参考文献是否正确
- [ ] 链接是否有效
- [ ] 作者信息是否正确

## 5. 后续迭代

提交v1后，收到审稿意见可以:
1. 修改内容
2. 重新编译
3. 提交 "v2" 更新（保留v1记录）

---

**创建日期**: 2026-06-16  
**作者**: 贾大林
