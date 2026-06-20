# PDF 生成过程总结

## 目标
将太初五岳(Dalin Soma)项目的5篇学术论文（Soma Engine、Soma Native、Soma Convergence、Soma LingYa、Soma Heritage）生成英文PDF格式。

## 遇到的问题

### 问题1：fpdf2 核心字体不支持CJK
- 使用 `fpdf2` 的内置核心字体（Courier）时，遇到 `FPDFUnicodeEncodingException` / `UnicodeEncodeError`
- 尝试更换为 DejaVu Sans 字体，仍然无法支持中文

### 问题2：外部工具不可用
- `wkhtmltopdf` 未安装
- `pandoc` 可用但未能解决核心问题
- 没有 `pdflatex` / `xelatex` 环境

### 问题3：混合中英文内容导致PDF中出现CJK泄漏
- Markdown中的中文内容在HTML→PDF转换后出现乱码（`****` 方框）
- 表格行中的中文标签（如"方案""计算复杂度"）泄露到PDF

### 问题4：Notes/Disclaimers 泄漏到 Abstract 区域
- 块引用解析顺序错误，导致元数据和免责声明被合并到 Abstract 文本中

### 问题5：fpdf2 API 错误
- `AttributeError: 'AcademicPDF' object has no attribute 'w_page'`
- `TypeError: 'int' object is not callable`
- `fpdf.errors.FPDFException: Not enough horizontal space`

## 尝试的方法

1. **直接 fpdf2 生成** — 使用内置字体 → CJK失败
2. **DejaVu Sans 字体** — 仍不支持中文 → 放弃
3. **wkhtmltopdf / pandoc** — 环境不可用 → 放弃
4. **Playwright HTML-to-PDF 管道** — Markdown → HTML → PDF，利用系统字体原生支持Unicode

## 最终解决方案

**采用 Playwright HTML-to-PDF 管道**：
1. 使用 Python `markdown` 库将 Markdown 转为 HTML
2. 通过 Playwright 将 HTML 渲染为 PDF（利用系统字体原生处理 Unicode）
3. 在 HTML 生成阶段实施激进的英文提取过滤：
   - 表格行：从括号中提取英文或回退到纯字母数字
   - 段落：动态 CJK 比例阈值（`cjk_count > total_alpha * 0.6`）过滤中文主导文本
   - Abstract：修正块引用解析顺序，防止元数据合并

## 最终结果

成功生成6个PDF文件：
- `Soma_Engine.pdf` (327KB)
- `Soma_LingYa.pdf` (256KB)
- `Soma_Native.pdf` (144KB)
- `Soma_Convergence.pdf` (144KB)
- `Soma_Heritage.pdf` (144KB)
- `Dalin_Soma_Five_Papers_Combined.pdf` (144KB)

## 遗留问题
1. 数学公式仍以原始 LaTeX 字符串形式显示（缺少 MathJax/KaTeX）
2. 部分代码块中的中文标签（"输入""输出"）仍未清除
3. 标题页格式不一致
4. 部分章节编号格式丢失

## 关键经验教训

1. **fpdf2 不适合 CJK 内容** — 除非嵌入自定义字体，否则不要用于中英文混合文档
2. **Playwright HTML-to-PDF 是可靠选择** — 原生 Unicode 支持，视觉渲染准确
3. **内容过滤应在 HTML 生成阶段完成** — 而不是在 PDF 渲染阶段补救
4. **MathJax 会增加复杂度** — 对于草稿/技术报告，原始 LaTeX 可接受
5. **环境依赖很重要** — 在脚本中加入优雅降级机制，避免硬依赖
