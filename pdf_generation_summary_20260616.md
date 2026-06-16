# PDF 生成过程总结

## 一、时间线概览

### 昨天（2026-06-15）
- 使用 `fpdf2` 编写 Python 脚本 `/tmp/gen_soma_pdf.py`，成功生成了 `SFA_Technical_Paper.pdf`（6页，13524字节）。
- 首次 PDF 生成基本成功。

### 今天（2026-06-16）

**阶段一：12:23 - 12:53 — 双语 PDF 尝试与 fpdf2 困境**
- 用户请求生成 Soma 论文的双语 PDF。
- 初始方案：用 `fpdf2` 直接生成 PDF。
- **问题 1**：`fpdf2` 核心字体（Courier）遇到 CJK 字符时报错 `UnicodeEncodeError`。
- **问题 2**：尝试使用 DejaVu Sans 字体，仍无法支持 CJK。
- **问题 3**：`wkhtmltopdf` 和 `pandoc` 在环境中不可用。
- **决策**：放弃 `fpdf2` 直接生成路线，转向 Playwright HTML-to-PDF 方案。

**阶段二：12:53 - 13:39 — Playwright 方案与英文提取策略**
- 创建 `generate_soma_pdfs.py`，采用 Playwright HTML-to-PDF 管线。
- **问题 4**：混合中英文 Markdown 内容导致 PDF 中出现 CJK 泄漏。
- **解决方案**：实施激进的英文提取过滤器：
  - **表格行**：从括号中提取英文，或回退到纯字母数字。
  - **段落**：动态 CJK 比例阈值（`cjk_count > total_alpha * 0.6`）过滤中文主导文本，保留技术英文术语。
  - **摘要块**：修正 blockquote 解析顺序，防止元数据和声明合并到 Abstract 正文。

**阶段三：13:39 - 14:07 — 最终打磨与生成**
- **问题 5**：Notes/Disclaimers 仍然泄漏到 Abstract 部分。
- **修复**：修正脚本第 212 行，遇到非 Abstract blockquote 时停止追加到 `abstract_text`。
- **结果**：成功生成 6 个 PDF 文件至 `/Users/apple/Desktop/太初五岳开源/`：
  1. `Soma_Engine.pdf`
  2. `Soma_LingYa.pdf`
  3. `Soma_Native.pdf`
  4. `Soma_Convergence.pdf`
  5. `Soma_Heritage.pdf`
  6. `Dalin_Soma_Five_Papers_Combined.pdf`

## 二、遇到的主要问题清单

| # | 问题 | 原因 | 解决方案 |
|---|------|------|----------|
| 1 | fpdf2 CJK 编码错误 | Core fonts 不支持中文 | 放弃 fpdf2 直接生成 |
| 2 | DejaVu Sans 仍不支持 CJK | 字体本身无 CJK 字形 | 彻底切换方案 |
| 3 | wkhtmltopdf/pandoc 不可用 | 环境未安装 | 改走 Playwright |
| 4 | PDF 中 CJK 泄漏 | 混合内容未经过滤 | 英文提取过滤器 |
| 5 | Abstract 混入元数据 | blockquote 解析顺序错误 | 修正第 212 行逻辑 |

## 三、关键经验教训

1. **fpdf2 不适合 CJK 场景**：除非配置复杂字体嵌入，否则应避免使用。
2. **Playwright HTML-to-PDF 是可靠替代方案**：利用浏览器渲染引擎处理排版，避免字体问题。
3. **英文提取策略**：对于中英文混合内容，激进过滤（CJK 比例阈值）比精细翻译更有效率。
4. **blockquote 解析需严格顺序控制**：元数据块和技术正文的边界容易混淆。
5. **剩余待解决问题**：数学公式仍为原始 LaTeX 字符串、代码块中仍有中文标签、标题页重复。

## 四、引用来源

- sum_0a926e662e2a8fc3（主压缩摘要）
- sum_16395a812c04125f（fpdf2 字体调试）
- sum_299a79193bcaabef（脚本修改记录）
- sum_65fdd25060d70a90（session 交接记录）
- sum_e6c0f67505a970aa（昨日首次 PDF 生成）
