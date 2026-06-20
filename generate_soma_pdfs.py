#!/usr/bin/env python3
"""
Generate formal English-only academic PDFs for Dalin Soma papers.

Pipeline: Markdown → HTML (with CSS) → PDF (via Playwright)
"""
import os
import re
import markdown
import subprocess

BASE = "/Users/apple/Desktop/QN1幻化引擎开源"

# Academic HTML template
HTML_TEMPLATE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>
@page {{
    size: A4;
    margin: 25mm 20mm 25mm 20mm;
    @bottom-center {{
        content: "Confidential — Property of Dalin Soma Project";
        font-size: 7pt;
        color: #999;
        font-style: italic;
    }}
}}
body {{
    font-family: Georgia, 'Times New Roman', serif;
    line-height: 1.55;
    max-width: none;
    margin: 0;
    padding: 0;
    color: #222;
    font-size: 10.5pt;
}}
h1 {{
    font-size: 20pt;
    color: #111;
    border-bottom: 2px solid #444;
    padding-bottom: 6px;
    margin-top: 30px;
    page-break-before: avoid;
}}
h2 {{
    font-size: 15pt;
    color: #1a1a6e;
    margin-top: 24px;
    page-break-before: avoid;
}}
h3 {{
    font-size: 12pt;
    color: #333;
    margin-top: 16px;
    page-break-before: avoid;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 9pt;
    page-break-inside: avoid;
}}
th, td {{
    border: 1px solid #bbb;
    padding: 4px 8px;
    text-align: center;
}}
th {{
    background: #e8e8f0;
    font-weight: bold;
}}
tr:nth-child(even) {{
    background: #fafafa;
}}
pre {{
    background: #f5f5f5;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 3px;
    font-size: 8pt;
    font-family: 'Courier New', monospace;
    overflow-x: auto;
    page-break-inside: avoid;
}}
code {{
    font-size: 8.5pt;
    font-family: 'Courier New', monospace;
}}
blockquote {{
    border-left: 3px solid #666;
    padding-left: 12px;
    color: #333;
    margin: 8px 0;
    font-style: italic;
}}
.note {{
    color: #b33;
    font-weight: bold;
    font-size: 9.5pt;
}}
.meta {{
    text-align: center;
    margin: 20px 0;
    font-size: 11pt;
    color: #444;
}}
.abstract-box {{
    background: #f9f9f9;
    border: 1px solid #ddd;
    padding: 15px;
    margin: 15px 0;
    border-radius: 4px;
}}
.abstract-title {{
    font-weight: bold;
    font-size: 12pt;
    text-align: center;
    margin-bottom: 8px;
}}
hr {{
    border: none;
    border-top: 1px solid #ccc;
    margin: 25px 0;
}}
ul, ol {{
    padding-left: 25px;
}}
li {{
    margin-bottom: 4px;
}}
</style>
</head><body>
{content}
</body></html>"""

# Title page template
TITLE_PAGE_TEMPLATE = """<div style="text-align:center; margin-top: 60px;">
<h1 style="border:none; font-size:24pt;">{title}</h1>
<p style="font-size:13pt; color:#446; font-style:italic;">{subtitle}</p>
<hr style="width:200px; margin:20px auto;">
<p style="font-size:11pt; margin:8px 0;">Author: {author}</p>
<p style="font-size:11pt; margin:8px 0;">Affiliation: {affiliation}</p>
<p style="font-size:11pt; margin:8px 0;">Email: {email}</p>
<p style="font-size:11pt; margin:8px 0;">Date: {date}</p>
<p style="font-size:11pt; margin:8px 0;">Version: {version}</p>
{abstract_html}
</div>"""


def strip_cjk(text):
    """Remove CJK characters."""
    return re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f\uff01-\uff5f\u2000-\u206f\u2700-\u27bf]', '', text).strip()


def extract_english_from_md(content):
    """Extract English content from bilingual paper, writing clean English HTML."""
    lines = content.split('\n')
    
    # Collect English content
    html_parts = []
    in_code = False
    code_buf = []
    in_table = False
    table_buf = []
    
    # Metadata
    meta = {'author': 'Dalin Jia', 'affiliation': 'Independent Researcher', 
            'email': '362118251@qq.com', 'date': 'June 2026', 'version': 'v3.0'}
    title = "Dalin Soma Technical Report"
    subtitle = ""
    abstract_text = ""
    keywords = ""
    
    for line in lines:
        s = line.strip()
        
        # Code blocks
        if s.startswith('```'):
            if in_code:
                html_parts.append('<pre>' + escape_html('\n'.join(code_buf)) + '</pre>')
                code_buf = []
                in_code = False
            else:
                in_code = True
                code_buf = []
            continue
        if in_code:
            code_buf.append(escape_html(re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', '', s)))
            continue
        
        # Headings - extract English
        if s.startswith('### '):
            html_parts.append(f'<h3>{escape_html(extract_english_heading(s[4:]))}</h3>')
            continue
        if s.startswith('## '):
            html_parts.append(f'<h2>{escape_html(extract_english_heading(s[3:]))}</h2>')
            continue
        if s.startswith('# '):
            h = s[2:]
            eng = extract_english_heading(h)
            html_parts.append(f'<h1>{escape_html(eng)}</h1>')
            if 'Soma' in h:
                title = eng
            continue
        
        # Abstract blockquote (MUST be before metadata check, since '> **Abstract:**' contains ** and :)
        if s.startswith('> **Abstract:**'):
            abstract_text = s.replace('> **Abstract:**', '').strip().replace('**', '')
            continue
        if s.startswith('> ') and abstract_text and not s.startswith('> **'):
            abstract_text += ' ' + s[2:].strip().replace('**', '')
            continue
        
        # Metadata
        if '**' in s and ':' in s:
            cleaned = strip_cjk(s)
            if 'Author:' in cleaned or 'Dalin' in cleaned:
                meta['author'] = re.search(r'(Dalin.*?)(?:\*|$)', cleaned).group(1) if re.search(r'(Dalin.*?)(?:\*|$)', cleaned) else meta['author']
            if 'Affiliation:' in cleaned:
                meta['affiliation'] = cleaned.split('Affiliation:')[-1].strip().replace('*', '')
            if 'Email:' in cleaned:
                meta['email'] = cleaned.split('Email:')[-1].strip().replace('*', '')
            if 'Date:' in cleaned and 'June' in cleaned:
                meta['date'] = cleaned.split('Date:')[-1].strip().replace('*', '')
            if 'Version:' in cleaned:
                meta['version'] = cleaned.split('Version:')[-1].strip().replace('*', '')
            continue
        
        # Keywords
        if 'Keywords:' in s or '关键词' in s:
            cleaned = strip_cjk(s)
            if 'Keywords:' in cleaned:
                keywords = cleaned.split('Keywords:')[-1].strip()
            continue
        
        # Tables
        if s.startswith('|'):
            if not in_table:
                in_table = True
                table_buf = [s]
            else:
                table_buf.append(s)
            continue
        if in_table:
            html_parts.append(render_table(table_buf))
            in_table = False
            table_buf = []
        
        # Empty lines
        if not s:
            continue
        
        # Skip pure Chinese lines (CJK-dominant paragraphs)
        cjk_count = len(re.findall(r'[\u4e00-\u9fff]', s))
        alpha_count = len(re.findall(r'[a-zA-Z]{3,}', s))
        # If line has CJK chars and fewer than 3 English words → skip
        if cjk_count > 0 and cjk_count > (cjk_count + alpha_count) * 0.6:
            continue
        
        # Block quotes (notes)
        if s.startswith('> '):
            text = s[2:].strip().replace('**', '')
            if text:
                html_parts.append(f'<blockquote class="note">{escape_html(text)}</blockquote>')
            continue
        
        # Bullet points
        if s.startswith('- ') or s.startswith('* '):
            html_parts.append(f'<li>{escape_html(strip_cjk(s[2:].strip()))}</li>')
            continue
        
        # Regular paragraphs - extract English
        cleaned = strip_cjk(s)
        if cleaned and alpha_count >= 2:
            html_parts.append(f'<p>{escape_html(cleaned)}</p>')
    
    # Flush remaining table
    if in_table and table_buf:
        html_parts.append(render_table(table_buf))
    
    # Build title page
    abstract_html = ''
    if abstract_text:
        abstract_html = f'''<div class="abstract-box">
<div class="abstract-title">ABSTRACT</div>
<p>{escape_html(abstract_text)}</p>
</div>'''
    
    title_html = TITLE_PAGE_TEMPLATE.format(
        title=title,
        subtitle=extract_english_heading(subtitle) if subtitle else "",
        author=meta['author'],
        affiliation=meta['affiliation'],
        email=meta['email'],
        date=meta['date'],
        version=meta['version'],
        abstract_html=abstract_html
    )
    
    return title_html + '\n' + '\n'.join(html_parts)


def escape_html(text):
    """Escape HTML special characters."""
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;'))


def extract_english_heading(text):
    """Extract English from bilingual heading."""
    # "1. 引言 (Introduction)" → "1. Introduction"
    paren = re.search(r'\(([^)]+)\)', text)
    if paren:
        prefix = re.match(r'^(\d+\.?\s*)', text)
        if prefix:
            return prefix.group(1).strip() + ' ' + paren.group(1).strip()
        return paren.group(1).strip()
    # Pure English
    if not re.search(r'[\u4e00-\u9fff]', text):
        return text
    # Mixed - extract English words
    words = re.findall(r'[A-Za-z][A-Za-z0-9_.\- ]+', text)
    nums = re.findall(r'\d+\.?\d*', text)
    result = ' '.join(nums + words)
    return result if result.strip() else text


def render_table(table_lines):
    """Render markdown table to HTML."""
    if not table_lines:
        return ''
    
    header = table_lines[0]
    cols = [c.strip() for c in header.split('|')[1:-1]]
    
    # Check if ALL header columns are Chinese → skip table
    all_cjk = all(re.search(r'[\u4e00-\u9fff]', c) and not re.search(r'[a-zA-Z]{3,}', c) for c in cols)
    if all_cjk:
        return ''
    
    header_eng = []
    for c in cols:
        if not re.search(r'[\u4e00-\u9fff]', c):
            header_eng.append(c)
        else:
            m = re.search(r'\(([^)]+)\)', c)
            header_eng.append(m.group(1).strip() if m else '')
    
    header_eng = [h for h in header_eng if h]
    if not header_eng:
        return ''
    
    data_rows = []
    for line in table_lines[2:]:
        if '|' not in line:
            continue
        cols = [c.strip() for c in line.split('|')[1:-1]]
        eng_row = []
        for c in cols:
            if not re.search(r'[\u4e00-\u9fff]', c):
                eng_row.append(c)
            else:
                m = re.search(r'\(([^)]+)\)', c)
                if m:
                    eng_row.append(m.group(1).strip())
                else:
                    words = re.findall(r'[A-Za-z0-9_.\-+×µ]', c)
                    eng_row.append(''.join(words))
        if any(eng_row):
            data_rows.append(eng_row)
    
    html = '<table>'
    html += '<thead><tr>' + ''.join(f'<th>{escape_html(c)}</th>' for c in header_eng) + '</tr></thead>'
    if data_rows:
        html += '<tbody>'
        for row in data_rows:
            html += '<tr>' + ''.join(f'<td>{escape_html(c)}</td>' for c in row) + '</tr>'
        html += '</tbody>'
    html += '</table>'
    return html


def generate_pdf_from_html(html_content, pdf_output):
    """Use Playwright to convert HTML to PDF."""
    from playwright.sync_api import sync_playwright
    
    # Write HTML to temp file
    html_path = pdf_output.replace('.pdf', '.html')
    full_html = HTML_TEMPLATE.format(content=html_content)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    # Use Playwright to generate PDF
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f'file://{os.path.abspath(html_path)}')
        # Wait for content to render
        page.wait_for_timeout(500)
        page.pdf(path=pdf_output, 
                 format='A4',
                 print_background=True,
                 margin={'top': '25mm', 'bottom': '20mm', 'left': '20mm', 'right': '20mm'})
        browser.close()
    
    # Clean up HTML
    if os.path.exists(html_path):
        os.remove(html_path)


def main():
    print("=" * 60)
    print("  Dalin Soma — PDF Generator (Markdown → HTML → PDF)")
    print("=" * 60)
    
    papers = [
        ("01_soma_engine/学术论文_v3.md", "Soma_Engine"),
        ("02_soma_lingya/学术论文_v3.md", "Soma_LingYa"),
        ("03_soma_native/学术论文_v3.md", "Soma_Native"),
        ("04_soma_convergence/学术论文.md", "Soma_Convergence"),
        ("05_soma_heritage/学术论文_v3.md", "Soma_Heritage"),
    ]
    
    eng_contents = {}
    
    for md_file, name in papers:
        md_path = os.path.join(BASE, md_file)
        if not os.path.exists(md_path):
            print(f"  ⚠️ Not found: {md_file}")
            continue
        
        print(f"\n📄 Processing: {name}")
        
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        html = extract_english_from_md(content)
        
        # Generate PDF
        pdf_path = os.path.join(BASE, f"{name}.pdf")
        generate_pdf_from_html(html, pdf_path)
        print(f"  ✅ {os.path.basename(pdf_path)} ({os.path.getsize(pdf_path) // 1024} KB)")
        
        eng_contents[name] = html
    
    # Combined
    print(f"\n📄 Generating combined paper...")
    combined = '\n<hr>\n'.join(eng_contents.values())
    combined_path = os.path.join(BASE, "Dalin_Soma_Five_Papers_Combined.pdf")
    generate_pdf_from_html(combined, combined_path)
    print(f"  ✅ Dalin_Soma_Five_Papers_Combined.pdf ({os.path.getsize(combined_path) // 1024} KB)")
    
    print(f"\n{'=' * 60}")
    print(f"  All PDFs generated in: {BASE}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
