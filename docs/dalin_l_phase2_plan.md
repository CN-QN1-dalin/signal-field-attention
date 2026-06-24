# Dalin L — Phase 2 作战计划

> **阶段**: Agent 特性开发
> **时间**: 2026-07 ~ 2026-09（3 个月）
> **团队**: 2 人全职
> **审查引擎**: 五方辩论共识引擎（持续运行）
> **状态**: 启动中

---

## 目标

完成 MVP 0.1.0 之后的所有 Agent 专属特性：

| 特性 | 负责人 | 状态 |
|------|--------|------|
| async/await + channel | Code | ⏳ 待开始 |
| 自我修复编译 | Fix | ⏳ 待开始 |
| 自动测试生成 | Test | ⏳ 待开始 |
| 统一 FFI（C） | FFI | ⏳ 待开始 |
| dalin fmt | Fmt | ⏳ 待开始 |
| dalin docs | Doc | ⏳ 待开始 |

---

## Month 1: 并发 + 自我修复

### Week 1-2: async/await + channel

**负责人**: Code

```dalin
// 目标：支持结构化并发
fn main() async {
    let (发送, 接收) = channel::<int>(16)
    
    spawn async {
        for i in 0..100 {
            发送.send(i).await
        }
    }
    
    for await 值 in 接收 {
        处理(值)
    }
}
```

**交付物**:
- ✅ `spawn` 关键字
- ✅ `async/await` 语法
- ✅ `channel` 类型
- ✅ 结构化并发生命周期管理

**审查标准**:
- Alpha: 并发安全，无数据竞争
- Beta: Agent 写并发代码零学习成本
- 混元: 编译速度 < 1s
- 元宝: 错误信息清晰

### Week 3-4: 自我修复编译

**负责人**: Fix

```rust
// src/auto_fix.rs

/// 自我修复引擎
pub struct AutoFix {
    errors: Vec<CompileError>,
    suggestions: Vec<Suggestion>,
}

impl AutoFix {
    /// 分析编译错误，生成修复建议
    pub fn analyze(&mut self, error: &CompileError) -> Vec<Suggestion> {
        match error.kind {
            ErrorKind::TypeMismatch { expected, actual } => {
                // 概率性修复
                let prob = self.calculate_fix_probability(expected, actual);
                if prob > 0.95 {
                    vec![Suggestion {
                        action: FixAction::ChangeType(expected.clone()),
                        probability: prob,
                        description: format!("将类型改为 {}", expected),
                    }]
                } else {
                    vec![
                        Suggestion {
                            action: FixAction::ChangeType(expected.clone()),
                            probability: prob,
                            description: format!("将类型改为 {}", expected),
                        },
                        Suggestion {
                            action: FixAction::ChangeValue(actual.clone()),
                            probability: prob * 0.5,
                            description: format!("将值改为 {}", expected.default_value()),
                        },
                    ]
                }
            }
            ErrorKind::UndefinedVariable(name) => {
                // 检查是否有相似变量名
                let similar = self.find_similar_names(&name);
                if !similar.is_empty() {
                    similar.into_iter().map(|n| Suggestion {
                        action: FixAction::RenameVariable(n.clone()),
                        probability: 0.9,
                        description: format!("是否想使用 '{}'?", n),
                    }).collect()
                } else {
                    vec![Suggestion {
                        action: FixAction::AddVariable(name.clone()),
                        probability: 0.7,
                        description: format!("添加变量 '{}' = ?", name),
                    }]
                }
            }
            _ => vec![],
        }
    }
    
    /// 计算修复概率（基于历史决策）
    fn calculate_fix_probability(&self, expected: &Type, actual: &Type) -> f64 {
        // 基于 Agent 的历史行为计算
        // 如果 Agent 之前多次将 string 改为 int，概率更高
        self.history.probability(expected, actual)
    }
    
    /// 查找相似变量名
    fn find_similar_names(&self, name: &str) -> Vec<String> {
        // Levenshtein 距离
        self.env.variables()
            .iter()
            .filter(|v| levenshtein(v, name) <= 2)
            .cloned()
            .collect()
    }
}
```

**交付物**:
- ✅ 概率性修复引擎
- ✅ 修复建议 UI（REPL + CLI）
- ✅ 历史决策学习

**审查标准**:
- Alpha: 修复不改变语义
- Beta: 修复准确率 > 99%
- 元宝: 用户确认流程简单

---

## Month 2: 自动测试 + 统一 FFI

### Week 5-6: 自动测试生成

**负责人**: Test

```dalin
// src/auto_test.rs

/// 自动测试生成器
pub struct AutoTestGen {
    functions: Vec<Function>,
}

impl AutoTestGen {
    /// 为函数自动生成测试用例
    pub fn generate_tests(&self, func: &Function) -> Vec<Test> {
        let mut tests = Vec::new();
        
        // 1. 正常路径测试
        tests.push(Test {
            name: format!("{}_normal", func.name),
            input: self.generate_normal_inputs(func),
            expected: self.infer_expected_output(func),
        });
        
        // 2. 边界条件测试
        tests.push(Test {
            name: format!("{}_boundary", func.name),
            input: self.generate_boundary_inputs(func),
            expected: self.infer_expected_output(func),
        });
        
        // 3. 异常输入测试
        tests.push(Test {
            name: format!("{}_error", func.name),
            input: self.generate_error_inputs(func),
            expected: Error,
        });
        
        // 4. 随机 fuzz 测试
        tests.extend(self.fuzz_test(func, 100));
        
        tests
    }
    
    /// 生成正常输入
    fn generate_normal_inputs(&self, func: &Function) -> Vec<Value> {
        func.params.iter().map(|param| {
            match param.ty {
                Type::Int => Value::Int(42),
                Type::Float => Value::Float(3.14),
                Type::String => Value::String("hello".to_string()),
                Type::Bool => Value::Bool(true),
                _ => Value::Unknown,
            }
        }).collect()
    }
    
    /// 生成边界输入
    fn generate_boundary_inputs(&self, func: &Function) -> Vec<Value> {
        func.params.iter().map(|param| {
            match param.ty {
                Type::Int => Value::Int(i64::MIN),  // 最小值
                Type::Float => Value::Float(f64::INFINITY),
                Type::String => Value::String("".to_string()),  // 空字符串
                Type::Bool => Value::Bool(false),
                _ => Value::Unknown,
            }
        }).collect()
    }
    
    /// 生成异常输入
    fn generate_error_inputs(&self, func: &Function) -> Vec<Value> {
        func.params.iter().map(|param| {
            match param.ty {
                Type::Int => Value::String("not_a_number".to_string()),
                Type::Float => Value::String("not_a_number".to_string()),
                Type::String => Value::Int(42),
                Type::Bool => Value::String("not_a_bool".to_string()),
                _ => Value::Unknown,
            }
        }).collect()
    }
    
    /// Fuzz 测试
    fn fuzz_test(&self, func: &Function, iterations: int) -> Vec<Test> {
        (0..iterations).map(|_| Test {
            name: format!("{}_fuzz_{}", func.name, rand::random::<u64>()),
            input: self.random_inputs(func),
            expected: self.infer_expected_output(func),
        }).collect()
    }
}
```

**交付物**:
- ✅ 边界分析引擎
- ✅ Fuzz 测试生成
- ✅ 覆盖率 > 90%

**审查标准**:
- Alpha: 测试用例覆盖所有分支
- Beta: Agent 零配置
- GPT: 边界分析理论正确
- 混元: 生成速度快
- 元宝: 测试报告清晰

### Week 7-8: 统一 FFI（C）

**负责人**: FFI

```dalin
// src/ffi.rs

/// C FFI 绑定生成器
pub struct CFFIGenerator {
    headers: Vec<CHeader>,
}

impl CFFIGenerator {
    /// 从 C 头文件生成 Dalin L 绑定
    pub fn generate_bindings(&self, header: &CHeader) -> Vec<DalinStmt> {
        let mut stmts = Vec::new();
        
        for func in &header.functions {
            stmts.push(DalinStmt::Fn {
                name: func.name.clone(),
                params: func.params.iter().map(|p| {
                    (p.name.clone(), Some(self.c_type_to_dalin(p.ty)))
                }).collect(),
                return_ty: Some(self.c_type_to_dalin(func.return_type)),
                body: self.generate_ffi_call(func),
            });
        }
        
        stmts
    }
    
    /// C 类型到 Dalin L 类型映射
    fn c_type_to_dalin(&self, c_type: &CStringType) -> String {
        match c_type {
            CStringType::Int => "int".to_string(),
            CStringType::Long => "int".to_string(),
            CStringType::Float => "float".to_string(),
            CStringType::Double => "float".to_string(),
            CStringType::Char => "char".to_string(),
            CStringType::Pointer => "*u8".to_string(),
            CStringType::Void => "unit".to_string(),
            _ => "unknown".to_string(),
        }
    }
    
    /// 生成 FFI 调用
    fn generate_ffi_call(&self, func: &CFunction) -> Expr {
        Expr::Call {
            func: Box::new(Expr::Ident(func.name.clone())),
            args: func.params.iter().map(|p| {
                Expr::Variable(p.name.clone())
            }).collect(),
        }
    }
}

// 使用示例
use c "stdio.h" as stdio

fn main() {
    stdio.printf("Hello, %s!\n", "Dalin L")
    let fd = stdio.fopen("test.txt", "w")
    stdio.fputs("Hello, World!\n", fd)
    stdio.fclose(fd)
}
```

**交付物**:
- ✅ C 头文件解析
- ✅ 类型映射
- ✅ FFI 调用生成

**审查标准**:
- Alpha: FFI 调用安全
- Beta: 零学习成本
- 混元: 性能无损

---

## Month 3: 格式化器 + 文档生成

### Week 9-10: dalin fmt

**负责人**: Fmt

```rust
// src/formatter.rs

/// 代码格式化器
pub struct Formatter {
    config: FormatConfig,
}

impl Formatter {
    /// 格式化源代码
    pub fn format(&self, source: &str) -> Result<String, FormatError> {
        let tokens = lex(source)?;
        let ast = parse(&tokens)?;
        let formatted = self.format_ast(&ast);
        Ok(formatted)
    }
    
    /// 格式化 AST
    fn format_ast(&self, ast: &Program) -> String {
        let mut output = String::new();
        
        for stmt in &ast.stmts {
            output.push_str(&self.format_stmt(stmt, 0));
            output.push('\n');
        }
        
        output
    }
    
    /// 格式化语句
    fn format_stmt(&self, stmt: &Stmt, indent: int) -> String {
        let indent_str = "  ".repeat(indent as usize);
        
        match stmt {
            Stmt::Let { name, ty, value } => {
                let mut s = format!("{}let {}{}", indent_str, name, 
                    if let Some(ty) = ty { format!(": {}", ty) } else { String::new() });
                if let Some(val) = value {
                    s.push_str(&format!(" = {}", self.format_expr(val, indent)));
                }
                s
            }
            Stmt::Fn { name, params, return_ty, body } => {
                let mut s = format!("{}fn {}(", indent_str, name);
                for (i, (param_name, param_ty)) in params.iter().enumerate() {
                    if i > 0 { s.push_str(", "); }
                    s.push_str(param_name);
                    if let Some(ty) = param_ty {
                        s.push_str(&format!(": {}", ty));
                    }
                }
                s.push_str(&format!(")"));
                if let Some(ty) = return_ty {
                    s.push_str(&format!(" -> {}", ty));
                }
                s.push_str(&format!(" {}", self.format_expr(body, indent + 1)));
                s
            }
            _ => format!("{}{:?}", indent_str, stmt),
        }
    }
    
    /// 格式化表达式
    fn format_expr(&self, expr: &Expr, indent: int) -> String {
        // 简化的表达式格式化
        format!("{:?}", expr)
    }
}
```

**交付物**:
- ✅ 统一代码风格
- ✅ `dalin fmt` 命令
- ✅ 配置文件支持

**审查标准**:
- Alpha: 格式化不改变语义
- Beta: 一键格式化
- 混元: 格式化速度快
- 元宝: 格式化结果一致

### Week 11-12: dalin docs

**负责人**: Doc

```rust
// src/docgen.rs

/// 文档生成器
pub struct DocGenerator {
    functions: Vec<Function>,
    structs: Vec<Struct>,
}

impl DocGenerator {
    /// 生成 HTML 文档
    pub fn generate_html(&self, output_path: &str) -> Result<(), DocError> {
        let html = self.build_html();
        std::fs::write(output_path, html)?;
        Ok(())
    }
    
    /// 构建 HTML
    fn build_html(&self) -> String {
        let mut html = String::from("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("  <meta charset=\"utf-8\">\n");
        html.push_str("  <title>Dalin L 文档</title>\n");
        html.push_str("  <style>\n");
        html.push_str("    body { font-family: sans-serif; margin: 2em; }\n");
        html.push_str("    h1 { color: #333; }\n");
        html.push_str("    pre { background: #f5f5f5; padding: 1em; }\n");
        html.push_str("  </style>\n");
        html.push_str("</head>\n<body>\n");
        
        html.push_str("<h1>Dalin L API 文档</h1>\n");
        
        // 函数文档
        html.push_str("<h2>函数</h2>\n");
        for func in &self.functions {
            html.push_str(&format!("<h3>{}</h3>\n", func.name));
            html.push_str(&format!("<pre><code>{}</code></pre>\n", func.signature));
            if let Some(doc) = &func.doc {
                html.push_str(&format!("<p>{}</p>\n", doc));
            }
        }
        
        // 结构体文档
        html.push_str("<h2>结构体</h2>\n");
        for struct_def in &self.structs {
            html.push_str(&format!("<h3>{}</h3>\n", struct_def.name));
            html.push_str(&format!("<pre><code>{}</code></pre>\n", struct_def.definition));
            if let Some(doc) = &struct_def.doc {
                html.push_str(&format!("<p>{}</p>\n", doc));
            }
        }
        
        html.push_str("</body>\n</html>");
        html
    }
}
```

**交付物**:
- ✅ HTML 文档生成
- ✅ `dalin docs` 命令
- ✅ 中文文档支持

**审查标准**:
- Alpha: 文档准确
- Beta: 零配置
- 豆包: 中文文档
- 元宝: 文档美观

---

## 资源需求

| 资源 | 数量 | 说明 |
|------|------|------|
| 团队 | 2 人 | Fix/Test/FFI + Fmt/Doc |
| 时间 | 3 个月 | 12 周 |
| 预算 | TBD | 取决于薪资 |

---

## 里程碑

| 日期 | 里程碑 | 状态 |
|------|--------|------|
| 2026-07 | Month 1: 并发 + 自我修复 | ⏳ 待开始 |
| 2026-08 | Month 2: 自动测试 + FFI | ⏳ 待开始 |
| 2026-09 | Month 3: 格式化 + 文档 | ⏳ 待开始 |
| 2026-10 | Phase 2 完成 | ⏳ 待开始 |

---

## 审查机制

每个功能完成后，必须通过以下审查：

| 审查方 | 重点 |
|--------|------|
| **Alpha** | 技术正确性 |
| **Beta** | Agent 友好度 |
| **豆包** | 中文支持 |
| **GPT** | 理论正确性 |
| **混元** | 工程实现 |
| **元宝** | 用户体验 |

---

**Phase 2 启动！全员就位！**

**执行。**
