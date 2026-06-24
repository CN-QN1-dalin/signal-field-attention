# Dalin L 仓库结构

## 目录结构

```
dalin-l/
├── .github/                    # GitHub 配置
│   ├── workflows/              # CI/CD
│   │   └── ci.yml              # 持续集成
│   ├── PULL_REQUEST_TEMPLATE.md # PR 模板
│   └── ISSUE_TEMPLATE.md       # Issue 模板
├── docs/                       # 文档
│   ├── language-reference.md   # 语言参考
│   ├── stdlib-reference.md     # 标准库参考
│   ├── agent-features.md       # Agent 特性
│   └── examples.md             # 示例代码
├── src/                        # 源代码
│   ├── main.rs                 # CLI 入口
│   ├── lexer.rs                # 词法分析器
│   ├── parser.rs               # 语法分析器
│   ├── type_checker.rs         # 类型检查器
│   ├── codegen.rs              # 代码生成器
│   ├── stdlib/                 # 标准库
│   │   ├── mod.rs
│   │   ├── collections.rs
│   │   ├── io.rs
│   │   ├── string.rs
│   │   └── option.rs
│   ├── cli/                    # CLI 模块
│   │   ├── mod.rs
│   │   ├── build.rs
│   │   ├── run.rs
│   │   ├── test.rs
│   │   ├── repl.rs
│   │   └── fmt.rs
│   ├── agent/                  # Agent 特性
│   │   ├── mod.rs
│   │   ├── auto_fix.rs
│   │   ├── auto_test.rs
│   │   ├── intent.rs
│   │   └── merge.rs
│   └── ffi/                    # FFI 模块
│       ├── mod.rs
│       └── c.rs
├── tests/                      # 测试
│   ├── lexer_tests.rs
│   ├── parser_tests.rs
│   ├── type_checker_tests.rs
│   ├── codegen_tests.rs
│   └── integration_tests.rs
├── examples/                   # 示例代码
│   ├── hello.dalan
│   ├── fibonacci.dalan
│   └── ...
├── std/                        # 标准库源码
│   ├── collections.dalan
│   ├── io.dalan
│   ├── string.dalan
│   └── ...
├── Cargo.toml                  # Rust 项目配置
├── Cargo.lock                  # 依赖锁定
├── README.md                   # 项目介绍
├── LICENSE                     # 许可证
├── CHANGELOG.md                # 变更日志
├── CONTRIBUTING.md             # 贡献指南
├── CODE_OF_CONDUCT.md          # 行为准则
├── SECURITY.md                 # 安全政策
├── .gitignore                  # Git 忽略
└── docs/                       # 设计文档
    ├── dalin_l_agent_native.md
    ├── dalin_l_debate_final.md
    ├── dalin_l_final_design.md
    ├── dalin_l_next_steps.md
    ├── dalin_l_team_handbook.md
    ├── dalin_l_lexer_prototype.md
    ├── dalin_l_parser_prototype.md
    ├── dalin_l_type_system_prototype.md
    ├── dalin_l_codegen_prototype.md
    ├── dalin_l_stdlib_prototype.md
    ├── dalin_l_cli_repl_prototype.md
    ├── dalin_l_delivery_report.md
    ├── dalin_l_phase2_plan.md
    ├── dalin_l_phase3_plan.md
    ├── dalin_l_phase4_plan.md
    ├── dalin_l_complete_roadmap.md
    ├── dalin_l_phase1_implementation_guide.md
    └── dalin_l_audit_report.md
```

## 文件说明

### 根目录
- `README.md` — 项目介绍和快速开始
- `LICENSE` — MIT 许可证
- `Cargo.toml` — Rust 项目配置
- `CHANGELOG.md` — 变更日志
- `CONTRIBUTING.md` — 贡献指南
- `CODE_OF_CONDUCT.md` — 行为准则
- `SECURITY.md` — 安全政策

### src/
- `main.rs` — CLI 入口点
- `lexer.rs` — 词法分析器
- `parser.rs` — 语法分析器
- `type_checker.rs` — 类型检查器
- `codegen.rs` — 代码生成器

### docs/
- `design.md` — 设计文档
- `reference.md` — 语言参考
- `examples/` — 示例代码

### tests/
- `lexer_tests.rs` — 词法分析器测试
- `parser_tests.rs` — 语法分析器测试
- `type_checker_tests.rs` — 类型检查器测试
- `codegen_tests.rs` — 代码生成器测试
- `integration_tests.rs` — 集成测试
