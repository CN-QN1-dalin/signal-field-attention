# Dalin L — Phase 3 作战计划

> **阶段**: 生态建设
> **时间**: 2026-10 ~ 2026-12（3 个月）
> **团队**: 2 人全职
> **审查引擎**: 五方辩论共识引擎（持续运行）
> **状态**: 规划中

---

## 目标

完成生态建设，让 Dalin L 真正可用：

| 特性 | 负责人 | 状态 |
|------|--------|------|
| 包管理器 | CLI | ⏳ 待开始 |
| 中文全面支持 | 豆包 | ⏳ 待开始 |
| VSCode 插件 | IDE | ⏳ 待开始 |
| 标准库扩展 | Std | ⏳ 待开始 |
| 文档站点 | Doc | ⏳ 待开始 |
| 社区建设 | 全员 | ⏳ 待开始 |

---

## Month 1: 包管理器 + 中文全面支持

### Week 1-2: dalin 包管理器

**负责人**: CLI

```rust
// src/package_manager.rs

/// 包管理器
pub struct PackageManager {
    registry_url: String,
    cache_dir: PathBuf,
    lock_file: PathBuf,
}

impl PackageManager {
    /// 添加依赖
    pub fn add(&mut self, name: &str, version: &str) -> Result<(), PackageError> {
        // 1. 从注册表下载
        let package = self.fetch_from_registry(name, version)?;
        
        // 2. 解析依赖树
        let deps = self.resolve_dependencies(&package)?;
        
        // 3. 下载到缓存
        for dep in &deps {
            self.download_to_cache(dep)?;
        }
        
        // 4. 更新 lock 文件
        self.update_lock_file(&package, &deps)?;
        
        Ok(())
    }
    
    /// 移除依赖
    pub fn remove(&mut self, name: &str) -> Result<(), PackageError> {
        // 1. 从 lock 文件移除
        // 2. 清理缓存
        // 3. 重新解析依赖树
        Ok(())
    }
    
    /// 更新依赖
    pub fn update(&mut self) -> Result<(), PackageError> {
        // 1. 检查注册表新版本
        // 2. 解析新依赖树
        // 3. 下载更新
        // 4. 更新 lock 文件
        Ok(())
    }
    
    /// 从注册表获取包
    fn fetch_from_registry(&self, name: &str, version: &str) -> Result<Package, PackageError> {
        let url = format!("{}/api/packages/{}/versions/{}", self.registry_url, name, version);
        // HTTP GET
        todo!()
    }
    
    /// 解析依赖树
    fn resolve_dependencies(&self, package: &Package) -> Result<Vec<Package>, PackageError> {
        // 拓扑排序 + 版本冲突检测
        todo!()
    }
    
    /// 下载到缓存
    fn download_to_cache(&self, package: &Package) -> Result<(), PackageError> {
        let cache_path = self.cache_dir.join(&package.name);
        // HTTP GET + 解压
        todo!()
    }
    
    /// 更新 lock 文件
    fn update_lock_file(&self, package: &Package, deps: &[Package]) -> Result<(), PackageError> {
        let lock_content = self.serialize_lock_file(package, deps);
        std::fs::write(&self.lock_file, lock_content)?;
        Ok(())
    }
}
```

**dalin.toml 配置**:
```toml
[package]
name = "my-project"
version = "0.1.0"
edition = "2026"

[dependencies]
http = "2.0"
serde = "1.0"
sqlite = "0.5"

[dev-dependencies]
test-utils = "1.0"
```

**交付物**:
- ✅ `dalin add` — 添加依赖
- ✅ `dalin remove` — 移除依赖
- ✅ `dalin update` — 更新依赖
- ✅ `dalin.toml` — 配置文件
- ✅ `dalin.lock` — 锁文件

**审查标准**:
- Alpha: 依赖解析正确
- Beta: 零配置
- 混元: 下载速度快
- 元宝: 错误信息清晰

### Week 3-4: 中文全面支持

**负责人**: 豆包

```rust
// src/chinese_support.rs

/// 中文支持模块
pub struct ChineseSupport {
    encoding: Encoding,
}

impl ChineseSupport {
    /// 验证中文标识符
    pub fn validate_identifier(&self, name: &str) -> Result<(), ValidationError> {
        // 检查是否为有效的中文标识符
        if name.is_empty() {
            return Err(ValidationError::EmptyIdentifier);
        }
        
        for ch in name.chars() {
            if !ch.is_alphanumeric() && !ch == '_' && !self.is_chinese_char(ch) {
                return Err(ValidationError::InvalidCharacter(ch));
            }
        }
        
        Ok(())
    }
    
    /// 判断是否为中文字符
    fn is_chinese_char(&self, ch: char) -> bool {
        let c = ch as u32;
        (c >= 0x4E00 && c <= 0x9FFF) ||  // CJK 统一汉字
        (c >= 0x3400 && c <= 0x4DBF) ||  // CJK 扩展 A
        (c >= 0x20000 && c <= 0x2A6DF) || // CJK 扩展 B
        (c >= 0xF900 && c <= 0xFAFF)      // CJK 兼容汉字
    }
    
    /// 生成中文错误信息
    pub fn error_message_cn(&self, code: &str, details: &str) -> String {
        match code {
            "E0001" => format!("类型不匹配：期望 {}, 得到 {}", details, details),
            "E0002" => format!("未定义的变量：{}", details),
            "E0003" => format!("函数参数数量不匹配：期望 {}, 得到 {}", details, details),
            _ => format!("错误 ({})：{}", code, details),
        }
    }
    
    /// 生成中文帮助信息
    pub fn help_message_cn(&self) -> String {
        r#"📝 Dalin L 中文帮助

可用命令:
  dalan build   编译程序
  dalan run     运行程序
  dalan test    运行测试
  dalan repl    交互式开发
  dalan fmt     格式化代码
  dalan docs    生成文档
  dalan add     添加依赖
  dalan remove  移除依赖
  dalan update  更新依赖

快捷键:
  Ctrl+D      退出 REPL
  Ctrl+C      取消当前操作
  Tab         自动补全
"#.to_string()
    }
}
```

**交付物**:
- ✅ 中文变量名/函数名
- ✅ 中文注释
- ✅ 中文错误信息
- ✅ 中文帮助文档

**审查标准**:
- 豆包: 中文支持全面
- Alpha: UTF-8 编码正确
- 混元: 性能无损

---

## Month 2: VSCode 插件 + 标准库扩展

### Week 5-6: VSCode 插件

**负责人**: IDE

```typescript
// src/ide/vscode-extension/src/extension.ts

import * as vscode from 'vscode';
import { DalinLServer } from './dalin-server';

export function activate(context: vscode.ExtensionContext) {
    const server = new DalinLServer();
    
    // 注册命令
    context.subscriptions.push(
        vscode.commands.registerCommand('dalin.build', () => {
            server.build();
        }),
        vscode.commands.registerCommand('dalin.run', () => {
            server.run();
        }),
        vscode.commands.registerCommand('dalin.test', () => {
            server.test();
        }),
        vscode.commands.registerCommand('dalin.fmt', () => {
            server.format();
        }),
        vscode.commands.registerCommand('dalin.docs', () => {
            server.generateDocs();
        }),
    );
    
    // 注册语言服务
    const languageClient = new LanguageClient({
        documentSelector: [{ scheme: 'file', language: 'dalin' }],
        uriConverter: {
            encode: (uri: vscode.Uri) => uri.toString(),
            decode: (str: string) => vscode.Uri.parse(str),
        },
        initializationOptions: {
            executable: 'dalin',
            args: ['language-server'],
        },
    });
    
    context.subscriptions.push(languageClient.start());
}
```

**功能**:
- ✅ 语法高亮
- ✅ 自动补全
- ✅ 跳转到定义
- ✅ 重构支持
- ✅ 实时错误检查
- ✅ 中文错误信息

**交付物**:
- ✅ VSCode 扩展
- ✅ 语法高亮
- ✅ 自动补全
- ✅ 错误检查

**审查标准**:
- Alpha: 语言服务器正确
- Beta: 用户体验好
- 豆包: 中文支持
- 元宝: 界面美观

### Week 7-8: 标准库扩展

**负责人**: Std

```dalin
// std/net.dalin

/// 网络模块
fn http_get(url: string) -> Result<Response, Error> {
    let socket = tcp_connect(url.host, url.port)
    let request = format!("GET {} HTTP/1.1\r\nHost: {}\r\n\r\n", url.path, url.host)
    socket.send(request)
    let response = socket.receive()
    tcp_close(socket)
    ok(parse_response(response))
}

fn http_post(url: string, body: string) -> Result<Response, Error> {
    let socket = tcp_connect(url.host, url.port)
    let request = format!("POST {} HTTP/1.1\r\nHost: {}\r\nContent-Length: {}\r\n\r\n{}", 
        url.path, url.host, body.len(), body)
    socket.send(request)
    let response = socket.receive()
    tcp_close(socket)
    ok(parse_response(response))
}

// std/crypto.dalin

/// 加密模块
fn sha256(data: &[u8]) -> [u8; 32] {
    // SHA-256 哈希
    ...
}

fn md5(data: &[u8]) -> [u8; 16] {
    // MD5 哈希
    ...
}

fn aes_encrypt(key: &[u8], data: &[u8]) -> Result<[u8], Error> {
    // AES 加密
    ...
}

fn aes_decrypt(key: &[u8], data: &[u8]) -> Result<[u8], Error> {
    // AES 解密
    ...
}

// std/sqlite.dalin

/// SQLite 数据库模块
use c "sqlite3.h" as sqlite3

fn db_open(path: string) -> Result<Database, Error> {
    let mut handle: *sqlite3.Sqlite3 = null
    let result = sqlite3.sqlite3_open(path, &handle)
    if result != 0 {
        error("无法打开数据库: " + sqlite3.sqlite3_errmsg(handle))
    } else {
        ok(Database { handle })
    }
}

fn db_execute(db: &Database, sql: string) -> Result<Vec<Row>, Error> {
    let mut result: [*char] = null
    let mut rows: int = 0
    let rc = sqlite3.sqlite3_exec(db.handle, sql, &result, &rows, null)
    if rc != 0 {
        error("执行失败: " + sqlite3.sqlite3_errmsg(db.handle))
    } else {
        ok(parse_rows(result, rows))
    }
}
```

**交付物**:
- ✅ std/net — 网络模块
- ✅ std/crypto — 加密模块
- ✅ std/sqlite — 数据库模块
- ✅ std/http — HTTP 客户端

**审查标准**:
- Alpha: 安全性
- Beta: 易用性
- 混元: 性能

---

## Month 3: 文档站点 + 社区建设

### Week 9-10: 文档站点

**负责人**: Doc

```bash
# 生成文档站点
dalin docs --site --output docs.dalin-lang.org

# 自动部署到 GitHub Pages
dalin docs --deploy
```

**站点结构**:
```
docs.dalin-lang.org/
├── index.html          # 首页
├── getting-started.html  # 入门指南
├── language-reference.html  # 语言参考
├── stdlib/             # 标准库文档
│   ├── collections.html
│   ├── io.html
│   ├── string.html
│   └── ...
├── api/                # API 文档
│   ├── functions.html
│   ├── structs.html
│   └── traits.html
└── examples/           # 示例代码
    ├── hello-world.dalin
    ├── fibonacci.dalin
    └── ...
```

**交付物**:
- ✅ 文档站点
- ✅ 中文文档
- ✅ 示例代码
- ✅ API 参考

**审查标准**:
- 豆包: 中文文档完整
- 元宝: 站点美观
- Beta: 易于学习

### Week 11-12: 社区建设

**负责人**: 全员

```
社区渠道:
- GitHub: https://github.com/CN-QN1-dalin/dalin-l
- Discord: https://discord.gg/dalin-l
- 微信群: 扫码加入
- 论坛: forum.dalin-lang.org
- 博客: blog.dalin-lang.org
```

**活动**:
- ✅ 开源项目启动
- ✅ 中文文档发布
- ✅ 教程系列
- ✅ 示例项目
- ✅ 社区贡献指南

**交付物**:
- ✅ GitHub 仓库
- ✅ Discord 服务器
- ✅ 微信群
- ✅ 论坛
- ✅ 博客
- ✅ 贡献指南

**审查标准**:
- 豆包: 中文社区活跃
- Beta: 新人友好
- 元宝: 社区氛围好

---

## 资源需求

| 资源 | 数量 | 说明 |
|------|------|------|
| 团队 | 2 人 | CLI/IDE + Std/Doc |
| 时间 | 3 个月 | 12 周 |
| 预算 | TBD | 取决于薪资 |
| 服务器 | 1 | 文档站点 + 论坛 |

---

## 里程碑

| 日期 | 里程碑 | 状态 |
|------|--------|------|
| 2026-10 | Month 1: 包管理器 + 中文 | ⏳ 待开始 |
| 2026-11 | Month 2: VSCode 插件 + 标准库 | ⏳ 待开始 |
| 2026-12 | Month 3: 文档站点 + 社区 | ⏳ 待开始 |
| 2027-01 | Phase 3 完成 | ⏳ 待开始 |

---

**Phase 3 规划完成！等待 Phase 2 完成后启动！**

**执行。**
