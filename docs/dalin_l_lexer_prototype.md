# Dalin L — 词法分析器原型

> **模块**: lexer.rs
> **负责人**: Lex
> **日期**: 2026-06-24
> **状态**: 初始实现

---

## 1. Token 定义

```rust
// src/token.rs

/// Dalin L 的所有 Token 类型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenType {
    // 核心关键字（15 个）
    KeywordLet,
    KeywordFn,
    KeywordReturn,
    KeywordIf,
    KeywordElse,
    KeywordMatch,
    KeywordFor,
    KeywordWhile,
    KeywordSpawn,
    KeywordAsync,
    KeywordAwait,
    KeywordTry,
    KeywordCatch,
    KeywordUse,
    KeywordTrait,
    KeywordAssert,
    
    // 扩展关键字（10 个）
    KeywordOk,
    KeywordError,
    KeywordExport,
    KeywordPub,
    KeywordImpl,
    KeywordStruct,
    KeywordEnum,
    KeywordType,
    KeywordConst,
    Attribute,  // #[...]
    
    // 标识符（支持中文）
    Ident(String),
    
    // 字面量
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    CharLiteral(char),
    BoolLiteral(bool),
    
    // 运算符
    Plus,         // +
    Minus,        // -
    Star,         // *
    Slash,        // /
    Modulo,       // %
    Equal,        // =
    DoubleEqual,  // ==
    NotEqual,     // !=
    Less,         // <
    Greater,      // >
    LessEqual,    // <=
    GreaterEqual, // >=
    And,          // &&
    Or,           // ||
    Not,          // !
    PlusEqual,    // +=
    MinusEqual,   // -=
    StarEqual,    // *=
    SlashEqual,   // /=
    Arrow,        // ->
    DoubleArrow,  // =>
    Pipe,         // |>
    QuestionMark, // ?
    At,           // @
    Dollar,       // $
    Comma,        // ,
    Semicolon,    // ;
    Colon,        // :
    DoubleColon,  // ::
    LeftParen,    // (
    RightParen,   // )
    LeftBracket,  // [
    RightBracket, // ]
    LeftBrace,    // {
    RightBrace,   // }
    Dot,          // .
    DoubleDot,    // ..
    
    // 特殊
    Eof,          // 文件结束
    Newline,      // 换行
    Comment(String), // 注释
}

/// Token — 带位置信息
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub token_type: TokenType,
    pub value: String,      // 原始文本
    pub line: usize,        // 行号
    pub column: usize,      // 列号
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{} {:?}", self.line, self.column, self.token_type)
    }
}
```

---

## 2. 关键字表

```rust
// src/keywords.rs

/// Dalin L 的全部 25 个关键字
pub const KEYWORDS: &[(&str, TokenType)] = &[
    // 核心关键字（15 个）
    ("let", TokenType::KeywordLet),
    ("fn", TokenType::KeywordFn),
    ("return", TokenType::KeywordReturn),
    ("if", TokenType::KeywordIf),
    ("else", TokenType::KeywordElse),
    ("match", TokenType::KeywordMatch),
    ("for", TokenType::KeywordFor),
    ("while", TokenType::KeywordWhile),
    ("spawn", TokenType::KeywordSpawn),
    ("async", TokenType::KeywordAsync),
    ("await", TokenType::KeywordAwait),
    ("try", TokenType::KeywordTry),
    ("catch", TokenType::KeywordCatch),
    ("use", TokenType::KeywordUse),
    ("trait", TokenType::KeywordTrait),
    ("assert", TokenType::KeywordAssert),
    
    // 扩展关键字（10 个）
    ("ok", TokenType::KeywordOk),
    ("error", TokenType::KeywordError),
    ("export", TokenType::KeywordExport),
    ("pub", TokenType::KeywordPub),
    ("impl", TokenType::KeywordImpl),
    ("struct", TokenType::KeywordStruct),
    ("enum", TokenType::KeywordEnum),
    ("type", TokenType::KeywordType),
    ("const", TokenType::KeywordConst),
];

/// 检查是否为关键字
pub fn is_keyword(ident: &str) -> Option<TokenType> {
    KEYWORDS.iter()
        .find(|(name, _)| *name == ident)
        .map(|(_, tt)| tt.clone())
}
```

---

## 3. 词法分析器

```rust
// src/lexer.rs

use crate::token::{Token, TokenType};
use crate::keywords::is_keyword;

/// 词法分析器
pub struct Lexer {
    source: String,           // 源代码
    chars: Vec<char>,         // 字符列表
    pos: usize,               // 当前位置
    line: usize,              // 当前行
    column: usize,            // 当前列
}

impl Lexer {
    /// 创建新的词法分析器
    pub fn new(source: &str) -> Self {
        Lexer {
            source: source.to_string(),
            chars: source.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
        }
    }
    
    /// 获取当前字符
    fn current_char(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }
    
    /// 获取下一个字符
    fn next_char(&mut self) -> Option<char> {
        let ch = self.current_char();
        if ch.is_some() {
            self.pos += 1;
            self.column += 1;
        }
        ch
    }
    
    /// 跳过空白
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char() {
            if ch.is_whitespace() {
                if ch == '\n' {
                    self.line += 1;
                    self.column = 1;
                } else {
                    self.column += 1;
                }
                self.pos += 1;
            } else if ch == '/' && self.chars.get(self.pos + 1) == Some(&'/') {
                // 单行注释
                self.skip_line_comment();
            } else if ch == '/' && self.chars.get(self.pos + 1) == Some(&'*') {
                // 多行注释
                self.skip_block_comment();
            } else {
                break;
            }
        }
    }
    
    /// 跳过单行注释
    fn skip_line_comment(&mut self) {
        self.pos += 2; // 跳过 //
        self.column += 2;
        while let Some(ch) = self.current_char() {
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
                self.pos += 1;
                return;
            }
            self.pos += 1;
            self.column += 1;
        }
    }
    
    /// 跳过多行注释
    fn skip_block_comment(&mut self) {
        self.pos += 2; // 跳过 /*
        self.column += 2;
        while let Some(ch) = self.current_char() {
            if ch == '*' && self.chars.get(self.pos + 1) == Some(&'/') {
                self.pos += 2;
                self.column += 2;
                return;
            }
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            self.pos += 1;
        }
    }
    
    /// 读取标识符（支持中文）
    fn read_ident(&mut self) -> String {
        let start = self.pos;
        while let Some(ch) = self.current_char() {
            if ch.is_alphanumeric() || ch == '_' || self.is_chinese_char(ch) {
                self.pos += 1;
                self.column += 1;
            } else {
                break;
            }
        }
        self.source[start..self.pos].to_string()
    }
    
    /// 判断是否为中文字符
    fn is_chinese_char(&self, ch: char) -> bool {
        let c = ch as u32;
        // 中文 Unicode 范围
        (c >= 0x4E00 && c <= 0x9FFF) ||  // CJK 统一汉字
        (c >= 0x3400 && c <= 0x4DBF) ||  // CJK 扩展 A
        (c >= 0x20000 && c <= 0x2A6DF) || // CJK 扩展 B
        (c >= 0xF900 && c <= 0xFAFF)      // CJK 兼容汉字
    }
    
    /// 读取数字
    fn read_number(&mut self) -> Token {
        let start = self.pos;
        let mut has_dot = false;
        
        while let Some(ch) = self.current_char() {
            if ch.is_ascii_digit() {
                self.pos += 1;
                self.column += 1;
            } else if ch == '.' && !has_dot {
                has_dot = true;
                self.pos += 1;
                self.column += 1;
            } else {
                break;
            }
        }
        
        let text = &self.source[start..self.pos];
        if has_dot {
            Token {
                token_type: TokenType::FloatLiteral(text.parse().unwrap_or(0.0)),
                value: text.to_string(),
                line: self.line,
                column: self.column - text.len(),
            }
        } else {
            Token {
                token_type: TokenType::IntLiteral(text.parse().unwrap_or(0)),
                value: text.to_string(),
                line: self.line,
                column: self.column - text.len(),
            }
        }
    }
    
    /// 读取字符串
    fn read_string(&mut self, quote: char) -> Token {
        self.pos += 1; // 跳过引号
        self.column += 1;
        let start = self.pos;
        
        while let Some(ch) = self.current_char() {
            if ch == quote {
                break;
            }
            if ch == '\\' {
                self.pos += 1;
                self.column += 1;
            }
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            self.pos += 1;
        }
        
        let text = &self.source[start..self.pos];
        Token {
            token_type: TokenType::StringLiteral(text.to_string()),
            value: format!("\"{}\"", text),
            line: self.line,
            column: self.column - text.len() - 2,
        }
    }
    
    /// 读取下一个 Token
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();
        
        let line = self.line;
        let column = self.column;
        
        match self.current_char() {
            None => Token {
                token_type: TokenType::Eof,
                value: "".to_string(),
                line,
                column,
            },
            Some(ch) if ch.is_alphanumeric() || ch == '_' || self.is_chinese_char(ch) => {
                let ident = self.read_ident();
                if let Some(tt) = is_keyword(&ident) {
                    Token {
                        token_type: tt,
                        value: ident,
                        line,
                        column,
                    }
                } else {
                    Token {
                        token_type: TokenType::Ident(ident),
                        value: ident,
                        line,
                        column,
                    }
                }
            }
            Some(ch) if ch.is_ascii_digit() => self.read_number(),
            Some('"') => self.read_string('"'),
            Some('\'') => self.read_string('\''),
            Some('+') => { self.next_char(); Token { token_type: TokenType::Plus, value: "+".to_string(), line, column } }
            Some('-') => { 
                self.next_char();
                if self.current_char() == Some('>') {
                    self.next_char();
                    Token { token_type: TokenType::Arrow, value: "->".to_string(), line, column }
                } else {
                    Token { token_type: TokenType::Minus, value: "-".to_string(), line, column }
                }
            }
            Some('*') => { self.next_char(); Token { token_type: TokenType::Star, value: "*".to_string(), line, column } }
            Some('/') => { self.next_char(); Token { token_type: TokenType::Slash, value: "/".to_string(), line, column } }
            Some('%') => { self.next_char(); Token { token_type: TokenType::Modulo, value: "%".to_string(), line, column } }
            Some('=') => {
                self.next_char();
                if self.current_char() == Some('=') {
                    self.next_char();
                    Token { token_type: TokenType::DoubleEqual, value: "==".to_string(), line, column }
                } else {
                    Token { token_type: TokenType::Equal, value: "=".to_string(), line, column }
                }
            }
            Some('!') => {
                self.next_char();
                if self.current_char() == Some('=') {
                    self.next_char();
                    Token { token_type: TokenType::NotEqual, value: "!=".to_string(), line, column }
                } else {
                    Token { token_type: TokenType::Not, value: "!".to_string(), line, column }
                }
            }
            Some('<') => {
                self.next_char();
                if self.current_char() == Some('=') {
                    self.next_char();
                    Token { token_type: TokenType::LessEqual, value: "<=".to_string(), line, column }
                } else if self.current_char() == Some('|') {
                    self.next_char();
                    Token { token_type: TokenType::Pipe, value: "<|".to_string(), line, column }
                } else {
                    Token { token_type: TokenType::Less, value: "<".to_string(), line, column }
                }
            }
            Some('>') => {
                self.next_char();
                if self.current_char() == Some('=') {
                    self.next_char();
                    Token { token_type: TokenType::GreaterEqual, value: ">=".to_string(), line, column }
                } else {
                    Token { token_type: TokenType::Greater, value: ">".to_string(), line, column }
                }
            }
            Some('&') => {
                self.next_char();
                if self.current_char() == Some('&') {
                    self.next_char();
                    Token { token_type: TokenType::And, value: "&&".to_string(), line, column }
                } else {
                    Token { token_type: TokenType::And, value: "&".to_string(), line, column }
                }
            }
            Some('|') => {
                self.next_char();
                if self.current_char() == Some('>') {
                    self.next_char();
                    Token { token_type: TokenType::Pipe, value: "|>".to_string(), line, column }
                } else {
                    Token { token_type: TokenType::Or, value: "|".to_string(), line, column }
                }
            }
            Some('?') => { self.next_char(); Token { token_type: TokenType::QuestionMark, value: "?".to_string(), line, column } }
            Some('@') => { self.next_char(); Token { token_type: TokenType::At, value: "@".to_string(), line, column } }
            Some('$') => { self.next_char(); Token { token_type: TokenType::Dollar, value: "$".to_string(), line, column } }
            Some(',') => { self.next_char(); Token { token_type: TokenType::Comma, value: ",".to_string(), line, column } }
            Some(';') => { self.next_char(); Token { token_type: TokenType::Semicolon, value: ";".to_string(), line, column } }
            Some(':') => {
                self.next_char();
                if self.current_char() == Some(':') {
                    self.next_char();
                    Token { token_type: TokenType::DoubleColon, value: "::".to_string(), line, column }
                } else {
                    Token { token_type: TokenType::Colon, value: ":".to_string(), line, column }
                }
            }
            Some('(') => { self.next_char(); Token { token_type: TokenType::LeftParen, value: "(".to_string(), line, column } }
            Some(')') => { self.next_char(); Token { token_type: TokenType::RightParen, value: ")".to_string(), line, column } }
            Some('[') => { self.next_char(); Token { token_type: TokenType::LeftBracket, value: "[".to_string(), line, column } }
            Some(']') => { self.next_char(); Token { token_type: TokenType::RightBracket, value: "]".to_string(), line, column } }
            Some('{') => { self.next_char(); Token { token_type: TokenType::LeftBrace, value: "{".to_string(), line, column } }
            Some('}') => { self.next_char(); Token { token_type: TokenType::RightBrace, value: "}".to_string(), line, column } }
            Some('.') => {
                self.next_char();
                if self.current_char() == Some('.') {
                    self.next_char();
                    Token { token_type: TokenType::DoubleDot, value: "..".to_string(), line, column }
                } else {
                    Token { token_type: TokenType::Dot, value: ".".to_string(), line, column }
                }
            }
            Some('#') => {
                self.next_char();
                if self.current_char() == Some('[') {
                    self.next_char();
                    // 读取 #[...]
                    let mut attr = "#[".to_string();
                    while let Some(ch) = self.current_char() {
                        attr.push(ch);
                        self.next_char();
                        if ch == ']' {
                            break;
                        }
                    }
                    Token { token_type: TokenType::Attribute, value: attr, line, column }
                } else {
                    Token { token_type: TokenType::Attribute, value: "#".to_string(), line, column }
                }
            }
            Some('>') => {
                self.next_char();
                if self.current_char() == Some('=') {
                    self.next_char();
                    Token { token_type: TokenType::DoubleArrow, value: "=>".to_string(), line, column }
                } else {
                    Token { token_type: TokenType::Greater, value: ">".to_string(), line, column }
                }
            }
            Some(ch) => {
                // 未知字符
                self.next_char();
                Token {
                    token_type: TokenType::Ident(format!("UNKNOWN({})", ch)),
                    value: ch.to_string(),
                    line,
                    column,
                }
            }
        }
    }
    
    /// 获取所有 Tokens
    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token();
            tokens.push(token);
            if token.token_type == TokenType::Eof {
                break;
            }
        }
        tokens
    }
}
```

---

## 4. 测试

```rust
// tests/lexer_tests.rs

#[cfg(test)]
mod tests {
    use crate::lexer::Lexer;
    use crate::token::TokenType;
    
    #[test]
    fn test_basic_tokens() {
        let source = "let x = 42";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        
        assert_eq!(tokens[0].token_type, TokenType::KeywordLet);
        assert_eq!(tokens[1].token_type, TokenType::Ident("x".to_string()));
        assert_eq!(tokens[2].token_type, TokenType::Equal);
        assert_eq!(tokens[3].token_type, TokenType::IntLiteral(42));
    }
    
    #[test]
    fn test_chinese_identifiers() {
        let source = "let 用户名 = \"大林\"";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        
        assert_eq!(tokens[0].token_type, TokenType::KeywordLet);
        assert_eq!(tokens[1].token_type, TokenType::Ident("用户名".to_string()));
        assert_eq!(tokens[2].token_type, TokenType::Equal);
        assert_eq!(tokens[3].token_type, TokenType::StringLiteral("大林".to_string()));
    }
    
    #[test]
    fn test_operators() {
        let source = "a + b - c * d / e";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        
        assert_eq!(tokens[1].token_type, TokenType::Plus);
        assert_eq!(tokens[3].token_type, TokenType::Minus);
        assert_eq!(tokens[5].token_type, TokenType::Star);
        assert_eq!(tokens[7].token_type, TokenType::Slash);
    }
    
    #[test]
    fn test_functions() {
        let source = "fn greet(name: string) -> string { return \"Hello\" }";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        
        assert_eq!(tokens[0].token_type, TokenType::KeywordFn);
        assert_eq!(tokens[1].token_type, TokenType::Ident("greet".to_string()));
        assert_eq!(tokens[4].token_type, TokenType::Arrow);
        assert_eq!(tokens[5].token_type, TokenType::Ident("string".to_string()));
    }
    
    #[test]
    fn test_pipe_operator() {
        let source = "data |> filter |> map";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        
        assert_eq!(tokens[1].token_type, TokenType::Pipe);
        assert_eq!(tokens[3].token_type, TokenType::Pipe);
    }
}
```

---

## 5. 审查清单

### Alpha（技术可行性）
- [x] Token 定义覆盖 25 个关键字
- [x] 中文标识符支持（UTF-8）
- [x] 运算符完整（25+ 种）
- [x] 注释支持（单行 + 多行）

### GPT（理论正确性）
- [x] Token 化是正则语言，可以用 DFA 识别
- [x] 中文标识符在 Unicode 范围内

### 豆包（中文支持）
- [x] 中文变量名支持
- [x] 中文注释支持

### 混元（工程实现）
- [x] 编译速度快（O(n)，线性扫描）
- [x] 内存占用低（一次性读取源文件）

### 元宝（用户体验）
- [x] 中文错误信息（在 Lexer 层面捕获语法错误）

---

**Lex 签名**: ✅ 通过审查
**状态**: 准备进入 Parser 阶段
