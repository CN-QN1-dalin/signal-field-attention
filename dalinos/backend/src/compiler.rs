// Dalin L Compiler Interface
// 作者: 混元 (后端专家)
// 日期: 2026-06-24
// 状态: MVP v1.0

/// 便捷函数：编译 Dalin L 代码
pub fn compile_dalin_l(source: &str, _version: &Option<String>) -> super::CompileResponse {
    let compiler = Compiler::new();
    let result = compiler.compile(source, CompileOptions::default());
    
    super::CompileResponse {
        success: result.success,
        output: result.output,
        errors: result.errors.iter().map(|e| e.message.clone()).collect(),
        warnings: result.warnings.clone(),
        suggestions: result.suggestions.clone(),
    }
}

use serde::{Deserialize, Serialize};
use std::process::Command;

// ==================== Response Types ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileResult {
    pub success: bool,
    pub output: String,
    pub errors: Vec<CompileError>,
    pub warnings: Vec<String>,
    pub suggestions: Vec<String>,
    pub stats: CompileStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileError {
    pub line: usize,
    pub column: usize,
    pub message: String,
    pub code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileStats {
    pub tokens_count: usize,
    pub ast_nodes: usize,
    pub compilation_time_ms: f64,
    pub output_size_bytes: usize,
}

// ==================== Compiler ====================

pub struct DalinLCompiler {
    compiler_path: String,
    version: String,
}

impl DalinLCompiler {
    /// Create new compiler instance
    pub fn new() -> Self {
        Self {
            compiler_path: std::env::var("DALINL_COMPILER_PATH")
                .unwrap_or_else(|_| "./dalinc".to_string()),
            version: "0.1.0".to_string(),
        }
    }

    /// Compile Dalin L source code
    pub fn compile(&self, source: &str, options: CompileOptions) -> CompileResult {
        let start_time = std::time::Instant::now();
        
        // Step 1: Lexical Analysis
        let lex_result = self.lex(source);
        if !lex_result.errors.is_empty() {
            return self.error_result(lex_result.errors, start_time);
        }
        
        // Step 2: Syntax Analysis
        let parse_result = self.parse(&lex_result.tokens);
        if !parse_result.errors.is_empty() {
            return self.error_result(parse_result.errors, start_time);
        }
        
        // Step 3: Semantic Analysis
        let semantic_result = self.analyze_semantics(&parse_result.ast);
        if !semantic_result.errors.is_empty() {
            return self.error_result(semantic_result.errors, start_time);
        }
        
        // Step 4: Code Generation
        let code_gen_result = self.generate_code(&parse_result.ast, &options);
        
        let compilation_time = start_time.elapsed().as_millis() as f64;
        
        CompileResult {
            success: code_gen_result.success,
            output: code_gen_result.output,
            errors: code_gen_result.errors,
            warnings: semantic_result.warnings,
            suggestions: self.generate_suggestions(&parse_result.ast),
            stats: CompileStats {
                tokens_count: lex_result.tokens.len(),
                ast_nodes: parse_result.ast.len(),
                compilation_time_ms: compilation_time,
                output_size_bytes: code_gen_result.output.len(),
            },
        }
    }

    /// Lexical Analysis - Tokenize source code
    fn lex(&self, source: &str) -> LexResult {
        let mut tokens = Vec::new();
        let mut errors = Vec::new();
        let mut line = 1;
        let mut column = 1;
        
        for ch in source.chars() {
            match ch {
                ' ' | '\t' => { column += 1; }
                '\n' => { line += 1; column = 1; }
                '=' => {
                    tokens.push(Token { kind: TokenType::Assign, value: "=".to_string(), line, column });
                    column += 1;
                }
                '+' => {
                    tokens.push(Token { kind: TokenType::Plus, value: "+".to_string(), line, column });
                    column += 1;
                }
                '-' => {
                    tokens.push(Token { kind: TokenType::Minus, value: "-".to_string(), line, column });
                    column += 1;
                }
                '*' => {
                    tokens.push(Token { kind: TokenType::Star, value: "*".to_string(), line, column });
                    column += 1;
                }
                '/' => {
                    tokens.push(Token { kind: TokenType::Slash, value: "/".to_string(), line, column });
                    column += 1;
                }
                '(' => {
                    tokens.push(Token { kind: TokenType::LParen, value: "(".to_string(), line, column });
                    column += 1;
                }
                ')' => {
                    tokens.push(Token { kind: TokenType::RParen, value: ")".to_string(), line, column });
                    column += 1;
                }
                ',' => {
                    tokens.push(Token { kind: TokenType::Comma, value: ",".to_string(), line, column });
                    column += 1;
                }
                ':' => {
                    tokens.push(Token { kind: TokenType::Colon, value: ":".to_string(), line, column });
                    column += 1;
                }
                ';' => {
                    tokens.push(Token { kind: TokenType::Semicolon, value: ";".to_string(), line, column });
                    column += 1;
                }
                '"' => {
                    // Parse string literal
                    let (string_val, end_col) = self.parse_string(source, column);
                    tokens.push(Token { kind: TokenType::String(string_val.clone()), value: string_val, line, column });
                    column = end_col;
                }
                _ if ch.is_alphabetic() || ch == '_' => {
                    // Parse identifier/keyword
                    let (ident, end_col) = self.parse_identifier(source, column);
                    let token_type = match ident.as_str() {
                        "让" => TokenType::Let,
                        "函数" => TokenType::Fn,
                        "类" => TokenType::Class,
                        "如果" => TokenType::If,
                        "否则" => TokenType::Else,
                        "当" => TokenType::While,
                        "返回" => TokenType::Return,
                        "真" => TokenType::True,
                        "假" => TokenType::False,
                        "空" => TokenType::Null,
                        _ => TokenType::Identifier(ident.clone()),
                    };
                    tokens.push(Token { kind: token_type, value: ident, line, column });
                    column = end_col;
                }
                _ if ch.is_ascii_digit() => {
                    // Parse number
                    let (num, end_col) = self.parse_number(source, column);
                    tokens.push(Token { kind: TokenType::Number(num.parse().unwrap_or(0.0)), value: num, line, column });
                    column = end_col;
                }
                _ => {
                    errors.push(CompileError {
                        line,
                        column,
                        message: format!("Unexpected character: {}", ch),
                        code: "LEX_001".to_string(),
                    });
                    column += 1;
                }
            }
        }
        
        LexResult { tokens, errors }
    }

    /// Parse string literal
    fn parse_string(&self, source: &str, start_col: usize) -> (String, usize) {
        let mut result = String::new();
        let mut col = start_col;
        
        for ch in source.chars().skip(col - 1) {
            if ch == '"' {
                col += 1;
                break;
            }
            result.push(ch);
            col += 1;
        }
        
        (result, col)
    }

    /// Parse identifier/keyword
    fn parse_identifier(&self, source: &str, start_col: usize) -> (String, usize) {
        let mut result = String::new();
        let mut col = start_col;
        
        for ch in source.chars().skip(col - 1) {
            if ch.is_alphanumeric() || ch == '_' {
                result.push(ch);
                col += 1;
            } else {
                break;
            }
        }
        
        (result, col)
    }

    /// Parse number
    fn parse_number(&self, source: &str, start_col: usize) -> (String, usize) {
        let mut result = String::new();
        let mut col = start_col;
        
        for ch in source.chars().skip(col - 1) {
            if ch.is_ascii_digit() || ch == '.' {
                result.push(ch);
                col += 1;
            } else {
                break;
            }
        }
        
        (result, col)
    }

    /// Syntax Analysis - Parse tokens into AST
    fn parse(&self, tokens: &[Token]) -> ParseResult {
        let mut ast = Vec::new();
        let mut errors = Vec::new();
        let mut pos = 0;
        
        // Simple recursive descent parser
        while pos < tokens.len() {
            match &tokens[pos].kind {
                TokenType::Let => {
                    pos += 1;
                    if pos >= tokens.len() {
                        errors.push(CompileError {
                            line: tokens[pos.min(tokens.len()-1)].line,
                            column: tokens[pos.min(tokens.len()-1)].column,
                            message: "Expected variable name after '让'".to_string(),
                            code: "PARSE_001".to_string(),
                        });
                        break;
                    }
                    let var_name = match &tokens[pos].kind {
                        TokenType::Identifier(name) => name.clone(),
                        _ => {
                            errors.push(CompileError {
                                line: tokens[pos].line,
                                column: tokens[pos].column,
                                message: "Expected variable name".to_string(),
                                code: "PARSE_002".to_string(),
                            });
                            pos += 1;
                            continue;
                        }
                    };
                    pos += 1;
                    
                    // Expect '='
                    if pos < tokens.len() && matches!(&tokens[pos].kind, TokenType::Assign) {
                        pos += 1;
                    }
                    
                    // Parse value
                    let value = if pos < tokens.len() {
                        match &tokens[pos].kind {
                            TokenType::Number(n) => format!("{}", n),
                            TokenType::String(s) => s.clone(),
                            TokenType::True => "true".to_string(),
                            TokenType::False => "false".to_string(),
                            TokenType::Null => "null".to_string(),
                            _ => "unknown".to_string(),
                        }
                    } else {
                        "unknown".to_string()
                    };
                    
                    ast.push(ASTNode::Let(var_name, value));
                }
                TokenType::Fn => {
                    // Parse function definition
                    pos += 1;
                    if pos < tokens.len() {
                        if let TokenType::Identifier(name) = &tokens[pos].kind {
                            ast.push(ASTNode::Function(name.clone(), Vec::new(), Vec::new()));
                            pos += 1;
                        }
                    }
                }
                _ => {
                    pos += 1;
                }
            }
        }
        
        ParseResult { ast, errors }
    }

    /// Semantic Analysis
    fn analyze_semantics(&self, ast: &[ASTNode]) -> SemanticResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Check for unused variables
        // Check for type mismatches
        // etc.
        
        SemanticResult { errors, warnings }
    }

    /// Code Generation
    fn generate_code(&self, ast: &[ASTNode], _options: &CompileOptions) -> CodeGenResult {
        let mut output = String::new();
        
        for node in ast {
            match node {
                ASTNode::Let(name, value) => {
                    output.push_str(&format!("let {} = {};\n", name, value));
                }
                ASTNode::Function(name, _params, _body) => {
                    output.push_str(&format!("fn {}() {{\n", name));
                    output.push_str("    // Function body\n");
                    output.push_str("}\n");
                }
            }
        }
        
        CodeGenResult {
            success: true,
            output,
            errors: Vec::new(),
        }
    }

    /// Generate suggestions based on AST
    fn generate_suggestions(&self, ast: &[ASTNode]) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        // Suggest optimizations
        if ast.len() > 100 {
            suggestions.push("考虑使用循环减少重复代码".to_string());
        }
        
        suggestions
    }

    /// Create error result
    fn error_result(&self, errors: Vec<CompileError>, start_time: std::time::Instant) -> CompileResult {
        CompileResult {
            success: false,
            output: String::new(),
            errors,
            warnings: Vec::new(),
            suggestions: Vec::new(),
            stats: CompileStats {
                tokens_count: 0,
                ast_nodes: 0,
                compilation_time_ms: start_time.elapsed().as_millis() as f64,
                output_size_bytes: 0,
            },
        }
    }
}

// ==================== Supporting Types ====================

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenType,
    pub value: String,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone)]
pub enum TokenType {
    Let,
    Fn,
    Class,
    If,
    Else,
    While,
    Return,
    True,
    False,
    Null,
    Assign,
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Comma,
    Colon,
    Semicolon,
    Identifier(String),
    Number(f64),
    String(String),
}

#[derive(Debug, Clone)]
pub enum ASTNode {
    Let(String, String),
    Function(String, Vec<String>, Vec<ASTNode>),
    If(Vec<ASTNode>, Vec<ASTNode>),
    While(Vec<ASTNode>, Vec<ASTNode>),
    Return(Option<Box<ASTNode>>),
}

#[derive(Debug, Clone)]
pub struct LexResult {
    pub tokens: Vec<Token>,
    pub errors: Vec<CompileError>,
}

#[derive(Debug, Clone)]
pub struct ParseResult {
    pub ast: Vec<ASTNode>,
    pub errors: Vec<CompileError>,
}

#[derive(Debug, Clone)]
pub struct SemanticResult {
    pub errors: Vec<CompileError>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CodeGenResult {
    pub success: bool,
    pub output: String,
    pub errors: Vec<CompileError>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompileOptions {
    pub optimize: bool,
    pub debug: bool,
    pub target: String,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            optimize: false,
            debug: true,
            target: "wasm".to_string(),
        }
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_compile() {
        let compiler = DalinLCompiler::new();
        let source = r#"
让 x = 1
让 y = 2
让 z = x + y
"#;
        let result = compiler.compile(source, CompileOptions::default());
        
        assert!(result.success);
        assert!(result.errors.is_empty());
        assert!(result.stats.tokens_count > 0);
    }
    
    #[test]
    fn test_syntax_error() {
        let compiler = DalinLCompiler::new();
        let source = r#"
让 x = 
"#;
        let result = compiler.compile(source, CompileOptions::default());
        
        assert!(!result.success);
        assert!(!result.errors.is_empty());
    }
}
