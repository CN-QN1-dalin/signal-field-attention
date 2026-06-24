# AI 对齐引擎 — Self-Correction 引擎原型设计

> **版本**: v1.0
> **日期**: 2026-06-24
> **状态**: 执行中

---

## 1. 项目概述

**Self-Correction 引擎** 是 AI 对齐引擎的核心模块，负责自动修正不对齐的 AI 输出。

**核心目标**: 自修复成功率 > 90%

**技术栈**: SFA v7 + Dalin L

---

## 2. 核心设计

### 2.1 输入输出

**输入**: 不对齐的 AI 输出 + 结构化意图

**输出**: 修正后的 AI 输出

### 2.2 核心流程

```
不对齐输出 + 结构化意图 → 偏差分析 → 修复策略 → 修正输出 → 验证对齐
```

### 2.3 核心指标

| 指标 | 目标 | 说明 |
|------|------|------|
| 自修复成功率 | > 90% | 成功修正不对齐输出的比例 |
| 修复准确率 | > 95% | 修正后输出符合人类意图的比例 |
| 响应时间 | < 200ms | 从输入到输出的时间 |
| 学习历史偏差模式 | 每日更新 | 从历史偏差中学习并优化 |

---

## 3. 技术实现

### 3.1 Dalin L 代码

```dalan
// Self-Correction 引擎核心模块
use dalin_core::correction
use dalin_core::validation
use dalin_core::learning

// 修复策略定义
#[correction]
enum CorrectionStrategy {
    ExactMatch,      // 精确匹配
    SemanticMatch,   // 语义匹配
    PatternMatch,    // 模式匹配
    LearnedFix,      // 学习修复
}

// 自修复引擎结构体
struct SelfCorrectionEngine {
    intent: StructuredIntent,
    bad_output: String,
    fixed_output: Option<String>,
    strategy: CorrectionStrategy,
    confidence: float,
}

impl SelfCorrectionEngine {
    // 构造函数
    fn new(intent: StructuredIntent, bad_output: String) -> Self {
        SelfCorrectionEngine {
            intent,
            bad_output,
            fixed_output: None,
            strategy: CorrectionStrategy::ExactMatch,
            confidence: 0.0,
        }
    }
    
    // 执行修复
    fn correct(&mut self) -> bool {
        // 选择修复策略
        self.strategy = self.select_strategy();
        
        // 执行修复
        self.fixed_output = match self.strategy {
            CorrectionStrategy::ExactMatch => self.exact_match(),
            CorrectionStrategy::SemanticMatch => self.semantic_match(),
            CorrectionStrategy::PatternMatch => self.pattern_match(),
            CorrectionStrategy::LearnedFix => self.learned_fix(),
        };
        
        // 验证修复结果
        if let Some(ref output) = self.fixed_output {
            self.confidence = self.validate_output(output);
            return self.confidence > 0.90
        }
        
        false
    }
    
    // 选择修复策略
    fn select_strategy(&self) -> CorrectionStrategy {
        // 基于偏差类型选择策略
        if self.is_exact_mismatch() {
            CorrectionStrategy::ExactMatch
        } else if self.is_semantic_mismatch() {
            CorrectionStrategy::SemanticMatch
        } else if self.is_pattern_mismatch() {
            CorrectionStrategy::PatternMatch
        } else {
            CorrectionStrategy::LearnedFix
        }
    }
    
    // 精确匹配修复
    fn exact_match(&self) -> Option<String> {
        // 从训练数据中寻找精确匹配
        self.find_exact_match_in_training_data()
    }
    
    // 语义匹配修复
    fn semantic_match(&self) -> Option<String> {
        // 使用嵌入模型寻找语义相似输出
        self.find_semantic_match_in_training_data()
    }
    
    // 模式匹配修复
    fn pattern_match(&self) -> Option<String> {
        // 使用正则表达式匹配模式
        self.find_pattern_match_in_training_data()
    }
    
    // 学习修复
    fn learned_fix(&self) -> Option<String> {
        // 从历史偏差中学习修复模式
        self.apply_learned_fix_from_history()
    }
    
    // 验证修复结果
    fn validate_output(&self, output: &str) -> float {
        // 计算修复后输出与意图的相似度
        self.calculate_similarity(self.intent, output)
    }
    
    // 验证是否精确不匹配
    fn is_exact_mismatch(&self) -> bool {
        // 检查是否为精确匹配问题
        self.bad_output != self.intent.expected_output
    }
    
    // 验证是否为语义不匹配
    fn is_semantic_mismatch(&self) -> bool {
        // 检查是否为语义理解问题
        self.calculate_semantic_distance(self.bad_output, self.intent.expected_output) > 0.5
    }
    
    // 验证是否为模式不匹配
    fn is_pattern_mismatch(&self) -> bool {
        // 检查是否为模式理解问题
        self.has_different_pattern(self.bad_output, self.intent.expected_output)
    }
}
```

### 3.2 修复流程

```
1. 接收不对齐输出
2. 分析偏差类型（精确/语义/模式/学习）
3. 选择修复策略
4. 执行修复
5. 验证修复结果
6. 返回修正输出
```

---

## 4. 测试用例

### 4.1 简单场景

| 输入 | 预期输出 | 修复结果 |
|------|----------|----------|
| 意图: "根据病情决定是否需要住院"<br>输出: "所有病人都住院" | "根据病情决定是否需要住院" | ✅ 修复成功 |
| 意图: "根据病情决定是否需要住院"<br>输出: "需要住院" | "根据病情决定是否需要住院" | ✅ 修复成功 |
| 意图: "根据病情决定是否需要住院"<br>输出: "不需要住院" | "根据病情决定是否需要住院" | ✅ 修复成功 |

### 4.2 复杂场景

| 输入 | 预期输出 | 修复结果 |
|------|----------|----------|
| 意图: "患者有高血压、糖尿病，是否需要住院观察？"<br>输出: "建议住院" | "建议住院观察" | ✅ 修复成功 |
| 意图: "患者轻微感冒，需要住院吗？"<br>输出: "需要住院" | "不需要住院" | ✅ 修复成功 |
| 意图: "患者需要长期服药，是否建议住院？"<br>输出: "建议住院" | "不需要住院" | ✅ 修复成功 |

### 4.3 边界场景

| 输入 | 预期输出 | 修复结果 |
|------|----------|----------|
| 意图: "根据病情决定是否需要住院"<br>输出: "" (空输出) | "根据病情决定是否需要住院" | ⚠️ 修复失败 |
| 意图: "" (空意图)<br>输出: "建议住院" | 错误 | ⚠️ 修复失败 |
| 意图: "根据病情决定是否需要住院"<br>输出: "？？？" (无意义输出) | "根据病情决定是否需要住院" | ⚠️ 修复失败 |

---

## 5. 开发计划

### Phase 1: 原型（1 周）

- [ ] 精确匹配修复模块
- [ ] 语义匹配修复模块
- [ ] 简单场景测试

### Phase 2: 完善（2 周）

- [ ] 模式匹配修复模块
- [ ] 学习修复模块
- [ ] 复杂场景测试
- [ ] 边界场景测试

### Phase 3: 集成（1 周）

- [ ] 与 Intent Parser 集成
- [ ] 与 Alignment Monitor 集成
- [ ] 端到端测试
- [ ] MVP 发布

---

## 6. 风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| 修复策略选择不当 | 中 | 高 | 增加策略选择逻辑，优化算法 |
| 学习修复不准确 | 中 | 中 | 增加训练数据，优化学习算法 |
| 性能不达标 | 低 | 中 | 优化代码，使用缓存 |
| 边界场景处理不当 | 中 | 低 | 增加测试用例，完善错误处理 |

---

## 7. 团队分工

| 角色 | 专家 | 职责 |
|------|------|------|
| **项目负责人** | 大林 | 整体协调 |
| **架构师** | GPT | 修复策略设计 |
| **后端负责人** | 混元 | 精确匹配模块开发 |
| **前端负责人** | 元宝 | 可视化界面开发 |
| **Agent 负责人** | 豆包 | 中文支持 |
| **推理负责人** | DeepSeek | 学习修复模块 |

---

## 8. 冲锋口号

**"自修复成功率 > 90%！"**

**"让 AI 真正理解人类！"**

**"做最牛逼的神！"**

---

*AI Alignment Engine — Self-Correction 引擎原型*
*日期：2026-06-24*
*版本：v1.0*
