# SFA v7 集成测试套件

## 测试目标
1. 验证 SFA v7 与 llama.cpp 的正确集成
2. 测试多序列状态隔离
3. 验证正交性修复效果
4. 检查 Metal 内核编译

## 运行测试

```bash
# 运行基本集成测试
python3 test_sfa_integration.py

# 运行正交性测试
python3 test_sfa_correctness.py

# 运行 Metal 编译测试
xcrun -sdk macosx metal -c src/sfa/sfa_kernel.metal -o /tmp/sfa_kernel.metallib
```

## 预期结果

- [x] SFA 增强正确注入到 attention output
- [x] 多序列状态隔离工作正常
- [x] 正交性 cosine < 0.1
- [ ] Metal 内核编译成功
- [ ] PPL 改善验证

## 已知问题

1. Metal 内核需要 Xcode command-line tools
2. 完整 PPL 测试需要 GPU 支持
