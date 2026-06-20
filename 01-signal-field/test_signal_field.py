#!/usr/bin/env python3
"""
Signal Field Experiment1: 信号场 v5d — 单元测试
"""

import unittest
from signal_field import (
    SignalFieldAttention, PredictiveCodingLayer,
    ConceptSignalGenerator, StandardAttention
)


class TestSignalFieldAttention(unittest.TestCase):
    def test_forward(self):
        sf = SignalFieldAttention(num_layers=4, dim=16)
        sig_gen = ConceptSignalGenerator(16)
        signals = sig_gen.batch_encode(["a", "b", "c"])
        result = sf.forward(signals)
        self.assertIn("anchor_layers", result)
        self.assertIn("total_compression", result)
        self.assertGreater(len(result["anchor_layers"]), 0)

    def test_compression_ratio(self):
        sf = SignalFieldAttention(num_layers=8, dim=32)
        sig_gen = ConceptSignalGenerator(32)
        signals = sig_gen.batch_encode(["test"] * 20)
        result = sf.forward(signals)
        # 重复信号应该高压缩
        self.assertGreater(result["total_compression"], 0.5)

    def test_anchor_layers(self):
        sf = SignalFieldAttention(num_layers=12, dim=16)
        sig_gen = ConceptSignalGenerator(16)
        signals = sig_gen.batch_encode(["x"] * 50)
        sf.forward(signals)
        # 锚点层应该在合理范围内
        for layer in sf.anchor_points:
            self.assertLess(layer, 12)


class TestConceptSignalGenerator(unittest.TestCase):
    def test_deterministic(self):
        gen = ConceptSignalGenerator(32)
        s1 = gen.encode("test")
        s2 = gen.encode("test")
        self.assertEqual(s1, s2)

    def test_different_concepts(self):
        gen = ConceptSignalGenerator(32)
        s1 = gen.encode("hello")
        s2 = gen.encode("world")
        # 不同概念应该有不同信号
        self.assertNotEqual(s1, s2)


if __name__ == "__main__":
    unittest.main()
