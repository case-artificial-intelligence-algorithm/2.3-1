#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from my_solution import solution

# 测试用例
def test_solution():
    # 正确答案
    correct_solution = [37, 89, 100, 103, 104]

    # 程序求解结果
    result = solution()

    assert (correct_solution == result).all()
    
