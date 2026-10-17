# Tiny Transformer (纯 C++ 手机端原生实现)

## ⚠️ 严重声明
1. **停止维护**：本项目已废弃，不再处理 Issues 和 PR。
2. **放弃版权**：作者放弃所有权利，无 License，随便用、随便改、随便商用，不需要署名，作者不承担任何责任。
3. **无英文支持**：本项目最初的训练数据和上下文完全是中文。代码本身是字符级的，不挑语言，但不要指望它能开箱即用地进行英文对话。
4. **完全由 AI 完成**：本项目的底层架构设计、代码编写、Bug 调试以及这份文档，完全由 DeepSeek 生成。人类用户只负责发提示词、跑编译器、敲回车。

## 项目简介
这是一个用 **纯 C++ (单文件)** 手写的、零第三方依赖的极简 Transformer 模型。专为在 Android 手机 (Termux) 端侧进行 CPU 训练和推理设计。

完全手写了前向与反向传播，没有使用 PyTorch、TensorFlow 等任何深度学习框架。

### 架构特性
对标现代主流 LLM (如 LLaMA 3 / Qwen) 的骨架：
*   **Decoder-only** 因果语言模型
*   **字符级 (Char-level)** Tokenizer
*   **RoPE** (旋转位置编码)
*   **RMSNorm** (Pre-Norm)
*   **GQA** (分组查询注意力)
*   **QK-Norm** (稳定注意力分数)
*   **SwiGLU** (门控前馈网络)
*   **KV Cache** (推理加速)
*   **AdamW** + Warmup + Cosine 学习率调度

## 手机上的实验结果
在联发科天玑 6300 (8核 CPU) 上跑出的真实泛化能力：
*   **0~9 加法**：约 5.5 万参数，学会进位，训练集外全对。
*   **4 轮对话**：约 64 万参数，学会对话轮次状态切换，训练集外拼接全对。
*   **倒序规则**：约 73 万参数，4000 条数据，完美学会反转规则 (如输入 QwErTy 输出 yTrEwQ)，即使该组合在训练集里没见过。

## 快速开始

### 1. 环境准备 (Termux)
pkg update && pkg upgrade -y
pkg install clang make git python -y

### 2. 编译
clang++ -O3 -march=armv8.2-a+dotprod -ffast-math -fno-math-errno -funroll-loops -fomit-frame-pointer -fno-exceptions -fno-rtti -o tiny tiny.cpp

*(注：如果手机架构不支持 -march 参数，直接去掉它即可。)*

### 3. 运行演示
run.sh 脚本会自动生成倒序任务数据、编译、训练并测试。

bash run.sh

或者手动执行：
# 训练
./tiny --mode train --data data.txt --model tiny.bin --D 128 --L 4 --nq 4 --nkv 2 --ctx 48 --epochs 100 --batch 16 --lr 1e-3 --stop-loss 0.02

# 推理
./tiny --mode gen --model tiny.bin --prompt "<u>Hello<a>" --greedy

## 目录结构
*   tiny.cpp : 核心代码（包含分词、自动求导、模型、训练、推理）。
*   run.sh : 一键生成数据、训练、测试的脚本。
*   gen_reverse.py : 生成倒序训练数据的脚本。