#!/bin/bash

# Tiny Transformer - One-Click Script
# Data Generation -> Compile -> Training -> Inference

set -e

echo "========================================="
echo "=== 1. Generating Reversal Dataset ==="
echo "========================================="
python gen_reverse.py

echo -e "\n========================================="
echo "=== 2. Compiling C++ Source ==="
echo "========================================="
# Try compiling with ARMv8.2-a optimizations
if clang++ -O3 -march=armv8.2-a+dotprod -ffast-math -fno-math-errno \
  -funroll-loops -fomit-frame-pointer -fno-exceptions -fno-rtti \
  -o tiny tiny.cpp; then
    echo "Compilation successful (with -march)."
else
    echo "Compilation failed with -march. Retrying without it..."
    clang++ -O3 -ffast-math -funroll-loops -o tiny tiny.cpp
    echo "Compilation successful (without -march)."
fi

echo -e "\n========================================="
echo "=== 3. Training ==="
echo "========================================="
rm -f model.ckpt tiny.bin

./tiny --mode train --data data.txt --model tiny.bin \
  --D 128 --L 4 --nq 4 --nkv 2 --ctx 48 \
  --epochs 100 --batch 16 --lr 1e-3 \
  --val-split 0.05 --stop-loss 0.02 \
  --grad-clip 1.0 --warmup 50 \
  --ckpt model.ckpt --ckpt-interval 20

echo -e "\n========================================="
echo "=== 4. Testing ==="
echo "========================================="

echo "--- Test 1: Hello -> olleH ---"
./tiny --mode gen --model tiny.bin --prompt "<u>Hello<a>" --greedy

echo -e "\n--- Test 2: QwErTy -> yTrEwQ ---"
./tiny --mode gen --model tiny.bin --prompt "<u>QwErTy<a>" --greedy

echo -e "\n--- Test 3: ZzYyXxWw -> wWxXyYzZ ---"
./tiny --mode gen --model tiny.bin --prompt "<u>ZzYyXxWw<a>" --greedy

echo -e "\nAll done!"