#!/bin/bash
set -e

echo "Compiling hello.cu..."
nvcc -o hello hello.cu

echo "Running..."
./hello
