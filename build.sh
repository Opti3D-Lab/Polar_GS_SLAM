#!/bin/bash

# 1. 定义颜色输出（可选，方便看进度）
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}开始编译 Cython 模块...${NC}"

# 2. 清理旧的编译产物
rm -rf build/
rm -f utils/*.c
rm -f utils/*.so

# 3. 执行编译
# --inplace 会根据 Extension 定义的名字(utils.xxx)将 .so 放在 utils 目录下
python setup.py build_ext --inplace

# 4. 检查编译是否成功并清理中间文件
if [ $? -eq 0 ]; then
    echo -e "${GREEN}编译成功，正在清理中间文件...${NC}"

    # 删除生成的 .c 源文件
    rm -f utils/normal_loss.c
    rm -f utils/slam_backend.c

    # 删除 build 临时文件夹
    rm -rf build/

    echo -e "${GREEN}完成！.so 文件已存放在 utils/ 目录下。${NC}"
    ls -lh utils/*.so
else
    echo "编译失败，请检查错误日志。"
    exit 1
fi