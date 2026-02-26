#!/bin/bash

echo "正在启动金融应用Python编程教学范例..."

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "未找到虚拟环境，正在创建..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 检查是否需要安装依赖
if [ ! -d "venv/lib/python3.*/site-packages/flask" ]; then
    echo "正在安装项目依赖..."
    pip install -r requirements.txt
fi

echo "启动Flask应用..."
python app.py
