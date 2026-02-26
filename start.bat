@echo off
echo 正在启动金融应用Python编程教学范例...

:: 检查虚拟环境是否存在
if not exist "venv\Scripts\python.exe" (
    echo 未找到虚拟环境，正在创建...
    python -m venv venv
)

:: 激活虚拟环境
call venv\Scripts\activate

:: 检查是否需要安装依赖
if not exist "venv\Lib\site-packages\flask\__init__.py" (
    echo 正在安装项目依赖...
    pip install -r requirements.txt
)

echo 启动Flask应用...
python app.py
