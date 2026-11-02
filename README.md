# Python 和 PyTorch 入门示例

- `python_example.py`：变量、列表、列表推导式、函数、循环和条件判断。
- `pytorch_example.py`：张量运算、自动求导，以及学习 `y = 2x + 1` 的线性回归。

已安装环境：Python 3.13.5、PyTorch 2.7.0+cpu。PyTorch 示例使用 CPU，无需 GPU 或下载数据。

## 在 VS Code 中运行

1. 打开此文件夹。
2. 打开“运行和调试”（Ctrl+Shift+D）。
3. 选择“Python 基础示例”或“PyTorch 入门示例”，按 F5。
4. 在“调试控制台”查看输出。配置使用现有的 `C:/ProgramData/miniconda3/python.exe`。

## 使用命令行运行

已通过 Windows 的 `assoc` / `ftype` 将 `.py` 文件关联到 Miniconda Python，并将 `.PY` 加入当前用户的 `PATHEXT`。
Windows“设置 → 应用 → 默认应用”中的 `.py` 也已设为 Python；系统的默认应用选择会覆盖传统文件关联。
在 VS Code 中新建一个 PowerShell 终端，即可直接运行：

```powershell
.\python_example.py
.\pytorch_example.py
```

文件关联固定使用 `C:/ProgramData/miniconda3/python.exe`。如需使用其他 Python 环境，仍可显式指定解释器：

```powershell
python python_example.py
python pytorch_example.py
```

训练结束后，模型的权重接近 2，偏置接近 1，输入 5 时预测值接近 11。
