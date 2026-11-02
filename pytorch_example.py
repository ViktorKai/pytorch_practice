"""PyTorch 入门：张量、自动求导和简单线性回归（CPU）。"""

import torch
from torch import nn


def main() -> None:
    torch.manual_seed(42)
    print("=== PyTorch 示例 ===")
    print(f"PyTorch 版本: {torch.__version__}")
    print("运行设备: CPU")

    # 张量可以看作支持自动求导的多维数组。
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.ones((2, 2))
    print(f"\n张量 a:\n{a}")
    print(f"a + b:\n{a + b}")
    print(f"a @ b（矩阵乘法）:\n{a @ b}")

    # y = x² 在 x = 3 处的导数为 2x = 6。
    value = torch.tensor(3.0, requires_grad=True)
    result = value**2
    result.backward()
    print(f"\n自动求导: x = {value.item():.1f}, y = x^2, dy/dx = {value.grad.item():.1f}")

    # 用少量样本学习 y = 2x + 1，不需要下载数据。
    x_train = torch.arange(5, dtype=torch.float32).reshape(-1, 1)
    y_train = 2 * x_train + 1
    model = nn.Linear(1, 1)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    print("\n训练线性回归，目标: y = 2x + 1")
    for epoch in range(201):
        prediction = model(x_train)
        loss = loss_function(prediction, y_train)
        optimizer.zero_grad()  # 清除上一次的梯度。
        loss.backward()  # 计算当前梯度。
        optimizer.step()  # 根据梯度更新参数。
        if epoch % 50 == 0:
            print(f"epoch {epoch:3d} | loss = {loss.item():.6f}")

    with torch.no_grad():
        predicted = model(torch.tensor([[5.0]])).item()
    print(f"学习结果: y = {model.weight.item():.4f}x + {model.bias.item():.4f}")
    print(f"输入 x = 5，预测 y = {predicted:.4f}（期望值: 11）")


if __name__ == "__main__":
    main()
