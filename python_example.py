"""Python 入门：变量、列表、循环和函数。"""


def calculate_average(numbers: list[float]) -> float:
    """计算非空列表的平均值。"""
    if not numbers:
        raise ValueError("列表不能为空")
    return sum(numbers) / len(numbers)


def main() -> None:
    print("=== Python 基础示例 ===")
    name = "Python"
    print(f"Hello, {name}!")

    numbers = [1, 2, 3, 4, 5]
    squares = [number**2 for number in numbers]
    print(f"原始列表: {numbers}")
    print(f"平方列表: {squares}")
    print(f"平均值: {calculate_average(numbers):.2f}")

    for number in numbers:
        kind = "偶数" if number % 2 == 0 else "奇数"
        print(f"{number} 是{kind}")


if __name__ == "__main__":
    main()
