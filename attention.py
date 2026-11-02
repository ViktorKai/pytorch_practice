import torch
import math

def attention(q, k, v, causal=True):
    """
    q, k, v: [B, H, T, D]
    返回:    [B, H, T, D]
    Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
    要求:
    1. 计算 QK^T
    2. 除以 sqrt(D)
    3. causal=True 时添加因果掩码
    4. 使用 softmax
    5. 与 V 矩阵相乘
    """

    k = k.transpose(-1, -2)  # 转置最后两维
    attention_scores = torch.matmul(q, k) / math.sqrt(q.size(-1))  # QK^T / sqrt(D)
    
    if causal:
        T = q.shape[-2]
        mask = torch.tril(torch.ones((T, T), dtype=torch.bool))
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))  # 因果掩码
    weights = attention_scores.softmax(dim=-1)  # softmax
    attention_output = torch.matmul(weights, v)  # 与 V 相乘
    # TODO: 独立实现
    print("attention_scores", attention_scores.shape)
    print("attention_output", attention_output.shape)
    return attention_output