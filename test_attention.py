import torch
import torch.nn.functional as F
from attention import attention

torch.manual_seed(42)

q = torch.randn(2, 4, 8, 16, requires_grad=True)
k = torch.randn(2, 4, 8, 16, requires_grad=True)
v = torch.randn(2, 4, 8, 16, requires_grad=True)

# Test 1: 输出形状
out = attention(q, k, v, causal=True)

assert out.shape == (2, 4, 8, 16)
print("PASS: output shape")

# Test 2: 与 PyTorch 官方实现对比
ref = F.scaled_dot_product_attention(
    q, k, v,
    dropout_p=0.0,
    is_causal=True
)

assert torch.allclose(
    out, ref, atol=1e-5, rtol=1e-5
)
print("PASS: numerical correctness")

# Test 3: 未来 Token 不影响历史 Token
v_changed = v.detach().clone()
v_changed[:, :, -1, :] += 100

out_changed = attention(
    q, k, v_changed, causal=True
)

assert torch.allclose(
    out[:, :, :-1, :],
    out_changed[:, :, :-1, :],
    atol=1e-5,
    rtol=1e-5
)
print("PASS: causal masking")

# Test 4: 反向传播
out.sum().backward()

for tensor in [q, k, v]:
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()

print("PASS: autograd")
print("ALL TESTS PASSED")