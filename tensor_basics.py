import torch
import math
#正索引从0开始，负索引从-1开始
#shape = [B, H, T, D] 
x = torch.randn(2, 8, 512)  # 随机生成一个形状为 (2, 8, 512) 的张量
y = torch.randn(2, 512, 8)
# shape、reshape、transpose、permute、matmul、softmax、contiguous、view


print("张量 x 的形状:", x.shape)
print("reshape 后的形状:", x.reshape(-1 ,2, 256).shape) # 不赋值就只是返回一个新的张量视图，原来的张量不变
print("张量 x 的形状:", x.shape)
print("transpose 后的形状:", x.transpose(2 ,1).shape) #交换两维
print("permute 后的形状:", x.permute(0, 2, 1).shape) #重排所有维度
print("矩阵乘法结果的形状:", torch.matmul(x, y).shape) #矩阵乘法  最后两维做矩阵乘，前面是 batch 维度，自动广播
print("矩阵乘法结果的形状:", torch.matmul(torch.randn(2, 2, 512, 8), torch.randn(2, 2, 8, 512)).shape)
# *：哈达玛积，同 shape 对应位置相乘
# @ / matmul：线性代数矩阵乘法，要满足 k 对齐
# torch.bmm 混淆：bmm 仅限3 维 batch 矩阵乘，不支持广播，matmul 支持广播，日常优先 matmul
print("softmax:",torch.softmax(x, dim=-1).shape) # dim=1表示对第 1 维做 softmax，dim=-1 表示对最后一维做 softmax
z = x.transpose(0, 1)
print("z", z.shape)
print("z", z.contiguous().view(-1, 512).shape) # view 只能在 contiguous 的张量上使用
print(x.contiguous().shape)
#print(x.view())
print("张量 x 的形状:", x.shape)
print(torch.__version__); print(torch.cuda.is_available())