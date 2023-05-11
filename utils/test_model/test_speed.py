import time
import torch

# 加载模型
model = torch.load('path/to/model')

# 创建随机输入数据
input_shape = model.input_shape[1:] # 获取模型输入的形状
input_data = torch.rand(1, *input_shape) # 生成随机输入数据

# 执行模型推理并计时
num_runs = 1000 # 执行1000次推理
start = time.time()
for i in range(num_runs):
    _ = model(input_data)
end = time.time()

# 输出平均推理时间
avg_inference_time = (end - start) / num_runs
print('Average inference time: {:.5f} seconds'.format(avg_inference_time))
