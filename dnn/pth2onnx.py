import torch
import onnx
import onnxruntime
import pandas as pd
from network import MyDataset, MyModel

# 加载PyTorch模型
input_dim = 36
hidden_dim = 72
output_dim = 1
model_name = './model/regression_model.pth'
model = MyModel(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load(model_name))

# 加载数据集
data = pd.read_csv("./data/infer_file.csv")
inputs = data.iloc[:, 3:-1].values
inputs = inputs.astype("float32")

# 将PyTorch模型转换为ONNX格式
onnx_filename = "./model/regression_model.onnx"
dummy_input = torch.randn(inputs.shape[0], input_dim)
torch.onnx.export(model, dummy_input, onnx_filename)


# 创建 ONNX Runtime Session
ort_session = onnxruntime.InferenceSession(onnx_filename)

# 构造输入张量字典
input_name = ort_session.get_inputs()[0].name
ort_inputs = {input_name: inputs}

# 运行模型
ort_outs = ort_session.run(None, ort_inputs)
output = ort_outs[0]

# 打印输出
print("预测结果:")
print(output)