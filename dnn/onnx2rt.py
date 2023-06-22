'''
这段代码的作用是将ONNX模型转换为TensorRT引擎，并在引擎上进行推理，输出模型的预测结果。
具体步骤如下：
    1. 导入需要的Python模块，包括tensorrt、pycuda.autoinit和pandas等。
    2. 使用TensorRT的OnnxParser方法加载ONNX模型，并获取模型的输入和输出张量的名称以及张量。
    3. 创建TensorRT Builder、Engine和Context，设置相关参数。
    4. 加载给定的数据集，并将其传递给TensorRT引擎。
    5. 定义输入、输出和缓存区，并在GPU上执行推理以获取预测结果。
    6. 打印预测结果。
'''
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import pandas as pd


# 加载 ONNX 模型
onnx_file_path = "./model/regression_model.onnx"
onnx_parser = trt.OnnxParser(trt.Logger(trt.Logger.WARNING))
with open(onnx_file_path, "rb") as f:
    onnx_parser.parse(f.read())
onnx_inputs = [onnx_parser.get_input_name(i) for i in range(onnx_parser.num_inputs)]
onnx_outputs = [onnx_parser.get_output_name(i) for i in range(onnx_parser.num_outputs)]
onnx_tensors = [onnx_parser.get_output(i) for i in range(onnx_parser.num_outputs)]

# 创建 TensorRT Builder、Engine 和 Context
trt_logger = trt.Logger(trt.Logger.WARNING)
trt_builder = trt.Builder(trt_logger)
trt_network = trt_builder.create_network()
trt_parser = trt.OnnxParser(trt_network, trt_logger)
trt_parser.parse(onnx_tensors[0].raw_data)
trt_builder.max_workspace_size = 1 << 28
trt_builder.max_batch_size = 1
trt_builder.fp16_mode = True
trt_engine = trt_builder.build_cuda_engine(trt_network)
trt_context = trt_engine.create_execution_context()

# 加载数据集
data = pd.read_csv("./data/infer_file.csv")
inputs = data.iloc[:, 3:-1].values
inputs = inputs.astype("float32")

# 定义输入、输出和缓存区
h_input = cuda.pagelocked_empty(trt.volume(trt_engine.get_binding_shape(0)), dtype=np.float32)
h_output = cuda.pagelocked_empty(trt.volume(trt_engine.get_binding_shape(1)), dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
stream = cuda.Stream()

# 输入数据传输
cuda.memcpy_htod_async(d_input, inputs.reshape(-1), stream)

# 运行 TensorRT 推理
trt_context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
cuda.memcpy_dtoh_async(h_output, d_output, stream)

# 同步 stream 和 CPU
stream.synchronize()

# 打印输出结果
output_data = h_output.reshape(trt_engine.get_binding_shape(1))
print("预测结果:")
print(output_data)