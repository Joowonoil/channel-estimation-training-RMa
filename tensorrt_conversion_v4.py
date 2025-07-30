import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import tensorrt as trt
from logzero import logger
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import onnx
import onnxruntime as ort
from model.estimator_v4 import Estimator_v4 # v4 모델 임포트
from peft import PeftModel # PeftModel 임포트

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def save_simple_test_model(file_name):
    model = SimpleModel().eval()
    path = Path(__file__).parents[0].resolve() / 'saved_model'
    torch.save(model, path / (file_name + '.pt'))


def load_model(file_name):
    path = Path(__file__).parents[0].resolve() / 'saved_model'
    # Load the pretrained model object
    model = torch.load(path / (file_name + '.pt'))
    return model


def export_onnx_model(file_name, input_shape, batch_size):
    # v4 모델 로딩 및 초기화
    conf_file = 'config_transfer_v4.yaml' # v4 설정 파일 사용
    model = Estimator_v4(conf_file).cuda().eval()

    # 사전 학습된 모델의 상태 사전 로드
    pretrained_model_path = Path(__file__).parents[0].resolve() / 'saved_model' / (file_name + '.pt')
    if not pretrained_model_path.exists():
         raise FileNotFoundError(f"Pretrained model file not found at {pretrained_model_path}")

    # Load the pretrained model's state dictionary
    pretrained_model_state_dict = torch.load(pretrained_model_path)

    # Load the state dict into the newly created model instance
    # Use strict=False in case the saved state_dict doesn't exactly match the current model structure
    try:
        model.load_state_dict(pretrained_model_state_dict, strict=True)
        print(f"Pretrained model state dict loaded successfully from {pretrained_model_path}")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}. Attempting non-strict loading.")
        model.load_state_dict(pretrained_model_state_dict, strict=False)
        print(f"Pretrained model state dict loaded non-strictly from {pretrained_model_path}")

    # LoRA 가중치 병합
    if isinstance(model.ch_tf, PeftModel):
        model.ch_tf = model.ch_tf.merge_and_unload()
        print("LoRA weights merged and unloaded.")

    inp = torch.randn(batch_size, *input_shape, dtype=torch.float32).cuda()
    path = Path(__file__).parents[0].resolve() / 'tensorrt_model' / (file_name + '.onnx')
    torch.onnx.export(
        model,
        inp,
        f=str(path),
        export_params=True,
        opset_version=18,
        input_names=['input'],
        # output_names=['ch_est', 'pn_est', 'rx_sig_comp'],
        output_names=['ch_est', 'rx_sig_comp'],
        # dynamic_axes={'input': {0: 'batch_size'}, 'ch_est': {0: 'batch_size'},
        #               'pn_est': {0: 'batch_size'}, 'rx_sig_comp': {0: 'batch_size'}}
        dynamic_axes={'input': {0: 'batch_size'}, 'ch_est': {0: 'batch_size'}, 'rx_sig_comp': {0: 'batch_size'}}
    )
    logger.info("Export ONNX model successfully")


def build_and_save_trt_engine(file_name, input_shape, batch_size, fp16=False):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 50)  # Set the maximum workspace size for the builder
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        config.default_device_type = trt.DeviceType.GPU  # Set the default device type to GPU
        if fp16:
            config.flags |= 1 << int(trt.BuilderFlag.FP16)  # Set the optimization precision to float16
        else:
            config.flags &= 0 << int(trt.BuilderFlag.FP16)

        # Load ONNX model and parsing
        onnx_path = Path(__file__).parents[0].resolve() / 'tensorrt_model' / (file_name + '.onnx')
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                logger.info('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    logger.info(parser.get_error(error))
                return None
        logger.info("ONNX parse ended")

        # Build TRT engine
        profile = builder.create_optimization_profile()
        network.add_input(name="input", dtype=trt.float32, shape=(-1, *input_shape))
        profile.set_shape(input="input", min=(1, *input_shape), opt=(batch_size, *input_shape),
                          max=(batch_size, *input_shape))
        config.add_optimization_profile(profile)
        logger.debug(f"config = {config}")
        logger.info("====================== building tensorrt engine... ====================")
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            logger.info('Tensorrt engine build failed.')
        else:
            logger.info('Tensorrt engine built successfully.')
            engine_path = Path(__file__).parents[0].resolve() / 'tensorrt_model' / (file_name + '.engine')
            with open(engine_path, 'wb') as f:
                f.write(bytearray(engine))


def convert_pytorch_model_to_trt_engine(file_name, input_shape, batch_size, fp16=False):
    export_onnx_model(file_name=file_name, input_shape=input_shape, batch_size=batch_size)
    build_and_save_trt_engine(file_name=file_name, input_shape=input_shape, batch_size=batch_size, fp16=fp16)


class HostDeviceMem():
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem


def trt_inference(file_name, input_tensor):
    with trt.Logger(trt.Logger.VERBOSE) as logger, trt.Runtime(logger) as runtime:
        engine_path = Path(__file__).parents[0].resolve() / 'tensorrt_model' / (file_name + '.engine')
        with open(engine_path, mode='rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()
    inputs = []
    outputs = []
    bindings = []
    output_shape = []
    batch_size = input_tensor.shape[0]
    stream = cuda.Stream()
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_shape = engine.get_tensor_shape(tensor_name)[1:]
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(tensor_name, input_tensor.shape)
        size = trt.volume((batch_size, *tensor_shape))
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
            output_shape.append((batch_size, *tensor_shape))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype) # page-locked memory buffer
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer address to device bindings.
        # When cast to int, it's a linear index into the context's memory (like memory address).
        bindings.append(int(device_mem))

       # Append to the appropriate input/output list.
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    np.copyto(inputs[0].host, input_tensor.ravel())
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    for i in range(engine.num_io_tensors):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
    context.execute_async_v3(stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    output_data = [np.reshape(data.host.copy(), shape) for data, shape in zip(outputs, output_shape)]
    return output_data


def onnx_inference(file_name, input_tensor):
    onnx_path = Path(__file__).parents[0].resolve() / 'tensorrt_model' / (file_name + '.onnx')
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    ort_sess = ort.InferenceSession(onnx_path)
    output = ort_sess.run(output_names=None, input_feed={'input': input_tensor})
    return output


def torch_inference(file_name, input_tensor):
    input_tensor = torch.tensor(input_tensor).cuda()
    # v4 모델 로딩 및 초기화
    conf_file = 'config_transfer_v4.yaml' # v4 설정 파일 사용
    model = Estimator_v4(conf_file=conf_file).cuda().eval()

    # 사전 학습된 모델의 상태 사전 로드
    pretrained_model_path = Path(__file__).parents[0].resolve() / 'saved_model' / (file_name + '.pt')
    if not pretrained_model_path.exists():
         raise FileNotFoundError(f"Pretrained model file not found at {pretrained_model_path}")

    # Load the pretrained model's state dictionary
    pretrained_model_state_dict = torch.load(pretrained_model_path)

    # Load the state dict into the newly created model instance
    # Use strict=False in case the saved state_dict doesn't exactly match the current model structure
    try:
        model.load_state_dict(pretrained_model_state_dict, strict=True)
        print(f"Pretrained model state dict loaded successfully from {pretrained_model_path}")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}. Attempting non-strict loading.")
        model.load_state_dict(pretrained_model_state_dict, strict=False)
        print(f"Pretrained model state dict loaded non-strictly from {pretrained_model_path}")

    # LoRA 가중치 병합
    if isinstance(model.ch_tf, PeftModel):
        model.ch_tf = model.ch_tf.merge_and_unload()
        print("LoRA weights merged and unloaded.")

    output = model(input_tensor)
    output = [data.detach().cpu().numpy() for data in output]
    return output


if __name__ == "__main__":
    # # Compile simple model
    # file_name = "simple_model"
    # input_shape = (10,)
    # batch_size = 16
    # save_simple_test_model(file_name)
    # convert_pytorch_model_to_trt_engine(file_name=file_name, input_shape=input_shape, batch_size=batch_size)

    # # Simple model inference test
    # file_name = "simple_model"
    # input_shape = (10,)
    # batch_size = 8
    # input_tensor = np.random.randn(batch_size, *input_shape).astype(np.float32)
    # output_trt = trt_inference(file_name, input_tensor)
    # output_onnx = onnx_inference(file_name, input_tensor)
    # output_torch = torch_inference(file_name, input_tensor)
    # pass


    # Compile estimator
    file_name = "InF_Nlos_RMa_Large_estimator_PreLN_LoRA_RMa"
    input_shape = (14, 3072, 2)
    batch_size = 8
    # input_tensor = torch.tensor((batch_size, *input_shape)).cuda() # 이 부분은 ONNX export에서 사용되므로 주석 처리
    convert_pytorch_model_to_trt_engine(file_name=file_name, input_shape=input_shape, batch_size=batch_size,
                                        fp16=False)

    # # Transformer channel model inference test
    file_name = "InF_Nlos_RMa_Large_estimator_PreLN_LoRA_RMa"
    input_shape = (14, 3072, 2)
    batch_size = 4
    input_tensor = np.random.randn(batch_size, *input_shape).astype(np.float32)
    output_trt = trt_inference(file_name, input_tensor)
    output_onnx = onnx_inference(file_name, input_tensor)
    output_torch = torch_inference(file_name, input_tensor)
    pass