import torch
# import openvino as ov
import torch
import tensorrt as trt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import os


def export_model(model, name):
    # export to tensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(name+".onnx", 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise Exception("Failed to parse the ONNX file")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 1GB
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise Exception("Failed to build serialized network.")

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    trt_engine_path = name + ".trt"
    with open(trt_engine_path, "wb") as f:
        f.write(engine.serialize())


