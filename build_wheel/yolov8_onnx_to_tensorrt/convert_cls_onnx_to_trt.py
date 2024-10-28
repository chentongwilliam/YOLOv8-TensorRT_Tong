import tensorrt as trt
import torch
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        type=str,
                        required=True,
                        help='Weights file')
    parser.add_argument('--output',
                        type=str,
                        default='',
                        required=False,
                        help='Output file Path')
    parser.add_argument('--input-shape',
                        nargs='+',
                        type=int,
                        default=[1, 3, 224, 224],
                        help='Model input shape only for api builder')
    
    # parser.add_argument('--conf-thres',
    #                     type=float,
    #                     default=0.25,
    #                     help='CONF threshoud for NMS plugin')
    # parser.add_argument('--fp16',
    #                     action='store_true',
    #                     help='Build model with fp16 mode')
    # parser.add_argument('--device',
    #                     type=str,
    #                     default='cuda:0',
    #                     help='TensorRT builder device')

    args = parser.parse_args()
    return args

def build_engine(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # 创建 builder 和网络定义
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 读取 ONNX 模型文件
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 配置 builder 和构建 engine
    config = builder.create_builder_config()
    total_memory = torch.cuda.get_device_properties(torch.device('cuda:0')).total_memory
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(total_memory * 0.75))
    serialized_engine = builder.build_serialized_network(network, config)

    # 将生成的 engine 保存到文件
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)

    # return serialized_engine

def main():
    args = parse_args()
    # 调用函数来转换模型
    if args.output == '':
        engine_file_path = Path(args.weights).with_suffix('.engine')
    else:
        engine_file_path = Path(args.output)
    build_engine(args.weights, engine_file_path)

if __name__ == "__main__":
    main()


