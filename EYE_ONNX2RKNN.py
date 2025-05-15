#!/usr/bin/env python3

from rknn.api import RKNN

# Paths
ONNX_MODEL_PATH = './Models/EYE_81.onnx'
RKNN_MODEL_PATH = './Models/EYE_81_quant.rknn'
QUANT_DATASET_PATH = './Models/eye_quant_data/dataset.txt'

# 1. Create RKNN object
rknn = RKNN(verbose=True)

# 2. Configure RKNN for RK3588
print('--> Configuring RKNN (FP16 for RK3588)')
rknn.config(
    target_platform='rk3588',
    float_dtype='float16',
    quantized_dtype='w8a8',
    optimization_level=3
)
print('    Configuration done.')

# 3. Load the ONNX model
print('--> Loading ONNX model')
ret = rknn.load_onnx(
    model=ONNX_MODEL_PATH,
    inputs=['input'],  # only image input
    input_size_list=[[3, 224, 224, 3]]  # batch size = 3
)
if ret != 0:
    print('*** Failed to load ONNX model')
    exit(ret)
print('    Model loaded successfully.')

# 4. Build the RKNN model (with quantization)
print('--> Building RKNN model with quantization')
ret = rknn.build(do_quantization=True, dataset=QUANT_DATASET_PATH)
if ret != 0:
    print('*** RKNN build failed')
    exit(ret)
print('    Build completed.')

# 5. Export the .rknn model
print('--> Exporting RKNN model')
ret = rknn.export_rknn(RKNN_MODEL_PATH)
if ret != 0:
    print('*** Export failed')
    exit(ret)
print(f'    RKNN model saved to: {RKNN_MODEL_PATH}')

# 6. Release resources
rknn.release()
print('All done.')
