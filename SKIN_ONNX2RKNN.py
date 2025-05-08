#!/usr/bin/env python3

from rknn.api import RKNN

# Paths
ONNX_MODEL_PATH = './Models/SKIN_72.onnx'
RKNN_MODEL_PATH = './Models/SKIN_72.rknn'


# 1. Create RKNN object
rknn = RKNN(verbose=True)

# 2. Configure for RK3588, FP16 (no INT8 quantization)
#    float_dtype='float16' ensures the model runs in FP16 on the NPU :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
print('--> Configuring RKNN (FP16 for RK3588)')
rknn.config(
    target_platform='rk3588',
    quantized_dtype='w16a16i_dfp'
)

print('    Configuration done.')

# 3. Load the ONNX model
#    Specify both inputs and their fixed shapes (batch=1)
print('--> Loading ONNX model')
ret = rknn.load_onnx(
    model=ONNX_MODEL_PATH,
    inputs=['image_input', 'meta_input'],
    input_size_list=[
        [3, 224, 224, 3],  # image_input: (1, H, W, C)
        [3, 18]            # meta_input: (1, feature_dim)
    ]
)

if ret != 0:
    print('*** Failed to load ONNX model')
    exit(ret)
print('    Model loaded successfully.')

# 4. Build the RKNN model in FP16 mode (no calibration, no INT8)
print('--> Building RKNN model (FP16 simulation)')
ret = rknn.build(do_quantization=False)
if ret != 0:
    print('*** RKNN build failed')
    exit(ret)
    
print('    Build completed.')

# 5. Export the .rknn file
print('--> Exporting RKNN model')
ret = rknn.export_rknn(RKNN_MODEL_PATH)
if ret != 0:
    print('*** Export failed')
    exit(ret)
print(f'    RKNN model saved to: {RKNN_MODEL_PATH}')

# 6. Clean up
rknn.release()
print('All done.')
