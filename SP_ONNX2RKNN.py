from rknn.api import RKNN

ONNX_MODEL_PATH = './Models/SP_91.onnx'
RKNN_MODEL_PATH = './Models/SP_91.rknn'

rknn = RKNN()

print('--> Configuring RKNN (FP16 quantization)')
rknn.config(
    target_platform='rk3588',
    float_dtype = 'float16',
)
print('Done.')

print('--> Loading ONNX model')
ret = rknn.load_onnx(
    model=ONNX_MODEL_PATH,
    inputs = ['input'],
    input_size_list=[[4, 19]]  # Set explicit shape
)
if ret != 0:
    print('Failed to load ONNX model')
    exit(ret)
print('Done.')

print('--> Building RKNN model with fp16 quantization')
ret = rknn.build(do_quantization=False)
if ret != 0:
    print('Build failed')
    exit(ret)
print('Done.')

print('--> Exporting RKNN model')
ret = rknn.export_rknn(RKNN_MODEL_PATH)
if ret != 0:
    print('Export failed')
    exit(ret)
print(f'RKNN model exported to {RKNN_MODEL_PATH}')

rknn.release()