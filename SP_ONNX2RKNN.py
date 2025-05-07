from rknn.api import RKNN

ONNX_MODEL_PATH = './Models/best_model_91.onnx'
RKNN_MODEL_PATH = './Models/best_model_91.rknn'

rknn = RKNN()

print('--> Configuring RKNN (FP16 quantization)')
rknn.config(
    target_platform='rk3588',
    quantized_dtype='w16a16i_dfp'
)
print('Done.')

print('--> Loading ONNX model')
ret = rknn.load_onnx(
    model=ONNX_MODEL_PATH,
    inputs = ['input'],
    input_size_list=[[1, 19]]  # Set explicit shape
)
if ret != 0:
    print('Failed to load ONNX model')
    exit(ret)
print('Done.')

print('--> Building RKNN model with fp16 quantization')
ret = rknn.build(do_quantization=True, dataset='./Models/dataset.txt')
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
