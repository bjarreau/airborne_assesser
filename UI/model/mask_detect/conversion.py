from tensorflow.python.compiler.tensorrt import trt_convert as trt

convert_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32, max_workspace_size_byte=8000000000)
converter = trt.TrtGraphConverterV2(input_saved_model_dir='.', conversion_params=convert_params)
converter.convert()
converter.save("./TRT")
print('Done Converting to TF-TRT FP32')