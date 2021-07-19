from tensorflow.contrib.tensorrt import trt_convert as trt
converter = trt.TrtGraphConverterV2(input_saved_model_dir=".")
converter.convert()
converter.save(".")