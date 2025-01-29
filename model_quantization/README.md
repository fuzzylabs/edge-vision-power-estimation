
# Quantized ONNX model

Download the ONNX model.

```bash
python download_onnx.py
```

Prepare calibration dataset for INT8 quantization.

> This will download COCO 2017 validation dataset for the first run.

```bash
python image_prep.py
```

The model can be quantized as an FP8, INT8 or INT4 model. For FP8 quantization `max` calibration is used. For INT8 quantization, you have choice between `max` and `entropy` calibration algorithms and for INT4, `awq_clip` or `rtn_dq` can be chosen.

```bash
python -m modelopt.onnx.quantization \
    --onnx_path=yolov5su.onnx \
    --quantize_mode=int8 \
    --calibration_data=calib.npy \
    --calibration_method=entropy \
    --output_path=yolov5su.quant.onnx
```

Run evaluation if the quantized model and unquantized model have the prediction within some tolerance and also test using a validation image.

```bash
python evaluate.py
```
