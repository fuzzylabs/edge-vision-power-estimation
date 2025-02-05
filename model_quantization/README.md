# Model Quantization

## Quantized PyTorch model

The following command creates a PTQ model for a PyTorch model converting it to a INT8 quantized model.

```bash
python pytorch_quantize.py
```

To run the evaluation on a custom image, run the following command.

```bash
python evaluate_pt.py
```

To run inference using quantized PyTorch model

```bash
python evaluate_pt.py --quant
```

## Quantized ONNX model

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

Run evaluation if the quantized model and unquantized model have the prediction using a validation image.

```bash
python evaluate_onnx.py
```

To run inference using quantized ONNX model

```bash
python evaluate_onnx.py --quant
```

## Quantized TRT model

To get TensorRT models for the original PyTorch models, run the following script.

```bash
python download_trt_engine.py
```

To convert quantized ONNX models to TensorRT models, we will use `trtexec` tool.

```bash
trtexec --onnx=quant.onnx --saveEngine=quant.engine --best
```
