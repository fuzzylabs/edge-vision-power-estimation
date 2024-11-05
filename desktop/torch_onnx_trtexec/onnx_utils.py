"""Functions to export and saving onnx model."""

from pathlib import Path
import time
import torch

DTYPE_MAPPING = {"float32": "fp32", "float16": "fp16", "bfloat16": "bfp16"}

def export_to_onnx(
    input_data,
    pt_model,
    onnx_file_path: str,
) -> None:
    """Export PyTorch model to ONNX model

    Args:
        input_data: Input data used for tracing
        pt_model: PyTorch model
        onnx_file_path: Path to save ONNX model
    """
    torch.onnx.export(
        pt_model,  # model being run
        input_data,  # model input (or a tuple for multiple inputs)
        onnx_file_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },  # variable length axes
    )


def save_onnx_model(args, model, input_data) -> str:
    st = time.perf_counter()
    dtype = DTYPE_MAPPING.get(args.dtype)
    onnx_dir_path = f"{args.model_dir}/{args.model}"
    onnx_file_path = f"{onnx_dir_path}/{args.model}_{dtype}.onnx"
    Path(onnx_dir_path).mkdir(exist_ok=True, parents=True)
    export_to_onnx(pt_model=model, input_data=input_data, onnx_file_path=onnx_file_path)
    onnx_convert_time = time.perf_counter() - st
    print(f"Time to taken convert PyTorch model to ONNX {onnx_convert_time:.3f} sec")
    return onnx_file_path