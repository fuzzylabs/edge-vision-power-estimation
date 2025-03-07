import torch
import torchvision.models as models
import torchvision.models.quantization as quant_models
from ecoml.model_summary.model_summary import get_summary

def main():
    model1 = models.resnet18(pretrained=True)
    model1.eval()

    input_shape1 = (1, 3, 224, 224)

    get_summary(model1, input_shape=input_shape1, summary_file_path="resnet18.json")
    print("Non quantised model generated")

    model2 = quant_models.resnet18(pretrained=True, quantize=True)
    model2.eval()

    get_summary(model2, input_shape=input_shape1, summary_file_path="resnet18_quantized.json")
    print("Quantized model generated")

if __name__ == "__main__":
    main()