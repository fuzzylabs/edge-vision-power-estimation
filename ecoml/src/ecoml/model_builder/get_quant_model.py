import torch
import torchvision.models as models
import torchvision.models.quantization as quant_models
from ecoml.model_summary.model_summary import get_summary

def generate_model_summaries():
    # Define standard input shape
    input_shape = (1, 3, 224, 224)
    
    # RESNET18
    model1 = models.resnet18(pretrained=True)
    model1.eval()
    get_summary(model1, input_shape=input_shape, summary_file_path="resnet18.json")
    print("ResNet18 Non-Quantized model generated")

    model2 = quant_models.resnet18(pretrained=True, quantize=True)
    model2.eval()
    get_summary(model2, input_shape=input_shape, summary_file_path="resnet18_quantized.json")
    print("ResNet18 Quantized model generated")

    # MOBILENETV2
    model3 = models.mobilenet_v2(pretrained=True)
    model3.eval()
    get_summary(model3, input_shape=input_shape, summary_file_path="mobilenetv2.json")
    print("MobileNetV2 Non-Quantized model generated")

    model4 = quant_models.mobilenet_v2(pretrained=True, quantize=True)
    model4.eval()
    get_summary(model4, input_shape=input_shape, summary_file_path="mobilenetv2_quantized.json")
    print("MobileNetV2 Quantized model generated")

    # INCEPTIONV3 (Requires 299x299 input)
    input_shape_inception = (1, 3, 299, 299)

    model5 = models.inception_v3(pretrained=True)
    model5.eval()
    get_summary(model5, input_shape=input_shape_inception, summary_file_path="inceptionv3.json")
    print("InceptionV3 Non-Quantized model generated")

    model6 = quant_models.inception_v3(pretrained=True, quantize=True)
    model6.eval()
    get_summary(model6, input_shape=input_shape_inception, summary_file_path="inceptionv3_quantized.json")
    print("InceptionV3 Quantized model generated")

if __name__ == "__main__":
    generate_model_summaries()
