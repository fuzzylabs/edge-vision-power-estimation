import torch
import torchvision.models as models
import torchvision.models.quantization as quant_models

model = models.resnet18(pretrained=True)
model.eval()
torch.save(model, "sample_data/resnet18.pt")
print("Model saved")

model2 = quant_models.resnet18(pretrained=True, quantize=True)
model2.eval()
torch.save(model2, "sample_data/resnet18_quantised.pt")
print("quantised model saved")