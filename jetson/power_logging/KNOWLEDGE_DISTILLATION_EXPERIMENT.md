# Knowledge Distillation Experiment 

Since distillation was done with older ultralytics/yolov5 library, it is not compatible with the new framework.
Hence we cannot perform the experiment in exactly the same manner automatically. However, we can make it reproducible 
with the following steps.

## Get the code
1. Copy the distilled model (e.g. `yolov5n-from-yolov5s.pt`) to this directory
2. Clone the exact code used for distillation
   ```
   git clone https://github.com/wonbeomjang/yolov5-knowledge-distillation.git
   cd yolov5-knowledge-distillation
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install 'numpy<2'
   ```
   
## Run evaluation
My suggestion is to use tmux for multiple terminals

1. Start power logging script in one terminal
   ```
   python measure_power.py --result-dir raw_data/distilled_models/yolov5n-from-yolov5s.pt
   ```
   
2. Start model validation script in another terminal
   ```
   cd yolov5-knowledge-distillation
   source .venv/bin/activate
   python val.py --weights ../yolov5n-from-yolov5s.pt --data coco.yaml
   ```
   
3. When validation is finished, Ctrl-C in the first terminal to stop power logging
4. Log the validation performance in `raw_data/distilled_models/yolov5n-from-yolov5s.pt/validation_results.json` in the following format
   ```
    {
        "metrics": {
            "metrics/precision(B)": 0.8022452631580933,
            "metrics/recall(B)": 0.6666666666666666,
            "metrics/mAP50(B)": 0.90053312264122,
            "metrics/mAP50-95(B)": 0.6284752778177719,
        },
        "speed": {
            "preprocess": 5.165755748748779,
            "inference": 156.59523010253906,
            "nms": 1.5867352485656738
        }
    }
   ```