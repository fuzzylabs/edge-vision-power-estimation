import argparse
import json
import time
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


baseline_json = "raw_data/prebuilt_models/resnet18/resnet18_baseline_results.json"
zkp_json = "raw_data/prebuilt_models/resnet18/resnet18_zkp_results.json"

def load_results(path: str):
    with open(path, "r") as file:
        return json.load(file)

def preprocess_data(data):
    memory_usage = data.get("memory_usage", {})
    if isinstance(memory_usage, dict):
        data["memory_usage_cpu"] = memory_usage.get("cpu_memory", 0.0)
        data["memory_usage_gpu"] = memory_usage.get("gpu_memory", 0.0)
    else:
        data["memory_usage_cpu"] = memory_usage or 0.0
        data["memory_usage_gpu"] = 0.0

    model_size = data.get("model_size", {})
    if isinstance(model_size, dict):
        data["model_size"] = model_size.get("size_MB", 0.0)
    else:
        data["model_size"] = model_size or 0.0

    for key in ["power_usage", "accuracy", "energy_efficiency", "flops"]:
        data[key] = data.get(key, 0.0)

    return data




def main():
    baseline_data = preprocess_data(load_results(baseline_json))
    zkp_data = preprocess_data(load_results(zkp_json))

    metrics = ["avg_latency", "avg_throughput", "total_time", "power_usage", "accuracy",
                "memory_usage_cpu", "memory_usage_gpu", "model_size", "flops", "energy_efficiency"
    ]

    metric_names = {
        "avg_latency": "Avg Latency (Sec)",
        "avg_throughput": "Avg Throughput (Samples/Sec)",
        "total_time": "Total Time (sec)",
        "power_usage": "Power Consumption (W)",
        "accuracy": "Accuracy (%)",
        "memory_usage": "Memory Usage (MB)",
        "model_size": "Model Size (MB)",
        "flops": "Floating Point Operations (FLOPS)",
        "energy_efficiency": "Energy Efficiency (W/Sample)"
    }

    baseline_values = [round(baseline_data.get(m, 0.0) or 0.0, 6) for m in metrics]
    zkp_values = [round(zkp_data.get(m, 0.0) or 0.0, 6) for m in metrics]

    percent_change = []
    for baseline, zkp in zip(baseline_values, zkp_values):
        if baseline != 0:
            change = ((zkp - baseline) / baseline) * 100
        else:
            change = 0.0
        percent_change.append(change)

    df = pd.DataFrame({
        "Metric": metrics,
        "Baseline": baseline_values,
        "ZKP": zkp_values,
        "Percent Change (%)": [round(p, 2) for p in percent_change],
    })
    pd.options.display.float_format = '{:,.6f}'.format
    print("- Comparison Table -\n")
    print(df.to_string(index=False))

    # fig, ax = plt.subplots(figsize=(8, 4))
    # ax.axis("tight")
    # ax.axis("off")
    # table = ax.table(
    #     cellText = df.values,
    #     colLabels = df.columns,
    #     cellLoc = "center",
    #     loc = "center",
    # )
    # table.auto_set_font_size(False)
    # table.set_fontsize(10)
    # table.auto_set_column_width(col=list(range(len(df.columns))))
    
    # plt.title("Comparison Table")
    # plt.tight_layout()
    # plt.savefig("comparison_table.png")
    # plt.show()

    # ax = df.plot(x="Metric", kind="bar", figsize=(12, 6))
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.6f'))
    # plt.title("Comparison of Baseline vs. ZKFP")
    # plt.ylabel("Metric Value")
    # plt.xticks(rotation=45, ha="right")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    # if len(metrics) == 1:
    #     axes = [axes]

    # for i, metric in enumerate(metrics):
    #     ax = axes[i]
    #     ax.bar([0], [baseline_data[metric]], width=0.4, color='blue', label='Baseline')
    #     ax.bar([1], [zkp_data[metric]], width=0.4, color='red', label='ZKP')
    #     ax.set_title(metric_names[metric])
    #     ax.set_xticks([0, 1])
    #     ax.set_xticklabels(["Baseline", "ZKP"])
    #     ax.set_ylabel("Value")
    #     if i == 0:
    #         ax.legend()

    # fig.suptitle("Metric-Wise Comparison", fontsize=14, y=1.05)
    # plt.tight_layout()
    # plt.show()


    # x = np.arange(len(metrics))
    # width = 0.35

    # plt.figure(figsize=(8, 6))

    # plt.bar(x - width/2, baseline_values, width, label="Baseline (Pre)")
    # plt.bar(x + width/2, zkp_values, width, label="ZKP (Post)")

    # plt.xticks(x, metrics)
    # plt.ylabel("Metric value")
    # plt.title("Pre vs Post Pruning")
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()



