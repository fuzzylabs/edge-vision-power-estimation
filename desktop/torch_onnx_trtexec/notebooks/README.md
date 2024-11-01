# Notebook

Thanks to [trt-engine-explorer](https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer) library that does all of the heavy lifting and producing beautiful plots for the TensorRT profiling.

Here we go through notebook on alexnet TensorRT engine. It requires following files for providing analysis saved during benchmarking

```bash
# Saved in models dir
.engine - the built engine file.

# Saved in results dir
.build.metadata.json - JSON of metadata parsed from the build log.
.graph.json - JSON of engine graph.
.profile.json - JSON of engine layers profiling.
.profile.metadata.json - JSON of metadata parsed from the profiling log.
.timing.json - JSON of engine profiling iteration timing.
```

## Plan Summary

We get a plan summary consisting of overview of model configuration, device information, builder configuration and performance summary.

We get a sunburst chart showing percantage of latency for each layer.

* The `gemm` layers take up to 69% of overall latency.
* The `Convolution` layers take up to 27% of overall latency.
* The `Pooling` layers take up to 1% of overall latency.
* The `Reformat` and `shape call` layers take remaining 3% share.

![percantage latency](../assets/alexnet_layer_avg_latencies.png)

## Performance and Memory Footprint

Performance provides further breakdown showing latency distribution, latency of layer grouped by type and individual layer latencies.

Memory footprint provides break down of weights and activations of layers grouped and individual layers in the model. It is measured in bytes where `total_footprint_bytes` calculates total bytes of the layer including input, output and weights.

We can further get memory consumed by filtering on different types of layers.

## Export plan

We can create a SVG of the TensorRT engine that provides various configuration such as including average latency time taken to run the layer.

An example of exported alexnet SVG that provides average latency taken by particular layer.

![alexnet](../results/alexnet/alexnet_avg_latency.png)
