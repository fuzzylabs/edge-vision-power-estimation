# trtexec

[trtexec](https://github.com/NVIDIA/TensorRT/tree/master/samples/trtexec) is a TensorRT command line wrapper to build and profile engines quickly. It is written in C++.

In the guide on [TensorRT](../../torch_onnx_trt/docs/TensorRT.md), we learnt that TensorRT consists of two phases: Build phase and Inference phase. `trtexec` tool works seamlessly for both the phases.

## Build phase

In build phase, we create a TensorRT engine from ONNX model.

In our code, the command used in [trt_utils.py](../trt_utils.py) `build_engine_cmd` function is the following

```bash
trtexec --verbose \
    --onnx={onnx_path} \
    --saveEngine={engine_path} \
    --exportLayerInfo={graph_json_fname} \
    --timingCacheFile={timing_cache_path} \
    --profilingVerbosity=detailed \
    --fp16
```

A comprehensive list of all flags used in build phase along with explaination for the flag can be found at `Flags for the Build Phase` section of the TensorRT documentation : <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-flags>

Let's take a look at build logs emitted while creating a TensorRT engine for alexnet model as example for following analysis. These logs are saved under `models/<model-name>` directory for each model with filename ending with `*.engine.build.log`.

The log file prints useful information back such as Model Options, Build Options, Device Information and Inference Options (not required for Build phase but for Inference phase). These logs are parsed into a metadata json file saved under `results/<model-name>` folder.

```txt
[11/01/2024-10:32:46] [I] === Model Options ===
[11/01/2024-10:32:46] [I] Format: ONNX
[11/01/2024-10:32:46] [I] Model: models/alexnet/alexnet_fp16.onnx
[11/01/2024-10:32:46] [I] Output:
[11/01/2024-10:32:46] [I] === Build Options ===
[11/01/2024-10:32:46] [I] Memory Pools: workspace: default, dlaSRAM: default, dlaLocalDRAM: default, dlaGlobalDRAM: default, tacticSharedMem: default
[11/01/2024-10:32:46] [I] avgTiming: 8
[11/01/2024-10:32:46] [I] Precision: FP32+FP16
[11/01/2024-10:32:46] [I] LayerPrecisions: 
[11/01/2024-10:32:46] [I] Layer Device Types: 
[11/01/2024-10:32:46] [I] Calibration: 
...more
[11/01/2024-10:32:46] [I] === System Options ===
[11/01/2024-10:32:46] [I] Device: 0
[11/01/2024-10:32:46] [I] DLACore: 
[11/01/2024-10:32:46] [I] Plugins:
[11/01/2024-10:32:46] [I] setPluginsToSerialize:
[11/01/2024-10:32:46] [I] dynamicPlugins:
[11/01/2024-10:32:46] [I] ignoreParsedPluginLibs: 0
[11/01/2024-10:32:46] [I] 
[11/01/2024-10:32:46] [I] === Inference Options ===
[11/01/2024-10:32:46] [I] Batch: Explicit
[11/01/2024-10:32:46] [I] Input inference shapes: model
[11/01/2024-10:32:46] [I] Iterations: 10
[11/01/2024-10:32:46] [I] Duration: 3s (+ 200ms warm up)
...more
[11/01/2024-10:32:46] [I] === Reporting Options ===
[11/01/2024-10:32:46] [I] Verbose: Enabled
[11/01/2024-10:32:46] [I] Averages: 10 inferences
[11/01/2024-10:32:46] [I] Percentiles: 90,95,99
...more
[11/01/2024-10:32:46] [I] === Device Information ===
[11/01/2024-10:32:46] [I] Available Devices: 
[11/01/2024-10:32:46] [I]   Device 0: "NVIDIA GeForce RTX 3060 Laptop GPU" UUID: GPU-2580ec36-5574-ddc7-4615-63d57b5969e2
[11/01/2024-10:32:46] [I] Selected Device: NVIDIA GeForce RTX 3060 Laptop GPU
[11/01/2024-10:32:46] [I] Selected Device ID: 0
[11/01/2024-10:32:46] [I] Selected Device UUID: GPU-2580ec36-5574-ddc7-4615-63d57b5969e2
[11/01/2024-10:32:46] [I] Compute Capability: 8.6
[11/01/2024-10:32:46] [I] SMs: 30
[11/01/2024-10:32:46] [I] Device Global Memory: 5937 MiB
[11/01/2024-10:32:46] [I] Shared Memory per SM: 100 KiB
[11/01/2024-10:32:46] [I] Memory Bus Width: 192 bits (ECC disabled)
[11/01/2024-10:32:46] [I] Application Compute Clock Rate: 1.425 GHz
[11/01/2024-10:32:46] [I] Application Memory Clock Rate: 7.001 GHz
[11/01/2024-10:32:46] [I] 
[11/01/2024-10:32:46] [I] Note: The application clock rates do not reflect the actual clock rates that the GPU is currently running at.
[11/01/2024-10:32:46] [I] 
```

> An example build metadata file for alexnet model is [alexnet_fp16.build.metadata.json](../results/alexnet/alexnet_fp16.build.metadata.json).

Similar to Python API, trtexec parses the ONNX model file to create a TensorRT network definition.

```txt
[11/01/2024-10:32:46] [I] [TRT] [MemUsageChange] Init CUDA: CPU +1, GPU +0, now: CPU 16, GPU 585 (MiB)
[11/01/2024-10:32:46] [V] [TRT] Trying to load shared library libnvinfer_builder_resource.so.10.4.0
[11/01/2024-10:32:46] [V] [TRT] Loaded shared library libnvinfer_builder_resource.so.10.4.0
[11/01/2024-10:32:48] [I] [TRT] [MemUsageChange] Init builder kernel library: CPU +2117, GPU +396, now: CPU 2289, GPU 981 (MiB)
[11/01/2024-10:32:48] [V] [TRT] CUDA lazy loading is enabled.
[11/01/2024-10:32:48] [I] Start parsing network model.
[11/01/2024-10:32:48] [I] [TRT] ----------------------------------------------------------------
[11/01/2024-10:32:48] [I] [TRT] Input filename:   models/alexnet/alexnet_fp16.onnx
[11/01/2024-10:32:48] [I] [TRT] ONNX IR version:  0.0.5
[11/01/2024-10:32:48] [I] [TRT] Opset version:    10
[11/01/2024-10:32:48] [I] [TRT] Producer name:    pytorch
[11/01/2024-10:32:48] [I] [TRT] Producer version: 2.5.0
[11/01/2024-10:32:48] [I] [TRT] Domain:           
[11/01/2024-10:32:48] [I] [TRT] Model version:    0
[11/01/2024-10:32:48] [I] [TRT] Doc string:       
[11/01/2024-10:32:48] [I] [TRT] ----------------------------------------------------------------
[11/01/2024-10:32:48] [V] [TRT] Adding network input: input with dtype: float16, dimensions: (-1, 3, 224, 224)
[11/01/2024-10:32:48] [V] [TRT] Registering tensor: input for ONNX tensor: input
...more
[11/01/2024-10:32:48] [V] [TRT] Marking output_7 as output: output
[11/01/2024-10:32:48] [I] Finished parsing network model. Parse time: 0.152975
[11/01/2024-10:32:48] [W] Dynamic dimensions required for input: input, but no shapes were provided. Automatically overriding shape to: 1x3x224x224
[11/01/2024-10:32:48] [I] Set shape of input tensor input for optimization profile 0 to: MIN=1x3x224x224 OPT=1x3x224x224 MAX=1x3x224x224
```

Parsing the ONNX model takes about 0.15 seconds. TensorRT measures the memory used before and after critical operations in the builder and runtime. These memory usage statistics are printed to TensorRT’s information logger. For example in above logs:

```txt
[11/01/2024-10:32:46] [I] [TRT] [MemUsageChange] Init CUDA: CPU +1, GPU +0, now: CPU 16, GPU 585 (MiB)
```

It indicates that memory use changes with CUDA initialization. `CPU +1, GPU +0` is the increased amount of memory after running CUDA initialization. The content after `now:` is the CPU/GPU memory usage snapshot after CUDA initialization.

As next step, TensorRT network definition graph is optimized using [internal optimization techniques](https://github.com/NVIDIA/TensorRT/issues/2576#issuecomment-1378529294) and graph fusions. It perform various fusion like Conv and RELU are combined to be not 2 separate layers but a single fused layer.

```txt
[11/01/2024-10:32:48] [W] [TRT] Could not read timing cache from: /tmp/alexnet_engine_cache. A new timing cache will be generated and written.
[11/01/2024-10:32:48] [V] [TRT] Original: 32 layers
[11/01/2024-10:32:48] [V] [TRT] After dead-layer removal: 32 layers
[11/01/2024-10:32:48] [V] [TRT] Graph construction completed in 0.000398764 seconds.
[11/01/2024-10:32:48] [V] [TRT] After adding DebugOutput nodes: 32 layers
[11/01/2024-10:32:48] [V] [TRT] Running: ConstShuffleFusion on classifier.1.bias
[11/01/2024-10:32:48] [V] [TRT] ConstShuffleFusion: Fusing classifier.1.bias with ONNXTRT_Broadcast
[11/01/2024-10:32:48] [V] [TRT] Running: ConstShuffleFusion on classifier.4.bias
[11/01/2024-10:32:48] [V] [TRT] ConstShuffleFusion: Fusing classifier.4.bias with ONNXTRT_Broadcast_4
[11/01/2024-10:32:48] [V] [TRT] Running: ConstShuffleFusion on classifier.6.bias
[11/01/2024-10:32:48] [V] [TRT] ConstShuffleFusion: Fusing classifier.6.bias with ONNXTRT_Broadcast_6
[11/01/2024-10:32:48] [V] [TRT] After Myelin optimization: 15 layers
[11/01/2024-10:32:48] [V] [TRT] Applying ScaleNodes fusions.
[11/01/2024-10:32:48] [V] [TRT] After scale fusion: 15 layers
[11/01/2024-10:32:48] [V] [TRT] Running: ConvReluFusion on /features/features.0/Conv
[11/01/2024-10:32:48] [V] [TRT] ConvReluFusion: Fusing /features/features.0/Conv with /features/features.1/Relu
[11/01/2024-10:32:48] [V] [TRT] Running: ConvReluFusion on /features/features.3/Conv
[11/01/2024-10:32:48] [V] [TRT] ConvReluFusion: Fusing /features/features.3/Conv with /features/features.4/Relu
[11/01/2024-10:32:48] [V] [TRT] Running: ConvReluFusion on /features/features.6/Conv
[11/01/2024-10:32:48] [V] [TRT] ConvReluFusion: Fusing /features/features.6/Conv with /features/features.7/Relu
[11/01/2024-10:32:48] [V] [TRT] Running: ConvReluFusion on /features/features.8/Conv
[11/01/2024-10:32:48] [V] [TRT] ConvReluFusion: Fusing /features/features.8/Conv with /features/features.9/Relu
[11/01/2024-10:32:48] [V] [TRT] Running: ConvReluFusion on /features/features.10/Conv
[11/01/2024-10:32:48] [V] [TRT] ConvReluFusion: Fusing /features/features.10/Conv with /features/features.11/Relu
[11/01/2024-10:32:48] [V] [TRT] Running: PoolingErasure on /avgpool/AveragePool
[11/01/2024-10:32:48] [V] [TRT] Removing /avgpool/AveragePool
[11/01/2024-10:32:48] [V] [TRT] After dupe layer removal: 9 layers
[11/01/2024-10:32:48] [V] [TRT] After final dead-layer removal: 9 layers
[11/01/2024-10:32:48] [V] [TRT] After tensor merging: 9 layers
[11/01/2024-10:32:48] [V] [TRT] After vertical fusions: 9 layers
[11/01/2024-10:32:48] [V] [TRT] After dupe layer removal: 9 layers
[11/01/2024-10:32:48] [V] [TRT] After final dead-layer removal: 9 layers
[11/01/2024-10:32:48] [V] [TRT] After tensor merging: 9 layers
[11/01/2024-10:32:48] [V] [TRT] After slice removal: 9 layers
[11/01/2024-10:32:48] [V] [TRT] After concat removal: 9 layers
[11/01/2024-10:32:48] [V] [TRT] Trying to split Reshape and strided tensor
[11/01/2024-10:32:48] [V] [TRT] Graph optimization time: 0.00064049 seconds.
```

Next, TensorRT performs autotuning. Consider following example for a convolution layer and RELU layer (`/features/features.0/Conv + /features/features.1/Relu`)

```txt
[11/01/2024-10:32:48] [V] [TRT] =============== Computing costs for /features/features.0/Conv + /features/features.1/Relu
[11/01/2024-10:32:48] [V] [TRT] *************** Autotuning format combination: Float(150528,50176,224,1) -> Float(193600,3025,55,1) ***************
[11/01/2024-10:32:48] [V] [TRT] --------------- Timing Runner: /features/features.0/Conv + /features/features.1/Relu (CaskConvolution[0x80000009])
[11/01/2024-10:32:48] [V] [TRT] Tactic Name: ampere_scudnn_128x64_relu_xregs_large_nn_v1 Tactic: 0x5deb29b7a8e275f7 Time: 0.0521752
[11/01/2024-10:32:48] [V] [TRT] Tactic Name: sm80_xmma_fprop_implicit_gemm_indexed_f32f32_f32f32_f32_nchwkcrs_nchw_tilesize64x64x8_stage3_warpsize1x4x1_g1_ffma_aligna4_alignc4 Tactic: 0xd828f024626fa982 Time: 0.0513463
[11/01/2024-10:32:48] [V] [TRT] Tactic Name: sm80_xmma_fprop_implicit_gemm_indexed_f32f32_f32f32_f32_nchwkcrs_nchw_tilesize128x64x8_stage2_warpsize2x2x1_g1_ffma_aligna4_alignc4_beta0_packed_stride Tactic: 0x31aa67f57c5aea77 Time: 0.0523215
[11/01/2024-10:32:48] [V] [TRT] Tactic Name: sm80_xmma_fprop_implicit_gemm_indexed_f32f32_f32f32_f32_nchwkcrs_nchw_tilesize128x16x8_stage3_warpsize4x1x1_g1_ffma_aligna4_alignc4 Tactic: 0x40a12e3938221818 Time: 0.0990354
[11/01/2024-10:32:48] [V] [TRT] Tactic Name: sm80_xmma_fprop_implicit_gemm_indexed_f32f32_f32f32_f32_nchwkcrs_nchw_tilesize128x64x8_stage2_warpsize1x4x1_g1_ffma_aligna4_alignc4_beta0_packed_stride Tactic: 0xede36641840ce3d2 Time: 0.0533455
[11/01/2024-10:32:48] [V] [TRT] Tactic Name: sm80_xmma_fprop_implicit_gemm_indexed_f32f32_f32f32_f32_nchwkcrs_nchw_tilesize32x32x8_stage3_warpsize1x2x1_g1_ffma_aligna4_alignc4 Tactic: 0xcb8a43f748d8a338 Time: 0.06656
[11/01/2024-10:32:48] [V] [TRT] Tactic Name: sm80_xmma_fprop_implicit_gemm_indexed_f32f32_f32f32_f32_nchwkcrs_nchw_tilesize128x32x8_stage3_warpsize2x2x1_g1_ffma_aligna4_alignc4 Tactic: 0xa9366041633a5135 Time: 0.0676328
[11/01/2024-10:32:48] [V] [TRT] Tactic Name: sm80_xmma_fprop_implicit_gemm_indexed_f32f32_f32f32_f32_nchwkcrs_nchw_tilesize128x64x8_stage2_warpsize4x1x1_g1_ffma_aligna4_alignc4_beta0_packed_stride Tactic: 0x1673e3594ce11cea Time: 0.0567589
[11/01/2024-10:32:48] [V] [TRT] /features/features.0/Conv + /features/features.1/Relu (CaskConvolution[0x80000009]) profiling completed in 0.0813502 seconds. Fastest Tactic: 0xd828f024626fa982 Time: 0.0513463
[11/01/2024-10:32:48] [V] [TRT] Skipping CaskFlattenConvolution: No valid tactics for /features/features.0/Conv + /features/features.1/Relu
[11/01/2024-10:32:48] [V] [TRT] >>>>>>>>>>>>>>> Chose Runner Type: CaskConvolution Tactic: 0xd828f024626fa982
[11/01/2024-10:32:48] [V] [TRT] *************** Autotuning format combination: Float(150528,1,672,3) -> Float(193600,1,3520,64) ***************
[11/01/2024-10:32:48] [V] [TRT] --------------- Timing Runner: /features/features.0/Conv + /features/features.1/Relu (CaskConvolution[0x80000009])
[11/01/2024-10:32:48] [V] [TRT] Tactic Name: sm80_xmma_fprop_implicit_gemm_indexed_f32f32_f32f32_f32_nhwckrsc_nhwc_tilesize32x32x8_stage3_warpsize1x2x1_g1_ffma_aligna4_alignc4 Tactic: 0x0a143be7a52f301a Time: 0.129463
[11/01/2024-10:32:48] [V] [TRT] Tactic Name: sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_f32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x32x16_stage1_warpsize4x1x1_g1_tensor16x8x8_aligna4_alignc4 Tactic: 0xf231cca3335919a4 Time: 0.0397166
[11/01/2024-10:32:48] [V] [TRT] Tactic Name: sm80_xmma_fprop_implicit_gemm_indexed_f32f32_f32f32_f32_nhwckrsc_nhwc_tilesize128x16x8_stage3_warpsize4x1x1_g1_ffma_aligna4_alignc4 Tactic: 0x4fd3c46622e98342 Time: 0.157842
...more
[11/01/2024-10:32:49] [V] [TRT] Tactic Name: sm75_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x16x32_stage1_warpsize4x1x1_g1_tensor16x8x8_aligna2_alignc4 Tactic: 0x4661a730b248432e Time: 0.0486644
[11/01/2024-10:32:49] [V] [TRT] Tactic Name: sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x32x64_stage1_warpsize4x1x1_g1_tensor16x8x16_aligna4_alignc4 Tactic: 0xa1c540a5038e4190 Time: 0.0270385
[11/01/2024-10:32:49] [V] [TRT] /features/features.0/Conv + /features/features.1/Relu (CaskConvolution[0x80000009]) profiling completed in 0.0677056 seconds. Fastest Tactic: 0xd4e0dd17e4d87903 Time: 0.0251124
[11/01/2024-10:32:49] [V] [TRT] Skipping CaskFlattenConvolution: No valid tactics for /features/features.0/Conv + /features/features.1/Relu
[11/01/2024-10:32:49] [V] [TRT] >>>>>>>>>>>>>>> Chose Runner Type: CaskConvolution Tactic: 0xd4e0dd17e4d87903
[11/01/2024-10:32:49] [V] [TRT] *************** Autotuning format combination: Half(50176,1:4,224,1) -> Half(48400,1:4,880,16) ***************
[11/01/2024-10:32:49] [V] [TRT] Skipping CaskConvolution: No valid tactics for /features/features.0/Conv + /features/features.1/Relu
[11/01/2024-10:32:49] [V] [TRT] Skipping CaskFlattenConvolution: No valid tactics for /features/features.0/Conv + /features/features.1/Relu
[11/01/2024-10:32:49] [V] [TRT] *************** Autotuning format combination: Half(50176,1:4,224,1) -> Half(24200,1:8,440,8) ***************
[11/01/2024-10:32:49] [V] [TRT] --------------- Timing Runner: /features/features.0/Conv + /features/features.1/Relu (CaskConvolution[0x80000009])
[11/01/2024-10:32:49] [V] [TRT] Tactic Name: sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x32x16_stage1_warpsize4x1x1_g1_tensor16x8x16_aligna8_alignc4 Tactic: 0x777aafa0d6de14bd Time: 0.0375589
[11/01/2024-10:32:49] [V] [TRT] Tactic Name: sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x16x64_stage1_warpsize4x1x1_g1_tensor16x8x16_aligna4_alignc2 Tactic: 0x17bdd86adff94950 Time: 0.0306907
...more
[11/01/2024-10:32:49] [V] [TRT] Tactic Name: sm75_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x16x32_stage1_warpsize4x1x1_g1_tensor16x8x8_aligna8_alignc4 Tactic: 0x886d7923447c7828 Time: 0.0225489
...more
[11/01/2024-10:32:49] [V] [TRT] Tactic Name: sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x16x32_stage1_warpsize4x1x1_g1_tensor16x8x16_aligna8_alignc2 Tactic: 0x955dd84f2f723fdc Time: 0.0362789
[11/01/2024-10:32:49] [V] [TRT] /features/features.0/Conv + /features/features.1/Relu (CaskConvolution[0x80000009]) profiling completed in 0.256518 seconds. Fastest Tactic: 0x1bf48a356bd0c083 Time: 0.0206263
[11/01/2024-10:32:49] [V] [TRT] Skipping CaskFlattenConvolution: No valid tactics for /features/features.0/Conv + /features/features.1/Relu
[11/01/2024-10:32:49] [V] [TRT] >>>>>>>>>>>>>>> Chose Runner Type: CaskConvolution Tactic: 0x1bf48a356bd0c083
```

TensorRT tests various strides of input (not to be confused with convolution or pooling strides), it means different representation of the same input (nchw, nhwc and [other tensort formats](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/LayerBase.html#tensorrt.TensorFormat)) and also the input precision from `float32` to `float16`.

We see the following line where for the layer `/features/features.0/Conv + /features/features.1/Relu` it chooses a tatic that is fastest overall across testing all tactic scenarios.

```txt
[11/01/2024-10:32:49] [V] [TRT] /features/features.0/Conv + /features/features.1/Relu (CaskConvolution[0x80000009]) profiling completed in 0.149464 seconds. Fastest Tactic: 0x280b0ad3c23d8442 Time: 0.0194743
```

> TensorRT optimizes for minimum timing of the whole network.

This process is repeated for all the layers in the network. Finally, we get following output where engine is built and a summary of all layers where the best selected tactic and all layer information with input, output, stride, padding, and, dilation are included.

```txt
[11/01/2024-10:32:57] [I] [TRT] Engine generation completed in 8.62539 seconds.
[11/01/2024-10:32:57] [V] [TRT] Layers:
Name: Reformatting CopyNode for Input Tensor 0 to /features/features.0/Conv + /features/features.1/Relu, LayerType: Reformat, Inputs: [ { Name: input, Location: Device, Dimensions: [1,3,224,224], Format/Datatype: Half }], Outputs: [ { Name: Reformatted Input Tensor 0 to /features/features.0/Conv + /features/features.1/Relu, Location: Device, Dimensions: [1,3,224,224], Format/Datatype: Half }], ParameterType: Reformat, Origin: REFORMAT, TacticValue: 0x00000000000003e8, StreamId: 0, Metadata: 
Name: /features/features.0/Conv + /features/features.1/Relu, LayerType: CaskConvolution, Inputs: [ { Name: Reformatted Input Tensor 0 to /features/features.0/Conv + /features/features.1/Relu, Location: Device, Dimensions: [1,3,224,224], Format/Datatype: Half }], Outputs: [ { Name: /features/features.1/Relu_output_0, Location: Device, Dimensions: [1,64,55,55], Format/Datatype: Half }], ParameterType: Convolution, Kernel: [11,11], PaddingMode: kEXPLICIT_ROUND_DOWN, PrePadding: [2,2], PostPadding: [2,2], Stride: [4,4], Dilation: [1,1], OutMaps: 64, Groups: 1, Weights: {"Type": "Half", "Count": 23232}, Bias: {"Type": "Half", "Count": 64}, HasBias: 1, HasReLU: 1, HasSparseWeights: 0, HasDynamicFilter: 0, HasDynamicBias: 0, HasResidual: 0, ConvXAsActInputIdx: -1, BiasAsActInputIdx: -1, ResAsActInputIdx: -1, Activation: RELU, TacticName: sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x32x64_stage1_warpsize4x1x1_g1_tensor16x8x16_aligna8_alignc8, TacticValue: 0x280b0ad3c23d8442, StreamId: 0, Metadata: [ONNX Layer: /features/features.0/Conv][ONNX Layer: /features/features.1/Relu]
...more
Name: /classifier/classifier_6/Gemm_myl9_4, LayerType: gemm, Inputs: [ { Name: output'.1_4, Dimensions: [1,4096], Format/Datatype: Half }, { Name: __mye595_dconst, Dimensions: [4096,1000], Format/Datatype: Half }, { Name: __mye574/classifier/classifier_6/Gemm_alpha, Dimensions: [1], Format/Datatype: Float }, { Name: __mye575/classifier/classifier_6/Gemm_beta, Dimensions: [1], Format/Datatype: Float }, { Name: classifier_6_bias _ ONNXTRT_Broadcast_6_constantHalf, Dimensions: [1,1000], Format/Datatype: Half }], Outputs: [ { Name: output, Dimensions: [1,1000], Format/Datatype: Half }], TacticName: sm50_xmma_cublas_gemvx_f16f16_f32_f32_tn_n_int32_unit_n_launch_param4x32x32_strided_unit_stride, StreamId: 0, Metadata: [ONNX Layer: /classifier/classifier.6/Gemm]
```

The output of layers is saved into a `*.graph.json` file under `results/<model-name>` folder. An example of alexnet layers is saved in [alexnet_fp16.graph.json](desktop/torch_onnx_trtexec/results/alexnet/alexnet_fp16.graph.json) file.

TensorRT engine for alexnet is built in approximately 9 seconds.

```txt
[11/01/2024-10:32:57] [I] Engine built in 8.69986 sec.
[11/01/2024-10:32:57] [I] Created engine with size: 117.087 MiB
```

There are few logs related to performance which will look over in inference phase. By default, `trtexec` warms up for at least 200 ms and runs inference for at least 10 iterations or at least 3 seconds, whichever is longer for profiling the built engine.

## Inference phase

In inference phase, we use the built TensorRT engine to run a profile the layer-wises latency and measure the performance of TensorRT model using `trtexec` CLI.

In our code, the command used in [trt_utils.py](../trt_utils.py) `profile_engine_cmd` function is the following

```bash
trtexec --verbose --noDataTransfers --useCudaGraph --warmUp=5000 \
    --iterations=100 --separateProfileRun --useSpinWait \
    --dumpLayerInfo --dumpProfile \ 
    --loadEngine=models/alexnet/alexnet_fp16.engine \
    --exportTimes=results/alexnet/alexnet_fp16.timing.json \
    --exportProfile=results/alexnet/alexnet_fp16.profile.json \
    --exportLayerInfo=results/alexnet/alexnet_fp16.graph.json \
    --timingCacheFile=/tmp/alexnet_engine_cache \
    --profilingVerbosity=detailed \
    --fp16
```

A comprehensive list of all flags used in inference phase along with explaination for the flag can be found at `Flags for the Inference Phase` section of the TensorRT documentation : <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-flags>

Let's take a look at the logs emitted while profiling TensorRT engine for Alexnet model as example for following analysis. These logs are saved under `models/<model-name>` directory for each model with filename ending with `*.engine.profile.log`.

Here is an example output for Alexnet TensorRT model after running this `trtexec` command:

```txt
[11/01/2024-10:33:09] [I] === Performance summary ===
[11/01/2024-10:33:09] [I] Throughput: 1995.29 qps
[11/01/2024-10:33:09] [I] Latency: min = 0.491211 ms, max = 0.506836 ms, mean = 0.498791 ms, median = 0.499023 ms, percentile(90%) = 0.501465 ms, percentile(95%) = 0.501953 ms, percentile(99%) = 0.503418 ms
[11/01/2024-10:33:09] [I] Enqueue Time: min = 0.000976562 ms, max = 0.0112305 ms, mean = 0.00168485 ms, median = 0.00146484 ms, percentile(90%) = 0.00195312 ms, percentile(95%) = 0.00244141 ms, percentile(99%) = 0.00390625 ms
[11/01/2024-10:33:09] [I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
[11/01/2024-10:33:09] [I] GPU Compute Time: min = 0.491211 ms, max = 0.506836 ms, mean = 0.498791 ms, median = 0.499023 ms, percentile(90%) = 0.501465 ms, percentile(95%) = 0.501953 ms, percentile(99%) = 0.503418 ms
[11/01/2024-10:33:09] [I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
[11/01/2024-10:33:09] [I] Total Host Walltime: 3.00106 s
[11/01/2024-10:33:09] [I] Total GPU Compute Time: 2.98676 s
[11/01/2024-10:33:09] [I] Explanations of the performance metrics are printed in the verbose logs.
```

The Throughput is **1995.29 inferences per second**, and the median Latency is **0.499023 ms** for Alexnet TensorRT model.

A helpful explaination for the performance metrics is also printed.

```txt
[11/01/2024-10:33:09] [V] === Explanations of the performance metrics ===
[11/01/2024-10:33:09] [V] Total Host Walltime: the host walltime from when the first query (after warmups) is enqueued to when the last query is completed.
[11/01/2024-10:33:09] [V] GPU Compute Time: the GPU latency to execute the kernels for a query.
[11/01/2024-10:33:09] [V] Total GPU Compute Time: the summation of the GPU Compute Time of all the queries. If this is significantly shorter than Total Host Walltime, the GPU may be under-utilized because of host-side overheads or data transfers.
[11/01/2024-10:33:09] [V] Throughput: the observed throughput computed by dividing the number of queries by the Total Host Walltime. If this is significantly lower than the reciprocal of GPU Compute Time, the GPU may be under-utilized because of host-side overheads or data transfers.
[11/01/2024-10:33:09] [V] Enqueue Time: the host latency to enqueue a query. If this is longer than GPU Compute Time, the GPU may be under-utilized.
[11/01/2024-10:33:09] [V] H2D Latency: the latency for host-to-device data transfers for input tensors of a single query.
[11/01/2024-10:33:09] [V] D2H Latency: the latency for device-to-host data transfers for output tensors of a single query.
[11/01/2024-10:33:09] [V] Latency: the summation of H2D Latency, GPU Compute Time, and D2H Latency. This is the latency to infer a single query.
```

It provides insights if we are underutilizing the GPU resources and where the bottlenecks for performance might be.

## Bonus

### trex

### modelopt
