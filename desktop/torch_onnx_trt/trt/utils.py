import os
import tensorrt as trt


def setup_timing_cache(config: trt.IBuilderConfig, timing_cache_path: str):
    """Sets up the builder to use the timing cache file, and creates it if it does not already exist.

    Args:
        config: Config used by trt engine builder
        timing_cache_path: Path to save timing cache
    """
    buffer = b""
    if os.path.exists(timing_cache_path):
        with open(timing_cache_path, mode="rb") as timing_cache_file:
            buffer = timing_cache_file.read()
    timing_cache: trt.ITimingCache = config.create_timing_cache(buffer)
    config.set_timing_cache(timing_cache, True)


def save_timing_cache(config: trt.IBuilderConfig, timing_cache_path: str):
    """Saves the config's timing cache to file.

    Args:
        config: Config used by trt engine builder
        timing_cache_path: Path to save timing cache
    """
    timing_cache: trt.ITimingCache = config.get_timing_cache()
    with open(timing_cache_path, "wb") as timing_cache_file:
        timing_cache_file.write(memoryview(timing_cache.serialize()))
