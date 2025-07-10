import torch
import openvino as ov
import openvino.frontend.pytorch.torchdynamo.execute as ov_ex
import os
from comfy_api.torch_helpers import set_torch_compile_wrapper


class TorchCompileModelOpenVINO:
    @classmethod
    def INPUT_TYPES(s):
        core = ov.Core()
        available_devices = core.available_devices
        model_cache_option = ["OFF", "ON"]
        return {
            "required": {
                "model": ("MODEL",),
                "device": (available_devices,),
                "model_cache": (model_cache_option,),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "OpenVINO"
    EXPERIMENTAL = True

    def patch(self, model, device, model_cache):
        model_cache_option = {"OFF": False, "ON": True}
        cache_path = os.path.join(os.getcwd(), "openvino_model_cache")
        options = {
            "device": device,
            "model_caching": model_cache_option[model_cache],
            "cache_dir": cache_path,
        }
        torch._dynamo.reset()
        ov_ex.compiled_cache.clear()
        ov_ex.req_cache.clear()
        ov_ex.partitioned_modules.clear()
        m = model.clone()

        if model_cache_option[model_cache]:
            print(f"Using the model cache from {cache_path}.")

        set_torch_compile_wrapper(
            m,
            backend="openvino",
            options=options,
        )
        return (m,)


# The node ID in NODE_CLASS_MAPPINGS should be globally unique across ComfyUI ecosystem
NODE_CLASS_MAPPINGS = {
    "OpenVINO_TorchCompileModel": TorchCompileModelOpenVINO,
}
