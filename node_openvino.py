import torch
import openvino as ov
import openvino.frontend.pytorch.torchdynamo.execute as ov_ex

from comfy_api.torch_helpers import set_torch_compile_wrapper


class TorchCompileModelOpenVINO:
    @classmethod
    def INPUT_TYPES(s):
        core = ov.Core()
        available_devices = core.available_devices

        return {
            "required": {
                "model": ("MODEL",),
                "device": (available_devices,),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "OpenVINO"
    EXPERIMENTAL = True

    def patch(self, model, device):
        options = {"device": device}
        torch._dynamo.reset()
        ov_ex.compiled_cache.clear()
        ov_ex.req_cache.clear()
        ov_ex.partitioned_modules.clear()
        m = model.clone()
        set_torch_compile_wrapper(m,
                                  backend="openvino",
                                  options=options,
                                  )
        return (m,)


# The node ID in NODE_CLASS_MAPPINGS should be globally unique across ComfyUI ecosystem
NODE_CLASS_MAPPINGS = {
    "OpenVINO_TorchCompileModel": TorchCompileModelOpenVINO,
}