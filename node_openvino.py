import torch
import openvino as ov
from typing_extensions import override
import openvino.frontend.pytorch.torchdynamo.execute as ov_ex
from comfy_api.latest import ComfyExtension, io
from comfy_api.torch_helpers import set_torch_compile_wrapper

class TorchCompileModelOpenVINO(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        core = ov.Core()
        available_devices = core.available_devices
        return io.Schema(
            node_id="OpenVINO_TorchCompileModel",
            category="OpenVINO",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "device",
                    options=available_devices,
                ),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, device) -> io.NodeOutput:
        torch._dynamo.reset()
        ov_ex.compiled_cache.clear()
        ov_ex.req_cache.clear()
        ov_ex.partitioned_modules.clear()
        m = model.clone()
        set_torch_compile_wrapper(model=m, backend="openvino", options={"device": device})
        return io.NodeOutput(m)

class OpenVINOTorchCompileExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TorchCompileModelOpenVINO,
        ]


async def comfy_entrypoint() -> OpenVINOTorchCompileExtension:
    return OpenVINOTorchCompileExtension()

NODE_CLASS_MAPPINGS = {
    "OpenVINO_TorchCompileModel": TorchCompileModelOpenVINO,
}
