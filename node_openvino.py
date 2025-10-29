import torch
import openvino as ov
from typing_extensions import override
import openvino.frontend.pytorch.torchdynamo.execute as ov_ex
from comfy_api.latest import ComfyExtension, io
from comfy_api.torch_helpers import set_torch_compile_wrapper


class TorchCompileDiffusionOpenVINO(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        core = ov.Core()
        available_devices = core.available_devices
        return io.Schema(
            node_id="TorchCompileDiffusionOpenVINO",
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
        set_torch_compile_wrapper(
            model=m, backend="openvino", options={"device": device}
        )
        return io.NodeOutput(m)


class TorchCompileVAEOpenVINO(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        core = ov.Core()
        available_devices = core.available_devices
        return io.Schema(
            node_id="TorchCompileVAEOpenVINO",
            category="OpenVINO",
            inputs=[
                io.Vae.Input("vae"),
                io.Combo.Input(
                    "device",
                    options=available_devices,
                ),
                io.Boolean.Input(
                    "compile_encoder",
                    default=True,
                ),
                io.Boolean.Input(
                    "compile_decoder",
                    default=True,
                ),
            ],
            outputs=[io.Vae.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, vae, device, compile_encoder, compile_decoder) -> io.NodeOutput:
        torch._dynamo.reset()
        ov_ex.compiled_cache.clear()
        ov_ex.req_cache.clear()
        ov_ex.partitioned_modules.clear()
        if compile_encoder:
            encoder_name = "encoder"
            if hasattr(vae.first_stage_model, "taesd_encoder"):
                encoder_name = "taesd_encoder"

            setattr(
                vae.first_stage_model,
                encoder_name,
                torch.compile(
                    getattr(vae.first_stage_model, encoder_name),
                    backend="openvino",
                    options={"device": device},
                ),
            )
        if compile_decoder:
            decoder_name = "decoder"
            if hasattr(vae.first_stage_model, "taesd_decoder"):
                decoder_name = "taesd_decoder"

            setattr(
                vae.first_stage_model,
                decoder_name,
                torch.compile(
                    getattr(vae.first_stage_model, decoder_name),
                    backend="openvino",
                    options={"device": device},
                ),
            )
        return io.NodeOutput(vae)


class OpenVINOTorchCompileExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [TorchCompileDiffusionOpenVINO, TorchCompileVAEOpenVINO]


async def comfy_entrypoint() -> OpenVINOTorchCompileExtension:
    return OpenVINOTorchCompileExtension()


NODE_CLASS_MAPPINGS = {
    "TorchCompileVAEOpenVINO": TorchCompileVAEOpenVINO,
    "TorchCompileDiffusionOpenVINO": TorchCompileDiffusionOpenVINO,
}
