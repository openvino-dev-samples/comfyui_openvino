import torch
import openvino as ov
from typing_extensions import override
from typing import Optional
import openvino.frontend.pytorch.torchdynamo.execute as ov_ex
from comfy_api.latest import ComfyExtension, io
from comfy_api.torch_helpers import set_torch_compile_wrapper

TORCH_COMPILE_KWARGS_VAE = "torch_compile_kwargs_vae"


class VAECompileWrapper:
    """
    VAE compiler wrapper that mirrors set_torch_compile_wrapper
    Compiles high-level encode/decode entry points so pipeline semantics stay intact
    """
    def __init__(self, vae):
        self.vae = vae
        self.first_stage = vae.first_stage_model
        self.compiled_modules = {}
        self.compile_kwargs = {}
        self.is_active = False

        # Store original forward methods
        self.original_encode = None
        self.original_decode = None

    def compile(self, backend: str, options: Optional[dict] = None,
               keys: Optional[list[str]] = None):
        """Compile specified VAE modules"""

        if self.is_active:
            self.remove()

        if keys is None:
            if hasattr(self.first_stage, "taesd_encoder"):
                keys = ["taesd_encoder", "taesd_decoder"]
            else:
                keys = ["encoder", "decoder"]

        compile_kwargs = {
            "backend": backend,
            "options": options,
        }
        compile_kwargs = {k: v for k, v in compile_kwargs.items() if v is not None}

        compiled_any = False

        for key in keys:
            if "encoder" in key and hasattr(self.first_stage, "encode"):
                try:
                    if self.original_encode is None:
                        self.original_encode = self.first_stage.encode
                    compiled_encode = torch.compile(self.original_encode, **compile_kwargs)
                    self.first_stage.encode = compiled_encode
                    self.compiled_modules["encode"] = compiled_encode
                    compiled_any = True
                    print("✅ Successfully compiled VAE.encode")
                except RuntimeError as e:
                    print(f"❌ Failed to compile VAE.encode: {e}")
                    if "encode" not in self.compiled_modules:
                        self.original_encode = None

            if "decoder" in key and hasattr(self.first_stage, "decode"):
                try:
                    if self.original_decode is None:
                        self.original_decode = self.first_stage.decode
                    compiled_decode = torch.compile(self.original_decode, **compile_kwargs)
                    self.first_stage.decode = compiled_decode
                    self.compiled_modules["decode"] = compiled_decode
                    compiled_any = True
                    print("✅ Successfully compiled VAE.decode")
                except RuntimeError as e:
                    print(f"❌ Failed to compile VAE.decode: {e}")
                    if "decode" not in self.compiled_modules:
                        self.original_decode = None

        if compiled_any:
            self.compile_kwargs = compile_kwargs
            self.is_active = True

            # Store into vae_options
            if not hasattr(self.vae, 'vae_options'):
                self.vae.vae_options = {}
            self.vae.vae_options[TORCH_COMPILE_KWARGS_VAE] = compile_kwargs

    def remove(self):
        """Remove compilation wrapper"""
        if not self.is_active:
            return

        if self.original_encode is not None and hasattr(self.first_stage, "encode"):
            self.first_stage.encode = self.original_encode
            self.original_encode = None
        if self.original_decode is not None and hasattr(self.first_stage, "decode"):
            self.first_stage.decode = self.original_decode
            self.original_decode = None

        self.compiled_modules.clear()
        self.compile_kwargs.clear()
        self.is_active = False

        if hasattr(self.vae, 'vae_options') and TORCH_COMPILE_KWARGS_VAE in self.vae.vae_options:
            del self.vae.vae_options[TORCH_COMPILE_KWARGS_VAE]

        print("✅ VAE compilation removed")


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
                io.Combo.Input("device", options=available_devices),
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
            model=m,
            backend="openvino",
            options={"device": device}
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
                io.Combo.Input("device", options=available_devices),
                io.Boolean.Input("compile_encoder", default=True),
                io.Boolean.Input("compile_decoder", default=True),
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

        # Get or create wrapper
        if not hasattr(vae, '_compile_wrapper'):
            vae._compile_wrapper = VAECompileWrapper(vae)

        wrapper = vae._compile_wrapper

        # Otherwise compile as requested
        keys = []
        first_stage = vae.first_stage_model
        has_taesd = hasattr(first_stage, "taesd_encoder")

        if compile_encoder:
            keys.append("taesd_encoder" if has_taesd else "encoder")

        if compile_decoder:
            keys.append("taesd_decoder" if has_taesd else "decoder")

        if keys:
            wrapper.compile(
                backend="openvino",
                options={"device": device},
                keys=keys,
            )

        return io.NodeOutput(vae)


class OpenVINOTorchCompileExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TorchCompileDiffusionOpenVINO,
            TorchCompileVAEOpenVINO,
        ]


async def comfy_entrypoint() -> OpenVINOTorchCompileExtension:
    return OpenVINOTorchCompileExtension()


NODE_CLASS_MAPPINGS = {
    "TorchCompileVAEOpenVINO": TorchCompileVAEOpenVINO,
    "TorchCompileDiffusionOpenVINO": TorchCompileDiffusionOpenVINO,
}
