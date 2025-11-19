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
    Dynamically swaps modules during forward instead of using setattr directly
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
               mode: Optional[str] = None, fullgraph=False, dynamic: Optional[bool] = None,
               keys: Optional[list[str]] = None):
        """Compile specified VAE modules"""

        # Clean previous compilation
        if self.is_active:
            self.remove()

        # Determine keys to compile
        if keys is None:
            keys = []
            if hasattr(self.first_stage, "taesd_encoder"):
                keys = ["taesd_encoder", "taesd_decoder"]
            else:
                keys = ["encoder", "decoder"]

        # Compile arguments
        compile_kwargs = {
            "backend": backend,
            "options": options,
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
        }
        compile_kwargs = {k: v for k, v in compile_kwargs.items() if v is not None}

        # Compile each module
        for key in keys:
            if not hasattr(self.first_stage, key):
                continue

            try:
                original_module = getattr(self.first_stage, key)
                # ✅ Only compile module without setattr
                compiled_module = torch.compile(original_module, **compile_kwargs)
                self.compiled_modules[key] = compiled_module
                print(f"✅ Successfully compiled VAE.{key}")
            except Exception as e:
                print(f"❌ Failed to compile VAE.{key}: {e}")

        if self.compiled_modules:
            self.compile_kwargs = compile_kwargs
            self._wrap_forward_methods()
            self.is_active = True

            # Store into vae_options
            if not hasattr(self.vae, 'vae_options'):
                self.vae.vae_options = {}
            self.vae.vae_options[TORCH_COMPILE_KWARGS_VAE] = compile_kwargs

    def _wrap_forward_methods(self):
        """Wrap encode/decode to use compiled modules at runtime"""

        # Save original methods
        if hasattr(self.first_stage, 'encode'):
            self.original_encode = self.first_stage.encode
            self.first_stage.encode = self._create_encode_wrapper()

        if hasattr(self.first_stage, 'decode'):
            self.original_decode = self.first_stage.decode
            self.first_stage.decode = self._create_decode_wrapper()

    def _create_encode_wrapper(self):
        """Create encode wrapper"""
        def encode_wrapper(x):
            # Determine which encoder to use
            encoder_key = "taesd_encoder" if "taesd_encoder" in self.compiled_modules else "encoder"

            if encoder_key in self.compiled_modules:
                # Temporarily replace encoder
                original_encoder = getattr(self.first_stage, encoder_key)
                try:
                    # ✅ Use compiled encoder
                    compiled_encoder = self.compiled_modules[encoder_key]
                    return compiled_encoder(x)
                except Exception as e:
                    print(f"Compiled encoder execution failed, falling back to original: {e}")
                    return original_encoder(x)
            else:
                # Use original method
                return self.original_encode(x)

        return encode_wrapper

    def _create_decode_wrapper(self):
        """Create decode wrapper"""
        def decode_wrapper(z):
            # Determine which decoder to use
            decoder_key = "taesd_decoder" if "taesd_decoder" in self.compiled_modules else "decoder"

            if decoder_key in self.compiled_modules:
                # Temporarily replace decoder
                original_decoder = getattr(self.first_stage, decoder_key)
                try:
                    # ✅ Use compiled decoder
                    compiled_decoder = self.compiled_modules[decoder_key]
                    return compiled_decoder(z)
                except Exception as e:
                    print(f"Compiled decoder execution failed, falling back to original: {e}")
                    return original_decoder(z)
            else:
                # Use original method
                return self.original_decode(z)

        return decode_wrapper

    def remove(self):
        """Remove compilation wrapper"""
        if not self.is_active:
            return

        # Restore original methods
        if self.original_encode is not None:
            self.first_stage.encode = self.original_encode
        if self.original_decode is not None:
            self.first_stage.decode = self.original_decode

        # Clean up
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
                io.Boolean.Input("remove_compile", default=False,
                               tooltip="Remove VAE compilation"),
            ],
            outputs=[io.Vae.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, vae, device, compile_encoder, compile_decoder, remove_compile) -> io.NodeOutput:
        torch._dynamo.reset()
        ov_ex.compiled_cache.clear()
        ov_ex.req_cache.clear()
        ov_ex.partitioned_modules.clear()

        # Get or create wrapper
        if not hasattr(vae, '_compile_wrapper'):
            vae._compile_wrapper = VAECompileWrapper(vae)

        wrapper = vae._compile_wrapper

        # Remove compilation if requested
        if remove_compile:
            wrapper.remove()
            return io.NodeOutput(vae)

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
