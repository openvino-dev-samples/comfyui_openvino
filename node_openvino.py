import torch
import openvino as ov
from typing_extensions import override
from typing import Optional
import openvino.frontend.pytorch.torchdynamo.execute as ov_ex
from comfy_api.latest import ComfyExtension, io
from comfy_api.torch_helpers import set_torch_compile_wrapper

TORCH_COMPILE_KWARGS_VAE = "torch_compile_kwargs_vae"


_dynamo_workarounds_applied = False


def _apply_dynamo_workarounds():
    """Work around two PyTorch 2.10 issues with the OpenVINO torch.compile backend:

    1. TorchDynamo bug in nn_module.py wrap_values() references undefined free variable
       'named_children' instead of 'result'. Triggered when tracing self.parameters()
       in ResBlock.forward(). Fix: monkey-patch ResBlock.forward to skip self.parameters()
       when not checkpointing (params are unused in that branch anyway).

    2. OpenVINO backend falls back to inductor (compile_fx) on any exception, but inductor's
       C++ codegen requires omp.h which may be missing on Windows MSVC setups.
       Fix: replace the inductor fallback with eager execution.
    """
    global _dynamo_workarounds_applied
    if _dynamo_workarounds_applied:
        return
    _dynamo_workarounds_applied = True

    # Fix 1: Patch ResBlock.forward to avoid self.parameters() call
    try:
        from comfy.ldm.modules.diffusionmodules.openaimodel import ResBlock
        from comfy.ldm.modules.diffusionmodules.util import checkpoint

        def _patched_forward(self, x, emb):
            if self.use_checkpoint:
                return checkpoint(
                    self._forward, (x, emb), self.parameters(), self.use_checkpoint
                )
            return self._forward(x, emb)

        ResBlock.forward = _patched_forward
    except (ImportError, AttributeError):
        pass

    # Fix 2: In PyTorch 2.10, make_fx(tracing_mode="fake") crashes with
    # "Cannot call numel() on tensor with symbolic sizes/strides" because
    # PyTorch 2.10's fake tracing creates symbolic-shaped tensors that the
    # C++ linear kernel can't handle. Fix: replace make_fx in the openvino
    # backend's namespace so it uses tracing_mode="symbolic" instead.
    # Also set allow_non_fake_inputs_override to prevent FakeTensor assertions.
    try:
        import openvino.frontend.pytorch.torchdynamo.backend as ov_backend
        from torch._subclasses.fake_tensor import fake_tensor_tls

        _original_make_fx = ov_backend.make_fx

        def _patched_make_fx(*args, **kwargs):
            if kwargs.get("tracing_mode") == "fake":
                kwargs["tracing_mode"] = "symbolic"
            return _original_make_fx(*args, **kwargs)

        ov_backend.make_fx = _patched_make_fx

        _original_fx_openvino = ov_backend.fx_openvino

        def _patched_fx_openvino(subgraph, example_inputs, options=None):
            old_override = fake_tensor_tls.allow_non_fake_inputs_override
            fake_tensor_tls.allow_non_fake_inputs_override = True
            try:
                result = _original_fx_openvino(subgraph, example_inputs, options)
                print(f"[OV-DEBUG] fx_openvino SUCCEEDED for subgraph")
                return result
            except Exception as e:
                print(f"[OV-DEBUG] fx_openvino FAILED: {type(e).__name__}: {e}")
                raise
            finally:
                fake_tensor_tls.allow_non_fake_inputs_override = old_override

        ov_backend.fx_openvino = _patched_fx_openvino

        # Fix 3: When fx_openvino fails, the openvino backend falls back to
        # compile_fx (inductor), which requires omp.h missing on Windows MSVC.
        # Replace compile_fx with eager execution so only the failing subgraphs
        # fall back, while successful ones still run on OpenVINO GPU.
        def _eager_fallback(subgraph, example_inputs):
            import sys, traceback as tb
            exc_info = sys.exc_info()
            if exc_info[1]:
                print(f"[OV-DEBUG] compile_fx fallback → eager. Exception: {exc_info[0].__name__}: {exc_info[1]}")
                tb.print_exc()
            else:
                print(f"[OV-DEBUG] compile_fx fallback → eager (no active exception)")
            return subgraph.forward

        ov_backend.compile_fx = _eager_fallback
    except (ImportError, AttributeError):
        pass

    # Fix 4: PyTorch 2.10 bug in symbolic_shapes.py — produce_guards_verbose()
    # crashes with IndexError when symbol_to_source[symbol] is an empty list
    # (triggered by conv shape guards). This happens AFTER successful backend
    # compilation but BEFORE guards are installed, so it would discard the
    # compiled OpenVINO code. Patch to return empty guards on IndexError,
    # which means "always use this compiled version" (fine for fixed-shape inference).
    try:
        from torch.fx.experimental.symbolic_shapes import ShapeEnv, _ShapeGuardsHelper

        _original_produce_guards_verbose = ShapeEnv.produce_guards_verbose

        def _patched_produce_guards_verbose(self, *args, **kwargs):
            try:
                return _original_produce_guards_verbose(self, *args, **kwargs)
            except IndexError:
                # symbol_to_source[symbol] is empty — return empty guards
                # so that compilation can proceed.
                langs = kwargs.get("langs", ("python", "verbose_python"))
                return [_ShapeGuardsHelper(exprs=[]) for _ in langs]

        ShapeEnv.produce_guards_verbose = _patched_produce_guards_verbose
    except (ImportError, AttributeError):
        pass


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
        _apply_dynamo_workarounds()
        ov_ex.compiled_cache.clear()
        ov_ex.req_cache.clear()
        ov_ex.partitioned_modules.clear()
        ov_ex.max_openvino_partitions = 0

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
        _apply_dynamo_workarounds()
        ov_ex.compiled_cache.clear()
        ov_ex.req_cache.clear()
        ov_ex.partitioned_modules.clear()
        ov_ex.max_openvino_partitions = 0

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
