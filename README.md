<div align="center">

# 🔥 OpenVINO Node for ComfyUI 🔥

**Optimize model inference in ComfyUI with [Intel OpenVINO toolkits](https://github.com/openvinotoolkit/openvino)**

![comfyui](https://github.com/user-attachments/assets/875384d2-f737-40ae-86ac-8d75ba1ae224)

---

[Supported Hardware](#-supported-hardware) &bull; [Install](#-install) &bull; [Instruction](#-instruction) &bull; [Q&A](#-qa)

</div>

<br/>

## 💻 Supported Hardware

This node can support running model on Intel CPU, GPU and NPU device. You can find more detailed informantion in [OpenVINO System Requirements](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html).

<br/>

## 🚗 Install

> **Prererquisites** &mdash; Install [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

The recommended installation method is to use the Comfy Registry.

### Comfy Registry

These nodes can be installed via the [Comfy Registry](https://registry.comfy.org/nodes/comfyui-openvino).

```bash
comfy node registry-install comfyui-openvino
```

### ComfyUI-Manager

This node can be installed via ComfyUI-Manager in the UI or via the CLI:

```bash
comfy node install comfyui-openvino
```

### Manual

This node can also be installed manually by copying them into your `custom_nodes` folder and then installing dependencies:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/openvino-dev-samples/comfyui_openvino
cd comfyui_openvino
pip install -r requirements.txt
```

<br/>

## 🚀 Instruction

To trigger OpenVINO Node for ComfyUI, you can follow the example as reference:

**1. Start a ComfyUI server.**

   - lanuch from source:
     ```bash
     cd ComfyUI
     python3 main.py --cpu --use-pytorch-cross-attention
     ```

   - lanuch from comfy-cli:
     ```bash
     comfy launch -- --cpu --use-pytorch-cross-attention
     ```

**2. Prepare a standard workflow in ComfyUI.**

   ![Step 1](https://github.com/user-attachments/assets/b2f7af47-08c3-4734-beca-ee4af596a6d1)

**3. Add OpenVINO Node.**

   ![Step 2](https://github.com/user-attachments/assets/6f485fcb-af62-4c3c-8486-88937eef218b)

**4. Connect `TorchCompileDiffusionOpenVINO` with Diffusion model and `TorchCompileVAEOpenVINO` with VAE model**

   ![Step 3](https://github.com/user-attachments/assets/3414811a-13c0-4643-805b-86e9694e09e6)

**5. Run workflow.** Please notice it may need an additional warm-up inference after switching new model.

   ![Step 4](https://github.com/user-attachments/assets/b8f40c64-47b4-48a8-9c8a-87ccba4650b6)

<br/>

## 🤔 Q&A

<details open>
<summary><b>1. Does it support LoRA loader ?</b></summary>

<br/>

Yes, and you can refer the following picture to add it into a workflow of OpenVINO node.

<img width="2000" height="805" alt="image" src="https://github.com/user-attachments/assets/ea1f7c6b-bbbc-4ee9-b5fd-62675d6aaa4e" />

</details>

---

<details>
<summary><b>2. How to install ComfyUI-OpenVINO via ComfyUI-Manager in the Portable version of ComfyUI?</b></summary>

<br/>

ComfyUI-Manager is **not enabled by default** in the Portable version. Follow the [official guide](https://docs.comfy.org/manager/install#portable-users) to enable it first.

Portable ComfyUI builds may not include Git, which is required for ComfyUI-Manager to work. It is recommended to download [Portable Git](https://git-scm.com/downloads/win) and explicitly set its path in your `.bat` launcher before starting ComfyUI:
```bat
set GIT_PYTHON_GIT_EXECUTABLE=C:\path\to\portable-git\bin\git.exe
```

After installing the extension via the Manager, make sure that the installed dependencies match the versions specified in:
```
ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui-openvino\requirements.txt
```
If newer versions were installed automatically, downgrade them to match. Note that the extension currently requires **torch <=2.6.0 and torchaudio <=2.6.0** specifically; if newer versions are already installed, downgrade them.

</details>

---

<details>
<summary><b>3. <code>RuntimeError: Compiler: 'cl' is not found</code> at higher resolutions — how to fix?</b></summary>

<br/>

At low resolutions (e.g. 512x512) the extension works fine. At higher resolutions, PyTorch's inductor triggers a JIT C++ compilation step that requires MSVC (`cl.exe`).

**Fix:** Install [Visual Studio](https://visualstudio.microsoft.com/) (Community edition is sufficient) or the standalone [MSVC Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). During installation, make sure to select the **"Desktop development with C++"** workload — this is what actually installs `cl.exe` and the required C++ components.
Then activate the MSVC environment in your `.bat` launcher **before** starting ComfyUI:
```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
```
Adjust the path to match your Visual Studio version and edition.

</details>

---

<details>
<summary><b>4. C++ compile error due to a space in the <code>torchinductor</code> cache folder name — how to fix?</b></summary>

<br/>

If your Windows display name (Full Name) contains a space (e.g. `John Doe`), PyTorch will create a temp folder like:
```
C:\Users\John\AppData\Local\Temp\torchinductor_John Doe
```
The space in the path breaks the MSVC compiler.

**Fix:** Override `USERNAME` in your `.bat` launcher before starting ComfyUI:
```bat
set USERNAME=User
```
This only affects the current session and does not change your Windows account. After this, PyTorch will create the temp folder as:
```
C:\Users\John\AppData\Local\Temp\torchinductor_User
```

If your **Windows account folder name** (e.g. `C:\Users\John Doe\`) also contains a space, overriding `USERNAME` alone is not enough — the temp path will still be broken. In that case, redirect the temp directory to a path without spaces:
```bat
set TEMP=C:\tmp
set TMP=C:\tmp
```
Make sure the folder `C:\tmp` exists before launching ComfyUI.

</details>

---

<details>
<summary><b>5. <code>python_embeded</code> is missing Python headers/libraries for C++ compilation — how to fix?</b></summary>

<br/>

Portable ComfyUI uses a stripped-down embedded Python that is missing the `include` and `libs` directories required for C++ compilation.

**Fix:**
1. Download the full Python installer for the **same version** as your `python_embeded`.
2. Copy the `include` and `libs` folders from the full Python installation into your `ComfyUI_windows_portable\python_embeded\` directory.

After this, `cl.exe` will be able to compile successfully.

<details>
<summary>📄 Example <code>run_cpu.bat</code> with all fixes applied</summary>

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set USERNAME=User
set GIT_PYTHON_GIT_EXECUTABLE=C:\Users\John\ComfyUI_windows_portable\git\bin\git.exe
.\python_embeded\python.exe -s ComfyUI\main.py --cpu --windows-standalone-build --enable-manager --use-pytorch-cross-attention
pause
```
</details>

</details>
