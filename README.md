# OpenVINO Node for ComfyUI

This node is designed for optimizing the performance of model inference in ComfyUI by leveraging [Intel OpenVINO toolkits](https://github.com/openvinotoolkit/openvino).


--- 

- [OpenVINO Node for ComfyUI](#openvino-node-for-comfyui)
  - [Supported Hardware](#supported-hardware)
  - [Install](#install)
    - [Comfy Registry](#comfy-registry)
    - [ComfyUI-Manager](#comfyui-manager)
    - [Manual](#manual)
  - [Instruction](#instruction)

## Supported Hardware

This node can support running model on Intel CPU, GPU and NPU device.You can find more detailed informantion in [OpenVINO System Requirements](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html).


## Install

**Prererquisites**

- Install [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

The recommended installation method is to use the Comfy Registry.

### Comfy Registry

These nodes can be installed via the [Comfy Registry](https://registry.comfy.org/nodes/comfyui-openvino).

```
comfy node registry-install comfyui-openvino
```

### ComfyUI-Manager

This node can be installed via ComfyUI-Manager in the UI or via the CLI:

```
comfy node install comfyui-openvino
```

### Manual

This node can also be installed manually by copying them into your `custom_nodes` folder and then installing dependencies:

```
cd ComfyUI/custom_nodes
git clone https://github.com/openvino-dev-samples/comfyui_openvino 
cd comfyui_openvino
pip install -r requirements.txt
```

## Instruction
To trigger OpenVINO Node for ComfyUI, you can follow the example as reference:
1. Start a ComfyUI server.
   - lanuch from source:
    ```
    cd ComfyUI
    python3 main.py --cpu --use-pytorch-cross-attention
    ```

   - lanuch from comfy-cli:
    ```
    comfy launch -- --cpu --use-pytorch-cross-attention
    ```
2. Prepare a standard workflow in ComfyUI.
   
    ![Step 1](https://github.com/user-attachments/assets/b2f7af47-08c3-4734-beca-ee4af596a6d1)

3. Add OpenVINO Node.
   
    ![Step 2](https://github.com/user-attachments/assets/6f485fcb-af62-4c3c-8486-88937eef218b)
   
4. Connect `TorchCompileDiffusionOpenVINO` with Diffusion model and `TorchCompileVAEOpenVINO` with VAE model
    ![Step 3](https://github.com/user-attachments/assets/3414811a-13c0-4643-805b-86e9694e09e6)

6. Run workflow. Please notice it may need an additional warm-up inference after switching new model.
   
    ![Step 3](https://github.com/user-attachments/assets/b8f40c64-47b4-48a8-9c8a-87ccba4650b6)
