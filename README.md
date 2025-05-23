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
    ```
    cd ComfyUI
    python3 main.py --cpu --use-pytorch-cross-attention
    ```
2. Prepare a standard workflow in ComfyUI.
   
    ![Step 1](https://github.com/user-attachments/assets/30137084-242b-48ef-8713-fd999168c070)

3. Add OpenVINO Node.
   
    ![Step 2](https://github.com/user-attachments/assets/0f9f2841-536b-4e05-8388-49ad219efefd)

4. Connect OpenVINO Node with Model/LoRa Loader.
   
    ![Step 3](https://github.com/user-attachments/assets/51d4de0f-c4d2-4e3a-9eb1-3942ef9354ca)

5. Run workflow. Please notice it may need an additional warm-up inference after switching new model.
   
    ![Step 4](https://github.com/user-attachments/assets/37a354f2-86eb-4d2a-8ddc-6fc31439ad08)