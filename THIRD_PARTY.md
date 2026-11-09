# 第三方组件说明

本文件只描述**本软件使用的外部组件**。主程序（界面、后端、工作区、AI 编辑、手绘引擎、
批量水印、封面、排序、账号与多语言、导出与缓存）为本项目原创实现。

---

## 1. 运行库

| 组件 | 用途 | 许可 |
|---|---|---|
| ONNX Runtime (`onnxruntime`) | 加载并执行 ONNX 模型推理 | MIT |
| OpenCV (`opencv-python`) | 图像解码、缩放、打码渲染 | Apache-2.0 |
| Pillow (`Pillow`) | 图片编解码、缩略图、PNG 导出 | HPND (MIT-CMU) |
| NumPy (`numpy`) | 数组与矩阵运算 | BSD-3-Clause |
| `tokenizers` | YOLO-World 自定义类别的文本编码 | Apache-2.0 |
| `imageio-ffmpeg`（可选） | 视频/GIF 转码 | BSD-2-Clause |

以上组件均按各自许可证使用，未做修改。

---

## 2. 打码模型应用部分（唯一需要单独说明的部分）

| 项 | 内容 |
|---|---|
| 模型 | **NudeNet**（YOLOv8 架构，ONNX 导出） |
| 权重 | `models/640m.onnx`（主）、`models/320n.onnx`（回退） |
| 来源 | https://github.com/notAI-tech/NudeNet |
| 许可 | **AGPL-3.0**（模型代码与权重版权归 notAI-tech 作者） |

本软件对这部分**只做应用层调用**：

- 用 ONNX Runtime 加载权重、按官方训练管线预处理（`cvtColor(RGBA2BGR)` + `swapRB=True`）、
  解析输出张量、把检测框交给自研的打码渲染器（马赛克 / 模糊 / 纯色 / 图片笔刷）；
- 未修改网络结构、未重新训练、未改动其推理算法；
- 权重文件不入库，首次启动按需下载（`app/censor_core.py` 的 `MODEL_SOURCES`）。

### 本项目的许可选择（已定）

本仓库自本版本起以 **AGPL-3.0-or-later** 发布（见 `LICENSE`）。这个选择让 NudeNet 的问题一次消失：

- NudeNet 权重与推理管线是 AGPL-3.0，本项目也是 AGPL-3.0，**两者兼容**，不需要隔离、不需要替换、不需要另买授权；
- 分发（含收费的整合包）时，只要把**对应版本的完整源码**一起给出去即可；
- 对外开 WebUI 提供服务时，页脚已放「取源码 / 对应版本」入口，满足 AGPL 第 13 条；
- 改动过的部分必须同样以 AGPL-3.0 公开，不能只公开原版。

### 如果将来要闭源卖

只有两条路，且都要动代码：

1. **换检测器**：YOLOX / RT-DETR / PP-YOLOE / PaddleDetection（Apache-2.0）或自训权重 —— 换完才能整包闭源；
   （注意：不要用 Ultralytics YOLOv8/v11 训练，那条线也是 AGPL）
2. **切分进程边界**：把模型应用层单独做成 AGPL 子进程（现有 `/infer` 就是这层边界），主程序通过 HTTP 调用，
   主程序才可另行闭源；但要注意「AGPL 子进程 + 闭源主程序」在分发时仍需给出子进程源码。

以上为工程与许可结构说明，正式商用前建议让律师过一遍服务条款与源码交付方式。
