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

### 分发时怎么办

| 你的分发方式 | 建议 |
|---|---|
| 只自己用 / 内部用，不对外发布 | 无额外义务 |
| 开源发布，主程序也选开源协议 | 在 README 里保留本文件与 NudeNet 的署名即可 |
| 闭源 / 收费分发 | 见下 |

闭源分发时的三条正路：

1. **把模型应用层做成可替换的独立模块**：该模块单独以 AGPL-3.0 提供源码（一个文件即可），
   主程序通过本地 HTTP / 子进程调用它（现有 `/infer` 接口天然就是这层边界），
   两部分的许可互不传染；
2. **换成宽松许可的检测器**：YOLOX、NanoDet、PP-YOLOE、RT-DETR、MMDetection、Detectron2 等
   （Apache-2.0 / MIT），或在自有数据上自训权重 —— 换掉后整包即可闭源；
3. **不打包模型**：程序只提供"用户自备 ONNX 权重"的加载能力，权重与许可责任交给用户。

以上为工程与许可结构建议，具体法律问题请自行咨询专业人士。
