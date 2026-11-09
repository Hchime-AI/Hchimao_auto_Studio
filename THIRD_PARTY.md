# 第三方组件说明

本文件只描述**本软件使用的外部组件**。主程序（界面、后端、工作区、AI 编辑、手绘引擎、
批量水印、封面、排序、账号与多语言、导出与缓存）为本项目原创实现。

> **先把结论放这儿**：自有代码用 **PolyForm Small Business 1.0.0**（见 `LICENSE`）；
> 打码引擎 NudeNet 是 **AGPL-3.0**，它和主程序之间的边界决定了你能不能闭源卖，见第 2 节。

---

## 1. 运行库

| 组件 | 用途 | 许可 |
|---|---|---|
| ONNX Runtime (`onnxruntime`) | 加载并执行 ONNX 模型推理 | MIT |
| OpenCV (`opencv-python`) | 图像解码、缩放、打码渲染 | Apache-2.0 |
| Pillow (`Pillow`) | 图片编解码、缩略图、PNG 导出 | HPND (MIT-CMU) |
| NumPy (`numpy`) | 数组与矩阵运算 | BSD-3-Clause |
| `tokenizers` | YOLO-World 自定义类别的文本编码 | Apache-2.0 |
| `imageio-ffmpeg`（可选） | 视频/GIF 转码 | BSD-2-Clause（其自带 ffmpeg 二进制另论，见第 4 节） |

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

### 边界在哪

PolyForm SB 只管**你写的代码**，管不了 NudeNet。分三种情况：

- **只自己用、对外不发包**：没有任何义务，PolyForm SB 怎么用都行；
- **把整合包发给别人**：`censor_core.py` 与 NudeNet 的预处理/后处理在同一个 Python 进程里、
  同一个作品里 → 法律上视为衍生作品，**整个包要按 AGPL-3.0 分发（附对应版本源码）**；
- **对外开 WebUI 收会员费**：AGPL 第 13 条生效，通过网络用它的用户有权拿到那一版的源码。

### 三条出路（想闭源卖就先做一条）

1. **拆成独立进程（改造量最小，推荐）**
   把检测做成单独的小程序/小服务（现在的 `webui.py` 的 `/infer` 已是这个形状）：
   它是 AGPL 的那一半，随包附它的源码；主程序按 PolyForm SB 卖，两边靠 HTTP 或 stdio 通话。
2. **换检测器**
   YOLOX / RT-DETR / PP-YOLOE / PaddleDetection（Apache-2.0），或用自己的数据训一个 —— 换完整包可闭源。
   （**别**换 Ultralytics YOLOv8/v11，那条线也是 AGPL-3.0。）
3. **干脆全开源**
   整个项目按 AGPL-3.0 发布（历史全文留在 `docs/AGPL-3.0.txt`）。一次合规、社区友好，
   代价是竞争对手（含大公司）也能直接拿去用。

> 在 1 或 2 完成之前，**对外发的安装包必须按 AGPL-3.0 处理**：附源码 + 附本文件 + 页脚给源码入口。

---

## 3. 主程序的授权（PolyForm Small Business 1.0.0）要点

- 从业人员 < 100 人、上一财年营收 < 100 万美元的公司或个人：**免费**使用，可商用；
- 超过门槛的公司：需要商业授权（这是变现点）；
- 不得把本软件或改写版当作**竞争性产品**对外提供（含做成同类 SaaS 转售）；
- 修改版必须改名，不得使用「Hchimao」「Hchime-AI」作为产品名。

---

## 4. 其它要注意的

- `imageio-ffmpeg` 自带的 `ffmpeg` 二进制可能是 GPL 构建：对外分发前用 `ffmpeg -L` 确认，
  是 GPL 就换成 LGPL 构建，或改成「用户自备 ffmpeg」；
- 界面用系统字体（微软雅黑等）时不要把字体文件打进安装包；
- 商标不随许可授权。

以上为工程与许可结构说明，正式商用前建议让律师过一遍服务条款与源码交付方式。
