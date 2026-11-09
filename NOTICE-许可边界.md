# 第三方许可与分发边界

本仓库**自己的代码**按根目录 `LICENSE`（MIT）授权。
但下面这些依赖带有更强的许可条款，**分发（尤其是对外提供网络服务）前必须逐条确认**。

## 1. NudeNet —— AGPL-3.0（最需要注意的一条）

- 组件：`models/640m.onnx` 等权重 + 打码推理管线（`app/auto_censor.py`）
- 许可：GNU Affero General Public License v3.0
- 含义：
  - **自己用**（本机跑、团队内部用）——不受限制
  - **对外提供服务**（哪怕只是网页接口）——整个衍生作品需按 AGPL 开源
  - **对外分发软件**——同样需要开源，或替换该组件
- 三条可选路线：
  1. 保持 AGPL，把本项目的打码部分也按 AGPL 发布
  2. 换成商业友好许可的模型（自训 ONNX / 购买授权）
  3. 只做本地离线工具售卖，不提供网络服务、不捆包分发

## 2. 其余依赖（均宽松许可，可商用）

| 组件 | 许可 |
|---|---|
| onnxruntime | MIT |
| opencv-python | Apache-2.0 |
| Pillow | HPND |
| numpy | BSD-3-Clause |
| tokenizers | Apache-2.0 |
| imageio-ffmpeg | BSD-2-Clause |
| huggingface_hub | Apache-2.0 |

详见 `THIRD_PARTY.md`。

## 3. 模型权重的分发

`models/*.onnx` 不进 git（见 `.gitignore`）。若对外发布安装包，需同时附上对应权重各自的许可文本与来源说明。

---

如需正式对外发布，建议在发布前把本文件更新为结论版（选定路线 + 附录许可全文）。
