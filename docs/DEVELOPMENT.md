# 开发文档

给接手这个项目的人看的：它由哪些部分组成、请求怎么走、数据放在哪、改代码时哪些地方不能碰。

---

## 1. 它是什么

一个本地跑的图片/视频打码工作站。核心能力是"检测出敏感部位 → 按配置遮住 → 批量出图"，
外面套了一层工作区管理：导入、分组启用、排序改名、水印、封面、打包导出，外加账号与多语言。

运行形态：**单机 Python 进程 + 浏览器前端**。没有外部服务、没有数据库，所有状态都是
`cache/` 下的 JSON 与文件。

---

## 2. 目录结构

```
app/
  webui.py          网页服务 + 全部前端（HTML/CSS/JS 都在这一个文件的 INDEX_HTML 里）
  censor_core.py    模型加载、检测、遮挡渲染、配置归一化
  yolo_world.py     开放词汇检测（自定义类别，可选）
  media_core.py     GIF / 视频逐帧处理
  auto_censor.py    命令行打码工具
  check_deps.py     环境自检 + 缺件自动安装
  selftest.py       端到端自测（单元 + HTTP + 工作区 + CLI + 媒体）
tools/
  start.ps1         启动脚本（挑 Python、装依赖、开服务）
models/             模型权重（首次启动自动下载）
cache/              运行期数据，可整个删掉重来
docs/               文档
```

`webui.py` 是唯一的前后端入口，**不拆分**是刻意的：改一个交互只需要动一个文件，
前端资源不走构建链，打开就是源码。代价是这个文件很大（当前约 4600 行），
所以下面会讲清楚它内部的区块顺序。

---

## 3. 启动链路

```
启动打码工作台.bat
  → tools/start.ps1
      → 找 Python（.venv 优先，其次系统 python）
      → 缺依赖时调用 app/check_deps.py
      → python app/webui.py 8080 [--lan]
```

`webui.py` 的 `__main__` 只做参数解析与 `serve_forever`；真正的初始化发生在模块导入时：
`Detector` 实例、目录常量、锁对象都在顶层建好，所以 `import webui` 就等于把服务准备好了
（`selftest.py` 正是靠这一点在进程内起服务）。

端口默认 8080；`--lan` 时绑 `0.0.0.0` 并在界面里给出局域网访问地址。

---

## 4. 后端

### 4.1 路由总览

`Handler` 继承 `BaseHTTPRequestHandler`，`do_GET` / `do_POST` 里用一串 `if parsed.path == ...`
分发。顺序很重要：**鉴权闸门在分发之前**，只有 `/`、`/infer`、`/api/auth/*` 免登录。

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/` | 返回 `INDEX_HTML`（`Cache-Control: no-store`） |
| GET | `/api/auth/state` | 是否已注册 / 当前是否已登录 / 当前用户 |
| GET | `/api/workspace` | 工作区元信息：目录列表、已启用、顺序、是否排过序 |
| GET | `/api/wsfiles?dirs=a\|b` | 按启用目录递归列出图片 |
| GET | `/api/fs`、`/api/wsbrowse` | 本地目录浏览（仓库页） |
| GET | `/api/thumb`、`/api/file`、`/asset/<id>` | 缩略图、原图、遮挡图资源 |
| GET | `/video/status`、`/video/result` | 媒体任务进度与结果 |
| POST | `/api/auth/register`、`login`、`logout`、`lang`、`pwd` | 账号 |
| POST | `/api/mirror` | 把选中的仓库图片复制进工作区批次 |
| POST | `/api/workspace/state` | 持久化 enabled / order / sorted |
| POST | `/api/workspace/put` | 写入一张新图（封面工具用） |
| POST | `/api/workspace/sort-rename` | 按给定顺序在磁盘上改成 `名字_01` |
| POST | `/api/workspace/rename-file` | 显式改名（批量） |
| POST | `/api/workspace/delete-files` / `restore` | 软删除进回收站 / 按 token 还原 |
| POST | `/api/put` | 暂存一张待导出图，返回 id |
| POST | `/api/export` | 把暂存的图按给定顺序打包成 zip |
| POST | `/infer` | 单图推理，返回 base64 结果 |
| POST | `/asset` | 上传遮挡图 |
| POST | `/video` | 提交 GIF / 视频任务 |

### 4.2 鉴权

- 账号：`cache/auth/users.json`，密码 `pbkdf2-hmac-sha256`（12 万轮 + 随机盐），只存散列。
- 会话：`cache/auth/sessions.json`，token 是 `secrets.token_urlsafe`。
- 传输：**Cookie**（`cb_token`，`Path=/`，`SameSite=Lax`，一年）。用 Cookie 而不是自定义头，
  是因为 `<img src="/api/thumb?...">`、`/asset/*` 这类浏览器自动发起的请求带不了自定义头；
  同时 `api()` 也会附带 `X-Auth-Token`，方便脚本调用。
- 注册必须带注册码（`webui.REG_CODE`），第一个账号也一样；退出登录清 cookie。

### 4.3 工作区

- 物理位置：`cache/mirror/<workspace-id>/<批次名>/<图片>`。
- 元信息：`cache/mirror/workspace.json`，字段 `id / name / created / enabled / order / sorted`。
  `enabled` 是"启用中的批次目录名"，`order` 是全局顺序（相对路径），`sorted` 表示最近是否确认过排序。
- 导入：`/api/mirror` 复制文件进新批次（母本不动），并把该批次自动加进 `enabled`、清空 `order/sorted`。
- 列表：`/api/wsfiles` 用 `os.walk` 递归，返回的 `rel` 一定带批次前缀（`批次甲/a1.png`），
  前端拿它当稳定主键。
- **排序导出规则**：`sorted=true` 时所有批次压平成一个层级、文件名前带序号；
  否则 zip 内保留批次子目录。

### 4.4 排序改名

`_ws_sort_rename(ws, order)` 做两阶段改名：先把所有目标改成 `.sort_tmp_%04d`，
再改成 `stem_%0Nd`，`N = max(2, len(str(总数)))`。两阶段是为了避免 A→B、B→A 这种环。
已有 `_\d{2,4}` 尾巴会先剥掉，重名自动加 `_k` 后缀。

`_ws_delete_files` / `_ws_restore` 用回收站实现：文件移到 `cache/trash/<token>/files/`，
同目录写 `manifest.json`；还原按 manifest 搬回去。界面的"撤回"就是拿 token 打这个接口。

### 4.5 导出

`/api/put` 把浏览器合成好的 PNG 落到 `cache/out/<uuid>.png` 并登记到内存表 `OUT_FILES`
（带 LRU 上限，超了删老文件）。`/api/export` 按 `files[{id,name}]` 的顺序写 zip，
写完在 `finally` 里删掉所有临时 png 与 zip 本体——所以 `cache/out` 平时是空的，属正常。

zip 名字由前端决定（默认取工作区名），响应头同时给 `filename` 与 RFC5987 的 `filename*`。

---

## 5. 前端

全部在 `INDEX_HTML` 这一个 raw 字符串里，顺序是：CSS → 骨架 HTML → `<script>`。

### 5.1 单例状态

```js
store = {
  work,                  // 工作区元信息
  pool:  Map(path→item), // 见过的所有图片
  items: [],             // 当前启用的图片（有顺序）
  enabled: Set,          // 启用的批次目录名
  order: [], sorted,     // 全局顺序与是否已确认
  sel: Set, curId,       // 勾选集合与当前单张
  mode, layout, ...
}
```

`syncItems()` 是唯一的同步入口：拉 `/api/wsfiles` → 按 `enabled` 过滤 → 按 `order` 排序 →
填 `store.items`。切换模式、导入、改名之后都调它，不要在别处手工拼列表。

### 5.2 模式（左侧 rail）

`repo / work / ai / manual / mark / cover / sort`，其中后五个是编辑模式（`EDIT_MODES`）。
`buildPanel()` 按 `store.mode` 决定右侧面板，`pumpPreview()` 按模式决定画哪种预览。

### 5.3 水印（分层）

水印不是"一个属性"，而是**图层数组** `item.wmLayers[]`：

- 调参数时画的是"待定层"，点"固化"才把当前配置 push 进数组；
- 预览时先画所有已固化层，再画待定层（正在编辑某层时跳过那一层，避免双画）；
- 右侧工作台的列表按层渲染，可勾选后批量删除，或点某层"编辑"把它读回参数区；
- 导出时 `composeCanvas()` 逐层重绘，所以改了水印不会影响已经固化的层。

### 5.4 封面工具

复制当前图再叠装饰：文字（标题/副标题 + 预设文案）、滤镜（暖/冷/黑白/鲜艳/胶片/模糊）、
压暗渐变、平台角标（B站/抖音/小红书/快手/微博/YouTube/X/Instagram，纯 Canvas 绘制的色块 + 文字）。
确认后 `POST /api/workspace/put?dir=&name=原名_封面.png`，返回的新条目直接进网格。

### 5.5 排序工具

- 只在"全部"布局可用（单张按钮会被置灰）；
- 卡片 HTML5 拖拽，勾选多张时整组跟随，落点区分"插到前 / 插到后"；
- 右侧还有置顶/上移/下移/置底、批量命名（前缀+起始号+位数）、批量改名、删除勾选；
- 双击卡片文件名可直接改；
- 所有会改变顺序或名字的操作都会 `pushW()` 记一份快照，撤回/重做就是按快照
  重新排 `store.items` 并把磁盘名改回去（`restoreSnap`）。

### 5.6 多语言

词表 `window.__TR__` 用**中文原文当键**：`{ "导出 ZIP": ["导出 ZIP","匯出 ZIP","ZIP 書き出し",...] }`。
`translateNow()` 遍历文本节点，把命中的原文换成目标语言，同时把原文记在 `node.__k` 上，
切回中文时按 `__k` 还原——所以可以来回切而不丢字。新 UI 文案要先在词表里加一条，否则翻译不到。

### 5.7 账号闸门

`bootAuth()` 在页面加载时问 `/api/auth/state`：
未注册 → 注册表单（含注册码）；已注册未登录 → 登录表单；已登录 → 进主界面。
登录后若该账号还没选过语言，先弹语言选择（六面国旗飞进左下角设置按钮），
选完记住，之后不再弹。

---

## 6. 运行期目录

```
cache/mirror/<ws>/<批次>/   工作区图片副本
cache/mirror/workspace.json 工作区元信息
cache/thumbs/               缩略图
cache/out/                  导出临时区（用完即删）
cache/trash/<token>/        软删除回收站
cache/auth/                 账号与会话
cache/backups_webui/        历史版本快照（回滚用，可删）
```

`cache/` 整个删掉不影响母本图片，只是工作区要重新导入。

---

## 7. 改代码时的检查清单

1. 改 `webui.py` 后：`python -c "import ast,io;ast.parse(io.open('app/webui.py',encoding='utf-8').read())"`
2. 前端部分：把 `<script>` 抽出来跑 `node --check`（`selftest.py` 不检查语法，只跑行为）
3. 跑一遍 `python app/selftest.py`，58 项应全绿
4. 版本号三处同步：`webui.py` 的 `APP_VERSION`、`INDEX_HTML` 里的 `<title>`、`README.md` 标题
5. 别在 `do_POST` / `do_GET` 里写 `import xxx` 或 `np = ...` 这种局部名——函数体内出现同名赋值
   会让整个函数里的模块名变局部变量，之前 `quote` 和 `np` 都踩过这个坑

---

## 8. 边界

- **不修改母本**：导入即复制，所有编辑都在 `cache/mirror` 里进行。
- **模型权重不入库**：`models/` 下的 `.onnx` 由 `ensure_model()` 按需下载，仓库里只有代码。
- 遮挡图资源、导出临时文件都有体积上限与清理逻辑，别把内存表当数据库用。
- 打码模型这一层的许可与分发限制见 `THIRD_PARTY.md`。
