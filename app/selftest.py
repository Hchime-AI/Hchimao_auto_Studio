"""端到端自测：单元 -> HTTP -> 工作区 -> CLI -> 媒体任务。

运行： python app/selftest.py        （不用先开服务，脚本自己起在随机端口）

自测会在真实目录里创建临时账号、临时工作区、临时图片与媒体文件，
结束时（包括中途抛异常）全部清理并把 cache/auth 与 workspace.json 复原，
不会动你自己的工作区与账号。
"""
import base64
import http.client
import io as _io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)          # 模型路径、cache 目录都相对 app/ 解析

from censor_core import build_cfg, censor_regions, parse_hex_color  # noqa: E402

PASS = FAIL = SKIP = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}   {detail}")


def skip(name, why):
    global SKIP
    SKIP += 1
    print(f"  [SKIP] {name}   {why}")


def banner(text):
    print(f"\n== {text} ==")


# ------------------------------------------------------------------ 造素材

def make_png(w=240, h=180, color=(60, 80, 160)):
    img = np.full((h, w, 3), color, np.uint8)
    cv2.putText(img, "T", (w // 3, int(h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (250, 250, 250), 3)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes(), img


def box_of(img, frac=(0.3, 0.3, 0.6, 0.6)):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = frac
    return [int(w * x1), int(h * y1), int(w * x2), int(h * y2)]


# 1 --------------------------------------------------------------- 参数校验

banner("1. 颜色解析与配置钳位")
check("白色 #ffffff", parse_hex_color("#ffffff") == (255, 255, 255))
check("红色 ff0000", parse_hex_color("ff0000") == (0, 0, 255))
check("三位缩写 #0f0", parse_hex_color("#0f0") == (0, 255, 0))
check("乱码颜色退黑", parse_hex_color("不是颜色") == (0, 0, 0))

check("未知 mode 退 mosaic", build_cfg(mode="???")["mode"] == "mosaic")
check("strength 上限 100", build_cfg(strength=999)["strength"] == 100)
check("strength 下限 1", build_cfg(strength=-20)["strength"] == 1)
check("strength 非数字退 35", build_cfg(strength="abc")["strength"] == 35)
check("margin 上限 60", build_cfg(margin=999)["margin"] == 60)
check("margin 负值归零", build_cfg(margin=-3)["margin"] == 0)
check("conf 钳位到 0.04~0.9",
      0.04 <= build_cfg(conf=-1)["conf"] <= 0.9 and
      0.04 <= build_cfg(conf=5)["conf"] <= 0.9)
check("classes=all 展开", build_cfg(classes=["all"])["classes"] != [])
check("默认类别非空", len(build_cfg()["classes"]) > 0)

# 2 --------------------------------------------------------------- 渲染

banner("2. 四种打码方式在图上留下的痕迹")
raw, base = make_png()
det = [{"class": "FEMALE_GENITALIA_EXPOSED", "score": 0.98, "box": box_of(base)}]
x1, y1, x2, y2 = det[0]["box"]

for mode in ("mosaic", "blur", "solid", "img"):
    img = base.copy()
    cfg = build_cfg(mode=mode, strength=40, margin=10, color="#ff0000")
    stamp = base if mode == "img" else None
    n = censor_regions(img, det, cfg, stamp=stamp)
    inside = float(np.abs(img[y1:y2, x1:x2].astype(int) - base[y1:y2, x1:x2]).mean())
    outside = float(np.abs(img[:5, :5].astype(int) - base[:5, :5]).mean())
    check(f"{mode}: 命中 1 处且只改框内", n == 1 and inside > 1 and outside == 0,
          f"n={n} in={inside:.1f} out={outside:.1f}")

img = base.copy()
check("空检测框不打码",
      censor_regions(img, [], build_cfg()) == 0 and np.array_equal(img, base))
img = base.copy()
check("非目标类别被跳过",
      censor_regions(img, [{"class": "FOOT", "score": 0.9, "box": box_of(base)}],
                     build_cfg()) == 0)

# 3 --------------------------------------------------------------- 服务

banner("3. 启动服务并处理鉴权")
import webui  # noqa: E402  (导入时初始化检测器)

AUTH_DIR = webui.AUTH_DIR
USERS, SESS = Path(webui.USERS_PATH), Path(webui.SESS_PATH)
SNAP = {p: (p.read_bytes() if p.exists() else None) for p in (USERS, SESS)}
WS_JSON = Path(webui.MIRROR_DIR) / "workspace.json"
WS_SNAP = WS_JSON.read_bytes() if WS_JSON.exists() else None
TMP_WS = None

server = ThreadingHTTPServer(("127.0.0.1", 0), webui.Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
PORT = server.server_address[1]
COOKIE = {"v": ""}


def call(method, path, body=None, ctype="application/octet-stream", raw=False):
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
    headers = {}
    if body is not None:
        headers["Content-Type"] = ctype
    if COOKIE["v"]:
        headers["Cookie"] = COOKIE["v"]
    conn.request(method, path, body=body, headers=headers)
    r = conn.getresponse()
    data = r.read()
    cookie = r.getheader("Set-Cookie") or ""
    if cookie.startswith("cb_token="):
        COOKIE["v"] = cookie.split(";")[0]
    conn.close()
    if raw:
        return r.status, data
    try:
        return r.status, json.loads(data or b"{}")
    except Exception:
        return r.status, {"_raw": data[:200]}


try:
    code, html = call("GET", "/", raw=True)
    text = html.decode("utf-8", "replace")
    check("首页可访问且是工作站界面",
          code == 200 and "Hchimao Studio" in text and 'id="rail"' in text,
          f"code={code}")

    code, j = call("GET", "/api/workspace")
    check("未登录访问接口被拒", code == 401 and j.get("need") == "auth", str(j))

    code, j = call("GET", "/api/auth/state")
    check("auth/state 可用", code == 200 and "registered" in j, str(j))

    body = json.dumps({"name": "selftest_bot", "pwd": "selftest",
                       "code": webui.REG_CODE}).encode()
    code, j = call("POST", "/api/auth/register", body, "application/json")
    check("用注册码注册临时账号", code == 200 and j.get("ok") and COOKIE["v"], str(j))

    bad = json.dumps({"name": "selftest_x", "pwd": "1234", "code": "nope"}).encode()
    code, j = call("POST", "/api/auth/register", bad, "application/json")
    check("错误注册码被拒", code >= 400, str(j))

    code, j = call("GET", "/api/workspace")
    check("登录后可读工作区", code == 200 and "dirs" in j, str(j))

    # 4 ----------------------------------------------------------- 推理接口
    banner("4. /infer 推理与输出格式")
    code, j = call("POST", "/infer", raw)
    check("默认参数 200", code == 200 and "censored" in j, str(j)[:120])
    if code == 200:
        out = cv2.imdecode(np.frombuffer(
            base64.b64decode(j["censored"].split(",")[1]), np.uint8), 1)
        check("返回图尺寸一致", out is not None and out.shape[:2] == base.shape[:2])
        check("返回耗时与计数", "elapsed_ms" in j and "censored_count" in j)

    code, j = call("POST", "/infer?mode=solid&color=%23ff0000&strength=90"
                              "&margin=30&conf=0.05&res=320&classes=all", raw)
    check("全参数组合 200", code == 200, str(j)[:120])

    code, j = call("POST", "/infer", b"not-an-image")
    check("坏图返回 400", code == 400 and "error" in j, str(j))

    code, j = call("POST", "/infer?mode=solid&strength=abc&conf=xyz&res=7777", raw)
    check("非法参数被钳位", code == 200, str(j)[:120])

    code, j = call("POST", "/infer?fmt=png", raw)
    check("fmt=png 生效", code == 200 and j["censored"].startswith("data:image/png"))
    code, j = call("POST", "/infer?fmt=bogus&quality=999", raw)
    check("非法 fmt 回退 jpeg",
          code == 200 and j["censored"].startswith("data:image/jpeg"))

    png_buf = _io.BytesIO()
    Image.new("RGB", (16, 16), (0, 200, 0)).save(png_buf, format="PNG")
    code, j = call("POST", "/asset", png_buf.getvalue())
    check("上传遮挡图拿到 id", code == 200 and j.get("id"), str(j))
    aid = j.get("id", "")
    code, j = call("POST", "/infer?mode=img&asset=" + str(aid), raw)
    check("mode=img 可推理", code == 200, str(j)[:120])
    code, j = call("POST", "/infer?mode=img", raw)
    check("mode=img 缺 asset 报 400", code == 400, str(j)[:120])

    # 5 ----------------------------------------------------------- 工作区
    banner("5. 工作区：排序改名 / 删除 / 还原 / 导出")
    TMP_WS = Path(tempfile.mkdtemp(prefix="selftest_ws_",
                                   dir=str(webui.MIRROR_DIR)))
    for batch, names in (("批次甲", ["a1.png", "a2.png"]), ("批次乙", ["b1.png"])):
        d = TMP_WS / batch
        d.mkdir(parents=True, exist_ok=True)
        for nm in names:
            payload, _ = make_png()
            (d / nm).write_bytes(payload)
    rel_ws = os.path.join("cache", "mirror", TMP_WS.name)
    WS_JSON.write_text(json.dumps(
        {"id": TMP_WS.name, "name": "自测工作区", "enabled": ["批次甲", "批次乙"],
         "order": [], "sorted": False}, ensure_ascii=False), encoding="utf-8")

    code, j = call("GET", "/api/workspace")
    check("工作区元信息可读", code == 200 and j.get("name") == "自测工作区", str(j)[:120])

    code, j = call("GET", "/api/wsfiles?dirs=%E6%89%B9%E6%AC%A1%E7%94%B2%7C"
                             "%E6%89%B9%E6%AC%A1%E4%B9%99")
    files = j.get("files", [])
    check("跨批次列出 3 张图", code == 200 and len(files) == 3, str(j)[:160])

    rels = [f["rel"] for f in files]
    code, j = call("POST", "/api/workspace/sort-rename",
                   json.dumps({"order": rels[::-1]}).encode(), "application/json")
    renamed = j.get("renamed", [])
    check("排序改名返回 3 条", code == 200 and len(renamed) == 3, str(j)[:160])
    if renamed:
        check("改名形如 名字_01",
              renamed[0]["name"].endswith("_01.png"), renamed[0]["name"])

    code, j = call("POST", "/api/workspace/rename-file",
                   json.dumps({"items": [{"rel": renamed[0]["rel"],
                                          "name": "封面主图.png"}]}).encode(),
                   "application/json")
    check("单个改名成功", code == 200 and j.get("ok"), str(j)[:120])

    code, j = call("POST", "/api/workspace/delete-files",
                   json.dumps({"rels": [renamed[-1]["rel"]]}).encode(),
                   "application/json")
    token = j.get("token", "")
    check("删除返回回收站 token", code == 200 and token, str(j)[:120])

    left = [p.name for p in (TMP_WS / "批次甲").iterdir()] + \
           [p.name for p in (TMP_WS / "批次乙").iterdir()]
    check("删除后磁盘少一个文件", len(left) == 2, str(left))

    code, j = call("POST", "/api/workspace/restore",
                   json.dumps({"token": token}).encode(), "application/json")
    left = [p.name for p in (TMP_WS / "批次甲").iterdir()] + \
           [p.name for p in (TMP_WS / "批次乙").iterdir()]
    check("还原后文件回来了", code == 200 and len(left) == 3, str(left))

    # 6 ----------------------------------------------------------- 导出
    banner("6. 导出 ZIP")
    code, j = call("GET", "/api/wsfiles?dirs=%E6%89%B9%E6%AC%A1%E7%94%B2%7C"
                             "%E6%89%B9%E6%AC%A1%E4%B9%99")
    files = j.get("files", [])
    payload, _ = make_png(120, 90)
    staged = []
    for idx, f in enumerate(files, 1):
        arc = "%02d_%s" % (idx, str(f["name"]).replace(".png", ".png"))
        code, pj = call("POST", "/api/put?name=" + quote(arc), payload)
        if code == 200 and pj.get("id"):
            staged.append({"id": pj["id"], "name": pj["name"]})
    check("导出前暂存 3 张", len(staged) == 3, str(len(staged)))

    code, data = call("POST", "/api/export",
                      json.dumps({"name": "自测导出.zip",
                                  "files": staged}).encode(),
                      "application/json", raw=True)
    check("导出返回 zip", code == 200 and data[:2] == b"PK", f"code={code} len={len(data)}")
    if data[:2] == b"PK":
        zf = zipfile.ZipFile(_io.BytesIO(data))
        names = zf.namelist()
        check("zip 内文件数一致", len(names) == 3, str(names))

    # 7 ----------------------------------------------------------- CLI
    banner("7. 命令行工具")
    r = subprocess.run([sys.executable, "auto_censor.py", "--list-classes"],
                       capture_output=True, text=True)
    check("--list-classes 正常",
          r.returncode == 0 and "FEMALE_BREAST_EXPOSED" in r.stdout, r.stderr[-200:])

    tdir = Path(tempfile.mkdtemp(prefix="selftest_cli_"))
    src = tdir / "in.png"
    payload, _ = make_png()
    src.write_bytes(payload)
    r = subprocess.run([sys.executable, "auto_censor.py", str(src), "--mode", "solid",
                        "--color", "00ff00", "--margin", "30"],
                       capture_output=True, text=True)
    check("CLI 处理图片成功",
          r.returncode == 0 and "已处理" in r.stdout, (r.stdout + r.stderr)[-300:])
    check("CLI 生成输出文件", (tdir / "in_censored.jpg").exists())

    r = subprocess.run([sys.executable, "auto_censor.py", "--mode", "bad"],
                       capture_output=True, text=True)
    check("CLI 缺图片参数返回帮助",
          r.returncode != 0 and "usage" in (r.stdout + r.stderr).lower())
    shutil.rmtree(tdir, ignore_errors=True)

    # 8 ----------------------------------------------------------- 媒体
    banner("8. GIF / 视频媒体任务")
    frames = [Image.fromarray(np.full((90, 120, 3), 30 + 25 * i, np.uint8))
              for i in range(8)]
    gbuf = _io.BytesIO()
    frames[0].save(gbuf, format="GIF", save_all=True,
                   append_images=frames[1:], duration=70, loop=0)
    code, j = call("POST", "/video?kind=gif", gbuf.getvalue())
    gid = j.get("id", "")
    check("提交 GIF 任务", code == 200 and gid, str(j)[:120])

    st = {}
    for _ in range(120):
        code, st = call("GET", "/video/status?id=" + gid)
        if st.get("st") in ("done", "error"):
            break
        time.sleep(0.5)
    check("GIF 任务完成", st.get("st") == "done", str(st)[:160])

    if st.get("st") == "done":
        code, data = call("GET", "/video/result?id=" + gid, raw=True)
        check("GIF 结果可下载",
              code == 200 and data[:3] == b"GIF", f"code={code} len={len(data)}")

    vdir = Path(tempfile.mkdtemp(prefix="selftest_vid_"))
    vpath = vdir / "in.mp4"
    vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 10, (128, 96))
    for i in range(12):
        vw.write(np.full((96, 128, 3), (20 * i) % 255, np.uint8))
    vw.release()
    code, j = call("POST", "/video?kind=video", vpath.read_bytes())
    vid = j.get("id", "")
    check("提交视频任务", code == 200 and vid, str(j)[:120])
    st = {}
    for _ in range(180):
        code, st = call("GET", "/video/status?id=" + vid)
        if st.get("st") in ("done", "error"):
            break
        time.sleep(0.5)
    check("视频任务完成", st.get("st") == "done", str(st)[:160])
    if st.get("st") == "done":
        code, data = call("GET", "/video/result?id=" + vid, raw=True)
        check("视频结果可下载", code == 200 and len(data) > 1000, f"code={code}")
    shutil.rmtree(vdir, ignore_errors=True)

    code, j = call("POST", "/video?kind=exe", b"x")
    check("非法 kind 被拒", code == 400, str(j)[:120])

finally:
    server.shutdown()
    if TMP_WS is not None:
        shutil.rmtree(TMP_WS, ignore_errors=True)
    if WS_SNAP is not None:
        WS_JSON.write_bytes(WS_SNAP)
    elif WS_JSON.exists():
        WS_JSON.unlink()
    for path, blob in SNAP.items():
        if blob is None:
            if path.exists():
                path.unlink()
        else:
            path.write_bytes(blob)

print(f"\n结果：通过 {PASS}，失败 {FAIL}，跳过 {SKIP}")
sys.exit(1 if FAIL else 0)
