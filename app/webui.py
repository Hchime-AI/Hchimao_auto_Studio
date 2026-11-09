# 本地网页版自动打码服务（仅使用标准库 + onnxruntime + opencv + Pillow，无新增依赖）
# 启动: python webui.py [端口]  然后浏览器打开 http://localhost:8080
# 模型: 默认 640m（缺失时自动从 hf-mirror 下载，失败回退 320n）
# 前端: AI 模式（批量图片/GIF/视频，手动点击开始）/ 手动模式（画笔编辑器，可叠加 AI 预打码）
import base64
import hashlib
import io
import json
import os
import secrets
import shutil
import socket
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote

APP_NAME = "Hchimao Studio"
APP_VERSION = "1.5.0"

import cv2
import numpy as np

from censor_core import (Detector, LABELS, DEFAULT_CLASSES,
                         ALLOWED_RESOLUTIONS, build_cfg, censor_regions, ensure_model)
from media_core import process_gif, process_video, _find_ffmpeg
from yolo_world import (normalize_world_classes, expand_words,
                        get_world_detector, WORLD_RESOLUTION, WORLD_PREFIX)

ensure_model()
detector = Detector()  # 640m @ 640，每次请求可按参数切换推理分辨率
HAS_FFMPEG = _find_ffmpeg() is not None


def _detect_combined(img, cfg):
    """NudeNet + YOLO-World（可选）合并检测。

    cfg["world_classes"] 为空时与原 detector.detect 行为完全一致（零开销）。
    非空时附加 YOLO-World 检测（同义词扩展 + world_conf 独立阈值）；
    模型缺失/下载失败抛 RuntimeError（由调用方转为任务失败并提示用户）。
    返回 (合并检测结果, 启用的 WORLD 类别标识列表)。
    """
    detector.resolution = cfg["res"]
    detections = detector.detect(img, conf=cfg["conf"])
    world_words = cfg.get("world_classes") or []
    if not world_words:
        return detections, []
    all_words = expand_words(world_words)
    wd = get_world_detector()
    wd.resolution = max(WORLD_RESOLUTION, cfg["res"])   # 高分辨率提升小目标召回
    embeds = wd.get_embeds(all_words)
    world_dets = wd.detect(img, all_words, embeds=embeds,
                           conf=cfg.get("world_conf", 0.15))
    enabled = [WORLD_PREFIX + w for w in all_words]
    return detections + world_dets, enabled

# ---------------- 媒体任务（GIF/视频） ----------------
MEDIA_ROOT = Path(tempfile.mkdtemp(prefix="censor_media_"))
MEDIA_LIMIT = 500 * 1024 * 1024   # 单个上传文件上限
MEDIA_KEEP = 6                    # 最多保留的已完成任务结果数
MEDIA_LOCK = threading.Lock()
MEDIA = {}                        # id -> job dict
MEDIA_MIME = {"gif": "image/gif", "webm": "video/webm", "mp4": "video/mp4"}
CURRENT_PORT = 8080   # 运行时由 __main__ 更新

# ---------------- 仓库 / 缩略图 / 导出 ----------------
try:  # 控制台编码兜底：GBK 终端下也不因 emoji/勾号崩掉
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

APP_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.dirname(APP_DIR)          # 程序根目录（app/ 的上一级）
CACHE_DIR = os.path.join(APP_ROOT, "cache")
MIRROR_DIR = os.path.join(CACHE_DIR, "mirror")
THUMB_DIR = os.path.join(CACHE_DIR, "thumbs")
OUT_DIR = os.path.join(CACHE_DIR, "out")
for _d in (CACHE_DIR, THUMB_DIR, OUT_DIR, MIRROR_DIR):
    try:
        os.makedirs(_d, exist_ok=True)
    except Exception:
        pass

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".avif")
THUMB_RAM = {}            # 内存缩略图缓存
THUMB_RAM_MAX = 300       # 超过后落盘到 _cache/thumbs
THUMB_LOCK = threading.Lock()
OUT_FILES = {}            # id -> {"path":..., "name":...}
OUT_LOCK = threading.Lock()

# ---------------- 账号 / 会话 ----------------
AUTH_DIR = os.path.join(CACHE_DIR, "auth")
USERS_PATH = os.path.join(AUTH_DIR, "users.json")
SESS_PATH = os.path.join(AUTH_DIR, "sessions.json")
TRASH_DIR = os.path.join(CACHE_DIR, "trash")
AUTH_LOCK = threading.Lock()
REG_CODE = "Hchime-ai"            # 注册码（所有新账号都必须填）
SESSION_TTL = 60 * 60 * 24 * 365  # 一年
for _d in (AUTH_DIR, TRASH_DIR):
    try:
        os.makedirs(_d, exist_ok=True)
    except Exception:
        pass


def _auth_load(path, default):
    try:
        with io.open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _auth_save(path, obj):
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _pwd_hash(pwd, salt=None, iters=120000):
    salt = salt or secrets.token_hex(8)
    h = hashlib.pbkdf2_hmac("sha256", (pwd or "").encode("utf-8"),
                            salt.encode("utf-8"), iters)
    return "pbkdf2$%d$%s$%s" % (iters, salt, base64.b64encode(h).decode())


def _pwd_ok(pwd, rec):
    try:
        _, it, salt, want = (rec or "").split("$")
        h = hashlib.pbkdf2_hmac("sha256", (pwd or "").encode("utf-8"),
                                salt.encode("utf-8"), int(it))
        return base64.b64encode(h).decode() == want
    except Exception:
        return False


def _auth_users():
    j = _auth_load(USERS_PATH, {"users": []})
    if not isinstance(j, dict):
        j = {"users": []}
    j.setdefault("users", [])
    return j


def _auth_pub(u):
    return {"name": u.get("name", ""), "lang": u.get("lang", ""),
            "created": u.get("created", 0)}


def _auth_find(name):
    name = (name or "").strip()
    for u in _auth_users()["users"]:
        if u.get("name", "").lower() == name.lower():
            return u
    return None


def _sess_new(uid):
    with AUTH_LOCK:
        s = _auth_load(SESS_PATH, {})
        if not isinstance(s, dict):
            s = {}
        now = time.time()
        s = {k: v for k, v in s.items()
             if isinstance(v, dict) and now - float(v.get("last", 0)) < SESSION_TTL}
        tok = secrets.token_urlsafe(24)
        s[tok] = {"uid": uid, "created": now, "last": now}
        _auth_save(SESS_PATH, s)
    return tok


def _sess_drop(tok):
    if not tok:
        return
    with AUTH_LOCK:
        s = _auth_load(SESS_PATH, {})
        if isinstance(s, dict) and tok in s:
            s.pop(tok, None)
            _auth_save(SESS_PATH, s)


def _auth_user_by_token(tok):
    if not tok:
        return None
    s = _auth_load(SESS_PATH, {})
    rec = s.get(tok) if isinstance(s, dict) else None
    if not isinstance(rec, dict):
        return None
    if time.time() - float(rec.get("last", 0)) > SESSION_TTL:
        return None
    for u in _auth_users()["users"]:
        if u.get("id") == rec.get("uid"):
            return u
    return None


# ---------------- 工作区文件：改名 / 软删 / 还原 ----------------
def _clean_name(name, ext_hint=""):
    """清洗成安全的单层文件名（去目录分隔、保留扩展名）。"""
    name = (name or "").strip().replace("\\", "/").split("/")[-1]
    for ch in '<>:"|?*':
        name = name.replace(ch, "_")
    name = name.strip(" .")
    if not name:
        name = "img"
    if not os.path.splitext(name)[1] and ext_hint:
        name += ext_hint
    return name[:180]


def _ws_rename_files(ws, items):
    """items=[{"rel":..,"name":..}]；同目录内批量改名，两阶段避免互相覆盖。"""
    root = os.path.abspath(ws["path"])
    staged = []
    for it in items or []:
        rel = (it.get("rel") or "").replace("\\", "/")
        full = _ws_safe_rel(ws, rel)
        if not full or not os.path.isfile(full):
            continue
        d = os.path.dirname(full)
        ext = os.path.splitext(full)[1]
        name = _clean_name(it.get("name"), ext)
        if not os.path.splitext(name)[1]:
            name += ext
        staged.append({"rel": rel, "dir": d, "name": name, "tmp": None})
    if not staged:
        return []
    # 第一阶段：全部挪到临时名，避免 a->b 与 b->a 互相踩
    for i, s in enumerate(staged):
        tmp = os.path.join(s["dir"], ".rn_tmp_%04d%s" % (i, os.path.splitext(s["name"])[1]))
        try:
            os.replace(os.path.join(s["dir"], os.path.basename(s["rel"])), tmp)
            s["tmp"] = tmp
        except OSError:
            s["tmp"] = None
    # 第二阶段：落成目标名（同目录重名加 (2)(3)）
    out = []
    used = set()
    for s in staged:
        if not s["tmp"]:
            continue
        dst = os.path.join(s["dir"], s["name"])
        stem, ext = os.path.splitext(s["name"])
        k = 2
        while os.path.normcase(dst) in used or (
                os.path.exists(dst) and os.path.normcase(dst) != os.path.normcase(s["tmp"])):
            dst = os.path.join(s["dir"], "%s(%d)%s" % (stem, k, ext))
            k += 1
        used.add(os.path.normcase(dst))
        try:
            os.replace(s["tmp"], dst)
        except OSError:
            continue
        rel_new = os.path.relpath(dst, root).replace(os.sep, "/")
        out.append({"from": s["rel"], "rel": rel_new, "name": os.path.basename(dst),
                    "path": dst, "batch": rel_new.split("/")[0] if "/" in rel_new else ""})
    return out


def _ws_delete_files(ws, rels):
    """软删：移到 cache/trash/<token>/files/<rel>，写 manifest，返回 token。"""
    token = uuid.uuid4().hex[:16]
    base = os.path.join(TRASH_DIR, token)
    moved = []
    for rel in rels or []:
        rel = (rel or "").replace("\\", "/")
        full = _ws_safe_rel(ws, rel)
        if not full or not os.path.isfile(full):
            continue
        dst = os.path.join(base, "files", rel.replace("/", os.sep))
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(full, dst)
        except Exception:
            continue
        moved.append({"rel": rel, "name": os.path.basename(rel)})
    if not moved:
        return None, []
    os.makedirs(base, exist_ok=True)
    _auth_save(os.path.join(base, "manifest.json"),
               {"token": token, "ws": ws.get("id", ""), "created": time.time(),
                "items": moved})
    return token, moved


def _ws_restore(token):
    """按 token 把软删的文件放回原处。"""
    token = "".join(ch for ch in (token or "") if ch.isalnum())
    base = os.path.join(TRASH_DIR, token)
    man = _auth_load(os.path.join(base, "manifest.json"), None)
    if not isinstance(man, dict):
        return []
    root = os.path.join(MIRROR_DIR, man.get("ws", ""))
    out = []
    for it in man.get("items", []):
        rel = it.get("rel", "").replace("\\", "/")
        srcf = os.path.join(base, "files", rel.replace("/", os.sep))
        dst = os.path.join(root, rel.replace("/", os.sep))
        if not os.path.isfile(srcf):
            continue
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(srcf, dst)
        except Exception:
            continue
        out.append({"rel": rel, "name": os.path.basename(rel), "path": dst})
    try:
        shutil.rmtree(base, ignore_errors=True)
    except Exception:
        pass
    return out


def _list_dir(path):
    """列出目录：子文件夹 + 图片文件（按名称排序）。"""
    path = os.path.abspath(os.path.expanduser(path or ""))
    if not os.path.isdir(path):
        raise FileNotFoundError("目录不存在：%s" % path)
    dirs, files, imgs = [], [], 0
    try:
        names = sorted(os.listdir(path), key=lambda n: n.lower())
    except PermissionError:
        raise PermissionError("没有权限读取：%s" % path)
    for n in names:
        if n.startswith("."):
            continue
        full = os.path.join(path, n)
        try:
            if os.path.isdir(full):
                dirs.append({"name": n, "path": full})
                continue
            ext = os.path.splitext(n)[1].lower()
            if ext not in IMG_EXTS:
                continue
            st = os.stat(full)
            files.append({"name": n, "path": full, "rel": n,
                          "size": st.st_size, "isImage": True})
            imgs += 1
        except OSError:
            continue
    parent = os.path.dirname(path)
    if parent == path:
        parent = ""
    return {"path": path, "parent": parent, "dirs": dirs,
            "files": files, "images": imgs}


def _img_size(path):
    """快速读图片宽高（优先 PIL 读文件头，失败回退 cv2）。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        pass
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            return int(w), int(h)
    except Exception:
        pass
    return 0, 0


MIRROR_KEEP = 4          # 只保留最近 N 次镜像会话（当前工作区永不清理）
WS_META = os.path.join(MIRROR_DIR, "workspace.json")
WS_LOCK = threading.Lock()


WS_KEYS = ("id", "name", "created", "enabled", "order", "sorted", "pkg")


def _ws_write(d):
    """写入工作区元信息（含启用文件夹 / 排序 / 是否已排序）。"""
    tmp = WS_META + ".tmp"
    payload = {
        "id": d.get("id"),
        "name": d.get("name") or "",
        "created": d.get("created") or time.time(),
        "enabled": [str(x) for x in (d.get("enabled") or [])],
        "order": [str(x) for x in (d.get("order") or [])],
        "sorted": bool(d.get("sorted")),
        "pkg": str(d.get("pkg") or ""),
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, WS_META)


def _ws_name_ok(name):
    import re as _re
    name = _re.sub(r'[\\/:*?"<>|\r\n\t]', "_", (name or "").strip())[:64].strip(" .")
    return name


def _ws_read():
    """当前编辑工作区（= 一个镜像会话目录）；没有就新建一个。"""
    d = {}
    try:
        with open(WS_META, encoding="utf-8") as f:
            d = json.load(f) or {}
    except Exception:
        d = {}
    sid = d.get("id") or ""
    path = os.path.join(MIRROR_DIR, sid) if sid else ""
    if not sid or not os.path.isdir(path):
        sid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        path = os.path.join(MIRROR_DIR, sid)
        os.makedirs(path, exist_ok=True)
        d = {"id": sid, "name": d.get("name") or "", "created": time.time()}
        _ws_write(d)
    d["path"] = path
    d.setdefault("enabled", [])
    d.setdefault("order", [])
    d.setdefault("sorted", False)
    try:
        names = set(os.listdir(path))
    except OSError:
        names = set()
    d["enabled"] = [n for n in d["enabled"] if n in names and os.path.isdir(os.path.join(path, n))]
    return d


def _ws_stats(d):
    n = 0
    size = 0
    for root, _dirs, files in os.walk(d["path"]):
        for fn in files:
            try:
                size += os.path.getsize(os.path.join(root, fn))
                n += 1
            except OSError:
                pass
    return n, size


def _mirror_files(paths, root_hint=""):
    """把选中的图片复制到 cache/mirror/<会话>/ 下，返回镜像副本信息。"""
    with WS_LOCK:
        ws = _ws_read()
        if not (ws.get("name") or "").strip():
            root_name = os.path.basename(os.path.abspath(root_hint)) if root_hint else ""
            ws["name"] = _ws_name_ok(root_name) or ("工作区-" + ws["id"][-6:])
            _ws_write(ws)
    dst_root = ws["path"]
    os.makedirs(dst_root, exist_ok=True)
    # 每次导入都放进一个批次文件夹，工作区根目录只显示文件夹
    batch = ""
    if root_hint:
        batch = _ws_name_ok(os.path.basename(os.path.abspath(root_hint)))
    if not batch:
        batch = "导入-" + time.strftime("%Y%m%d-%H%M%S")
    bdst = os.path.join(dst_root, batch)
    if os.path.isdir(bdst) and os.listdir(bdst):
        k = 2
        while os.path.isdir(os.path.join(dst_root, "%s (%d)" % (batch, k))):
            k += 1
        bdst = os.path.join(dst_root, "%s (%d)" % (batch, k))
    dst_root = bdst
    batch_final = os.path.basename(bdst)
    os.makedirs(dst_root, exist_ok=True)
    base = os.path.abspath(root_hint) if root_hint else ""
    out = []
    for src in paths:
        try:
            src = os.path.abspath(src)
            if not os.path.isfile(src):
                continue
            if base and os.path.normcase(src).startswith(os.path.normcase(base)):
                rel = os.path.relpath(src, base)
            else:
                rel = os.path.basename(src)
            rel = rel.replace("..", "_")
            dst = os.path.join(dst_root, rel)
            d = os.path.dirname(dst)
            if d:
                os.makedirs(d, exist_ok=True)
            shutil.copy2(src, dst)
            w, h = _img_size(dst)
            out.append({"name": os.path.basename(src),
                        "rel": batch_final + "/" + rel.replace("\\", "/"),
                        "path": dst, "src": src, "w": w, "h": h, "batch": batch_final,
                        "thumb": "/api/thumb?w=420&path=" + quote(dst)})
        except Exception:
            continue
    # 只保留最近几次镜像，避免缓存无限膨胀
    try:
        cur = os.path.basename(ws["path"])
        sessions = sorted([d for d in os.listdir(MIRROR_DIR)
                           if os.path.isdir(os.path.join(MIRROR_DIR, d)) and d != cur])
        for old in sessions[:max(0, len(sessions) - (MIRROR_KEEP - 1))]:
            shutil.rmtree(os.path.join(MIRROR_DIR, old), ignore_errors=True)
    except Exception:
        pass
    with WS_LOCK:
        cur = _ws_read()
        en = [n for n in (cur.get("enabled") or []) if n != batch_final]
        en.append(batch_final)
        cur["enabled"] = en
        cur["sorted"] = False          # 新批次进来，之前的排序失效
        cur["order"] = []
        _ws_write(cur)
    return {"dir": dst_root, "items": out, "batch": batch_final,
            "workspace": ws.get("name") or ""}


def _ws_dirs(ws):
    """工作区根目录下的批次文件夹（只有这些可以被"启用"）。"""
    root = ws["path"]
    out = []
    try:
        for n in sorted(os.listdir(root), key=lambda x: x.lower()):
            if n.startswith("."):
                continue
            full = os.path.join(root, n)
            if os.path.isdir(full):
                out.append(n)
    except OSError:
        pass
    return out


def _ws_items(ws, names=None):
    """列出（启用的）文件夹里的图片；rel 统一带批次前缀，如 `批次A/01.jpg`。"""
    root = ws["path"]
    todo = list(names) if names else _ws_dirs(ws)
    out = []
    for batch in todo:
        bdir = os.path.join(root, batch)
        if not os.path.isdir(bdir):
            continue
        for dirpath, dirnames, files in os.walk(bdir):
            dirnames.sort(key=lambda x: x.lower())
            for fn in sorted(files, key=lambda x: x.lower()):
                if os.path.splitext(fn)[1].lower() not in IMG_EXTS:
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    rel_in = os.path.relpath(full, bdir).replace(os.sep, "/")
                except Exception:
                    rel_in = fn
                rel = batch + "/" + rel_in
                w = h = 0
                try:
                    w, h = _img_size(full)
                except Exception:
                    pass
                out.append({"name": fn, "rel": rel, "path": full, "batch": batch,
                            "w": w, "h": h,
                            "thumb": "/api/thumb?w=420&path=" + quote(full)})
    return out


def _ws_safe_rel(ws, rel):
    """把前端传来的 rel 解析成工作区内绝对路径，越界返回 None。"""
    rel = str(rel or "").replace("\\", "/").strip("/")
    if not rel:
        return None
    root = os.path.abspath(ws["path"])
    full = os.path.abspath(os.path.join(root, rel.replace("/", os.sep)))
    try:
        if os.path.commonpath([os.path.normcase(full), os.path.normcase(root)]) != os.path.normcase(root):
            return None
    except ValueError:
        return None
    return full


def _ws_sort_rename(ws, order):
    """按给定顺序把图片重命名成 `原名_NN.png`，两阶段改名避免互相覆盖。"""
    import re as _re
    seq = _re.compile(r"_\d{2,4}$")
    pairs = []
    for rel in order:
        full = _ws_safe_rel(ws, rel)
        if not full or not os.path.isfile(full):
            continue
        pairs.append((rel, full))
    if not pairs:
        return []
    width = max(2, len(str(len(pairs))))
    staged = []
    for i, (rel, full) in enumerate(pairs):
        d = os.path.dirname(full)
        try:
            tmp = os.path.join(d, ".sort_tmp_%04d" % i)
            os.replace(full, tmp)
            staged.append((rel, tmp))
        except OSError:
            staged.append((rel, full))
    renamed = []
    used = set()
    for i, (rel, tmp) in enumerate(staged):
        d = os.path.dirname(tmp)
        ext = os.path.splitext(rel)[1] or ".png"
        stem = os.path.splitext(os.path.basename(rel))[0]
        stem = seq.sub("", stem) or "img"
        name = "%s_%0*d%s" % (stem, width, i + 1, ext)
        dst = os.path.join(d, name)
        k = 2
        while os.path.normcase(dst) in used or (os.path.exists(dst) and os.path.normcase(dst) != os.path.normcase(tmp)):
            name = "%s_%0*d_%d%s" % (stem, width, i + 1, k, ext)
            dst = os.path.join(d, name)
            k += 1
        used.add(os.path.normcase(dst))
        try:
            os.replace(tmp, dst)
        except OSError:
            continue
        batch = os.path.basename(os.path.dirname(dst)) if os.path.dirname(dst) != os.path.abspath(ws["path"]) else ""
        # rel 重新按"批次/文件"拼（批次内可能有子目录，用真实相对路径）
        rel_new = os.path.relpath(dst, ws["path"]).replace(os.sep, "/")
        renamed.append({"from": rel, "rel": rel_new, "name": name, "path": dst,
                        "batch": batch})
    return renamed


def _thumb(path, width=420):
    """生成缩略图字节；内存优先，超过上限落盘。"""
    path = os.path.abspath(path)
    try:
        mtime = int(os.path.getmtime(path))
    except OSError:
        raise FileNotFoundError("文件不存在")
    key = (path, mtime, width)
    with THUMB_LOCK:
        hit = THUMB_RAM.get(key)
    if hit:
        return hit
    import hashlib
    disk = os.path.join(THUMB_DIR, hashlib.md5(
        ("%s|%s|%s" % (path, mtime, width)).encode("utf-8", "ignore")).hexdigest() + ".jpg")
    if os.path.exists(disk):
        try:
            data = open(disk, "rb").read()
            with THUMB_LOCK:
                if len(THUMB_RAM) < THUMB_RAM_MAX:
                    THUMB_RAM[key] = data
            return data
        except OSError:
            pass
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解析图片：%s" % os.path.basename(path))
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise ValueError("缩略图编码失败")
    data = buf.tobytes()
    with THUMB_LOCK:
        if len(THUMB_RAM) < THUMB_RAM_MAX:
            THUMB_RAM[key] = data
        else:
            for k in list(THUMB_RAM)[:40]:
                THUMB_RAM.pop(k, None)
    try:
        open(disk, "wb").write(data)
    except OSError:
        pass
    return data


# ---------------- 局域网访问 ----------------
LAN_HOST = "0.0.0.0"
LOOPBACK_HOST = "127.0.0.1"
CURRENT_LAN = False   # 运行时由 __main__ 更新：是否允许局域网访问


def _lan_ips():
    """返回本机局域网 IP 列表（尽力而为，优先真实局域网段）。"""
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass
    # 补充 UDP connect 法拿到的出口 IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    # 去重并排序：优先 192.168/10./172.16-31 真实局域网段，其次其他
    uniq = sorted(set(ips))
    def rank(ip):
        if ip.startswith("192.168."):
            return 0
        if ip.startswith("10."):
            return 1
        if ip.startswith("172."):
            try:
                o = int(ip.split(".")[1])
                if 16 <= o <= 31:
                    return 2
            except Exception:
                pass
        return 3
    return sorted(uniq, key=rank)


def _restart_with_mode(mode):
    """LAN 模式切换：用 os.execv 替换自身进程（同端口、不同绑定），
    旧进程随即退出，避免新旧进程争抢端口。"""
    import os as _os
    python = sys.executable
    args = [python, _os.path.abspath(__file__), str(CURRENT_PORT)]
    if mode == "lan":
        args.append("--lan")
    _os.execv(python, args)   # 替换当前进程，不会返回


# ---------------- 遮挡图片素材（图片打码模式） ----------------
ASSET_LIMIT = 8 * 1024 * 1024     # 单张遮挡图上限
ASSET_KEEP = 8                    # 最多缓存的素材数
ASSETS = {}                       # id -> {"img": ndarray, "ts": float}


def _media_worker(job_id, data, kind, cfg, stamp):
    job = MEDIA[job_id]

    def cb(done, total):
        with MEDIA_LOCK:
            job["frame"], job["frames"] = done, total
            job["pct"] = min(99, int(done * 100 / max(1, total)))

    try:
        # YOLO-World 懒加载放在任务线程里做（下载可能耗时，避免卡 HTTP 响应）
        world_words = expand_words(cfg.get("world_classes") or [])
        world_detector = get_world_detector() if world_words else None
        if world_detector:
            world_detector.resolution = max(WORLD_RESOLUTION, cfg["res"])
        world_embeds = (world_detector.get_embeds(world_words)
                        if world_detector else None)
        enabled_world = [WORLD_PREFIX + w for w in world_words]
        if enabled_world and cfg["classes"] is not None:
            cfg["classes"] = list(cfg["classes"]) + enabled_world
        cfg["world_classes"] = world_words
        detector.resolution = cfg["res"]
        if kind == "gif":
            out, ext, info = process_gif(data, cfg, detector, stamp=stamp,
                                         progress=cb, world=world_detector,
                                         world_embeds=world_embeds)
        else:
            out, ext, info = process_video(data, cfg, detector, stamp=stamp,
                                           progress=cb, prefer=cfg["vfmt"],
                                           world=world_detector,
                                           world_embeds=world_embeds)
        # job_id 是 uuid hex，ext 只可能来自 {"gif","webm","mp4"}
        path = MEDIA_ROOT / (job_id + "." + ext)
        path.write_bytes(out)
        with MEDIA_LOCK:
            job.update(st="done", pct=100, path=str(path), ext=ext, info=info)
    except Exception as e:
        with MEDIA_LOCK:
            job.update(st="fail", err=str(e))
    finally:
        with MEDIA_LOCK:  # 只保留最近 MEDIA_KEEP 个已产出结果的任务
            finished = sorted((j for j in MEDIA.values() if j.get("path")),
                              key=lambda j: j["ts"])
            for old in finished[:-MEDIA_KEEP]:
                try:
                    os.remove(old["path"])
                except OSError:
                    pass
                MEDIA.pop(old["id"], None)


def _media_snapshot(job):
    """读取任务快照；调用方必须已持有 MEDIA_LOCK（Lock 不可重入）。"""
    return {"st": job["st"], "pct": job.get("pct", 0),
            "frame": job.get("frame"), "frames": job.get("frames"),
            "err": job.get("err"), "ext": job.get("ext"),
            "info": job.get("info")}


def _resolve_stamp(cfg):
    """图片遮挡模式下按 cfg["asset"] 取出遮挡图；其余模式返回 None。"""
    if cfg["mode"] != "img":
        return None, None
    with MEDIA_LOCK:
        entry = ASSETS.get(cfg["asset"] or "")
    if entry is None:
        return None, "请先上传遮挡图片（打码方式 → 图片遮挡）"
    return entry["img"], None

# ---------------- 页面 ----------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hchimao Studio V1.5.0</title>
<style>
:root{
  --bg:#0c0f13; --panel:#12161c; --panel2:#181d25; --line:#242b36; --line2:#333c4a;
  --text:#e6ebf2; --dim:#8d97a6; --amber:#6ee7c4; --amber2:#0b2b24;
  --danger:#f0666a; --ok:#41d17f; --shadow:0 6px 22px rgba(0,0,0,.35);
}
html[data-theme="light"]{
  --bg:#f2f4f7; --panel:#ffffff; --panel2:#f7f8fa; --line:#e2e6ec; --line2:#cfd6e0;
  --text:#1e242c; --dim:#6b7480; --amber:#0f9f7f; --amber2:#e6f7f2;
  --danger:#d94a4f; --ok:#12915c; --shadow:0 6px 22px rgba(20,30,50,.10);
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--text);font:13px/1.5 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
     display:flex;overflow:hidden}
button{font:inherit;color:inherit;background:transparent;border:1px solid var(--line2);border-radius:8px;
       padding:5px 11px;cursor:pointer;transition:.14s}
button:hover{border-color:var(--amber);color:var(--amber)}
button.primary{background:var(--amber);border-color:var(--amber);color:#04211a;font-weight:600}
button.primary:hover{filter:brightness(1.08);color:#04211a}
button:disabled{opacity:.4;cursor:not-allowed;border-color:var(--line)}

/* ---------- 左侧导航 ---------- */
.rail{width:76px;flex:0 0 76px;background:var(--panel);border-right:1px solid var(--line);
      display:flex;flex-direction:column;align-items:center;gap:6px;padding:12px 6px}
.rail-btn{width:100%;padding:9px 2px;border:0;border-radius:10px;color:var(--dim);
          display:flex;flex-direction:column;align-items:center;gap:4px;font-size:11px;line-height:1.2}
.rail-btn .ic{font-size:17px;line-height:1}
.rail-btn:hover{background:var(--panel2);color:var(--text)}
.rail-btn.on{background:var(--amber2);color:var(--amber)}
.rail .spacer{flex:1}

/* ---------- 主区 ---------- */
.main{flex:1;min-width:0;display:flex;flex-direction:column}
.topbar{height:46px;flex:0 0 46px;display:flex;align-items:center;gap:10px;padding:0 14px;
        border-bottom:1px solid var(--line);background:var(--panel)}
#curPath{flex:1;min-width:0;color:var(--dim);font-size:12px;white-space:nowrap;overflow:hidden;
         text-overflow:ellipsis}
#curPath b{color:var(--text);font-weight:600}
.viewbar{height:44px;flex:0 0 44px;display:flex;align-items:center;gap:8px;padding:0 14px;
         border-bottom:1px solid var(--line)}
.seg{display:flex;background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:2px}
.seg button{border:0;border-radius:7px;padding:4px 16px;color:var(--dim)}
.seg button.on{background:var(--amber);color:#04211a;font-weight:600}
.spacer{flex:1}
.arw{width:34px;height:30px;padding:0;display:grid;place-items:center;font-size:15px}

/* ---------- 工作区 ---------- */
.workzone{position:relative;flex:1;min-height:0;display:flex;overflow:hidden}
.vp{flex:1;min-width:0;position:relative;overflow:hidden}
#stage{position:absolute;left:0;top:0;transform-origin:0 0;will-change:transform}
#stage img,#stage canvas{position:absolute;left:0;top:0;display:block}
#cvPaint{cursor:crosshair}
#cvWm{position:absolute;left:0;top:0;pointer-events:none;z-index:3;display:block}
body.mark #cvPaint{cursor:default}
body.mark.mk-drag #cvPaint{cursor:move}
.danger{color:#ff6b6b!important;border-color:#573034!important}
.danger:hover{background:#3a1f22!important}
.navbtn{
  position:absolute;top:50%;transform:translateY(-50%);z-index:9;
  width:40px;height:64px;border-radius:12px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  font-size:26px;line-height:1;padding:0 0 4px;
  color:var(--fg);background:var(--panel);border:1px solid var(--line2);
  opacity:.62;transition:opacity .15s,transform .15s;
  box-shadow:0 6px 18px rgba(0,0,0,.22)
}
.navbtn:hover{opacity:1}
.navbtn:active{transform:translateY(-50%) scale(.96)}
.navbtn.prev{left:10px}
.navbtn.next{right:10px}
.navbtn.hide{display:none}
.sbar{position:absolute;border-radius:99px;background:var(--line2);opacity:.55;transition:opacity .15s}
.sbar:hover,.sbar.drag{opacity:1}
.sbar.v{width:9px;right:1px;top:4px;bottom:4px}
.sbar.h{height:9px;left:4px;right:14px;bottom:1px}
.sbar i{position:absolute;background:var(--amber);border-radius:99px;opacity:.85}
.sbar.v i{left:0;right:0}
.sbar.h i{top:0;bottom:0}
.sbar.hide{display:none}

/* 全部视图 */
#gridWrap{position:absolute;inset:0;overflow:auto;scrollbar-width:none;padding:14px}
#gridWrap::-webkit-scrollbar{width:0;height:0}
#grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(clamp(110px,var(--tile,168px),260px),1fr))}
.card{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:10px;
      overflow:hidden;cursor:pointer}
.card:hover{border-color:var(--amber)}
.card .box{position:relative;aspect-ratio:4/3;background:var(--panel2)}
.card .box img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
.card .nm{padding:5px 8px;font-size:11px;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card.done{border-color:var(--ok)}
.card.done .nm{color:var(--ok)}
.card.cur{border-color:var(--amber);box-shadow:0 0 0 1px var(--amber) inset}
.pick{position:absolute;right:6px;top:6px;width:20px;height:20px;border-radius:6px;border:1.5px solid #fff9;
      background:#0007;display:none;place-items:center;font-size:12px;color:#fff}
body.ai .pick{display:grid}
.pick.on{background:var(--amber);border-color:var(--amber);color:#04211a;font-weight:700}
.badge{position:absolute;left:6px;top:6px;font-size:10px;padding:1px 6px;border-radius:99px;
       background:#0009;color:#fff;display:none}
.card.done .badge{display:block}
.dir-card .dirbox{display:grid;place-items:center;font-size:30px;color:var(--dim)}
.dir-card:hover .dirbox{color:var(--amber)}
.empty{position:absolute;inset:0;display:grid;place-items:center;color:var(--dim);font-size:13px;text-align:center;padding:30px}

/* ---------- 右侧面板 ---------- */
.panel{width:288px;flex:0 0 288px;background:var(--panel);border-left:1px solid var(--line);
       overflow:auto;padding:14px}
.panel h3{margin:0 0 10px;font-size:12px;color:var(--dim);font-weight:600;letter-spacing:.04em}
.grp{margin-bottom:18px}
.row{display:flex;align-items:center;gap:8px;margin:8px 0}
.row label{flex:0 0 62px;color:var(--dim);font-size:12px}
.row input[type=range]{flex:1;min-width:0;accent-color:var(--amber)}
.val{flex:0 0 40px;text-align:right;font-variant-numeric:tabular-nums;color:var(--text);font-size:12px}
.seg2{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.seg2 button{padding:7px 0}
.seg2 button.on{background:var(--amber2);border-color:var(--amber);color:var(--amber)}
.chips{display:flex;flex-wrap:wrap;gap:5px;max-height:190px;overflow:auto}
.chip{font-size:11px;padding:3px 8px;border-radius:99px;border:1px solid var(--line2);color:var(--dim);cursor:pointer}
.chip.on{border-color:var(--amber);color:var(--amber);background:var(--amber2)}
.dirs{display:flex;flex-direction:column;gap:2px;max-height:210px;overflow:auto}
.dir{display:flex;gap:6px;align-items:center;padding:5px 7px;border-radius:7px;cursor:pointer;color:var(--dim);font-size:12px}
.dir:hover{background:var(--panel2);color:var(--text)}
.hint{color:var(--dim);font-size:11px;line-height:1.6}
.hint.strong{color:var(--text)}
select{background:var(--panel2);border:1px solid var(--line2);border-radius:8px;color:var(--text);padding:4px 6px;font:inherit}
input[type=color]{width:34px;height:24px;padding:0;border:1px solid var(--line2);border-radius:6px;background:var(--panel2)}
.card.enabled{border-color:var(--amber)}
.card.enabled .nm{color:var(--amber)}
.card.dir-card .pick{background:#0007}
.card .seq{position:absolute;left:6px;bottom:6px;font-size:11px;font-weight:700;padding:1px 7px;border-radius:99px;background:var(--amber);color:#04211a;z-index:4}
.card .badge.wm{left:6px;top:auto;bottom:6px;display:block;background:#0009;color:#9ff0db}
.card.dragging{opacity:.4}
.card.dropbefore{box-shadow:-3px 0 0 0 var(--amber)}
.card.dropafter{box-shadow:3px 0 0 0 var(--amber)}
.card.sel{outline:2px solid var(--amber);outline-offset:-2px}
.layers{display:flex;flex-direction:column;gap:4px;max-height:230px;overflow:auto;margin:6px 0}
.layer{display:flex;align-items:center;gap:6px;padding:5px 7px;border:1px solid var(--line);border-radius:8px;background:var(--panel2)}
.layer.on{border-color:var(--amber)}
.layer .lt{flex:1;min-width:0;font-size:11px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.layer button{padding:2px 7px;font-size:11px;border-radius:6px}
.editbar{display:flex;align-items:center;gap:8px;margin:8px 0;padding:6px 8px;border-radius:8px;border:1px dashed var(--amber);background:var(--amber2)}
.chips.small .chip{font-size:10px;padding:2px 7px}
.pfchip{display:inline-flex;align-items:center;gap:5px;font-size:10px;padding:3px 8px;border-radius:99px;
        border:1px solid var(--line2);color:var(--dim);cursor:pointer;background:var(--panel2)}
.pfchip i{width:8px;height:8px;border-radius:2.5px;display:block;flex:0 0 8px}
.pfchip.on{color:var(--text);border-color:var(--amber);background:var(--amber2)}
.pfbox{border:1px solid var(--line);border-radius:9px;background:var(--panel2);padding:6px;margin:6px 0;
       display:flex;flex-direction:column;gap:5px}
.pfbox .hd{display:flex;align-items:center;gap:6px}
.pfbox .hd i{width:9px;height:9px;border-radius:3px;display:block;flex:0 0 9px}
.pfbox .hd .nm{flex:1;min-width:0;font-size:11px;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pfbox select{font-size:11px;padding:2px 4px;border-radius:6px}
.pfbox input[type=text]{font-size:11px;padding:3px 6px;border-radius:6px}
.pfbox button{padding:1px 8px;font-size:12px;line-height:1.4}
.panel .grp > .row > button, .panel .grp > .row > .chip{white-space:nowrap}
.panel .row .val{white-space:nowrap}
.need{border-color:#ff6b6b!important;box-shadow:0 0 0 3px #ff6b6b33}
input[type=text]{width:100%;background:var(--panel2);border:1px solid var(--line2);border-radius:8px;
                 padding:6px 8px;color:var(--text);font:inherit}
#prog{height:4px;border-radius:99px;background:var(--panel2);overflow:hidden;margin-top:8px;display:none}
#prog i{display:block;height:100%;width:0;background:var(--amber);transition:width .2s}
#toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);background:var(--panel);
       border:1px solid var(--line2);box-shadow:var(--shadow);padding:9px 16px;border-radius:10px;
       font-size:12px;opacity:0;pointer-events:none;transition:.2s;z-index:99}
#toast.on{opacity:1}

/* ===== 账号闸门 / 语言 / 设置（V1.5.0）===== */
#gateWrap{position:fixed;inset:0;z-index:200;display:none;align-items:center;justify-content:center;
  background:radial-gradient(1100px 560px at 32% 8%,rgba(110,231,196,.10),transparent 62%),var(--bg)}
#gateWrap.on{display:flex}
.gate-card{width:386px;background:var(--panel);border:1px solid var(--line);border-radius:16px;
  padding:26px 24px 20px;box-shadow:0 26px 70px rgba(0,0,0,.5)}
.gate-card h1{font-size:19px;margin:0 0 6px;letter-spacing:.5px}
.gate-card .sub{color:var(--dim);font-size:12.5px;line-height:1.6;margin-bottom:16px}
.fld{margin-bottom:11px}
.fld label{display:block;font-size:12px;color:var(--dim);margin-bottom:5px}
.fld input{width:100%;height:38px;background:var(--panel2);border:1px solid var(--line2);
  border-radius:9px;color:var(--text);padding:0 12px;font-size:13.5px;outline:none;box-sizing:border-box}
.fld input:focus{border-color:var(--amber)}
.gate-err{min-height:15px;font-size:12px;color:var(--danger);margin:2px 0 8px}
.gate-card button.primary{width:100%;height:40px;font-size:14px;border-radius:10px}
.gate-foot{margin-top:11px;font-size:12px;color:var(--dim);text-align:center}
.gate-foot a{color:var(--amber);cursor:pointer}
.flag{width:24px;height:16px;border-radius:3px;display:inline-block;position:relative;
  overflow:hidden;box-shadow:0 0 0 1px rgba(255,255,255,.18) inset;flex:0 0 auto}
.flag.cn{background:#de2910}
.flag.cn:after{content:"★";position:absolute;left:1px;top:-4px;color:#ffde00;font-size:9px}
.flag.tw{background:#fe0000}
.flag.tw:before{content:"";position:absolute;left:0;top:0;width:12px;height:8px;background:#000095}
.flag.tw:after{content:"☀";position:absolute;left:2px;top:-4px;color:#fff;font-size:8px}
.flag.jp{background:#fff}
.flag.jp:after{content:"";position:absolute;left:50%;top:50%;width:9px;height:9px;margin:-4.5px 0 0 -4.5px;
  border-radius:50%;background:#bc002d}
.flag.us{background:repeating-linear-gradient(#b22234 0 2px,#fff 2px 4px)}
.flag.us:after{content:"";position:absolute;left:0;top:0;width:10px;height:7px;background:#3c3b6e}
.flag.kr{background:#fff}
.flag.kr:after{content:"";position:absolute;left:50%;top:50%;width:8px;height:8px;margin:-4px 0 0 -4px;
  border-radius:50%;background:linear-gradient(#cd2e3a 50%,#0047a0 50%)}
.flag.ru{background:linear-gradient(#fff 0 33%,#0039a6 33% 66%,#d52b1e 66%)}
#setFlag{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px}
#setFlag .flag{width:20px;height:14px}
#langWrap{position:fixed;inset:0;z-index:210;display:none;align-items:center;justify-content:center;
  background:rgba(4,6,9,.72);backdrop-filter:blur(3px)}
#langWrap.on{display:flex}
.lang-card{width:520px;background:var(--panel);border:1px solid var(--line);border-radius:16px;
  padding:24px;box-shadow:0 26px 70px rgba(0,0,0,.55)}
.lang-card h2{font-size:17px;margin:0 0 6px}
.lang-card .sub{color:var(--dim);font-size:12.5px;margin-bottom:16px}
.lgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.lbtn{display:flex;align-items:center;gap:9px;padding:11px 12px;background:var(--panel2);
  border:1px solid var(--line2);border-radius:10px;color:var(--text);cursor:pointer;font-size:13px;
  transition:transform .12s,border-color .12s,background .12s}
.lbtn:hover{transform:translateY(-1px);border-color:var(--amber)}
.lbtn.on{border-color:var(--amber);background:var(--amber2)}
#setPop{position:fixed;left:74px;bottom:12px;z-index:150;width:306px;display:none;
  background:var(--panel);border:1px solid var(--line2);border-radius:13px;padding:12px;
  box-shadow:0 20px 50px rgba(0,0,0,.5)}
#setPop.on{display:block}
#setPop .k{font-size:11px;color:var(--dim);letter-spacing:.6px;margin-bottom:7px}
#setPop .sec{padding:8px 2px 10px;border-bottom:1px solid var(--line)}
#setPop .sec:last-child{border-bottom:0;padding-bottom:2px}
#setPop .acc{display:flex;align-items:center;justify-content:space-between;font-size:13px}
#setPop .acc a{color:var(--dim);font-size:12px;cursor:pointer}
#setPop .acc a:hover{color:var(--danger)}
#setPop .lgrid{grid-template-columns:repeat(3,1fr);gap:7px}
#setPop .lbtn{padding:6px 4px;font-size:11px;gap:5px;justify-content:center;white-space:nowrap}
#setPop .lbtn .flag{width:18px;height:12px}
#themeSeg{display:flex;gap:7px}
#themeSeg button{flex:1;height:30px;font-size:12.5px}
.card .nm[contenteditable="true"]{outline:1px dashed var(--amber);border-radius:4px;padding:0 3px;background:var(--panel2)}
.panel .row3{display:flex;gap:6px;margin-bottom:6px}
.panel input.inp{height:30px;background:var(--panel2);border:1px solid var(--line2);border-radius:7px;
  color:var(--text);padding:0 8px;font-size:12.5px;width:100%;box-sizing:border-box;margin:0}
.panel input.inp:focus{border-color:var(--amber);outline:none}
.panel button.danger{color:var(--danger);border-color:rgba(240,102,106,.45)}
.panel button.danger:hover{background:rgba(240,102,106,.12)}
</style>
</head>
<body>
<nav class="rail" id="rail">
  <button class="rail-btn on" data-mode="repo"><span class="ic">▤</span>仓库</button>
  <button class="rail-btn" data-mode="work"><span class="ic">🗂</span>工作区</button>
  <button class="rail-btn" data-mode="ai"><span class="ic">✦</span>AI 编辑</button>
  <button class="rail-btn" data-mode="manual"><span class="ic">✎</span>手动编辑</button>
  <button class="rail-btn" data-mode="mark"><span class="ic">◈</span>批量水印</button>
  <button class="rail-btn" data-mode="cover"><span class="ic">▣</span>封面</button>
  <button class="rail-btn" data-mode="sort"><span class="ic">⇅</span>排序</button>
  <div class="spacer"></div>
  <button class="rail-btn" id="setBtn" title="设置"><span class="ic" id="setFlag"></span>设置</button>
</nav>

<div class="main">
  <div class="topbar">
    <div id="curPath">未选择仓库目录</div>
    <button id="exportBtn" class="primary" title="只导出打过码的图片">导出打码后文件</button>
    <button id="exportAllBtn" title="导出全部图片，并应用各自已经做过的更改（笔迹 / 水印等）">导出所有已做更改</button>
  </div>

  <div class="viewbar">
    <div class="seg" id="viewseg">
      <button data-layout="all" class="on">全部</button>
      <button data-layout="single">单张</button>
    </div>
    <button id="selAllBtn" title="全选 / 取消全选">全选</button>
    <button id="importBtn" class="primary" style="display:none" title="把勾选的图片复制到程序内缓存并导入">导入所选</button>
    <span id="selInfo" class="hint"></span>
    <div class="spacer"></div>
    <button id="dlOneBtn" style="display:none" title="把当前这张按看到的画面存成 PNG">下载这张</button>
    <button class="arw" id="undoBtn" title="撤销 (Ctrl+Z)">↶</button>
    <button class="arw" id="redoBtn" title="恢复 (Ctrl+Shift+Z)">↷</button>
  </div>

  <div class="workzone" id="zone">
    <div class="vp" id="vp">
      <div id="gridWrap"><div id="grid"></div>
        <div class="empty" id="emptyTip">还没有图片 —— 在左侧「仓库」里选一个本地文件夹导入</div>
      </div>
      <div id="stage" style="display:none">
        <img id="cvBase" alt="">
        <canvas id="cvPaint"></canvas>
        <canvas id="cvWm"></canvas>
      </div>
    </div>
    <button class="navbtn prev hide" id="navPrev" title="上一张 (←)">‹</button>
    <button class="navbtn next hide" id="navNext" title="下一张 (→)">›</button>
    <div class="sbar v hide" id="sbV"><i></i></div>
    <div class="sbar h hide" id="sbH"><i></i></div>
  </div>
</div>

<aside class="panel" id="panel"></aside>
<div id="toast"></div>

<div id="setPop">
  <div class="sec">
    <div class="k">账号</div>
    <div class="acc"><b id="setUser">—</b><a id="setLogout">退出登录</a></div>
  </div>
  <div class="sec">
    <div class="k">语言</div>
    <div class="lgrid" id="setLangs"></div>
  </div>
  <div class="sec">
    <div class="k">主题</div>
    <div id="themeSeg">
      <button data-theme="dark">深色</button>
      <button data-theme="light">浅色</button>
    </div>
  </div>
</div>

<div id="gateWrap"><div class="gate-card" id="gateCard"></div></div>

<div id="langWrap"><div class="lang-card">
  <h2>选择语言</h2>
  <div class="sub">选好之后，语言标志会缩进左下角的「设置」里，之后随时可以改。</div>
  <div class="lgrid" id="langGrid"></div>
</div></div>

<script>
window.__TR__ = {"仓库":["仓库","倉庫","リポジトリ","Files","파일","Файлы"],"工作区":["工作区","工作區","ワークスペース","Workspace","작업 공간","Рабочая область"],"AI 编辑":["AI 编辑","AI 編輯","AI 編集","AI Edit","AI 편집","AI-обработка"],"手动编辑":["手动编辑","手動編輯","手動編集","Manual","수동 편집","Вручную"],"批量水印":["批量水印","批次浮水印","一括透かし","Watermark","일괄 워터마크","Водяные знаки"],"封面":["封面","封面","カバー","Cover","커버","Обложка"],"排序":["排序","排序","並べ替え","Order","정렬","Порядок"],"设置":["设置","設定","設定","Settings","설정","Настройки"],"账号":["账号","帳號","アカウント","Account","계정","Аккаунт"],"语言":["语言","語言","言語","Language","언어","Язык"],"主题":["主题","主題","テーマ","Theme","테마","Тема"],"深色":["深色","深色","ダーク","Dark","다크","Тёмная"],"浅色":["浅色","淺色","ライト","Light","라이트","Светлая"],"退出登录":["退出登录","登出","ログアウト","Sign out","로그아웃","Выйти"],"选择语言":["选择语言","選擇語言","言語を選択","Choose language","언어 선택","Выбор языка"],"创建账号":["创建账号","建立帳號","アカウント作成","Create account","계정 만들기","Создать аккаунт"],"登录":["登录","登入","ログイン","Sign in","로그인","Войти"],"用户名":["用户名","使用者名稱","ユーザー名","Username","사용자 이름","Имя"],"密码":["密码","密碼","パスワード","Password","비밀번호","Пароль"],"注册码":["注册码","註冊碼","登録コード","Invite code","등록 코드","Код"],"创建并进入":["创建并进入","建立並進入","作成して開始","Create & enter","만들고 시작","Создать"],"用户名 2-24 个字符":["用户名 2-24 个字符","使用者名稱 2-24 個字元","ユーザー名は2〜24文字","Username 2-24 chars","사용자 이름 2-24자","Имя 2–24 символа"],"密码至少 4 位":["密码至少 4 位","密碼至少 4 位","パスワードは4文字以上","Password min 4 chars","비밀번호 4자 이상","Пароль от 4 символов"],"已经有账号？去登录":["已经有账号？去登录","已經有帳號？去登入","アカウントをお持ちですか？","Have an account? Sign in","계정이 있나요? 로그인","Уже есть аккаунт?"],"创建新账号":["创建新账号","建立新帳號","新規アカウント","New account","새 계정","Новый аккаунт"],"未选择仓库目录":["未选择仓库目录","未選擇倉庫目錄","フォルダ未選択","No folder selected","폴더 미선택","Папка не выбрана"],"导出打码后文件":["导出打码后文件","匯出已打碼檔案","モザイク済みを書き出し","Export censored","모자이크 내보내기","Экспорт обработанных"],"导出所有":["导出所有","匯出全部","すべて書き出し","Export all","전체 내보내기","Экспорт всех"],"导出 ZIP":["导出 ZIP","匯出 ZIP","ZIP 書き出し","Export ZIP","ZIP 내보내기","Экспорт ZIP"],"全部":["全部","全部","すべて","All","전체","Все"],"单张":["单张","單張","1枚","Single","단일","Одна"],"全选":["全选","全選","全選択","Select all","전체 선택","Выбрать всё"],"取消全选":["取消全选","取消全選","選択解除","Clear","선택 해제","Снять выбор"],"导入所选":["导入所选","匯入所選","選択を読み込み","Import selected","선택 가져오기","Импорт выбранных"],"下载这张":["下载这张","下載這張","この画像を保存","Download","이 이미지 저장","Скачать"],"上一级":["上一级","上一層","上へ","Up","상위","Вверх"],"刷新":["刷新","重新整理","更新","Refresh","새로고침","Обновить"],"打开":["打开","開啟","開く","Open","열기","Открыть"],"已启用的文件夹里没有图片":["已启用的文件夹里没有图片","已啟用的資料夾裡沒有圖片","有効なフォルダに画像がありません","No images in enabled folders","활성 폴더에 이미지가 없습니다","В активных папках нет изображений"],"还没有启用任何文件夹 —— 去左侧「工作区」勾选要编辑的文件夹":["还没有启用任何文件夹 —— 去左侧「工作区」勾选要编辑的文件夹","尚未啟用任何資料夾 —— 到左側「工作區」勾選要編輯的資料夾","有効なフォルダがありません — 左の「ワークスペース」で選択してください","No folders enabled — pick them in Workspace","활성 폴더가 없습니다 — 왼쪽 '작업 공간'에서 선택하세요","Нет активных папок — выберите в «Рабочей области»"],"先在工作区勾选文件夹，再来加水印":["先在工作区勾选文件夹，再来加水印","先在工作區勾選資料夾，再來加水印","先にワークスペースでフォルダを選んでください","Pick a folder in Workspace first","먼저 작업 공간에서 폴더를 선택하세요","Сначала выберите папку"],"当前工作区":["当前工作区","目前工作區","現在のワークスペース","Current workspace","현재 작업 공간","Текущая область"],"工作区名称（导出 zip 就用它）":["工作区名称（导出 zip 就用它）","工作區名稱（匯出 zip 就用它）","ワークスペース名（zip 名になります）","Workspace name (used for the zip)","작업 공간 이름 (zip 이름)","Имя области (для zip)"],"重命名":["重命名","重新命名","名前を変更","Rename","이름 변경","Переименовать"],"只导已打码":["只导已打码","只匯出已打碼","モザイク済みのみ","Censored only","모자이크만","Только обработанные"],"删除工作区":["删除工作区","刪除工作區","ワークスペースを削除","Delete workspace","작업 공간 삭제","Удалить область"],"仓库目录":["仓库目录","倉庫目錄","フォルダ","Source folder","소스 폴더","Папка-источник"],"导入本目录全部":["导入本目录全部","匯入本目錄全部","この階層をすべて読み込み","Import whole folder","이 폴더 전체 가져오기","Импорт всей папки"],"左上":["左上","左上","左上","Top-left","좌상","Сверху слева"],"右上":["右上","右上","右上","Top-right","우상","Сверху справа"],"左下":["左下","左下","左下","Bottom-left","좌하","Снизу слева"],"右下":["右下","右下","右下","Bottom-right","우하","Снизу справа"],"水印文字":["水印文字","浮水印文字","透かし文字","Watermark text","워터마크 텍스트","Текст водяного знака"],"字体":["字体","字型","フォント","Font","글꼴","Шрифт"],"黑体":["黑体","黑體","ゴシック","Sans","고딕","Гротеск"],"宋体":["宋体","宋體","明朝","Serif","명조","Антиква"],"等宽":["等宽","等寬","等幅","Mono","고정폭","Моно"],"颜色":["颜色","顏色","色","Color","색상","Цвет"],"字号":["字号","字級","サイズ","Size","크기","Размер"],"密度":["密度","密度","密度","Density","밀도","Плотность"],"角度":["角度","角度","角度","Angle","각도","Угол"],"边距":["边距","邊距","余白","Margin","여백","Отступ"],"速度":["速度","速度","速度","Speed","속도","Скорость"],"固化到选中":["固化到选中","固化到選取","選択に焼き付け","Burn to selected","선택에 확정","Закрепить"],"固化到全部":["固化到全部","固化到全部","すべてに焼き付け","Burn to all","전체에 확정","Закрепить на всех"],"清除水印":["清除水印","清除浮水印","透かしを消す","Clear watermark","워터마크 지우기","Убрать знак"],"导出所选":["导出所选","匯出所選","選択を書き出し","Export selected","선택 내보내기","Экспорт выбранных"],"导出全部":["导出全部","匯出全部","すべて書き出し","Export all","전체 내보내기","Экспорт всех"],"预览动画":["预览动画","預覽動畫","アニメ表示","Preview","미리보기","Просмотр"],"取消编辑":["取消编辑","取消編輯","編集を終了","Cancel edit","편집 취소","Отменить правку"],"编辑":["编辑","編輯","編集","Edit","편집","Изменить"],"删":["删","刪","削除","Del","삭제","Удал."],"删除勾选":["删除勾选","刪除勾選","選択を削除","Delete checked","선택 삭제","Удалить отмеченные"],"清空这张":["清空这张","清空這張","この画像を全消去","Clear this image","이 이미지 비우기","Очистить"],"这张图的水印":["这张图的水印","這張圖的浮水印","この画像の透かし","Watermarks on this image","이 이미지의 워터마크","Знаки на этом изображении"],"先打开一张图":["先打开一张图","先開啟一張圖","まず画像を開いてください","Open an image first","먼저 이미지를 여세요","Откройте изображение"],"封面工具":["封面工具","封面工具","カバーツール","Cover tool","커버 도구","Обложка"],"封面标题":["封面标题","封面標題","カバー見出し","Cover title","커버 제목","Заголовок"],"封面主标题":["封面主标题","封面主標題","メイン見出し","Main title","메인 제목","Главный заголовок"],"副标题（可空）":["副标题（可空）","副標題（可空）","サブタイトル（任意）","Subtitle (optional)","부제 (선택)","Подзаголовок"],"显示标题":["显示标题","顯示標題","見出しを表示","Show title","제목 표시","Показать заголовок"],"位置":["位置","位置","位置","Position","위치","Положение"],"底部":["底部","底部","下","Bottom","하단","Снизу"],"顶部":["顶部","頂部","上","Top","상단","Сверху"],"整屏":["整屏","整屏","全画面","Full","전체","Весь экран"],"左对齐":["左对齐","左對齊","左揃え","Left","왼쪽","По левому краю"],"居中":["居中","居中","中央","Center","가운데","По центру"],"右对齐":["右对齐","右對齊","右揃え","Right","오른쪽","По правому краю"],"标题大小":["标题大小","標題大小","見出しサイズ","Title size","제목 크기","Размер заголовка"],"压暗":["压暗","壓暗","暗くする","Dim","어둡게","Затемнение"],"滤镜":["滤镜","濾鏡","フィルター","Filter","필터","Фильтр"],"平台标识":["平台标识","平台標識","プラットフォーム表示","Platform badge","플랫폼 배지","Значок платформы"],"标识大小":["标识大小","標識大小","表示サイズ","Badge size","배지 크기","Размер значка"],"确认封面并复制一张":["确认封面并复制一张","確認封面並複製一張","カバーを確定して複製","Apply cover & copy","커버 확정 후 복사","Готово и копия"],"重置":["重置","重設","リセット","Reset","초기화","Сброс"],"原图":["原图","原圖","オリジナル","None","원본","Оригинал"],"暖调":["暖调","暖調","暖色","Warm","따뜻하게","Тёплый"],"冷调":["冷调","冷調","寒色","Cool","차갑게","Холодный"],"黑白":["黑白","黑白","モノクロ","Mono","흑백","Ч/Б"],"浓郁":["浓郁","濃郁","ビビッド","Vivid","선명하게","Насыщенный"],"胶片":["胶片","底片","フィルム","Film","필름","Плёнка"],"轻虚化底":["轻虚化底","輕虛化底","ぼかし背景","Soft blur","부드러운 블러","Мягкое размытие"],"必看":["必看","必看","必見","Must see","필수","Обязательно"],"精选合集":["精选合集","精選合集","厳選まとめ","Best of","엄선 모음","Подборка"],"高清无水印":["高清无水印","高清無浮水印","高画質・透かしなし","HD no mark","고화질 무워터마크","HD без знака"],"完整版":["完整版","完整版","完全版","Full ver.","완전판","Полная версия"],"新作上线":["新作上线","新作上線","新作公開","New","신작","Новинка"],"私藏推荐":["私藏推荐","私藏推薦","秘蔵おすすめ","Picks","추천","Рекомендую"],"排序工具":["排序工具","排序工具","並べ替えツール","Order tool","정렬 도구","Сортировка"],"置顶":["置顶","置頂","先頭へ","Top","맨 위","В начало"],"上移":["上移","上移","上へ","Up","위로","Выше"],"下移":["下移","下移","下へ","Down","아래로","Ниже"],"置底":["置底","置底","末尾へ","Bottom","맨 아래","В конец"],"确认排序并改名":["确认排序并改名","確認排序並改名","確定して改名","Apply & rename","정렬 확정 및 이름변경","Применить"],"恢复文件名顺序":["恢复文件名顺序","恢復檔名順序","元の名前順に戻す","Reset names","원래 이름순","Исходный порядок"],"改名中…":["改名中…","改名中…","名前変更中…","Renaming…","이름 변경 중…","Переименование…"],"打包中…":["打包中…","打包中…","圧縮中…","Zipping…","압축 중…","Упаковка…"],"生成中…":["生成中…","產生中…","生成中…","Working…","생성 중…","Готовим…"],"复制中…":["复制中…","複製中…","コピー中…","Copying…","복사 중…","Копирование…"],"还没有打码过的图片，可点「导出所有」":["还没有打码过的图片，可点「导出所有」","還沒有打碼過的圖片，可點「匯出全部」","モザイク済みがありません。「すべて書き出し」を","No censored images yet","모자이크된 이미지가 없습니다","Обработанных пока нет"],"工作区里还没有图片":["工作区里还没有图片","工作區裡還沒有圖片","ワークスペースに画像がありません","No images in workspace","작업 공간에 이미지가 없습니다","В области нет изображений"],"没有可导出的图片":["没有可导出的图片","沒有可匯出的圖片","書き出す画像がありません","Nothing to export","내보낼 이미지가 없습니다","Нечего экспортировать"],"没有可排序的图片":["没有可排序的图片","沒有可排序的圖片","並べ替える画像がありません","Nothing to sort","정렬할 이미지가 없습니다","Нечего сортировать"],"先在「全部」里勾选要移动的图":["先在「全部」里勾选要移动的图","先在「全部」裡勾選要移動的圖","「すべて」で移動する画像を選択","Select images in All view","'전체'에서 선택하세요","Выберите в «Все»"],"这些图没有水印":["这些图没有水印","這些圖沒有浮水印","透かしがありません","No watermarks","워터마크 없음","Знаков нет"],"先选中要加水印的图片":["先选中要加水印的图片","先選取要加水印的圖片","透かしを入れる画像を選択","Select images first","먼저 이미지를 선택","Выберите изображения"],"水印文字不能为空":["水印文字不能为空","浮水印文字不能為空","透かし文字が空です","Watermark text is empty","워터마크 텍스트가 비었습니다","Текст пуст"],"还没确认":["还没确认","還沒確認","未確定","Not applied","미확정","Не применено"],"识别与打码":["识别与打码","辨識與打碼","検出とモザイク","Detect & censor","감지 및 모자이크","Детекция"],"高斯模糊":["高斯模糊","高斯模糊","ガウスぼかし","Gaussian","가우시안","Гаусс"],"纯色":["纯色","純色","単色","Solid","단색","Заливка"],"图片":["图片","圖片","画像","Image","이미지","Картинка"],"强度":["强度","強度","強さ","Strength","강도","Сила"],"边缘外扩":["边缘外扩","邊緣外擴","マージン","Margin","여백 확장","Расширение"],"阈值":["阈值","閾值","しきい値","Threshold","임계값","Порог"],"分辨率":["分辨率","解析度","解像度","Resolution","해상도","Разрешение"],"打码类别":["打码类别","打碼類別","対象カテゴリ","Categories","카테고리","Категории"],"开始打码":["开始打码","開始打碼","実行","Start","시작","Старт"],"未勾选时对全部图片执行":["未勾选时对全部图片执行","未勾選時對全部圖片執行","未選択時は全画像に実行","Runs on all when nothing checked","선택 없으면 전체 실행","Без выбора — на все"],"画笔":["画笔","畫筆","ブラシ","Brush","브러시","Кисть"],"色彩":["色彩","色彩","色","Color","색","Цвет"],"马赛克":["马赛克","馬賽克","モザイク","Mosaic","모자이크","Мозаика"],"模糊":["模糊","模糊","ぼかし","Blur","블러","Размытие"],"橡皮":["橡皮","橡皮","消しゴム","Eraser","지우개","Ластик"],"遮挡图片":["遮挡图片","遮擋圖片","隠す画像","Cover image","가림 이미지","Перекрытие"],"选择图片…":["选择图片…","選擇圖片…","画像を選択…","Choose image…","이미지 선택…","Выбрать…"],"笔刷大小":["笔刷大小","筆刷大小","ブラシサイズ","Brush size","브러시 크기","Размер кисти"],"导出已打码":["导出已打码","匯出已打碼","モザイク済みを書き出し","Export censored","모자이크 내보내기","Экспорт"],"码的粗细":["码的粗细","碼的粗細","モザイクの粗さ","Grain","모자이크 굵기","Крупность"],"删除勾选的文件":["删除勾选的文件","刪除勾選的檔案","選択ファイルを削除","Delete checked files","선택 파일 삭제","Удалить отмеченные файлы"],"撤回":["撤回","撤回","元に戻す","Undo","실행 취소","Отменить"],"重做":["重做","重做","やり直す","Redo","다시 실행","Повторить"],"批量命名":["批量命名","批次命名","一括リネーム","Batch rename","일괄 이름변경","Пакетный переименование"],"前缀":["前缀","前綴","接頭辞","Prefix","접두어","Префикс"],"起始号":["起始号","起始號","開始番号","Start","시작 번호","Начало"],"位数":["位数","位數","桁数","Digits","자릿수","Разряды"],"应用到勾选":["应用到勾选","套用到勾選","選択に適用","Apply to checked","선택에 적용","К отмеченным"],"应用到全部":["应用到全部","套用到全部","すべてに適用","Apply to all","전체에 적용","Ко всем"],"双击卡片上的文件名可以直接改名":["双击卡片上的文件名可以直接改名","雙擊卡片上的檔名可以直接改名","カードのファイル名をダブルクリックで改名","Double-click a filename to rename","카드의 파일명을 더블클릭해 이름 변경","Двойной клик по имени — переименовать"],"这个文件名已经存在":["这个文件名已经存在","這個檔名已經存在","その名前は既に存在します","Name already exists","이미 있는 이름입니다","Такое имя уже есть"],"已改名":["已改名","已改名","改名しました","Renamed","이름 변경됨","Переименовано"],"已删除":["已删除","已刪除","削除しました","Deleted","삭제됨","Удалено"],"已撤回":["已撤回","已撤回","元に戻しました","Undone","실행 취소됨","Отменено"],"没有可撤回的操作":["没有可撤回的操作","沒有可撤回的操作","元に戻す操作がありません","Nothing to undo","취소할 작업이 없습니다","Нечего отменять"]};

Object.assign(window.__TR__, {"共":["共","共","計","Total","총","Всего"],"张":["张","張","枚","items","장","шт."],"启用":["启用","啟用","有効","on","활성","активных"],"个文件夹":["个文件夹","個資料夾","フォルダ","folders","폴더","папок"],"已选":["已选","已選","選択","Selected","선택","Выбрано"],"未启用文件夹":["未启用文件夹","未啟用資料夾","有効なフォルダなし","No folders on","활성 폴더 없음","Нет активных папок"],"已勾选":["已勾选","已勾選","選択済み","Picked","선택됨","Отмечено"],"个文件":["个文件","個檔案","ファイル","files","개 파일","файлов"],"个图片文件":["个图片文件","個圖片檔案","画像ファイル","image files","이미지 파일","файлов изображений"],"当前":["当前","目前","現在","Now","현재","Сейчас"],"已确认排序":["已确认排序","已確認排序","確定済み","Ordered","정렬 확정","Отсортировано"],"还未确认排序":["还未确认排序","尚未確認排序","未確定","Not applied","미확정","Не отсортировано"],"已重做":["已重做","已重做","やり直しました","Redone","다시 실행됨","Повторено"],"删除失败":["删除失败","刪除失敗","削除に失敗","Delete failed","삭제 실패","Ошибка удаления"],"改名失败":["改名失败","改名失敗","改名に失敗","Rename failed","이름 변경 실패","Ошибка переименования"],"写入失败":["写入失败","寫入失敗","書き込み失敗","Write failed","쓰기 실패","Ошибка записи"],"请求出错：":["请求出错：","請求錯誤：","リクエストエラー：","Request error: ","요청 오류: ","Ошибка запроса: "],"脚本错误：":["脚本错误：","腳本錯誤：","スクリプトエラー：","Script error: ","스크립트 오류: ","Ошибка скрипта: "],"读取目录失败：":["读取目录失败：","讀取目錄失敗：","フォルダ読込失敗：","Load failed: ","폴더 읽기 실패: ","Ошибка чтения: "],"读取启用文件夹失败：":["读取启用文件夹失败：","讀取啟用資料夾失敗：","有効フォルダ読込失敗：","Load failed: ","활성 폴더 읽기 실패: ","Ошибка чтения: "],"图片加载失败：":["图片加载失败：","圖片載入失敗：","画像読込失敗：","Image load failed: ","이미지 로드 실패: ","Ошибка загрузки: "],"导出失败：":["导出失败：","匯出失敗：","書き出し失敗：","Export failed: ","내보내기 실패: ","Ошибка экспорта: "],"下载失败：":["下载失败：","下載失敗：","保存失敗：","Download failed: ","다운로드 실패: ","Ошибка загрузки: "],"排序失败：":["排序失败：","排序失敗：","並べ替え失敗：","Sort failed: ","정렬 실패: ","Ошибка сортировки: "],"导入失败：":["导入失败：","匯入失敗：","読込失敗：","Import failed: ","가져오기 실패: ","Ошибка импорта: "],"封面生成失败：":["封面生成失败：","封面產生失敗：","カバー生成失敗：","Cover failed: ","커버 생성 실패: ","Ошибка обложки: "],"已生成封面：":["已生成封面：","已產生封面：","カバー生成：","Cover created: ","커버 생성됨: ","Обложка создана: "],"已下载 ":["已下载 ","已下載 ","保存しました ","Downloaded ","저장됨 ","Скачано "],"已导出 ":["已导出 ","已匯出 ","書き出しました ","Exported ","내보냄 ","Экспортировано "],"已按新顺序改名 ":["已按新顺序改名 ","已按新順序改名 ","新しい順に改名 ","Renamed in order ","새 순서로 이름 변경 ","Переименовано по порядку "]," 张":[" 张"," 張"," 枚"," items"," 장"," шт."],"选中一张图，在它上面叠封面文字/滤镜/平台标识；确认后复制出一张新图「原名_封面.png」，原图不动。":["选中一张图，在它上面叠封面文字/滤镜/平台标识；确认后复制出一张新图「原名_封面.png」，原图不动。","選取一張圖，在它上面疊封面文字/濾鏡/平台標識；確認後複製出一張新圖「原名_封面.png」，原圖不動。","画像を選び、見出し・フィルター・プラットフォーム表示を重ねます。確定すると「元名_カバー.png」として複製され、元画像は変わりません。","Pick an image and overlay cover text, filters and platform badges. Applying makes a copy named “name_cover.png”; the original stays untouched.","이미지를 선택해 커버 문구·필터·플랫폼 배지를 얹습니다. 확정하면 '원본명_커버.png'로 복사되며 원본은 그대로입니다.","Выберите изображение и наложите текст, фильтр и значок платформы. При подтверждении создаётся копия «имя_обложка.png», оригинал не меняется."],"勾选＝启用；启用后这个文件夹的图片才会出现在 AI 编辑等工具里":["勾选＝启用；启用后这个文件夹的图片才会出现在 AI 编辑等工具里","勾選＝啟用；啟用後這個資料夾的圖片才會出現在 AI 編輯等工具裡","チェック＝有効。有効にしたフォルダの画像が AI 編集などのツールに表示されます。","Checked = enabled. Only enabled folders show up in AI Edit and the other tools.","체크 = 활성. 활성화한 폴더의 이미지만 AI 편집 등 도구에 나타납니다.","Галочка = активна. Только активные папки видны в AI-обработке и других инструментах."],"中间工作台里点文件夹进入、点图片勾选，勾好后点顶部「导入所选」；选中的图片会复制到程序内 cache 缓存再编辑，母本不动。":["中间工作台里点文件夹进入、点图片勾选，勾好后点顶部「导入所选」；选中的图片会复制到程序内 cache 缓存再编辑，母本不动。","中間工作台裡點資料夾進入、點圖片勾選，勾好後點頂部「匯入所選」；選取的圖片會複製到程式內 cache 快取再編輯，母本不動。","中央でフォルダを開き、画像を選んで上部の「選択を読み込み」を押します。選択画像はプログラム内の cache に複製され、元ファイルは変更されません。","Open a folder in the middle pane, tick images, then press Import selected. Copies go into the app cache for editing; originals stay untouched.","가운데 영역에서 폴더를 열고 이미지를 선택한 뒤 상단의 '선택 가져오기'를 누르세요. 선택한 이미지는 앱 cache로 복사되며 원본은 그대로입니다.","Откройте папку в центре, отметьте изображения и нажмите «Импорт выбранных». Копии попадут в cache программы, оригиналы не изменятся."],"在「全部」界面拖动卡片排好顺序（勾选多张可整组拖动），双击卡片文件名能直接改名。确认后按顺序把图片改名成「文件名_01」这种格式，导出时顺序就会直接生效。":["在「全部」界面拖动卡片排好顺序（勾选多张可整组拖动），双击卡片文件名能直接改名。确认后按顺序把图片改名成「文件名_01」这种格式，导出时顺序就会直接生效。","在「全部」介面拖動卡片排好順序（勾選多張可整組拖動），雙擊卡片檔名能直接改名。確認後按順序把圖片改名成「檔名_01」這種格式，匯出時順序就會直接生效。","「すべて」でカードをドラッグして順番を決めます（複数選択でまとめて移動）。カードのファイル名をダブルクリックすると改名できます。確定すると「名前_01」の形式に改名され、書き出し順に反映されます。","Drag cards in the All view to set the order (multi-select moves as a group). Double-click a filename to rename it. Applying renames files to “name_01” so the order carries into the export.","'전체' 화면에서 카드를 끌어 순서를 정합니다(여러 개 선택 시 함께 이동). 파일명을 더블클릭하면 바로 이름을 바꿀 수 있습니다. 확정하면 '이름_01' 형식으로 바뀌어 내보내기 순서에 반영됩니다.","Перетаскивайте карточки в режиме «Все» (несколько выбранных двигаются группой). Двойной клик по имени переименовывает. После применения файлы получают имена «имя_01», и порядок переходит в экспорт."]});
Object.assign(window.__TR__, {"导出所有已做更改": ["导出所有已做更改", "匯出所有已做更改", "変更をすべて書き出し", "Export all edits", "모든 편집 내보내기", "Экспорт всех изменений"], "只导已打码": ["只导已打码", "只匯出已打碼", "モザイク済みのみ", "Censored only", "모자이크만", "Только обработанные"], "单独导出已做更改": ["单独导出已做更改", "單獨匯出已做更改", "変更分だけ書き出し", "Export edits only", "편집본만 내보내기", "Только изменения"], "排序包": ["排序包", "排序包", "並べ替えパック", "Sort package", "정렬 패키지", "Пакет сортировки"], "排序包文件名（必填，例：小雅的夏天）": ["排序包文件名（必填，例：小雅的夏天）", "排序包檔名（必填，例：小雅的夏天）", "パック名（必須・例：小雅の夏）", "Package name (required, e.g. Xiaoya-Summer)", "패키지 이름(필수, 예: 소야의 여름)", "Имя пакета (обязательно)"], "单独导出排序后的包": ["单独导出排序后的包", "單獨匯出排序後的包", "並べ替え済みを書き出し", "Export sorted package", "정렬된 패키지 내보내기", "Экспорт отсортированного пакета"], "镜像目录里存的已经是排好的顺序，导出时直接按镜像文件打包，不再逐张重新合成，几秒就能出包。": ["镜像目录里存的已经是排好的顺序，导出时直接按镜像文件打包，不再逐张重新合成，几秒就能出包。", "鏡像目錄裡存的就是排好的順序，匯出時直接按鏡像檔案打包，不再逐張重新合成，幾秒就能出包。", "ミラー側は既に並べ替え済み。書き出しはそのファイルをそのまま固めるだけで、1枚ずつ合成し直しません。", "The mirror folder is already in order, so exporting just zips those files - no per-image re-compositing.", "미러 폴더가 이미 정렬되어 있어 그대로 압축합니다. 한 장씩 다시 합성하지 않습니다.", "В зеркале уже нужный порядок - упаковываем файлы напрямую, без повторной сборки."], "包内文件按当前顺序命名为「文件名_1、_2 … _n」，序号由排列自动生成，你不用定位数。确认排序前必须填好上面的文件名。": ["包内文件按当前顺序命名为「文件名_1、_2 … _n」，序号由排列自动生成，你不用定位数。确认排序前必须填好上面的文件名。", "包內檔案按目前順序命名為「檔名_1、_2 … _n」，序號由排列自動產生，你不用決定位數。確認排序前必須先填好上面的檔名。", "パック内は現在の順序で「名前_1、_2 … _n」。番号は自動で、桁数を決める必要はありません。確定前に上の名前を入れてください。", "Files inside are named NAME_1, NAME_2 ... NAME_n in the current order - numbering is automatic. Fill in the name above before confirming.", "압축 안 파일은 현재 순서대로 「이름_1, _2 … _n」로 저장됩니다. 번호는 자동입니다. 확정 전에 위 이름을 채우세요.", "Внутри архива - ИМЯ_1, ИМЯ_2 … ИМЯ_n по текущему порядку, нумерация автоматическая. Заполните имя до подтверждения."], "先填排序包文件名，再确认排序": ["先填排序包文件名，再确认排序", "先填排序包檔名，再確認排序", "先にパック名を入れてください", "Fill the package name first", "먼저 패키지 이름을 입력하세요", "Сначала укажите имя пакета"], "先勾选要导出的图片；整包导出用顶栏的「导出所有已做更改」": ["先勾选要导出的图片；整包导出用顶栏的「导出所有已做更改」", "先勾選要匯出的圖片；整包匯出用頂欄的「匯出所有已做更改」", "書き出す画像を先に選択してください。全体は上部の「変更をすべて書き出し」から。", "Pick the images to export first; use Export all edits in the top bar for everything.", "먼저 내보낼 이미지를 선택하세요. 전체는 상단의 '모든 편집 내보내기'를 쓰세요.", "Сначала отметьте изображения; всё сразу - кнопкой в верхней панели."], "标题样式": ["标题样式", "標題樣式", "タイトル様式", "Title style", "제목 스타일", "Стиль заголовка"], "刊头": ["刊头", "刊頭", "紙面ロゴ", "Masthead", "간판", "Шапка"], "极简": ["极简", "極簡", "ミニマル", "Minimal", "미니멀", "Минимал"], "空心": ["空心", "空心", "中抜き", "Outline", "외곽선", "Контур"], "印章": ["印章", "印章", "判子", "Stamp", "도장", "Штамп"], "竖排": ["竖排", "直排", "縦組み", "Vertical", "세로쓰기", "Вертикально"], "经典": ["经典", "經典", "クラシック", "Classic", "클래식", "Классика"], "平台标识": ["平台标识", "平台標識", "プラットフォーム", "Platform badges", "플랫폼 배지", "Значки платформ"], "默认样式": ["默认样式", "預設樣式", "既定スタイル", "Default style", "기본 스타일", "Стиль по умолчанию"], "官方牌": ["官方牌", "官方牌", "公式プレート", "Official", "공식 플레이트", "Официальный"], "细描边": ["细描边", "細描邊", "細枠", "Hairline", "얇은 테두리", "Тонкая рамка"], "色块印": ["色块印", "色塊印", "カラーブロック", "Color plate", "색면 스탬프", "Цветной блок"], "极简字": ["极简字", "極簡字", "文字のみ", "Text only", "텍스트만", "Только текст"], "贴纸": ["贴纸", "貼紙", "ステッカー", "Sticker", "스티커", "Стикер"], "跟随默认": ["跟随默认", "跟隨預設", "既定に従う", "Use default", "기본값 따르기", "Как по умолчанию"], "排布": ["排布", "排布", "配置", "Flow", "배열", "Раскладка"], "横排": ["横排", "橫排", "横組み", "Horizontal", "가로쓰기", "Горизонтально"], "整图链接行（单独显示在角落，可空）": ["整图链接行（单独显示在角落，可空）", "整圖連結行（單獨顯示在角落，可空）", "全体リンク行（隅に表示・空可）", "Full-image link line (corner, optional)", "전체 링크 줄(모서리, 선택)", "Строка ссылки (в углу, необязательно)"], "链接（可空，例：patreon.com/xxx）": ["链接（可空，例：patreon.com/xxx）", "連結（可空，例：patreon.com/xxx）", "リンク（空可・例：patreon.com/xxx）", "Link (optional, e.g. patreon.com/xxx)", "링크(선택, 예: patreon.com/xxx)", "Ссылка (необязательно)"], "确认封面并复制一张": ["确认封面并复制一张", "確認封面並複製一張", "カバーを確定して複製", "Apply cover & copy", "커버 확정 후 복사", "Применить обложку"], "重置封面设置": ["重置封面设置", "重設封面設定", "カバー設定をリセット", "Reset cover", "커버 설정 초기화", "Сбросить обложку"], "选中一张图，在它上面叠标题、滤镜和平台标识；确认后复制出一张新图「原名_封面.png」，原图不动。": ["选中一张图，在它上面叠标题、滤镜和平台标识；确认后复制出一张新图「原名_封面.png」，原图不动。", "選中一張圖，在它上面疊標題、濾鏡和平台標識；確認後複製出一張新圖「原名_封面.png」，原圖不動。", "画像を選び、タイトル・フィルタ・プラットフォームを重ねます。確定すると「元名_cover.png」として複製され、元画像はそのまま。", "Pick an image and layer a title, filter and platform badges on it. Confirming copies out NAME_cover.png and leaves the original untouched.", "이미지를 고르고 제목·필터·플랫폼 배지를 얹으세요. 확정하면 '원본명_cover.png'로 복사되고 원본은 그대로입니다.", "Выберите изображение и наложите заголовок, фильтр и значки. При подтверждении создаётся копия ИМЯ_cover.png, оригинал не меняется."], "「单独导出已做更改」只导出画面里勾选的那些图，并套上现在的封面设置（勾选后再点）；点「确认封面并复制一张」则是把封面焊进一张新图。": ["「单独导出已做更改」只导出画面里勾选的那些图，并套上现在的封面设置（勾选后再点）；点「确认封面并复制一张」则是把封面焊进一张新图。", "「單獨匯出已做更改」只匯出畫面裡勾選的那些圖，並套上目前的封面設定（勾選後再點）；點「確認封面並複製一張」則是把封面焊進一張新圖。", "「変更分だけ書き出し」は選択中の画像だけに現在のカバー設定を適用して書き出します。「カバーを確定して複製」は1枚に焼き込みます。", "Export edits only exports the ticked images with the current cover settings; Apply cover & copy burns the cover into a new image.", "'편집본만 내보내기'는 선택한 이미지에 현재 커버 설정을 적용해 내보냅니다. '커버 확정 후 복사'는 한 장에 새깁니다.", "«Только изменения» экспортирует отмеченные изображения с текущими настройками; «Применить обложку» впечатывает её в новую копию."]});


Object.assign(window.__TR__, {"在右侧粘贴一个本地路径，或点「打开」开始浏览":["在右侧粘贴一个本地路径，或点「打开」开始浏览","在右側貼上一個本機路徑，或點「開啟」開始瀏覽","右側にローカルパスを貼るか「開く」を押してください","Paste a local path on the right, or press Open to browse","오른쪽에 로컬 경로를 붙여넣거나 '열기'를 누르세요","Вставьте локальный путь справа или нажмите «Открыть»"],"粘贴本地路径，如 E:\\图片\\待打码":["粘贴本地路径，如 E:\\图片\\待打码","貼上本機路徑，如 E:\\圖片\\待打碼","ローカルパスを貼り付け（例 E:\\画像\\素材）","Paste a local path, e.g. E:\\images\\todo","로컬 경로 붙여넣기 (예: E:\\images\\todo)","Вставьте путь, напр. E:\\images\\todo"]});

Object.assign(window.__TR__, {"新文件名（不含扩展名）":["新文件名（不含扩展名）","新檔名（不含副檔名）","新しいファイル名（拡張子なし）","New file name (no ext)","새 파일 이름 (확장자 제외)","Новое имя (без расширения)"],"应用改名":["应用改名","套用改名","改名を適用","Apply rename","이름 변경 적용","Применить"],"重命名勾选的文件":["重命名勾选的文件","重新命名勾選的檔案","選択ファイルを改名","Rename checked files","선택 파일 이름변경","Переименовать отмеченные"],"先输入新文件名":["先输入新文件名","先輸入新檔名","新しい名前を入力","Type a name first","먼저 이름을 입력","Сначала имя"],"勾选一张就是这个名字；勾选多张会自动加 _1、_2 后缀。改名会同步到磁盘，仓库/工作区里看到的都是新名字。":["勾选一张就是这个名字；勾选多张会自动加 _1、_2 后缀。改名会同步到磁盘，仓库/工作区里看到的都是新名字。","勾選一張就是這個名字；勾選多張會自動加 _1、_2 後綴。改名會同步到磁碟，倉庫/工作區看到的都是新名字。","1枚ならその名前、複数なら _1、_2 を付けます。ディスクにも反映されます。","One file keeps the name; several get _1, _2 suffixes. Applies to disk too.","1장이면 그 이름, 여러 장이면 _1, _2 접미사가 붙습니다. 디스크에도 반영됩니다.","Один файл — это имя, несколько — с _1, _2. Меняется и на диске."]});
Object.assign(window.__TR__, {"标题": ["标题", "標題", "タイトル", "Title", "제목", "Заголовок"], "必看": ["必看", "必看", "必見", "Must see", "필수", "Must see"], "精选合集": ["精选合集", "精選合集", "厳選まとめ", "Best of", "엄선 모음", "Подборка"], "高清无水印": ["高清无水印", "高畫質無浮水印", "高画質・透かしなし", "HD, no watermark", "고화질·워터마크 없음", "HD без водяных знаков"], "完整版": ["完整版", "完整版", "完全版", "Full ver.", "완전판", "Полная версия"], "新作上线": ["新作上线", "新作上線", "新作公開", "New release", "신작 공개", "Новинка"], "私藏推荐": ["私藏推荐", "私藏推薦", "秘蔵おすすめ", "Personal pick", "개인 추천", "Личная рекомендация"]});

const ALL_CLASSES = __ALL_CLASSES__;
const DEF_CLASSES = __DEFAULT_CLASSES__;
const ZH = __ZH__;

const $ = id => document.getElementById(id);
const el = (tag, cls, txt) => { const n = document.createElement(tag); if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; };
const sleep = ms => new Promise(r => setTimeout(r, ms));
window.addEventListener("error", e => {
  try { toast("脚本错误：" + (e.message || e), 6000); } catch (_) {}
});
window.addEventListener("unhandledrejection", e => {
  const r = e && e.reason;
  try { toast("请求出错：" + ((r && r.message) || r), 6000); } catch (_) {}
});


/* ================= 语言（i18n） ================= */
const LANGS = ["zh-CN", "zh-TW", "ja", "en", "ko", "ru"];
const LANG_META = [
  { id: "zh-CN", name: "简体中文", cls: "cn" },
  { id: "zh-TW", name: "繁體中文", cls: "tw" },
  { id: "ja", name: "日本語", cls: "jp" },
  { id: "en", name: "English", cls: "us" },
  { id: "ko", name: "한국어", cls: "kr" },
  { id: "ru", name: "Русский", cls: "ru" },
];
/* TR[中文原文] = [简中, 繁中, 日, 英, 韩, 俄]；缺省回落中文 */
const TR = window.__TR__ || {};
let LANG = 0;
function tf(s) { const a = TR[s]; return a && a[LANG] ? a[LANG] : s; }
function langIdx(id) { const i = LANGS.indexOf(id); return i < 0 ? 0 : i; }

function flagEl(cls) { const s = document.createElement("span"); s.className = "flag " + cls; return s; }

const _TRREV = (() => {
  const m = Object.create(null);
  for (const k in TR) { const a = TR[k]; for (let i = 0; i < a.length; i++) { const v = a[i]; if (v && !(v in m)) m[v] = k; } }
  return m;
})();
function _trKey(s) { if (Object.prototype.hasOwnProperty.call(TR, s)) return s; const k = _TRREV[s]; return k || null; }
function translateNow(root) {
  if (!root) return;
  const walk = node => {
    for (let n = node.firstChild; n; n = n.nextSibling) {
      if (n.nodeType === 3) {
        const raw = n.nodeValue, s = raw.trim();
        if (!s) continue;
        const k = _trKey(n.__k !== undefined ? n.__k : s);
        if (!k) continue;
        n.__k = k;
        const a = TR[k], dn = a ? (a[LANG] || a[0]) : null;
        if (dn && dn !== s) n.nodeValue = raw.replace(s, dn);
      } else if (n.nodeType === 1 && n.tagName !== "SCRIPT" && n.tagName !== "STYLE") {
        walk(n);
      }
    }
  };
  walk(root);
  const els = root.querySelectorAll ? [root].concat(Array.from(root.querySelectorAll("[title],[placeholder]"))) : [root];
  for (const n of els) {
    if (!n.getAttribute) continue;
    for (const at of ["title", "placeholder"]) {
      const v = n.getAttribute(at); if (!v) continue;
      n.__at = n.__at || {};
      const k = _trKey(n.__at[at] !== undefined ? n.__at[at] : v.trim());
      if (!k) continue;
      n.__at[at] = k;
      const a = TR[k], dn = a ? (a[LANG] || a[0]) : null;
      if (dn) n.setAttribute(at, dn);
    }
  }
}
let _trT = 0;
function trSoon() {
  clearTimeout(_trT);
  _trT = setTimeout(() => { try { translateNow(document.body); } catch (e) {} }, 25);
}
const _mo = new MutationObserver(() => trSoon());
window.addEventListener("load", () => {
  try { _mo.observe(document.body, { childList: true, subtree: true, characterData: true }); } catch (e) {}
});

function renderLangButtons(host) {
  host.innerHTML = "";
  LANG_META.forEach((L, i) => {
    const b = el("button", "lbtn" + (i === LANG ? " on" : ""));
    b.appendChild(flagEl(L.cls));
    b.appendChild(el("span", null, L.name));
    b.onclick = () => pickLang(i, b);
    host.appendChild(b);
  });
}
async function pickLang(i, fromEl) { await setLang(i, { animate: true, from: fromEl }); }
function paintSetFlag() {
  const box = $("setFlag"); if (!box) return;
  box.innerHTML = ""; box.appendChild(flagEl(LANG_META[LANG].cls));
}
async function setLang(i, opts) {
  LANG = i;
  document.documentElement.lang = LANGS[i];
  try { localStorage.setItem("cb-lang", LANGS[i]); } catch (e) {}
  paintSetFlag();
  renderLangButtons($("setLangs"));
  trSoon();
  try { await api("/api/auth/lang", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ lang: LANGS[i] }) }); } catch (e) {}
  if (opts && opts.animate) await flyFlagToSettings(opts.from);
  renderGrid(); buildPanel();
}

/* 语言标志从选择弹窗"缩"进左下角设置 */
function flyFlagToSettings(fromEl) {
  return new Promise(res => {
    const target = $("setBtn"); if (!target) return res();
    const tr = target.getBoundingClientRect();
    const src = fromEl ? fromEl.getBoundingClientRect() : { left: innerWidth / 2, top: innerHeight / 2, width: 24, height: 16 };
    const f = flagEl(LANG_META[LANG].cls);
    f.style.cssText = "position:fixed;left:" + src.left + "px;top:" + src.top + "px;width:24px;height:16px;z-index:260;border-radius:3px;transition:none";
    document.body.appendChild(f);
    const dx = tr.left + 2 - src.left, dy = tr.top + (tr.height - 16) / 2 - src.top;
    const anim = f.animate(
      [{ transform: "translate(0,0) scale(1)", opacity: 1 },
       { transform: `translate(${dx * .55}px,${dy * .55}px) scale(.55)`, opacity: .9, offset: .6 },
       { transform: `translate(${dx}px,${dy}px) scale(.34)`, opacity: 0 }],
      { duration: 720, easing: "cubic-bezier(.5,.05,.2,1)" });
    anim.onfinish = () => { f.remove(); paintSetFlag(); res(); };
  });
}

/* ================= 账号 ================= */
const AUTH = { token: localStorage.getItem("cb-token") || "", user: null, registered: false };

function showGate() {
  const w = $("gateWrap"); if (!w) return;
  w.classList.add("on");
  renderGate();
}
function hideGate() { $("gateWrap").classList.remove("on"); }

function renderGate() {
  const card = $("gateCard"); card.innerHTML = "";
  const first = !AUTH.registered;
  card.appendChild(el("h1", null, first ? "创建账号" : "登录"));
  card.appendChild(el("div", "sub", first
    ? "第一次使用请先创建账号（需要注册码），创建后再选语言。之后打开本页面会自动登录。"
    : "输入账号密码继续。账号只保存在本机程序目录里。"));
  const mk = (label, type, ph) => {
    const f = el("div", "fld");
    f.appendChild(el("label", null, label));
    const i = el("input"); i.type = type; i.placeholder = ph || "";
    f.appendChild(i); card.appendChild(f); return i;
  };
  const u = mk("用户名", "text", "2-24 个字符");
  const p = mk("密码", "password", "至少 4 位");
  const c = first ? mk("注册码", "text", "Hchime-ai") : null;
  const err = el("div", "gate-err"); card.appendChild(err);
  const btn = el("button", "primary", first ? "创建并进入" : "登录");
  card.appendChild(btn);
  const foot = el("div", "gate-foot");
  const a = el("a", null, first ? "已经有账号？去登录" : "创建新账号");
  a.onclick = async () => {
    if (AUTH.registered) {
      // 已有账号体系下"创建新账号"同样需要注册码
      AUTH.registered = false; renderGate();
    } else {
      AUTH.registered = true; renderGate();
    }
  };
  foot.appendChild(a); card.appendChild(foot);

  const go = async () => {
    err.textContent = "";
    const name = u.value.trim(), pwd = p.value;
    if (name.length < 2) { err.textContent = tf("用户名 2-24 个字符"); return; }
    if (pwd.length < 4) { err.textContent = tf("密码至少 4 位"); return; }
    btn.disabled = true; const old = btn.textContent; btn.textContent = "…";
    try {
      const url = first ? "/api/auth/register" : "/api/auth/login";
      const body = first ? { name, pwd, code: (c ? c.value : "").trim() } : { name, pwd };
      const j = await fetch(url, {
        method: "POST", credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then(r => r.json());
      if (!j.ok) throw new Error(j.error || "失败");
      AUTH.token = j.token || ""; AUTH.user = j.user || { name };
      try { localStorage.setItem("cb-token", AUTH.token); } catch (e) {}
      AUTH.registered = true;
      hideGate();
      await enterApp(j.first === true);
    } catch (e) {
      err.textContent = e.message;
    } finally { btn.disabled = false; btn.textContent = old; }
  };
  btn.onclick = go;
  [u, p, c].forEach(i => i && (i.onkeydown = ev => { if (ev.key === "Enter") go(); }));
  setTimeout(() => u.focus(), 60);
}

/* 进入主界面；first=true 时先选语言 */
async function enterApp(first) {
  if (first && !localStorage.getItem("cb-lang-done")) {
    const w = $("langWrap");
    w.classList.add("on");
    const grid = $("langGrid");
    renderLangButtons2(grid);
    return;
  }
  const saved = langIdx(AUTH.user && AUTH.user.lang || localStorage.getItem("cb-lang") || "zh-CN");
  LANG = saved;
  document.documentElement.lang = LANGS[LANG];
  paintSetFlag(); renderLangButtons($("setLangs"));
  startApp();
  trSoon();
}
function renderLangButtons2(host) {
  host.innerHTML = "";
  LANG_META.forEach((L, i) => {
    const b = el("button", "lbtn");
    b.appendChild(flagEl(L.cls));
    b.appendChild(el("span", null, L.name));
    b.onclick = async () => {
      $("langWrap").classList.remove("on");
      try { localStorage.setItem("cb-lang-done", "1"); } catch (e) {}
      LANG = i; document.documentElement.lang = LANGS[i];
      try { localStorage.setItem("cb-lang", LANGS[i]); } catch (e) {}
      paintSetFlag(); renderLangButtons($("setLangs"));
      startApp();
      await flyFlagToSettings(b);
      trSoon();
      renderGrid(); buildPanel();
    };
    host.appendChild(b);
  });
}

/* ================= 设置浮层 ================= */
function openSetPop(on) {
  const p = $("setPop");
  const isOn = on == null ? !p.classList.contains("on") : on;
  p.classList.toggle("on", isOn);
  if (isOn) {
    $("setUser").textContent = (AUTH.user && AUTH.user.name) || "—";
    renderLangButtons($("setLangs"));
    paintThemeSeg();
  }
}
function paintThemeSeg() {
  const cur = document.documentElement.dataset.theme || "dark";
  document.querySelectorAll("#themeSeg button").forEach(b => {
    b.classList.toggle("primary", b.dataset.theme === cur);
  });
}
$("setBtn").onclick = () => openSetPop();
$("setLogout").onclick = async () => {
  try { await fetch("/api/auth/logout", { method: "POST", credentials: "same-origin" }); } catch (e) {}
  AUTH.token = ""; AUTH.user = null;
  try { localStorage.removeItem("cb-token"); } catch (e) {}
  openSetPop(false);
  showGate();
};
document.querySelectorAll("#themeSeg button").forEach(b => {
  b.onclick = () => { applyTheme(b.dataset.theme); paintThemeSeg(); };
});
document.addEventListener("click", ev => {
  const p = $("setPop");
  if (!p.classList.contains("on")) return;
  if (p.contains(ev.target) || ev.target.closest("#setBtn")) return;
  openSetPop(false);
});
document.addEventListener("keydown", ev => { if (ev.key === "Escape") openSetPop(false); });

const EDIT_MODES = ["ai", "manual", "mark", "cover", "sort"];
const store = {
  work: { name: "", count: 0, bytes: 0, path: "", folder: "", enabled: [], order: [], sorted: false },
  wsb: { path: "", rel: "", parent: "", dirs: [], files: [] },
  mode: "repo",
  layout: "all",
  root: "",
  items: [],
  pool: new Map(),
  enabled: new Set(),
  order: [],
  sorted: false,
  pkg: "",
  curId: null,
  sel: new Set(),
  anchor: null,
  seq: 0,
  browse: { path: "", parent: "", dirs: [], files: [], sel: new Set(), anchor: null },
};
function isEdit(m) { return EDIT_MODES.includes(m || store.mode); }
let _saveT = 0;
function saveState() {
  clearTimeout(_saveT);
  _saveT = setTimeout(() => {
    api("/api/workspace/state", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: [...store.enabled], order: store.order || [], sorted: !!store.sorted, pkg: store.pkg || "" }),
    }).catch(() => {});
  }, 260);
}

/* ---------------- 基础 UI ---------------- */
function toast(msg, ms = 2000) {
  const t = $("toast"); t.textContent = msg; t.classList.add("on");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("on"), ms);
}
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("cb-theme", t); } catch (e) {}
}
applyTheme(localStorage.getItem("cb-theme") || "dark");

function setMode(m) {
  store.mode = m;
  if ((m === "repo" || m === "sort") && store.layout === "single") setLayout("all");
  document.querySelectorAll("#rail .rail-btn[data-mode]").forEach(b => b.classList.toggle("on", b.dataset.mode === m));
  document.body.classList.toggle("ai", EDIT_MODES.includes(m) || m === "work");
  document.body.classList.toggle("mark", m === "mark");
  document.body.classList.toggle("sort", m === "sort");
  applyCoverPreviewFilter();
  if (m === "mark") {
    if (!store.enabled.size) toast("先在工作区勾选文件夹，再来加水印");
    setTimeout(pumpPreview, 60);
  } else if (wmPlay) { toggleWmPlay(false); }
  if (m === "work") {
    loadWork();
    if (!store.wsb || !store.wsb.path) wsOpen(""); else wsOpen(store.wsb.path);
  }
  if (EDIT_MODES.includes(m)) {
    syncItems().then(() => {
      if (store.mode !== m) return;
      buildPanel(); renderGrid(); updateSelInfo(); pumpPreview();
      if (m === "cover") requestAnimationFrame(fitWholeForCover);
    });
    return;
  }
  buildPanel();
  renderGrid();
  updateSelInfo();
}
document.querySelectorAll("#rail .rail-btn[data-mode]").forEach(b => b.onclick = () => setMode(b.dataset.mode));

function setLayout(l) {
  if (l === "single" && (store.mode === "repo" || store.mode === "sort")) l = "all";  // 仓库 / 排序没有单图画布
  store.layout = l;
  document.querySelectorAll("#viewseg button").forEach(b => b.classList.toggle("on", b.dataset.layout === l));
  const single = l === "single";
  $("stage").style.display = single ? "block" : "none";
  $("gridWrap").style.display = single ? "none" : "block";
  const dl = $("dlOneBtn");
  if (dl) dl.style.display = (single && store.mode !== "repo") ? "" : "none";
  if (single) { openSingle(store.curId || (store.items[0] && store.items[0].id)); }
  else { pane.reset(); renderGrid(); }
  updateNavBtns();
}
document.querySelectorAll("#viewseg button").forEach(b => b.onclick = () => setLayout(b.dataset.layout));

/* ---------------- 单图翻页 ---------------- */
function navList() {
  if (store.mode === "repo") {
    return store.browse.files.map(f => ({ id: f.path, name: f.name, path: f.path, w: 0, h: 0, repo: true }));
  }
  return store.items;
}
function updateNavBtns() {
  const single = store.layout === "single" && store.mode !== "repo";
  const list = single ? store.items : [];
  const i = list.findIndex(x => x.id === store.curId);
  const pv = $("navPrev"), nx = $("navNext");
  if (!pv || !nx) return;
  pv.classList.toggle("hide", !single);
  nx.classList.toggle("hide", !single);
  pv.disabled = !single || i <= 0;
  nx.disabled = !single || i < 0 || i >= list.length - 1;
  pv.style.opacity = pv.disabled ? ".28" : "";
  nx.style.opacity = nx.disabled ? ".28" : "";
}
function stepImage(dir) {
  if (store.mode === "repo") return;
  const i = store.items.findIndex(x => x.id === store.curId);
  const j = Math.min(store.items.length - 1, Math.max(0, i + dir));
  if (j === i || i < 0) return;
  const it = store.items[j];
  store.curId = it.id;
  if (it.repo) { toast("这张要先导入进缓存才能编辑"); return; }
  openSingle(it.id);
}
$("navPrev").onclick = () => stepImage(-1);
$("navNext").onclick = () => stepImage(1);

/* ---------------- 仓库 ---------------- */
async function api(url, opt) {
  opt = opt || {};
  opt.credentials = "same-origin";
  opt.headers = Object.assign({ "X-Auth-Token": (window.AUTH && AUTH.token) || "" }, opt.headers || {});
  const r = await fetch(url, opt);
  if (r.status === 401) { try { showGate(); } catch (e) {} }
  const ct = r.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    const j = await r.json();
    if (j.error) throw new Error(j.error);
    return j;
  }
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r;
}

function setPath(p) {
  store.root = p || "";
  const box = $("curPath");
  box.innerHTML = "";
  if (!p) { box.textContent = "未选择仓库目录"; return; }
  const b = el("b", null, p);
  box.appendChild(b);
}

async function openDir(path) {
  try {
    const j = await api("/api/fs?path=" + encodeURIComponent(path || ""));
    setPath(j.path);
    store.browse = {
      path: j.path || "", parent: j.parent || "", dirs: j.dirs || [],
      files: (j.files || []).filter(f => f.isImage), sel: store.browse.sel, anchor: null,
    };
    store.root = j.path || store.root;
    if (store.mode === "repo") { renderGrid(); renderRepoPanel(j); updateSelInfo(); }
    return j;
  } catch (e) { toast("读取目录失败：" + e.message, 3000); }
}

let _uidSeq = 0;
function mkItem(f) {
  const p = f.path;
  return {
    id: ++store.seq, uid: "u" + (++_uidSeq), name: f.name, rel: f.rel || f.name, path: p, src: f.src || p,
    w: f.w || 0, h: f.h || 0,
    thumb: f.thumb || ("/api/thumb?w=420&path=" + encodeURIComponent(p)),
    sticks: [], redo: [], baseSrc: null, paintDirty: false, done: false, busy: false,
    thumbCache: null, wmLayers: [],
  };
}
function dirOf(it) {
  const r = String(it.rel || "").replace(/\\/g, "/");
  const i = r.indexOf("/");
  return i < 0 ? "" : r.slice(0, i);
}
function poolItem(f) {
  let it = store.pool.get(f.path);
  if (!it) { it = mkItem(f); store.pool.set(f.path, it); }
  else {
    it.name = f.name || it.name;
    it.rel = f.rel || it.rel;
    it.w = f.w || it.w; it.h = f.h || it.h;
  }
  return it;
}
function addItems(files) {
  let added = 0;
  for (const f of files) {
    if (store.pool.has(f.path)) continue;
    store.pool.set(f.path, mkItem(f)); added++;
  }
  return added;
}
async function syncItems() {
  if (!store.enabled.size) {
    store.items = [];
    store.curId = null;
    cur = null;
    return store.items;
  }
  try {
    const q = "/api/wsfiles?dirs=" + encodeURIComponent([...store.enabled].join("|"));
    const j = await api(q);
    for (const f of j.files || []) poolItem(f);
  } catch (e) { toast("读取启用文件夹失败：" + e.message, 3000); }
  const eds = store.enabled;
  let list = [...store.pool.values()].filter(it => eds.has(dirOf(it)));
  if (store.order && store.order.length) {
    const idx = new Map(store.order.map((p, i) => [p, i]));
    list.sort((a, b) => (idx.has(a.rel) ? idx.get(a.rel) : 1e9) - (idx.has(b.rel) ? idx.get(b.rel) : 1e9));
  } else {
    list.sort((a, b) => String(a.rel).localeCompare(String(b.rel), "zh-CN"));
  }
  store.items = list;
  if (!store.items.some(i => i.id === store.curId)) store.curId = store.items[0] ? store.items[0].id : null;
  if (cur && !store.items.includes(cur)) cur = null;
  return store.items;
}

/* ---------------- 网格 ---------------- */
function renderGrid() {
  if (store.mode === "repo") { renderBrowse(); return; }
  if (store.mode === "work") { renderWork(); return; }
  document.body.classList.toggle("ai", EDIT_MODES.includes(store.mode) || store.mode === "work");
  const _sb = document.querySelector('#viewseg button[data-layout="single"]');
  if (_sb) _sb.disabled = store.mode === "sort";
  const grid = $("grid");
  grid.innerHTML = "";
  $("emptyTip").style.display = store.items.length ? "none" : "grid";
  $("emptyTip").textContent = store.enabled.size
    ? "已启用的文件夹里没有图片"
    : "还没有启用任何文件夹 —— 去左侧「工作区」勾选要编辑的文件夹";
  store.items.forEach((it, idx) => {
    const c = el("div", "card");
    c.dataset.id = it.id;
    if (it.done) c.classList.add("done");
    if (it.id === store.curId) c.classList.add("cur");
    if (store.sel.has(it.id)) c.classList.add("sel");
    const box = el("div", "box");
    const img = el("img"); img.loading = "lazy"; img.src = it.thumbCache || it.thumb;
    box.appendChild(img);
    box.appendChild(el("div", "badge", "已打码"));
    const nw = (it.wmLayers || []).length;
    if (nw) box.appendChild(el("div", "badge wm", "水印 " + nw));
    if (store.mode === "sort") box.appendChild(el("div", "seq", String(idx + 1)));
    const pick = el("div", "pick" + (store.sel.has(it.id) ? " on" : ""), store.sel.has(it.id) ? "✓" : "");
    pick.dataset.pick = it.id;
    box.appendChild(pick);
    c.appendChild(box);
    const nmEl = el("div", "nm", it.name);
    if (store.mode === "sort") nmEl.ondblclick = ev => { ev.stopPropagation(); startRename(it, nmEl); };
    c.appendChild(nmEl);
    if (store.mode === "sort") { c.draggable = true; bindSortCard(c, it); }
    c.onclick = ev => {
      if (ev.target.dataset.pick) { togglePick(+ev.target.dataset.pick, ev.shiftKey); ev.stopPropagation(); return; }
      if (store.mode === "sort") { togglePick(it.id, ev.shiftKey); return; }
      if (isEdit(store.mode) && ev.shiftKey) { togglePick(it.id, true); return; }
      store.curId = it.id; setLayout("single");
    };
    grid.appendChild(c);
  });
}

/* ---------------- 工作区浏览（缓存镜像目录） ---------------- */
async function wsOpen(path) {
  const q = path ? "?path=" + encodeURIComponent(path) : "";
  try {
    const j = await api("/api/wsbrowse" + q);
    store.wsb = { path: j.path, rel: j.rel || "", parent: j.parent || "",
                  dirs: j.dirs || [], files: j.files || [] };
    renderGrid(); updateSelInfo();
  } catch (e) { toast("打不开这个文件夹：" + e.message, 3000); }
}

function renderWork() {
  const grid = $("grid");
  const w = store.wsb;
  grid.innerHTML = "";
  const total = (w.dirs ? w.dirs.length : 0) + (w.files ? w.files.length : 0);
  $("emptyTip").style.display = total ? "none" : "grid";
  $("emptyTip").textContent = w.rel
    ? "这个文件夹是空的"
    : "工作区还是空的 —— 去左侧「仓库」里选图片导入，导入会在这里生成一个批次文件夹";

  if (w.rel) {
    const up = el("div", "card dir-card");
    const box = el("div", "box"); box.textContent = "⬆"; box.classList.add("dirbox");
    up.appendChild(box);
    up.appendChild(el("div", "nm", "上一级"));
    up.onclick = () => wsOpen(w.parent);
    grid.appendChild(up);
  }
  for (const d of w.dirs || []) {
    const c = el("div", "card dir-card");
    const on = store.enabled.has(d.name);
    if (on) c.classList.add("enabled");
    const box = el("div", "box"); box.textContent = "📁"; box.classList.add("dirbox");
    const pick = el("div", "pick" + (on ? " on" : ""), on ? "✓" : "");
    pick.dataset.dirpick = d.name;
    pick.style.display = "grid";
    pick.title = "勾选＝启用；启用后这个文件夹的图片才会出现在 AI 编辑等工具里";
    box.appendChild(pick);
    c.appendChild(box);
    c.appendChild(el("div", "nm", d.name + (on ? " · 已启用" : "")));
    c.onclick = ev => {
      if (ev.target.dataset.dirpick) { toggleDir(ev.target.dataset.dirpick); ev.stopPropagation(); return; }
      wsOpen(d.path);
    };
    grid.appendChild(c);
  }
  for (const f of w.files || []) {
    const it = store.items.find(x => x.path === f.path);
    const c = el("div", "card");
    if (it) { c.dataset.id = it.id; if (it.done) c.classList.add("done"); if (it.id === store.curId) c.classList.add("cur"); }
    const box = el("div", "box");
    const img = el("img"); img.loading = "lazy";
    img.src = (it && (it.thumbCache || it.thumb)) || ("/api/thumb?w=420&path=" + encodeURIComponent(f.path));
    box.appendChild(img);
    box.appendChild(el("div", "badge", "已打码"));
    c.appendChild(box);
    c.appendChild(el("div", "nm", f.name));
    c.onclick = () => openWorkImage(f);
    grid.appendChild(c);
  }
}

function toggleDir(name) {
  if (!name) return;
  if (store.enabled.has(name)) store.enabled.delete(name); else store.enabled.add(name);
  saveState();
  renderWork(); updateSelInfo();
  if (isEdit(store.mode)) syncItems().then(() => { renderGrid(); updateSelInfo(); });
}

function openWorkImage(f) {
  let it = store.pool.get(f.path);
  if (!it) {
    const relRoot = store.wsb ? (store.wsb.rel ? store.wsb.rel + "/" : "") : "";
    it = mkItem({ ...f, rel: relRoot + f.name, src: "" });
    store.pool.set(it.path, it);
  }
  store.curId = it.id;
  setLayout("single");
}

/* ---------------- 仓库浏览（中央工作台） ---------------- */
function renderBrowse() {
  const grid = $("grid");
  const b = store.browse;
  grid.innerHTML = "";
  const has = b.dirs.length + b.files.length > 0;
  $("emptyTip").style.display = has ? "none" : "grid";
  $("emptyTip").textContent = b.path ? "这个文件夹里没有图片或子文件夹" : "在右侧粘贴一个本地路径，或点「打开」开始浏览";

  if (b.path) {
    const up = el("div", "card dir-card");
    const box = el("div", "box"); box.textContent = "⬆"; box.classList.add("dirbox");
    up.appendChild(box);
    up.appendChild(el("div", "nm", ".. 上一级"));
    up.onclick = () => openDir(b.parent);
    grid.appendChild(up);
  }
  for (const d of b.dirs) {
    const c = el("div", "card dir-card");
    const box = el("div", "box"); box.textContent = "📁"; box.classList.add("dirbox");
    c.appendChild(box);
    c.appendChild(el("div", "nm", d.name));
    c.ondblclick = () => openDir(d.path);
    c.onclick = ev => { if (ev.detail >= 2) return; openDir(d.path); };
    grid.appendChild(c);
  }
  for (const f of b.files) {
    const c = el("div", "card");
    const box = el("div", "box");
    const img = el("img"); img.loading = "lazy";
    img.src = f.thumb || ("/api/thumb?w=420&path=" + encodeURIComponent(f.path));
    box.appendChild(img);
    const on = b.sel.has(f.path);
    const pick = el("div", "pick" + (on ? " on" : ""), on ? "✓" : "");
    pick.style.display = "grid";
    box.appendChild(pick);
    c.appendChild(box);
    c.appendChild(el("div", "nm", f.name));
    c.onclick = ev => { toggleBrowsePick(f.path, ev.shiftKey); };
    grid.appendChild(c);
  }
}

function toggleBrowsePick(path, shift) {
  const b = store.browse;
  const list = b.files.map(f => f.path);
  const idx = list.indexOf(path);
  if (shift && b.anchor != null) {
    const a = list.indexOf(b.anchor);
    if (a >= 0 && idx >= 0) {
      const [s, e] = a < idx ? [a, idx] : [idx, a];
      const on = !b.sel.has(path);
      for (let k = s; k <= e; k++) { if (on) b.sel.add(list[k]); else b.sel.delete(list[k]); }
    }
  } else if (b.sel.has(path)) b.sel.delete(path);
  else b.sel.add(path);
  b.anchor = path;
  renderBrowse(); updateSelInfo();
}

async function importSelected() {
  const paths = [...store.browse.sel];
  store.browse.sel = new Set();
  if (!paths.length) { toast("先在中间勾选要导入的图片"); return; }
  const btn = $("importBtn");
  btn.disabled = true; btn.textContent = "复制中…";
  try {
    const r = await fetch("/api/mirror", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paths, root: store.browse.path }),
    });
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || ("HTTP " + r.status));
    const n = addItems(j.items);
    if (j.batch) store.enabled.add(j.batch);
    store.order = []; store.sorted = false;
    saveState();
    store.browse.sel.clear();
    store.root = store.browse.path;
    renderBrowse(); updateSelInfo();
    await loadWork(true);
    wsOpen("");
    if (isEdit(store.mode)) { await syncItems(); renderGrid(); updateSelInfo(); }
    toast(`已复制 ${j.count} 张到缓存并导入${n < j.count ? "（" + (j.count - n) + " 张重复）" : ""}`);
  } catch (e) {
    toast("导入失败：" + e.message, 3000);
  } finally {
    btn.disabled = false;
    updateSelInfo();
  }
}

function togglePick(id, shift) {
  const idx = store.items.findIndex(i => i.id === id);
  if (shift && store.anchor != null) {
    const a = store.items.findIndex(i => i.id === store.anchor);
    if (a >= 0) {
      const [s, e] = a < idx ? [a, idx] : [idx, a];
      const on = !store.sel.has(id);
      for (let k = s; k <= e; k++) { if (on) store.sel.add(store.items[k].id); else store.sel.delete(store.items[k].id); }
    }
  } else {
    if (store.sel.has(id)) store.sel.delete(id); else store.sel.add(id);
  }
  store.anchor = id;
  renderGrid(); updateSelInfo();
}

function updateSelInfo() {
  const n = store.sel.size, t = store.items.length;
  const repo = store.mode === "repo";
  const bt = store.browse.files.length;
  const bn = store.browse.files.filter(f => store.browse.sel.has(f.path)).length;
  const edit = isEdit(store.mode);
  $("selInfo").textContent = repo
    ? (bt ? (bn ? `已勾选 ${bn} / ${bt} 个文件` : `${bt} 个图片文件`) : "")
    : (edit ? (store.enabled.size
        ? (n ? `${tf("已选")} ${n} / ${t}` : `${tf("共")} ${t} ${tf("张")} · ${tf("启用")} ${store.enabled.size} ${tf("个文件夹")}`)
        : tf("未启用文件夹")) : "");
  $("selAllBtn").style.display = (edit || repo) ? "" : "none";
  $("selAllBtn").textContent = repo
    ? (bn && bn === bt ? "取消全选" : "全选")
    : ((n && n === t) ? "取消全选" : "全选");
  const ib = $("importBtn");
  ib.style.display = repo ? "" : "none";
  ib.textContent = bn ? `导入所选 (${bn})` : "导入所选";
  ib.disabled = !bn;
}

$("selAllBtn").onclick = () => {
  if (store.mode === "repo") {
    const b = store.browse;
    if (b.sel.size === b.files.length) b.sel.clear();
    else b.files.forEach(f => b.sel.add(f.path));
    renderBrowse(); updateSelInfo();
    return;
  }
  if (store.sel.size === store.items.length) store.sel.clear();
  else store.items.forEach(i => store.sel.add(i.id));
  renderGrid(); updateSelInfo();
};

$("importBtn").onclick = importSelected;

/* ---------------- 单张 / 缩放 ---------------- */
const pane = (() => {
  const vp = $("vp"), stage = $("stage"), barV = $("sbV"), barH = $("sbH");
  const st = { k: 1, x: 0, y: 0, fit: 1 };
  let nat = { w: 1, h: 1 };
  let touched = false, raf = 0;

  const VW = () => vp.clientWidth, VH = () => vp.clientHeight;

  function apply() {
    stage.style.transform = `translate3d(${st.x}px,${st.y}px,0) scale(${st.k})`;
    syncBars();
  }
  function clampXY() {
    const cw = nat.w * st.k, ch = nat.h * st.k, vw = VW(), vh = VH();
    st.x = cw <= vw ? (vw - cw) / 2 : Math.min(Math.max(st.x, vw - cw), 0);
    st.y = ch <= vh ? (vh - ch) / 2 : Math.min(Math.max(st.y, vh - ch), 0);
  }
  function fit() {
    const vw = VW(), vh = VH();
    if (!nat.w || !nat.h || vw < 2) return;
    st.fit = Math.max(vw / nat.w, vh / nat.h);
    st.k = st.fit;
    const cw = nat.w * st.k, ch = nat.h * st.k;
    st.x = cw <= vw ? (vw - cw) / 2 : 0;
    st.y = ch <= vh ? (vh - ch) / 2 : 0;
    touched = false;
    apply();
  }
  function syncBars() {
    const cw = nat.w * st.k, ch = nat.h * st.k, vw = VW(), vh = VH();
    const ovY = Math.max(0, ch - vh), ovX = Math.max(0, cw - vw);
    barV.classList.toggle("hide", ovY < 2);
    barH.classList.toggle("hide", ovX < 2);
    const trackV = vh - 8, trackH = vw - 18;
    if (ovY >= 2) {
      const th = Math.max(28, trackV * (vh / ch));
      const i = barV.firstElementChild;
      i.style.height = th + "px";
      i.style.top = ((0 - st.y) / ovY) * (trackV - th) + "px";
    }
    if (ovX >= 2) {
      const tw = Math.max(28, trackH * (vw / cw));
      const i = barH.firstElementChild;
      i.style.width = tw + "px";
      i.style.left = ((0 - st.x) / ovX) * (trackH - tw) + "px";
    }
  }
  function bindBar(bar, horiz) {
    bar.addEventListener("pointerdown", e => {
      if (e.button !== 0) return;
      e.preventDefault(); e.stopPropagation();
      bar.setPointerCapture(e.pointerId);
      bar.classList.add("drag");
      const cw = nat.w * st.k, ch = nat.h * st.k;
      const ov = horiz ? Math.max(0, cw - VW()) : Math.max(0, ch - VH());
      const track = horiz ? VW() - 18 : VH() - 8;
      const thumb = horiz ? bar.firstElementChild.offsetWidth : bar.firstElementChild.offsetHeight;
      const span = Math.max(1, track - thumb);
      const p0 = horiz ? e.clientX : e.clientY;
      const o0 = horiz ? st.x : st.y;
      const move = ev => {
        const d = (horiz ? ev.clientX : ev.clientY) - p0;
        const v = o0 - (d / span) * ov;
        if (horiz) st.x = v; else st.y = v;
        clampXY(); apply();
      };
      const end = () => {
        bar.classList.remove("drag");
        bar.removeEventListener("pointermove", move);
        bar.removeEventListener("pointerup", end);
        bar.removeEventListener("pointercancel", end);
      };
      bar.addEventListener("pointermove", move);
      bar.addEventListener("pointerup", end);
      bar.addEventListener("pointercancel", end);
    });
  }
  bindBar(barV, false); bindBar(barH, true);

  function onWheel(e) {
    if ($("stage").style.display === "none") return;
    e.preventDefault();
    const r = vp.getBoundingClientRect();
    const vw = VW(), vh = VH();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let d = e.deltaY;
    if (e.deltaMode === 1) d *= 16; else if (e.deltaMode === 2) d *= vh;

    // 指针在图片“画面”之外（含四周空白）→ 只纵向平移，不缩放
    const cw0 = nat.w * st.k, ch0 = nat.h * st.k;
    const inImage = mx >= st.x && mx <= st.x + cw0 && my >= st.y && my <= st.y + ch0;
    if (!inImage) {
      const ovY = Math.max(0, ch0 - vh);
      if (ovY < 2) return;
      const cy = v => Math.min(0, Math.max(vh - ch0, v));
      st.y = cy(st.y - d);
      apply(); syncBars();
      return;
    }
    const k2 = Math.min(st.fit * 14, Math.max(st.fit * .12, st.k * Math.exp(-d * 0.0022)));
    if (Math.abs(k2 - st.k) < 1e-6) return;
    const cw = nat.w * k2, ch = nat.h * k2;
    const cx = v => (cw <= vw ? (vw - cw) / 2 : Math.min(Math.max(v, vw - cw), 0));
    const cy = v => (ch <= vh ? (vh - ch) / 2 : Math.min(Math.max(v, vh - ch), 0));
    const ux = (mx - st.x) / st.k, uy = (my - st.y) / st.k;
    let nx = mx - ux * k2, ny = my - uy * k2;
    if (Math.abs(cx(nx) - nx) > 2) { const cu = (vw / 2 - st.x) / st.k; nx = vw / 2 - cu * k2; }
    if (Math.abs(cy(ny) - ny) > 2) { const cv = (vh / 2 - st.y) / st.k; ny = vh / 2 - cv * k2; }
    st.k = k2; st.x = cx(nx); st.y = cy(ny);
    touched = true;
    apply();
    document.documentElement.style.setProperty("--tile", Math.round(40 * st.k * 2) + "px");
  }
  $("zone").addEventListener("wheel", onWheel, { passive: false });
  new ResizeObserver(() => { if (!touched) fit(); else { clampXY(); apply(); } }).observe(vp);

  return {
    setNat(w, h) { nat = { w: Math.max(1, w), h: Math.max(1, h) }; fit(); },
    fit, apply, reset() { const s = $("stage"); if (s) s.style.display = "none"; },
    state: st,
  };
})();

/* ---------------- 单张编辑 ---------------- */
const cvBase = $("cvBase"), cvPaint = $("cvPaint");
const pctx = cvPaint.getContext("2d");
let cur = null;          // 当前 item
let baseCanvas = null;   // 底图 canvas（用于马赛克取样）
let tool = { type: "color", size: 36, strength: 35, color: "#ff3b30", stampId: "", stampImg: null, stampName: "" };
let drawing = null;

function curItem() { return store.items.find(i => i.id === store.curId) || null; }

async function openSingle(id) {
  const it = store.items.find(i => i.id === id) || [...store.pool.values()].find(i => i.id === id);
  if (!it) { $("stage").style.display = "none"; return; }
  if (cur && cur.id !== it.id) { layerSel.clear(); wmEdit = null; }
  store.curId = it.id;
  cur = it;
  const src = it.baseSrc || "/api/file?path=" + encodeURIComponent(it.path);
  await new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => { setBase(im); res(); };
    im.onerror = () => { toast("图片加载失败：" + it.name); res(); };
    im.src = src;
  });
  redrawPaint();
  $("stage").style.display = "block";
  $("gridWrap").style.display = "none";
  pane.setNat(it.w || cvBase.naturalWidth, it.h || cvBase.naturalHeight);
  pumpPreview();
  // 单张打开后布局才定型，重画一次预览并刷新右侧工作台（水印层列表依赖 cur）
  if (store.mode === "mark" || store.mode === "cover") { buildPanel(); requestAnimationFrame(pumpPreview); }
  if (store.mode === "cover") requestAnimationFrame(fitWholeForCover);
  updateUndoBtns();
  updateNavBtns();
}

function setBase(im) {
  const w = im.naturalWidth || im.width, h = im.naturalHeight || im.height;
  cvBase.src = im.src; cvBase.width = w; cvBase.height = h;
  cvBase.style.width = w + "px"; cvBase.style.height = h + "px";
  if (!baseCanvas) baseCanvas = document.createElement("canvas");
  baseCanvas.width = w; baseCanvas.height = h;
  baseCanvas.getContext("2d").drawImage(im, 0, 0, w, h);
  cvPaint.width = w; cvPaint.height = h;
  cvPaint.style.width = w + "px"; cvPaint.style.height = h + "px";
  if (cur) { cur.w = w; cur.h = h; }
  const _wm = $("cvWm");
  if (_wm) {
    _wm.width = w; _wm.height = h;
    _wm.style.width = w + "px"; _wm.style.height = h + "px";
  }
}

/* 笔迹重放：撤销/恢复都是从头重画，保证马赛克取样一致 */
function redrawPaint() {
  pctx.setTransform(1, 0, 0, 1, 0, 0);
  pctx.clearRect(0, 0, cvPaint.width, cvPaint.height);
  if (!cur) return;
  for (const s of cur.sticks) drawStroke(s);
  cur.paintDirty = cur.sticks.length > 0;
}
const _tmpA = document.createElement("canvas"), _tmpB = document.createElement("canvas");

/* 把一条笔迹画到任意 ctx 上；src 提供马赛克/模糊的取样底图 */
function renderStroke(ctx, s, src, w, h) {
  if (!s || !s.pts || !s.pts.length) return;
  if (s.type === "eraser") {
    ctx.save(); ctx.globalCompositeOperation = "destination-out"; ctx.fillStyle = "#000";
    for (const p of s.pts) { ctx.beginPath(); ctx.arc(p.x, p.y, s.size / 2, 0, 7); ctx.fill(); }
    ctx.restore(); return;
  }
  if (s.type === "color") {
    ctx.save(); ctx.fillStyle = s.color;
    for (const p of s.pts) { ctx.beginPath(); ctx.arc(p.x, p.y, s.size / 2, 0, 7); ctx.fill(); }
    ctx.restore(); return;
  }
  if (s.type === "img") {
    const im = s.stampImg;
    if (!im) return;
    const rad = s.size / 2;
    for (const p of s.pts) {
      ctx.save(); ctx.beginPath(); ctx.arc(p.x, p.y, rad, 0, 7); ctx.clip();
      const k = Math.max((rad * 2) / im.width, (rad * 2) / im.height);
      ctx.drawImage(im, p.x - im.width * k / 2, p.y - im.height * k / 2, im.width * k, im.height * k);
      ctx.restore();
    }
    return;
  }
  if (!src) return;
  const rad = s.size / 2;
  if (s.type === "blur") {
    const pad = Math.ceil(rad) + 8;
    for (const p of s.pts) {
      const x0 = Math.max(0, Math.floor(p.x - rad - pad)), y0 = Math.max(0, Math.floor(p.y - rad - pad));
      const x1 = Math.min(w, Math.ceil(p.x + rad + pad)), y1 = Math.min(h, Math.ceil(p.y + rad + pad));
      const bw = x1 - x0, bh = y1 - y0;
      if (bw <= 0 || bh <= 0) continue;
      _tmpB.width = bw; _tmpB.height = bh;
      const c2 = _tmpB.getContext("2d");
      c2.clearRect(0, 0, bw, bh);
      c2.filter = "blur(" + Math.max(2, Math.round(s.strength / 5)) + "px)";
      c2.drawImage(src, x0, y0, bw, bh, 0, 0, bw, bh);
      c2.filter = "none";
      const g = c2.createRadialGradient(p.x - x0, p.y - y0, rad * .72, p.x - x0, p.y - y0, rad * 1.02);
      g.addColorStop(0, "rgba(0,0,0,1)"); g.addColorStop(1, "rgba(0,0,0,0)");
      c2.globalCompositeOperation = "destination-in";
      c2.fillStyle = g; c2.fillRect(0, 0, bw, bh);
      c2.globalCompositeOperation = "source-over";
      ctx.drawImage(_tmpB, x0, y0);
    }
    return;
  }
  // 马赛克：全局网格对齐
  const block = Math.max(3, Math.round(s.strength / 2.5));
  for (const p of s.pts) {
    const pad = Math.ceil(rad) + 8;
    const x0 = Math.max(0, Math.floor((p.x - rad - pad) / block) * block);
    const y0 = Math.max(0, Math.floor((p.y - rad - pad) / block) * block);
    const x1 = Math.min(w, Math.ceil((p.x + rad + pad) / block) * block);
    const y1 = Math.min(h, Math.ceil((p.y + rad + pad) / block) * block);
    const bw = x1 - x0, bh = y1 - y0;
    if (bw <= 0 || bh <= 0) continue;
    const tw = Math.max(1, Math.ceil(bw / block)), th = Math.max(1, Math.ceil(bh / block));
    _tmpA.width = tw; _tmpA.height = th;
    const c3 = _tmpA.getContext("2d");
    c3.clearRect(0, 0, tw, th);
    c3.drawImage(src, x0, y0, bw, bh, 0, 0, tw, th);
    ctx.save();
    ctx.beginPath(); ctx.arc(p.x, p.y, rad, 0, 7); ctx.clip();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(_tmpA, 0, 0, tw, th, x0, y0, tw * block, th * block);
    ctx.restore();
  }
}

function renderStrokes(ctx, sticks, src, w, h) {
  for (const s of sticks || []) renderStroke(ctx, s, src, w, h);
}

function drawStroke(s) {
  renderStroke(pctx, s, baseCanvas, cvPaint.width, cvPaint.height);
}

function ptOf(e) {
  const r = cvPaint.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) / (r.width / cvPaint.width),
    y: (e.clientY - r.top) / (r.height / cvPaint.height),
  };
}

cvPaint.addEventListener("pointerdown", e => {
  if (!cur || e.button !== 0 || store.mode === "mark") return;
  try { cvPaint.setPointerCapture(e.pointerId); } catch (err) {}
  const s = { type: tool.type, size: +tool.size, strength: +tool.strength, color: tool.color,
              stampImg: tool.type === "img" ? tool.stampImg : null, pts: [ptOf(e)] };
  cur.sticks.push(s); cur.redo = [];
  drawing = s;
  drawStroke(s);
  cur.done = true;
  markDirty(cur);
  updateUndoBtns();
});
cvPaint.addEventListener("pointermove", e => {
  if (!drawing) return;
  const p = ptOf(e);
  const last = drawing.pts[drawing.pts.length - 1];
  if (Math.hypot(p.x - last.x, p.y - last.y) < 1.2) return;
  drawing.pts.push(p);
  drawStroke({ ...drawing, pts: [p] });
});
cvPaint.addEventListener("pointerup", () => { if (drawing) { drawing = null; refreshThumb(cur); } });
cvPaint.addEventListener("pointercancel", () => { drawing = null; });

function markDirty(it) {
  it.done = true;
  const card = document.querySelector(`.card[data-id="${it.id}"] .badge`);
  if (card) card.parentElement.parentElement.classList.add("done");
}

function updateUndoBtns() {
  $("undoBtn").disabled = !(cur && cur.sticks.length);
  $("redoBtn").disabled = !(cur && cur.redo.length);
}

$("undoBtn").onclick = () => {
  if (!cur || !cur.sticks.length) return;
  cur.redo.push(cur.sticks.pop());
  redrawPaint(); updateUndoBtns(); refreshThumb(cur);
};
$("redoBtn").onclick = () => {
  if (!cur || !cur.redo.length) return;
  cur.sticks.push(cur.redo.pop());
  redrawPaint(); updateUndoBtns(); refreshThumb(cur);
};
document.addEventListener("keydown", e => {
  if (store.layout === "single" && $("stage").style.display !== "none" && !e.ctrlKey && !e.metaKey) {
    if (e.key === "ArrowLeft") { e.preventDefault(); stepImage(-1); return; }
    if (e.key === "ArrowRight") { e.preventDefault(); stepImage(1); return; }
  }
  if ((e.ctrlKey || e.metaKey) && String(e.key).toLowerCase() === "z" && store.mode === "sort") {
    e.preventDefault(); e.shiftKey ? redoWork() : undoWork(); return;
  }
  if (!(e.ctrlKey || e.metaKey) || $("stage").style.display === "none") return;
  if (e.key.toLowerCase() === "z") { e.preventDefault(); e.shiftKey ? $("redoBtn").click() : $("undoBtn").click(); }
  if (e.key.toLowerCase() === "y") { e.preventDefault(); $("redoBtn").click(); }
});

/* 缩略图刷新（卡片上直接看到打码结果） */
async function refreshThumb(it) {
  if (!it) return;
  const w = 420;
  const c = document.createElement("canvas");
  const scale = Math.min(1, w / Math.max(1, (it.w || 800)));
  c.width = Math.max(1, Math.round((it.w || 800) * scale));
  c.height = Math.max(1, Math.round((it.h || 600) * scale));
  const x = c.getContext("2d");
  try {
    // 底图（AI 结果优先）+ 全部笔迹，按缩略图尺寸重放，和导出结果一致
    const full = await composeCanvas(it, { noResize: true });
    x.drawImage(full, 0, 0, c.width, c.height);
    it.thumbCache = c.toDataURL("image/jpeg", .8);
    const img = document.querySelector(`.card[data-id="${it.id}"] img`);
    if (img) img.src = it.thumbCache;
  } catch (e) {}
}

function drawBaseOf(it, ctx, w, h) {
  return new Promise(res => {
    const im = new Image();
    im.onload = () => { ctx.drawImage(im, 0, 0, w, h); res(); };
    im.onerror = () => res();
    im.src = it.baseSrc || ("/api/file?path=" + encodeURIComponent(it.path));
  });
}

/* ---------------- 面板 ---------------- */
function buildPanel() {
  const p = $("panel");
  p.innerHTML = "";
  if (store.mode === "repo") { buildRepoPanel(); return; }
  if (store.mode === "work") { buildWorkPanel(p); return; }
  if (store.mode === "mark") { buildMarkPanel(p); return; }
  if (store.mode === "ai") { buildAIPanel(p); return; }
  if (store.mode === "cover") { buildCoverPanel(p); return; }
  if (store.mode === "sort") { buildSortPanel(p); return; }
  buildBrushPanel(p);
}

function buildMarkPanel(p) {
  p.innerHTML = "";
  p.appendChild(el("h3", null, "批量水印"));

  const seg = el("div", "seg2");
  for (const t of WM_TEMPLATES) {
    const b = el("button", wm.template === t.id ? "on" : "", t.name);
    b.title = t.hint;
    b.onclick = () => { wm.template = t.id; if (wmPlay && t.id !== "scroll" && t.id !== "bounce") toggleWmPlay(false); renderWmPreview(); buildMarkPanel(p); };
    seg.appendChild(b);
  }
  p.appendChild(seg);
  p.appendChild(el("div", "hint", WM_TEMPLATES.find(t => t.id === wm.template).hint));

  if (wm.template === "corner1") {
    const cs = el("div", "seg2");
    cs.style.marginTop = "8px";
    for (const [k, nm] of [["tl", "左上"], ["tr", "右上"], ["bl", "左下"], ["br", "右下"]]) {
      const b = el("button", wm.corner === k ? "on" : "", nm);
      b.onclick = () => { wm.corner = k; renderWmPreview(); buildMarkPanel(p); };
      cs.appendChild(b);
    }
    p.appendChild(cs);
  }

  const g = el("div", "grp");
  const ti = el("input"); ti.type = "text"; ti.value = wm.text;
  ti.placeholder = "水印文字";
  ti.oninput = () => { wm.text = ti.value; renderWmPreview(); };
  g.appendChild(ti);
  p.appendChild(g);

  const g2 = el("div", "grp");
  const fr = el("div", "row");
  fr.appendChild(el("label", null, "字体"));
  const fsel = el("select");
  for (const [k, v] of [["hei", "黑体"], ["song", "宋体"], ["mono", "等宽"]]) {
    const o = el("option", null, v); o.value = k; if (wm.font === k) o.selected = true; fsel.appendChild(o);
  }
  fsel.onchange = () => { wm.font = fsel.value; renderWmPreview(); };
  fr.appendChild(fsel); g2.appendChild(fr);

  const cr = el("div", "row");
  cr.appendChild(el("label", null, "颜色"));
  const col = el("input"); col.type = "color"; col.value = wm.color;
  col.oninput = () => { wm.color = col.value; renderWmPreview(); };
  cr.appendChild(col);
  const op = el("input"); op.type = "range"; op.min = 4; op.max = 100; op.value = Math.round(wm.opacity * 100);
  const opv = el("span", "hint", op.value + "%");
  op.oninput = () => { wm.opacity = +op.value / 100; opv.textContent = op.value + "%"; renderWmPreview(); };
  cr.appendChild(op); cr.appendChild(opv);
  g2.appendChild(cr);

  g2.appendChild(wmSlider("字号", 10, 140, 1, wm.size, v => { wm.size = +v; renderWmPreview(); }));
  if (wm.template === "tile") {
    g2.appendChild(wmSlider("密度", 0.5, 5, 0.1, wm.density, v => { wm.density = +v; renderWmPreview(); }));
    g2.appendChild(wmSlider("角度", -90, 90, 1, wm.angle, v => { wm.angle = +v; renderWmPreview(); }));
  }
  if (wm.template === "corner1" || wm.template === "corner") {
    g2.appendChild(wmSlider("边距", 0, 0.2, 0.005, wm.margin, v => { wm.margin = +v; renderWmPreview(); }));
  }
  if (wm.template === "scroll" || wm.template === "bounce") {
    g2.appendChild(wmSlider("条数 / 密度", 1, 6, 1, wm.density, v => { wm.density = +v; renderWmPreview(); }));
    g2.appendChild(wmSlider("速度", 0.02, 1, 0.01, wm.speed, v => { wm.speed = +v; }));
  }
  p.appendChild(g2);

  const g3 = el("div", "grp");
  const r1 = el("div", "row");
  const bSel = el("button", "primary", "固化到选中");
  const bAll = el("button", null, "固化到全部");
  r1.append(bSel, bAll); g3.appendChild(r1);
  const r2 = el("div", "row");
  const bClr = el("button", null, "清除水印");
  const bExpWm = el("button", "primary", "单独导出已做更改");
  r2.append(bClr, bExpWm); g3.appendChild(r2);
  const r3 = el("div", "row");
  const bExpAll = el("button", null, "导出全部");
  const bPlay = el("button", wmPlay ? "on" : "", "预览动画");
  bPlay.onclick = () => {
    if (wm.template !== "scroll" && wm.template !== "bounce") { toast("滚动 / 弹动模板才有动画"); return; }
    toggleWmPlay(!wmPlay); bPlay.classList.toggle("on", !!wmPlay);
  };
  r3.append(bPlay, bExpAll); g3.appendChild(r3);
  p.appendChild(g3);
  p.appendChild(el("div", "hint", "模板之间的字体、颜色、透明度是同一套，切换模板不会丢。单个角落可以在画布上直接拖动。点「固化到选中 / 全部」会把这层水印焊死在图上，之后再改参数只会往上叠新的一层；已固化的层在下方工作台里可以逐层编辑或删除。"));

  if (wmEdit) {
    const bar = el("div", "editbar");
    bar.appendChild(el("div", "hint strong", `编辑第 ${wmEdit.index + 1} 层`));
    const bc = el("button", null, "取消编辑");
    bc.onclick = () => { wmEdit = null; renderWmPreview(); buildPanel(); };
    bar.appendChild(bc);
    p.appendChild(bar);
  }
  p.appendChild(buildLayerList(p));

  const sel = () => store.sel.size ? store.items.filter(i => store.sel.has(i.id)) : store.items.slice();
  bSel.onclick = () => applyWm(store.sel.size ? store.items.filter(i => store.sel.has(i.id)) : store.items.slice(), bSel);
  bAll.onclick = () => applyWm(store.items.slice(), bAll);
  bClr.onclick = () => clearWm(sel());
  bExpWm.onclick = () => exportEdited(bExpWm, "mark");
  bExpAll.onclick = () => exportZip(store.items.slice(), "导出全部", bExpAll);
}

function wmSlider(label, min, max, step, val, oninput) {
  const row = el("div", "row");
  row.appendChild(el("label", null, label));
  const inp = el("input"); inp.type = "range";
  inp.min = min; inp.max = max; inp.step = step; inp.value = val;
  const v = el("div", "val", String(val));
  inp.oninput = () => { v.textContent = inp.value; oninput(inp.value); };
  row.appendChild(inp); row.appendChild(v);
  return row;
}

async function loadWork(rebuild) {
  try {
    const j = await api("/api/workspace");
    store.work = { name: j.name || "", folder: j.folder || "", count: j.count || 0,
                   bytes: j.bytes || 0, path: j.path || "" };
  } catch (e) {}
  if (rebuild && store.mode === "work") buildWorkPanel($("panel"));
  return store.work;
}

function fmtSize(n) {
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0, v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return v.toFixed(v < 10 && i ? 1 : 0) + " " + u[i];
}

function buildWorkPanel(p) {
  p.innerHTML = "";
  p.appendChild(el("h3", null, "当前工作区"));
  const w = store.work || {};
  const inp = el("input"); inp.type = "text"; inp.id = "wsName";
  inp.placeholder = "工作区名称（导出压缩包就用它）";
  inp.value = w.name || "";
  p.appendChild(inp);

  const row = el("div", "row");
  const rename = el("button", "primary", "重命名");
  row.appendChild(rename);
  p.appendChild(row);

  const info = el("div", "hint");
  info.textContent = `${w.count || 0} 个文件 · ${fmtSize(w.bytes || 0)}${w.path ? "\n" + w.path : ""}`;
  info.style.whiteSpace = "pre-wrap";
  p.appendChild(info);

  const r2 = el("div", "row");
  const all = el("button", "primary", "导出所有已做更改");
  const cen = el("button", null, "只导已打码");
  r2.append(all, cen);
  p.appendChild(r2);

  const r3 = el("div", "row");
  const del = el("button", "danger", "删除工作区");
  r3.appendChild(del);
  p.appendChild(r3);

  p.appendChild(el("div", "hint", "工作区里放的是导入时复制到 cache 的副本；重命名、删除、导出都只动副本，母本文件不受影响。"));

  rename.onclick = async () => {
    const nm = inp.value.trim();
    if (!nm) { toast("名称不能为空"); return; }
    try {
      const j = await api("/api/workspace/rename", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: nm }),
      });
      store.work.name = j.name;
      toast("已重命名为 " + j.name);
      buildWorkPanel(p);
    } catch (e) { toast("重命名失败：" + e.message, 3000); }
  };
  all.onclick = () => exportAll(all);
  cen.onclick = () => exportCensored(cen);
  del.onclick = async () => {
    if (!confirm("删除工作区里的全部副本？母本文件不会被删。")) return;
    try {
      await api("/api/workspace/delete", { method: "POST", body: "{}" });
      store.items = []; store.sel.clear(); store.curId = 0;
      store.work = { name: store.work.name || "", count: 0, bytes: 0, path: store.work.path || "" };
      renderGrid(); updateSelInfo(); buildWorkPanel(p); updateNavBtns();
      toast("工作区已清空（母本未改动）");
    } catch (e) { toast("删除失败：" + e.message, 3000); }
  };
  loadWork(false).then(() => {
    if (store.mode !== "work") return;
    info.textContent = `${store.work.count || 0} 个文件 · ${fmtSize(store.work.bytes || 0)}${store.work.path ? "\n" + store.work.path : ""}`;
    if (!inp.value && store.work.name) inp.value = store.work.name;
  });
}

function buildRepoPanel() {
  const p = $("panel"); p.innerHTML = "";
  p.appendChild(el("h3", null, "仓库目录"));
  const inp = el("input"); inp.type = "text"; inp.id = "repoPath";
  inp.placeholder = "粘贴本地路径，如 E:\\图片\\待打码";
  inp.value = store.browse.path || store.root || ""; p.appendChild(inp);
  inp.onkeydown = ev => { if (ev.key === "Enter") openDir(inp.value.trim()); };
  const row = el("div", "row");
  const go = el("button", null, "打开"), up = el("button", null, "上一级"), rf = el("button", null, "刷新");
  row.append(go, up, rf); p.appendChild(row);
  go.onclick = () => openDir(inp.value.trim());
  up.onclick = () => openDir(store.browse.parent || (store.browse.path || "").replace(/[\\/][^\\/]*$/, ""));
  rf.onclick = () => openDir(store.browse.path || inp.value.trim());
  const info = el("div", "hint"); info.id = "repoInfo"; p.appendChild(info);
  const all = el("button", "primary", "导入本目录全部");
  all.style.margin = "6px 0";
  all.onclick = async () => {
    const b = store.browse;
    if (!b.files.length) { toast("这个文件夹里没有图片"); return; }
    b.sel = new Set(b.files.map(f => f.path));
    await importSelected();
  };
  p.appendChild(all);
  p.appendChild(el("div", "hint", "中间工作台里点文件夹进入、点图片勾选，勾好后点顶部「导入所选」；选中的图片会复制到程序内 cache 缓存再编辑，母本不动。"));
  if (store.browse.path) renderRepoPanel({ images: store.browse.files.length, dirs: store.browse.dirs });
}

function renderRepoPanel(j) {
  const info = $("repoInfo"); if (!info) return;
  const imgs = j.images != null ? j.images : store.browse.files.length;
  const dn = (j.dirs || []).length;
  const cn = store.browse.sel.size;
  info.textContent = `本目录 ${imgs} 张图片 · ${dn} 个子文件夹${cn ? ` · 已勾选 ${cn} 张` : ""}`;
}

function buildBrushPanel(p) {
  p.appendChild(el("h3", null, "画笔"));
  const seg = el("div", "seg2");
  [["color", "色彩"], ["mosaic", "马赛克"], ["blur", "模糊"], ["img", "图片"], ["eraser", "橡皮"]].forEach(([k, t]) => {
    const b = el("button", tool.type === k ? "on" : "", t);
    b.onclick = () => { tool.type = k; buildPanel(); };
    seg.appendChild(b);
  });
  p.appendChild(seg);
  const g1 = el("div", "grp");
  g1.appendChild(slider("颜色", "colorPick", "color", 1, 1, 1, tool.color));
  p.appendChild(g1);
  const g2 = el("div", "grp");
  if (tool.type === "img") {
    const row = el("div", "row");
    row.appendChild(el("label", null, "遮挡图片"));
    const btn = el("button", null, "选择图片…");
    row.appendChild(btn);
    g2.appendChild(row);
    const f = el("input");
    f.type = "file"; f.accept = "image/*"; f.style.display = "none";
    f.onchange = () => loadBrushStamp(f.files && f.files[0]);
    btn.onclick = () => f.click();
    g2.appendChild(f);
    if (tool.stampImg) {
      const pv = el("div", "row");
      const im = el("img");
      im.src = tool.stampImg.src;
      im.style.cssText = "width:48px;height:48px;object-fit:contain;border-radius:8px;border:1px solid var(--line2);background:#0002";
      pv.appendChild(im);
      const tx = el("div", "hint");
      tx.style.cssText = "flex:1;min-width:0";
      tx.textContent = tool.stampName + " · 笔刷大小决定铺开范围";
      pv.appendChild(tx);
      g2.appendChild(pv);
    } else {
      g2.appendChild(el("div", "hint", "先选一张图片作为遮挡图案，再用笔画。"));
    }
  }
  g2.appendChild(slider("笔刷大小", "bSize", "size", 2, 240, 1, tool.size));
  const gx = el("div", "grp");
  const rE = el("div", "row");
  const bd = el("button", null, "导出已打码");
  const ba = el("button", "primary", "导出所有");
  rE.append(bd, ba); gx.appendChild(rE);
  gx.appendChild(el("div", "hint", "「导出所有」会把没打码的图也一起打包，适合整批交付。"));
  p.appendChild(gx);
  bd.onclick = () => exportCensored(bd);
  ba.onclick = () => exportAll(ba);
  if (tool.type === "mosaic" || tool.type === "blur") {
    g2.appendChild(slider("码的粗细", "bStr", "strength", 1, 100, 1, tool.strength));
    g2.appendChild(el("div", "hint", "码的粗细只影响马赛克块大小 / 模糊半径；橡皮只擦手动画的笔迹"));
  }
  p.appendChild(g2);
}

function slider(label, id, key, min, max, step, val) {
  const row = el("div", "row");
  row.appendChild(el("label", null, label));
  const inp = el("input"); inp.type = "range"; inp.id = id;
  inp.min = min; inp.max = max; inp.step = step; inp.value = val;
  if (key === "color") {
    inp.type = "color";
    inp.oninput = () => { tool.color = inp.value; };
  } else {
    inp.oninput = () => { tool[key] = +inp.value; row.querySelector(".val").textContent = inp.value; };
  }
  row.appendChild(inp);
  const v = el("div", "val", key === "color" ? "" : String(val));
  row.appendChild(v);
  return row;
}

function buildAIPanel(p) {
  p.appendChild(el("h3", null, "识别与打码"));
  const seg = el("div", "seg2");
  [["mosaic", "马赛克"], ["blur", "高斯模糊"], ["solid", "纯色"], ["img", "图片"]].forEach(([k, t]) => {
    const b = el("button", ai.mode === k ? "on" : "", t);
    b.onclick = () => { ai.mode = k; buildPanel(); };
    seg.appendChild(b);
  });
  p.appendChild(seg);

  const g = el("div", "grp");
  if (ai.mode === "img") {
    const lab = el("div", "row");
    lab.appendChild(el("label", null, "遮挡图片"));
    const btn = el("button", null, "选择图片…");
    lab.appendChild(btn);
    g.appendChild(lab);
    const inp = el("input");
    inp.type = "file"; inp.accept = "image/*"; inp.style.display = "none";
    inp.onchange = () => uploadAsset(inp.files && inp.files[0]);
    btn.onclick = () => inp.click();
    g.appendChild(inp);
    if (ai.asset) {
      const pv = el("div", "row");
      const im = el("img");
      im.src = "/asset/" + ai.asset;
      im.style.cssText = "width:54px;height:54px;object-fit:contain;border-radius:8px;border:1px solid var(--line2);background:#0002";
      pv.appendChild(im);
      const tx = el("div", "hint");
      tx.style.cssText = "flex:1;min-width:0";
      tx.textContent = `${ai.assetName}（${ai.assetW}×${ai.assetH}）按检测框等比铺满`;
      pv.appendChild(tx);
      g.appendChild(pv);
    } else {
      g.appendChild(el("div", "hint", "还没选遮挡图：先在上一行选一张图（PNG/JPG），否则只会退化成纯色块。"));
    }
  } else {
    g.appendChild(slider("强度", "aiStrength", "strength", 1, 100, 1, ai.strength));
  }
  g.appendChild(slider("边缘外扩", "aiMargin", "margin", 0, 60, 1, ai.margin));
  g.appendChild(slider("阈值", "aiConf", "conf", 0.05, 0.9, 0.05, ai.conf));
  const rr = el("div", "row");
  rr.appendChild(el("label", null, "分辨率"));
  const sel = el("select");
  ["320", "640", "960", "1280"].forEach(v => { const o = el("option", null, v); o.value = v; if (+ai.res === +v) o.selected = true; sel.appendChild(o); });
  sel.onchange = () => ai.res = sel.value;
  rr.appendChild(sel); g.appendChild(rr);
  p.appendChild(g);

  p.appendChild(el("h3", null, "打码类别"));
  const chips = el("div", "chips");
  for (const k of ALL_CLASSES) {
    const on = ai.classes.includes(k);
    const c = el("div", "chip" + (on ? " on" : ""), ZH[k] || k);
    c.onclick = () => {
      const i = ai.classes.indexOf(k);
      if (i >= 0) ai.classes.splice(i, 1); else ai.classes.push(k);
      c.classList.toggle("on");
    };
    chips.appendChild(c);
  }
  p.appendChild(chips);

  const run = el("button", "primary", "开始打码");
  run.style.width = "100%"; run.style.marginTop = "12px"; run.id = "runAI";
  p.appendChild(run);
  const prog = el("div"); prog.id = "prog"; prog.appendChild(el("i")); p.appendChild(prog);
  const st = el("div", "hint"); st.id = "aiStatus"; st.style.marginTop = "8px";
  st.textContent = store.sel.size ? `将对已勾选的 ${store.sel.size} 张执行` : "未勾选时对全部图片执行";
  p.appendChild(st);
  run.onclick = startAI;
}

const ai = {
  mode: "mosaic", strength: 35, margin: 15, conf: 0.25, res: "640",
  asset: "", assetName: "", assetW: 0, assetH: 0,
  classes: DEF_CLASSES.slice(),
};

function loadBrushStamp(file) {
  if (!file) return;
  const fr = new FileReader();
  fr.onload = () => {
    const im = new Image();
    im.onload = () => {
      tool.stampImg = im; tool.stampName = file.name;
      buildPanel();
      toast("画笔遮挡图案就绪：" + file.name);
    };
    im.onerror = () => toast("这张图读不出来，换一张 PNG/JPG 试试", 3000);
    im.src = fr.result;
  };
  fr.readAsDataURL(file);
}

async function uploadAsset(file) {
  if (!file) return;
  if (file.size > 8 * 1024 * 1024) { toast("遮挡图片超过 8MB"); return; }
  try {
    const r = await fetch("/asset", { method: "POST", body: file });
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || ("HTTP " + r.status));
    ai.asset = j.id; ai.assetName = file.name; ai.assetW = j.w; ai.assetH = j.h;
    toast("遮挡图片已就绪：" + file.name);
    buildPanel();
  } catch (e) { toast("上传遮挡图片失败：" + e.message, 3000); }
}

async function startAI() {
  if (ai.mode === "img" && !ai.asset) { toast("「图片」模式要先在右侧选一张遮挡图片"); return; }
  const targets = store.sel.size ? store.items.filter(i => store.sel.has(i.id)) : store.items.slice();
  if (!targets.length) { toast("没有可处理的图片"); return; }
  const prog = $("prog"), bar = prog.querySelector("i"), st = $("aiStatus");
  prog.style.display = "block";
  let ok = 0, fail = 0;
  for (let i = 0; i < targets.length; i++) {
    const it = targets[i];
    st.textContent = `正在打码 ${i + 1} / ${targets.length} · ${it.name}`;
    bar.style.width = Math.round((i / targets.length) * 100) + "%";
    try {
      const blob = await (await fetch("/api/file?path=" + encodeURIComponent(it.path))).blob();
      const q = new URLSearchParams({
        mode: ai.mode, strength: ai.strength, margin: ai.margin, color: tool.color,
        conf: ai.conf, res: ai.res, classes: ai.classes.join(","), fmt: "png", quality: 95,
        vfmt: "webm", asset: ai.asset || "",
      });
      const r = await fetch("/infer?" + q, { method: "POST", body: blob });
      const j = await r.json();
      if (j.error) throw new Error(j.error);
      it.baseSrc = j.censored; it.done = true;
      await refreshThumb(it);
      ok++;
    } catch (e) { fail++; it.fail = e.message; }
    bar.style.width = Math.round(((i + 1) / targets.length) * 100) + "%";
  }
  st.textContent = `完成 ${ok} 张${fail ? "，失败 " + fail + " 张" : ""}`;
  renderGrid();
  if (cur && store.curId) openSingle(store.curId);
  setTimeout(() => { prog.style.display = "none"; bar.style.width = "0"; }, 1200);
}

/* ---------------- 水印 ---------------- */
const cvWm = $("cvWm"), wctx = cvWm.getContext("2d");

const WM_FONTS = {
  hei: '"Microsoft YaHei","PingFang SC","Noto Sans SC",system-ui,sans-serif',
  song: '"SimSun","Songti SC","Noto Serif SC",serif',
  mono: '"Cascadia Mono",Consolas,"Courier New",monospace',
};
const WM_TEMPLATES = [
  { id: "corner1", name: "四角任意", hint: "任选一个角，只打一个水印，位置固定带边距" },
  { id: "tile", name: "斜侧满屏", hint: "整屏斜向平铺，可调密度与角度" },
  { id: "scroll", name: "滚动水印", hint: "横向走马灯，导出静帧取当前相位" },
  { id: "bounce", name: "弹动水印", hint: "在画面里来回弹动，导出静帧取当前相位" },
  { id: "corner", name: "单个角落", hint: "只有一个水印，可在画布上直接拖动" },
];

const wm = {
  template: "corner1",
  text: "CONFIDENTIAL",
  font: "hei",
  size: 34,
  color: "#ffffff",
  opacity: 0.35,
  density: 2.2,     // tile: 行距倍数；scroll: 条数
  angle: -28,
  margin: 0.045,    // 角落边距（相对短边）
  corner: "br",     // 四角任意：tl / tr / bl / br
  x: 0.78, y: 0.86, // 单个角落的水印中心（0~1）
  speed: 0.18,      // 滚动/弹动速度（画面宽度/秒）
  phase: 0,         // 动画相位 0~1
  stroke: false,
  strokeColor: "#000000",
};
let wmDirty = true;
let wmPlay = 0;
function toggleWmPlay(on) {
  if (wmPlay) { cancelAnimationFrame(wmPlay); wmPlay = 0; }
  if (!on) { renderWmPreview(); return; }
  let last = performance.now();
  const step = now => {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    wm.phase = (wm.phase + dt * wm.speed) % 1;
    renderWmPreview();
    wmPlay = requestAnimationFrame(step);
  };
  wmPlay = requestAnimationFrame(step);
}

let wmEdit = null;             // {itemId, index} 正在编辑的已固化水印
const layerSel = new Set();     // 勾选的水印图层（按 index）
function wmLayers(it) {
  if (!it) return [];
  if (!Array.isArray(it.wmLayers)) it.wmLayers = [];
  return it.wmLayers;
}
function wmOf(it) { return wmLayers(it); }

function wmFont(cfg) {
  return `600 ${Math.max(8, Math.round(cfg.size))}px ${WM_FONTS[cfg.font] || WM_FONTS.hei}`;
}

function wmStyle(ctx, cfg, alpha) {
  ctx.font = wmFont(cfg);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.globalAlpha = alpha;
  ctx.fillStyle = cfg.color;
  if (cfg.stroke) { ctx.lineWidth = Math.max(1, cfg.size / 12); ctx.strokeStyle = cfg.strokeColor; }
}

function wmText(ctx, cfg, x, y, rot) {
  ctx.save();
  if (rot) { ctx.translate(x, y); ctx.rotate(rot); ctx.translate(-x, -y); }
  if (cfg.stroke) ctx.strokeText(cfg.text, x, y);
  ctx.fillText(cfg.text, x, y);
  ctx.restore();
}

/* 把水印画到 ctx（尺寸 w×h）；cfg 缺省用全局 wm，t 为动画相位 0~1 */
function drawWatermark(ctx, w, h, cfg, t) {
  if (!cfg || !cfg.text) return;
  const phase = (t == null ? (cfg.phase || 0) : t) % 1;
  const short = Math.min(w, h);
  const margin = Math.max(6, Math.round(short * cfg.margin));
  ctx.save();
  wmStyle(ctx, cfg, cfg.opacity);

  // 长文字自动缩字号，避免小图上四个角互相压字
  const _maxW = w * (cfg.template === "tile" ? 0.34 : 0.42);
  let _tw = ctx.measureText(cfg.text).width;
  if (_tw > _maxW && _tw > 0) {
    const k = _maxW / _tw;
    ctx.font = `600 ${Math.max(8, Math.round(cfg.size * k))}px ${WM_FONTS[cfg.font] || WM_FONTS.hei}`;
  }

  if (cfg.template === "corner1") {
    const tw = ctx.measureText(cfg.text).width, th = cfg.size;
    const pad = margin + Math.max(tw, th) / 2;
    const c = cfg.corner || "br";
    const x = (c === "tl" || c === "bl") ? pad : w - pad;
    const y = (c === "tl" || c === "tr") ? pad : h - pad;
    wmText(ctx, cfg, Math.max(pad * 0.6, Math.min(w - pad * 0.6, x)),
                     Math.max(pad * 0.6, Math.min(h - pad * 0.6, y)), 0);
  } else if (cfg.template === "corner") {
    const tw = ctx.measureText(cfg.text).width, th = cfg.size;
    const pad = margin + Math.max(tw, th) / 2;
    wmText(ctx, cfg, Math.max(pad * 0.4, Math.min(w - pad * 0.4, cfg.x * w)),
                     Math.max(pad * 0.4, Math.min(h - pad * 0.4, cfg.y * h)), 0);
  } else if (cfg.template === "tile") {
    const tw = ctx.measureText(cfg.text).width;
    const stepX = Math.max(60, tw + short * 0.10 * (1 / Math.max(0.4, cfg.density / 2.2)));
    const stepY = Math.max(40, cfg.size * (1.2 + 1.0 / Math.max(0.4, cfg.density)));
    const rot = cfg.angle * Math.PI / 180;
    const diag = Math.hypot(w, h);
    ctx.save();
    ctx.translate(w / 2, h / 2); ctx.rotate(rot);
    const cols = Math.ceil(diag / stepX) + 2, rows = Math.ceil(diag / stepY) + 2;
    for (let r = -rows; r <= rows; r++) {
      for (let c = -cols; c <= cols; c++) {
        const x = c * stepX + (r % 2 ? stepX / 2 : 0);
        const y = r * stepY;
        if (Math.hypot(x, y) > diag * 0.62) continue;
        if (cfg.stroke) ctx.strokeText(cfg.text, x, y);
        ctx.fillText(cfg.text, x, y);
      }
    }
    ctx.restore();
  } else if (cfg.template === "scroll") {
    const tw = ctx.measureText(cfg.text).width + short * 0.18;
    ctx.textAlign = "left";
    const lines = Math.max(1, Math.round(cfg.density));
    for (let i = 0; i < lines; i++) {
      const y = h * (i + 1) / (lines + 1);
      let x = ((w - phase * tw * 2) % tw) - tw;
      while (x < w + tw) { wmText(ctx, cfg, x + tw / 2, y, 0); x += tw; }
    }
  } else if (cfg.template === "bounce") {
    const tw = ctx.measureText(cfg.text).width;
    const pad = margin + tw / 2;
    const spanX = Math.max(1, w - pad * 2), spanY = Math.max(1, h - pad * 2);
    // 三角波：来回弹
    const tri = v => { const m = v % 2; return m < 1 ? m : 2 - m; };
    const x = pad + spanX * tri(phase * 2 + 0.25);
    const y = pad + spanY * tri(phase * 3 + 0.6);
    wmText(ctx, cfg, x, y, 0);
  }
  ctx.restore();
}

/* 预览：已固化的水印全部画上，再叠加"还没确认"的这一层（只在单图视图可见） */
function pumpPreview() {
  if (store.mode === "cover") { renderCoverPreview(); return; }
  if (store.mode === "mark") { renderWmPreview(); return; }
  if (store.mode === "ai" || store.mode === "manual") renderWmPreview();
}
function renderWmPreview() {
  if (!cur || !cvWm) return;
  const w = cvPaint.width, h = cvPaint.height;
  if (!w || !h) return;
  cvWm.width = w; cvWm.height = h;
  cvWm.style.width = w + "px"; cvWm.style.height = h + "px";
  wctx.clearRect(0, 0, w, h);
  const layers = wmLayers(cur);
  const editing = wmEdit && wmEdit.itemId === cur.id ? wmEdit.index : -1;
  layers.forEach((cfg, i) => { if (i !== editing) drawWatermark(wctx, w, h, cfg, cfg.phase || 0); });
  if (store.mode === "mark" && wm.text.trim()) {
    drawWatermark(wctx, w, h, wm, wm.phase || 0);
  }
}

/* 固化一层水印：确认后这层写进 it.wmLayers，之后再改水印只会叠新的一层 */
async function applyWm(list, btn) {
  if (!list.length) { toast("先选中要加水印的图片"); return; }
  if (!wm.text.trim()) { toast("水印文字不能为空"); return; }
  const old = btn ? btn.textContent : "";
  if (btn) { btn.disabled = true; }
  let mode = "add";
  for (const it of list) {
    const ls = wmLayers(it);
    const cfg = JSON.parse(JSON.stringify(wm));
    if (wmEdit && wmEdit.itemId === it.id && ls[wmEdit.index]) { ls[wmEdit.index] = cfg; mode = "edit"; }
    else ls.push(cfg);
  }
  wmEdit = null;
  layerSel.clear();
  renderWmPreview();
  for (let i = 0; i < list.length; i++) {
    if (btn) btn.textContent = `水印 ${i + 1}/${list.length}`;
    await refreshThumb(list[i]);
  }
  if (btn) { btn.disabled = false; btn.textContent = old; }
  renderGrid(); buildPanel(); updateSelInfo();
  toast(mode === "edit"
    ? `已更新这层水印（${list.length} 张）`
    : `已固化第 ${wmLayers(list[0]).length} 层水印（${list.length} 张 · ${WM_TEMPLATES.find(t => t.id === wm.template).name}）`);
}

/* 删除：勾选的图层优先，没有勾选就清空这张图的全部水印 */
async function clearWm(list) {
  let n = 0;
  for (const it of list) {
    const ls = wmLayers(it);
    if (!ls.length) continue;
    if (layerSel.size && it.id === store.curId) {
      const keep = ls.filter((_, i) => !layerSel.has(i));
      n += ls.length - keep.length;
      it.wmLayers = keep;
    } else { n += ls.length; it.wmLayers = []; }
  }
  layerSel.clear(); wmEdit = null;
  renderWmPreview();
  for (const it of list) refreshThumb(it);
  renderGrid(); buildPanel();
  toast(n ? `已删除 ${n} 层水印` : "这些图没有水印");
}

/* 水印工作台：列出这张图本次工作里已经固化的水印层 */
function buildLayerList(host) {
  const it = cur;
  const box = el("div", "grp");
  box.appendChild(el("div", "hint strong", "这张图的水印"));
  if (!it) { box.appendChild(el("div", "hint", "先打开一张图")); return box; }
  const ls = wmLayers(it);
  box.appendChild(el("div", "hint strong", `这张图本次工作共有 ${ls.length} 个水印`));
  layerSel.forEach(i => { if (i >= ls.length) layerSel.delete(i); });
  const wrap = el("div", "layers");
  ls.forEach((cfg, i) => {
    const row = el("div", "layer" + (layerSel.has(i) ? " on" : ""));
    const cb = el("input"); cb.type = "checkbox"; cb.checked = layerSel.has(i);
    cb.onchange = () => { cb.checked ? layerSel.add(i) : layerSel.delete(i); row.classList.toggle("on", cb.checked); };
    const t = el("div", "lt", `#${i + 1} ${WM_TEMPLATES.find(x => x.id === cfg.template)?.name || cfg.template} · ${cfg.text}`);
    t.title = `${cfg.text}（${cfg.template}）`;
    const bE = el("button", null, "编辑");
    bE.onclick = () => {
      Object.assign(wm, JSON.parse(JSON.stringify(cfg)));
      wmEdit = { itemId: it.id, index: i };
      if (wm.template === "scroll" || wm.template === "bounce") toggleWmPlay(false);
      renderWmPreview(); buildPanel();
      toast(`正在编辑第 ${i + 1} 层水印，改动只覆盖这一层`);
    };
    const bD = el("button", null, "删");
    bD.onclick = async () => {
      ls.splice(i, 1);
      layerSel.clear(); wmEdit = null;
      await refreshThumb(it);
      renderWmPreview(); renderGrid(); buildPanel();
      toast(`已删除第 ${i + 1} 层水印`);
    };
    row.append(cb, t, bE, bD);
    wrap.appendChild(row);
  });
  if (!ls.length) wrap.appendChild(el("div", "hint", "还没有水印 —— 调好参数点「固化到选中/全部」"));
  box.appendChild(wrap);
  const row = el("div", "row");
  const bDel = el("button", null, "删除勾选");
  bDel.disabled = !layerSel.size;
  bDel.onclick = () => clearWm([it]);
  const bClr = el("button", null, "清空这张");
  bClr.disabled = !ls.length;
  bClr.onclick = () => { it.wmLayers = []; layerSel.clear(); wmEdit = null; renderWmPreview(); refreshThumb(it); renderGrid(); buildPanel(); };
  row.append(bDel, bClr);
  box.appendChild(row);
  return box;
}

/* 单个角落：在画布上拖动 */
let wmDrag = null;
function wmPointOf(e) {
  const r = cvPaint.getBoundingClientRect();
  return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
}
$("cvPaint").addEventListener("pointerdown", e => {
  if (store.mode !== "mark" || e.button !== 0) return;
  const cfg = wm;
  if (cfg.template !== "corner") { toast("这个模板位置固定；要自由挪位置请切到「单个角落」", 2200); return; }
  wmDrag = wmPointOf(e);
}, true);
$("cvPaint").addEventListener("pointermove", e => {
  if (!wmDrag || store.mode !== "mark") return;
  const p = wmPointOf(e);
  if (Math.hypot(p.x - wmDrag.x, p.y - wmDrag.y) < 0.004) return;
  wm.x = Math.min(0.97, Math.max(0.03, p.x));
  wm.y = Math.min(0.97, Math.max(0.03, p.y));
  wmDrag = p;
  renderWmPreview();
});
$("cvPaint").addEventListener("pointerup", () => { wmDrag = null; });

/* ---------------- 封面工具 ---------------- */
const PLATFORMS = [
  // 赞助 / 会员平台
  { id: "patreon", name: "Patreon", word: "Patreon", mark: "P", color: "#F96854", kind: "sponsor" },
  { id: "fanbox", name: "FANBOX", word: "FANBOX", mark: "F", color: "#0096FA", kind: "sponsor" },
  { id: "fantia", name: "Fantia", word: "Fantia", mark: "Fa", color: "#F0527A", kind: "sponsor" },
  { id: "kofi", name: "Ko-fi", word: "Ko-fi", mark: "K", color: "#FF5E5B", kind: "sponsor" },
  { id: "substar", name: "SubscribeStar", word: "SubStar", mark: "S", color: "#0F6BFF", kind: "sponsor" },
  { id: "boosty", name: "Boosty", word: "Boosty", mark: "Bo", color: "#FF4D2D", kind: "sponsor" },
  { id: "gumroad", name: "Gumroad", word: "Gumroad", mark: "G", color: "#FF90E8", kind: "sponsor" },
  // 插图站 / 同人商店
  { id: "pixiv", name: "pixiv", word: "pixiv", mark: "p", color: "#0096FA", kind: "art" },
  { id: "dlsite", name: "DLsite", word: "DLsite", mark: "DL", color: "#E4007F", kind: "art" },
  { id: "booth", name: "BOOTH", word: "BOOTH", mark: "B", color: "#FC4F4F", kind: "art" },
  { id: "e621", name: "e621", word: "e621", mark: "e6", color: "#FFCB0B", kind: "booru" },
  { id: "rule34", name: "rule34", word: "rule34", mark: "r34", color: "#006FF9", kind: "booru" },
  { id: "fa", name: "FurAffinity", word: "FurAffinity", mark: "FA", color: "#FF7A00", kind: "booru" },
  { id: "da", name: "DeviantArt", word: "DeviantArt", mark: "DA", color: "#05CC47", kind: "art" },
  // 社交 / 视频
  { id: "x", name: "X", word: "X", mark: "X", color: "#0B0B0B", kind: "social" },
  { id: "bsky", name: "Bluesky", word: "Bluesky", mark: "bs", color: "#0085FF", kind: "social" },
  { id: "ins", name: "Instagram", word: "Instagram", mark: "IG", color: "#C13584", kind: "social" },
  { id: "youtube", name: "YouTube", word: "YouTube", mark: "YT", color: "#FF0000", kind: "social" },
  { id: "bili", name: "bilibili", word: "bilibili", mark: "b", color: "#FB7299", kind: "social" },
  { id: "douyin", name: "抖音", word: "抖音", mark: "抖", color: "#161823", kind: "social" },
  { id: "xhs", name: "小红书", word: "小红书", mark: "红", color: "#FF2442", kind: "social" },
  { id: "kuaishou", name: "快手", word: "快手", mark: "快", color: "#FF4906", kind: "social" },
  { id: "weibo", name: "微博", word: "微博", mark: "微", color: "#E6162D", kind: "social" },
];
const BADGE_STYLES = [
  { id: "official", name: "官方牌" },
  { id: "outline", name: "细描边" },
  { id: "plate", name: "色块印" },
  { id: "minimal", name: "极简字" },
  { id: "sticker", name: "贴纸" },
];
const COVER_TITLE_STYLES = [
  { id: "editorial", name: "刊头" },
  { id: "minimal", name: "极简" },
  { id: "outline", name: "空心" },
  { id: "stamp", name: "印章" },
  { id: "vertical", name: "竖排" },
  { id: "classic", name: "经典" },
];
const COVER_TEXTS = ["必看", "精选合集", "高清无水印", "完整版", "新作上线", "私藏推荐"];
const COVER_FILTERS = [
  { id: "none", name: "原图" },
  { id: "warm", name: "暖调" },
  { id: "cool", name: "冷调" },
  { id: "mono", name: "黑白" },
  { id: "vivid", name: "浓郁" },
  { id: "film", name: "胶片" },
  { id: "blur", name: "轻虚化底" },
];
const cover = {
  title: "", sub: "", showTitle: true, titleStyle: "editorial",
  font: "hei", size: 0.075, color: "#ffffff", align: "left",
  pos: "bottom", filter: "none", dim: 0.28,
  badges: [], badgePos: "tr", badgeSize: 0.085, badgeLayout: "v",
  badgeStyle: "official", link: "",
};
const _badgeMsr = document.createElement("canvas").getContext("2d");
function hexRGB(hex) {
  const s = String(hex || "#ffffff").replace("#", "");
  const n = s.length === 3 ? s.split("").map(c => c + c).join("") : (s + "000000").slice(0, 6);
  const v = parseInt(n, 16) || 0;
  return { r: (v >> 16) & 255, g: (v >> 8) & 255, b: v & 255 };
}
function withA(hex, a) {
  const c = hexRGB(hex);
  return "rgba(" + c.r + "," + c.g + "," + c.b + "," + a + ")";
}
function lumOf(hex) { const c = hexRGB(hex); return (0.299 * c.r + 0.587 * c.g + 0.114 * c.b) / 255; }
function inkFor(hex) { return lumOf(hex) > 0.62 ? "#111111" : "#ffffff"; }
function coverFontPx(h, k) { return Math.max(10, Math.round(h * k)); }
function coverFam() { return WM_FONTS[cover.font] || WM_FONTS.hei; }
function platformOf(b) {
  const id = typeof b === "string" ? b : (b && b.id);
  return PLATFORMS.find(v => v.id === id) || null;
}
function normBadges() {
  const arr = cover.badges || [];
  for (let i = 0; i < arr.length; i++) {
    const b = arr[i];
    if (typeof b === "string") arr[i] = { id: b, style: "", tag: "", link: "" };
    else if (b) {
      if (b.style == null) b.style = "";
      if (b.tag == null) b.tag = "";
      if (b.link == null) b.link = "";
    }
  }
  cover.badges = arr;
  return arr;
}
function coverAccent() {
  const list = cover.badges || [];
  const p = platformOf(list[0]);
  return p ? p.color : "#ffffff";
}
function applyCoverFilter(ctx, w, h, id) {
  if (id === "warm") { ctx.fillStyle = "rgba(255,150,60,.16)"; ctx.fillRect(0, 0, w, h); }
  else if (id === "cool") { ctx.fillStyle = "rgba(60,150,255,.16)"; ctx.fillRect(0, 0, w, h); }
  else if (id === "mono") { ctx.fillStyle = "rgba(0,0,0,.10)"; ctx.fillRect(0, 0, w, h); }
  else if (id === "vivid") {
    ctx.fillStyle = "rgba(255,60,120,.10)"; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "rgba(60,220,255,.10)"; ctx.fillRect(0, 0, w, h);
  } else if (id === "film") {
    ctx.fillStyle = "rgba(20,10,0,.16)"; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "rgba(255,240,200,.08)"; ctx.fillRect(0, 0, w, h);
  } else if (id === "blur") {
    ctx.fillStyle = "rgba(0,0,0,.18)"; ctx.fillRect(0, 0, w, h);
  }
}
function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.max(0, Math.min(r, Math.min(w, h) / 2));
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}
function coverWrap(ctx, text, maxW) {
  const out = [];
  let line = "";
  for (const ch of String(text)) {
    if (ch === "\n") { out.push(line); line = ""; continue; }
    const t = line + ch;
    if (ctx.measureText(t).width > maxW && line) { out.push(line); line = ch; } else line = t;
  }
  if (line) out.push(line);
  return out.slice(0, 4);
}
function drawCoverDeco(ctx, w, h) {
  if (cover.dim <= 0.001) return;
  const g = ctx.createLinearGradient(0, cover.pos === "top" ? 0 : h, 0, cover.pos === "top" ? h * 0.5 : h * 0.5);
  if (cover.pos === "bottom") { g.addColorStop(0, "rgba(0,0,0,0)"); g.addColorStop(1, "rgba(0,0,0," + cover.dim + ")"); }
  else if (cover.pos === "top") { g.addColorStop(0, "rgba(0,0,0," + cover.dim + ")"); g.addColorStop(1, "rgba(0,0,0,0)"); }
  else { g.addColorStop(0, "rgba(0,0,0," + (cover.dim * 0.85) + ")"); g.addColorStop(1, "rgba(0,0,0," + (cover.dim * 0.85) + ")"); }
  ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
}
function titleFont(style, px) {
  const fam = coverFam();
  if (style === "minimal") return "600 " + px + "px " + fam;
  if (style === "classic") return "700 " + px + "px " + fam;
  return "800 " + px + "px " + fam;
}
function drawCoverText(ctx, w, h) {
  const fam = coverFam();
  const px = coverFontPx(h, cover.size);
  const pad = Math.round(w * 0.055);
  const style = cover.titleStyle || "editorial";
  const accent = coverAccent();
  const align = cover.align === "center" ? "center" : cover.align === "right" ? "right" : "left";
  const x = align === "center" ? w / 2 : align === "right" ? w - pad : pad;
  const maxW = w - pad * 2;
  const top = cover.pos !== "bottom";
  const title = cover.showTitle ? String(cover.title || "").trim() : "";
  const sub = String(cover.sub || "").trim();
  ctx.save();
  ctx.textBaseline = "alphabetic";
  ctx.textAlign = align;

  if (style === "vertical") {
    const chars = Array.from(title).filter(c => c !== "\n").slice(0, 16);
    const vpx = Math.round(px * 0.96);
    const step = Math.round(vpx * 1.12);
    const totalH = chars.length * step;
    const vx = align === "center" ? w / 2 : align === "right" ? w - pad - vpx * 0.5 : pad + vpx * 0.5;
    const yTop = top ? pad : h - pad - totalH;
    ctx.font = "700 " + vpx + "px " + fam;
    ctx.shadowColor = "rgba(0,0,0,.55)"; ctx.shadowBlur = vpx * 0.16;
    chars.forEach((ch, i) => {
      ctx.fillStyle = cover.color;
      ctx.fillText(ch, vx, yTop + i * step + vpx * 0.84);
    });
    ctx.shadowBlur = 0;
    const lx = align === "right" ? vx - vpx * 0.78 : vx + vpx * 0.78;
    ctx.fillStyle = withA(accent, .9);
    ctx.fillRect(lx, yTop, Math.max(2, Math.round(vpx * 0.06)), totalH);
    if (sub) {
      const spx2 = Math.max(9, Math.round(px * 0.32));
      ctx.font = "500 " + spx2 + "px " + fam;
      ctx.fillStyle = "rgba(255,255,255,.86)";
      ctx.fillText(sub.slice(0, 40), vx, top ? yTop + totalH + spx2 * 1.8 : yTop - spx2 * 1.1);
    }
    ctx.restore();
    return;
  }

  let lines = [];
  if (title) {
    ctx.font = titleFont(style, px);
    lines = coverWrap(ctx, title, maxW);
  }
  const lh = Math.round(px * (style === "editorial" ? 1.10 : 1.2));
  const subPx = Math.max(9, Math.round(px * 0.36));
  const blockH = lines.length ? lines.length * lh : 0;
  const total = blockH + (sub ? subPx * 1.9 + px * 0.14 : 0) + (style === "stamp" ? px * 0.18 : 0);
  let y0;
  if (cover.pos === "top") y0 = pad;
  else if (cover.pos === "center") y0 = Math.max(pad, (h - total) / 2);
  else y0 = h - pad - total;

  const one = (text, yy) => {
    if (style === "outline") {
      ctx.font = titleFont(style, px);
      ctx.lineWidth = Math.max(2, px * 0.05);
      ctx.strokeStyle = cover.color;
      ctx.shadowColor = "rgba(0,0,0,.35)"; ctx.shadowBlur = px * 0.16;
      ctx.strokeText(text, x, yy);
      ctx.shadowBlur = 0;
      ctx.lineWidth = Math.max(1, px * 0.014);
      ctx.strokeStyle = withA(accent, .9);
      ctx.strokeText(text, x, yy);
    } else if (style === "stamp") {
      ctx.font = titleFont(style, px);
      ctx.save();
      ctx.textAlign = "left";
      const tw = ctx.measureText(text).width;
      const bw = tw + px * 0.62, bh = px * 1.16;
      const bx = align === "center" ? x - bw / 2 : align === "right" ? x - bw : x;
      ctx.strokeStyle = cover.color;
      ctx.lineWidth = Math.max(2, px * 0.05);
      ctx.shadowColor = "rgba(0,0,0,.3)"; ctx.shadowBlur = px * 0.12;
      roundRect(ctx, bx, yy - px * 0.92, bw, bh, px * 0.08); ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.fillStyle = cover.color;
      ctx.fillText(text, bx + px * 0.31, yy);
      ctx.restore();
    } else {
      ctx.font = titleFont(style, px);
      ctx.fillStyle = cover.color;
      ctx.shadowColor = "rgba(0,0,0,.5)";
      ctx.shadowBlur = px * (style === "minimal" ? 0.1 : 0.18);
      ctx.fillText(text, x, yy);
      ctx.shadowBlur = 0;
    }
  };
  lines.forEach((ln, i) => one(ln, y0 + px * 0.86 + i * lh));

  if (blockH && (style === "editorial" || style === "minimal")) {
    const rw = Math.min(maxW, Math.max(px * 2.2, px * 0.35 * Math.min(6, title.length)));
    const rh = Math.max(2, Math.round(px * 0.075));
    const rx = align === "center" ? x - rw / 2 : align === "right" ? x - rw : x;
    ctx.fillStyle = withA(accent, .95);
    ctx.fillRect(rx, y0 + blockH + px * 0.10, rw, rh);
  }
  if (sub) {
    ctx.font = "500 " + subPx + "px " + fam;
    ctx.fillStyle = "rgba(255,255,255,.9)";
    ctx.shadowColor = "rgba(0,0,0,.45)"; ctx.shadowBlur = subPx * 0.9;
    ctx.fillText(sub.slice(0, 64), x, y0 + blockH + subPx * 1.55 + px * 0.12);
    ctx.shadowBlur = 0;
  }
  ctx.restore();
}
function badgeTag(b) {
  const t = String(b.tag || "").trim();
  if (!t) return "";
  return /^[@＠]/.test(t) ? t : "@" + t;
}
function badgeGeo(p, b, bh) {
  const fam = coverFam();
  const style = b.style || cover.badgeStyle || "official";
  const tag = badgeTag(b);
  const fs = Math.max(7, Math.round(bh * 0.40));
  const padx = bh * 0.48, gap = bh * 0.30;
  _badgeMsr.font = "700 " + fs + "px " + fam;
  const wordW = _badgeMsr.measureText(p.word).width;
  _badgeMsr.font = "500 " + Math.round(fs * 0.84) + "px " + fam;
  const tagW = tag ? _badgeMsr.measureText(tag).width : 0;
  let w = 0, boxH = bh, mk = 0;
  if (style === "official") w = padx * 2 + wordW + (tag ? gap * 1.1 + tagW : 0);
  else if (style === "outline") w = padx * 2.1 + wordW + (tag ? gap + tagW : 0);
  else if (style === "plate") { w = padx * 2 + Math.max(wordW, tagW); boxH = Math.round(bh * 1.34); }
  else if (style === "minimal") { mk = bh * 0.82; w = mk + bh * 0.30 + wordW + (tag ? gap + tagW : 0); }
  else { mk = bh * 0.62; w = bh * 0.22 + mk + bh * 0.18 + wordW + (tag ? gap + tagW : 0) + bh * 0.26; boxH = Math.round(bh * 1.04); }
  const link = String(b.link || "").trim();
  let linkH = 0;
  if (link) {
    _badgeMsr.font = "500 " + Math.round(bh * 0.30) + "px " + (WM_FONTS.mono || fam);
    linkH = Math.round(bh * 0.46);
    w = Math.max(w, _badgeMsr.measureText(link).width);
  }
  return { style, tag, fs, w: Math.ceil(w), boxH, h: boxH + linkH, wordW, tagW, padx, gap, mk, link };
}
function drawBadge(ctx, p, b, g, x, y) {
  const fam = coverFam();
  const ink = inkFor(p.color);
  const boxH = g.boxH;
  ctx.save();
  if (g.style === "official") {
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,.38)"; ctx.shadowBlur = boxH * 0.26; ctx.shadowOffsetY = boxH * 0.08;
    ctx.fillStyle = p.color;
    roundRect(ctx, x, y, g.w, boxH, boxH * 0.24); ctx.fill();
    ctx.restore();
    ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillStyle = ink;
    ctx.font = "700 " + g.fs + "px " + fam;
    ctx.fillText(p.word, x + g.padx, y + boxH / 2 + 1);
    if (g.tag) {
      const sx = x + g.padx + g.wordW + g.gap * 0.55;
      ctx.globalAlpha = 0.42;
      ctx.fillRect(sx, y + boxH * 0.26, Math.max(1, Math.round(boxH * 0.03)), boxH * 0.48);
      ctx.globalAlpha = 0.92;
      ctx.font = "500 " + Math.round(g.fs * 0.84) + "px " + fam;
      ctx.fillText(g.tag, sx + g.gap * 0.6, y + boxH / 2 + 1);
      ctx.globalAlpha = 1;
    }
  } else if (g.style === "outline") {
    ctx.lineWidth = Math.max(1.5, boxH * 0.055);
    ctx.strokeStyle = p.color;
    ctx.shadowColor = "rgba(0,0,0,.45)"; ctx.shadowBlur = boxH * 0.22;
    roundRect(ctx, x, y, g.w, boxH, boxH * 0.3); ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillStyle = p.color;
    ctx.font = "600 " + g.fs + "px " + fam;
    ctx.fillText(p.word, x + g.padx, y + boxH / 2 + 1);
    if (g.tag) {
      ctx.globalAlpha = 0.85;
      ctx.font = "400 " + Math.round(g.fs * 0.82) + "px " + fam;
      ctx.fillText(g.tag, x + g.padx + g.wordW + g.gap * 0.8, y + boxH / 2 + 1);
      ctx.globalAlpha = 1;
    }
  } else if (g.style === "plate") {
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,.4)"; ctx.shadowBlur = boxH * 0.22; ctx.shadowOffsetY = boxH * 0.07;
    ctx.fillStyle = p.color;
    roundRect(ctx, x, y, g.w, boxH, boxH * 0.14); ctx.fill();
    ctx.restore();
    ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillStyle = ink;
    ctx.font = "800 " + g.fs + "px " + fam;
    ctx.fillText(p.word, x + g.padx, y + boxH * 0.38);
    ctx.globalAlpha = 0.8;
    ctx.font = "600 " + Math.round(g.fs * 0.58) + "px " + fam;
    ctx.fillText(g.tag || "★", x + g.padx, y + boxH * 0.70);
    ctx.globalAlpha = 1;
  } else if (g.style === "minimal") {
    const mk = g.mk;
    ctx.fillStyle = p.color;
    roundRect(ctx, x, y + (boxH - mk) / 2, mk, mk, mk * 0.28); ctx.fill();
    ctx.fillStyle = inkFor(p.color);
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.font = "800 " + Math.round(mk * 0.58) + "px " + fam;
    ctx.fillText(p.mark, x + mk / 2, y + boxH / 2 + 1);
    ctx.textAlign = "left";
    const tx = x + mk + boxH * 0.30;
    ctx.fillStyle = "#ffffff";
    ctx.shadowColor = "rgba(0,0,0,.5)"; ctx.shadowBlur = boxH * 0.18;
    ctx.font = "700 " + g.fs + "px " + fam;
    ctx.fillText(p.word, tx, y + boxH / 2 + 1);
    ctx.shadowBlur = 0;
    if (g.tag) {
      ctx.font = "400 " + Math.round(g.fs * 0.8) + "px " + fam;
      ctx.fillStyle = "rgba(255,255,255,.75)";
      ctx.fillText(g.tag, tx + g.wordW + g.gap * 0.55, y + boxH / 2 + 1);
    }
    ctx.fillStyle = withA(p.color, .95);
    ctx.fillRect(x, y + boxH + Math.round(boxH * 0.10), g.w, Math.max(1, Math.round(boxH * 0.05)));
  } else {
    ctx.save();
    ctx.translate(x + g.w / 2, y + boxH / 2);
    ctx.rotate(-0.045);
    ctx.shadowColor = "rgba(0,0,0,.35)"; ctx.shadowBlur = boxH * 0.3; ctx.shadowOffsetY = boxH * 0.1;
    ctx.fillStyle = "rgba(255,255,255,.96)";
    roundRect(ctx, -g.w / 2, -boxH / 2, g.w, boxH, boxH * 0.34); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = withA(p.color, .5); ctx.lineWidth = Math.max(1, boxH * 0.03);
    roundRect(ctx, -g.w / 2, -boxH / 2, g.w, boxH, boxH * 0.34); ctx.stroke();
    const mk = boxH * 0.6;
    ctx.fillStyle = p.color;
    roundRect(ctx, -g.w / 2 + boxH * 0.24, -mk / 2, mk, mk, mk * 0.3); ctx.fill();
    ctx.fillStyle = inkFor(p.color);
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.font = "800 " + Math.round(mk * 0.58) + "px " + fam;
    ctx.fillText(p.mark, -g.w / 2 + boxH * 0.24 + mk / 2, 1);
    ctx.textAlign = "left";
    const tx0 = -g.w / 2 + boxH * 0.24 + mk + boxH * 0.16;
    ctx.fillStyle = "#141414";
    ctx.font = "800 " + g.fs + "px " + fam;
    ctx.fillText(p.word, tx0, 1);
    if (g.tag) {
      ctx.font = "500 " + Math.round(g.fs * 0.8) + "px " + fam;
      ctx.fillStyle = withA(p.color, .95);
      ctx.fillText(g.tag, tx0 + g.wordW + g.gap * 0.5, 1);
    }
    ctx.restore();
  }
  if (g.link) {
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.font = "500 " + Math.round(boxH * 0.30) + "px " + (WM_FONTS.mono || fam);
    ctx.fillStyle = "rgba(255,255,255,.8)";
    ctx.shadowColor = "rgba(0,0,0,.55)"; ctx.shadowBlur = boxH * 0.22;
    ctx.fillText(g.link, x, y + boxH + Math.round(boxH * 0.42));
    ctx.shadowBlur = 0;
  }
  ctx.restore();
}
function drawCoverBadges(ctx, w, h) {
  const list = normBadges().filter(b => platformOf(b));
  if (!list.length) return;
  const bh = Math.max(8, Math.round(h * cover.badgeSize));
  const pad = Math.round(w * 0.035);
  const gap = Math.round(bh * 0.30);
  const right = cover.badgePos === "tr" || cover.badgePos === "br";
  const bottom = cover.badgePos === "bl" || cover.badgePos === "br";
  const blocks = list.map(b => { const p = platformOf(b); return { b, p, g: badgeGeo(p, b, bh) }; });
  ctx.save();
  if (cover.badgeLayout === "h") {
    const totalW = blocks.reduce((a, v) => a + v.g.w + gap, -gap);
    let x = right ? w - pad - totalW : pad;
    const maxH = Math.max.apply(null, blocks.map(v => v.g.h));
    const y = bottom ? h - pad - maxH : pad;
    blocks.forEach(v => { drawBadge(ctx, v.p, v.b, v.g, x, y + (maxH - v.g.h) / 2); x += v.g.w + gap; });
  } else {
    const totalH = blocks.reduce((a, v) => a + v.g.h + gap, -gap);
    let y = bottom ? h - pad - totalH : pad;
    blocks.forEach(v => {
      const bx = right ? w - pad - v.g.w : pad;
      drawBadge(ctx, v.p, v.b, v.g, bx, y);
      y += v.g.h + gap;
    });
  }
  ctx.restore();
}
function drawCoverLink(ctx, w, h) {
  const txt = String(cover.link || "").trim();
  if (!txt) return;
  const pad = Math.round(w * 0.035);
  const fs = Math.max(10, Math.round(h * 0.022));
  const rightSide = cover.badgePos === "tr" || cover.badgePos === "br";
  ctx.save();
  ctx.font = "500 " + fs + "px " + (WM_FONTS.mono || coverFam());
  ctx.fillStyle = "rgba(255,255,255,.82)";
  ctx.shadowColor = "rgba(0,0,0,.55)"; ctx.shadowBlur = fs * 0.6;
  ctx.textBaseline = "alphabetic";
  ctx.textAlign = rightSide ? "left" : "right";
  ctx.fillText(txt, rightSide ? pad : w - pad, h - pad);
  ctx.restore();
}
function drawCover(ctx, w, h) {
  applyCoverFilter(ctx, w, h, cover.filter);
  drawCoverDeco(ctx, w, h);
  drawCoverText(ctx, w, h);
  drawCoverBadges(ctx, w, h);
  drawCoverLink(ctx, w, h);
}
/* 封面模式用「整图适应」而不是铺满，否则标题/标识会跑到视口外面 */
function fitWholeForCover() {
  const vp = $("vp");
  if (!vp || !cur) return;
  const vw = vp.clientWidth, vh = vp.clientHeight;
  const w = cvPaint.width || cur.w || 0, h = cvPaint.height || cur.h || 0;
  if (!vw || !vh || !w || !h) return;
  const st = pane.state;
  const k = Math.min(vw / w, vh / h);
  if (!isFinite(k) || k <= 0) return;
  st.k = k;
  st.x = (vw - w * k) / 2;
  st.y = (vh - h * k) / 2;
  pane.apply();
}
function renderCoverPreview() {
  if (!cur || !cvWm) return;
  const w = cvPaint.width, h = cvPaint.height;
  if (!w || !h) return;
  cvWm.width = w; cvWm.height = h;
  cvWm.style.width = w + "px"; cvWm.style.height = h + "px";
  wctx.clearRect(0, 0, w, h);
  if (store.mode !== "cover") return;
  wmLayers(cur).forEach(cfg => drawWatermark(wctx, w, h, cfg, cfg.phase || 0));
  drawCover(wctx, w, h);
}
function applyCoverPreviewFilter() {
  if (store.mode === "cover") document.body.classList.add("cover");
  else document.body.classList.remove("cover");
}
function buildCoverPanel(p) {
  p.innerHTML = "";
  p.appendChild(el("h3", null, "封面工具"));
  p.appendChild(el("div", "hint", "选中一张图，在它上面叠标题、滤镜和平台标识；确认后复制出一张新图「原名_封面.png」，原图不动。"));

  /* ---- 标题 ---- */
  const g0 = el("div", "grp");
  g0.appendChild(el("div", "hint strong", "标题"));
  const ti = el("input"); ti.type = "text"; ti.value = cover.title; ti.placeholder = "封面主标题";
  ti.oninput = () => { cover.title = ti.value; renderCoverPreview(); };
  g0.appendChild(ti);
  const si = el("input"); si.type = "text"; si.value = cover.sub; si.placeholder = "副标题 / 一行备注（可空）";
  si.oninput = () => { cover.sub = si.value; renderCoverPreview(); };
  g0.appendChild(si);
  const chips = el("div", "chips small");
  for (const t of COVER_TEXTS) {
    const c = el("button", "chip", t);
    c.onclick = () => { cover.title = t; ti.value = t; renderCoverPreview(); };
    chips.appendChild(c);
  }
  g0.appendChild(chips);

  g0.appendChild(el("div", "hint", "标题样式"));
  const stChips = el("div", "chips small");
  for (const st of COVER_TITLE_STYLES) {
    const c = el("button", "chip" + ((cover.titleStyle || "editorial") === st.id ? " on" : ""), st.name);
    c.onclick = () => { cover.titleStyle = st.id; renderCoverPreview(); buildPanel(); };
    stChips.appendChild(c);
  }
  g0.appendChild(stChips);

  const r1 = el("div", "row");
  const ck = el("input"); ck.type = "checkbox"; ck.checked = cover.showTitle;
  ck.onchange = () => { cover.showTitle = ck.checked; renderCoverPreview(); };
  r1.append(ck, el("label", null, "显示标题"));
  const col = el("input"); col.type = "color"; col.value = cover.color;
  col.oninput = () => { cover.color = col.value; renderCoverPreview(); };
  r1.appendChild(col);
  g0.appendChild(r1);

  const r2 = el("div", "row");
  r2.appendChild(el("label", null, "位置"));
  const ps = el("select");
  for (const [k, v] of [["bottom", "底部"], ["top", "顶部"], ["center", "整屏"]]) {
    const o = el("option", null, v); o.value = k; if (cover.pos === k) o.selected = true; ps.appendChild(o);
  }
  ps.onchange = () => { cover.pos = ps.value; renderCoverPreview(); };
  r2.appendChild(ps);
  const al = el("select");
  for (const [k, v] of [["left", "左对齐"], ["center", "居中"], ["right", "右对齐"]]) {
    const o = el("option", null, v); o.value = k; if (cover.align === k) o.selected = true; al.appendChild(o);
  }
  al.onchange = () => { cover.align = al.value; renderCoverPreview(); };
  r2.appendChild(al);
  g0.appendChild(r2);

  g0.appendChild(wmSlider("标题大小", 0.03, 0.18, 0.005, cover.size, v => { cover.size = +v; renderCoverPreview(); }));
  g0.appendChild(wmSlider("压暗", 0, 0.8, 0.02, cover.dim, v => { cover.dim = +v; renderCoverPreview(); }));

  const r3 = el("div", "row");
  r3.appendChild(el("label", null, "滤镜"));
  const fs = el("select");
  for (const f of COVER_FILTERS) { const o = el("option", null, f.name); o.value = f.id; if (cover.filter === f.id) o.selected = true; fs.appendChild(o); }
  fs.onchange = () => { cover.filter = fs.value; renderCoverPreview(); };
  r3.appendChild(fs);
  g0.appendChild(r3);
  p.appendChild(g0);

  /* ---- 平台标识 ---- */
  const g2 = el("div", "grp");
  g2.appendChild(el("div", "hint strong", "平台标识"));
  const list = normBadges();
  const chips2 = el("div", "chips small");
  for (const pf of PLATFORMS) {
    const on = list.some(b => b.id === pf.id);
    const c = el("button", "pfchip" + (on ? " on" : ""));
    const dot = el("i"); dot.style.background = pf.color; c.appendChild(dot);
    c.appendChild(document.createTextNode(pf.name));
    c.onclick = () => {
      const i = list.findIndex(b => b.id === pf.id);
      if (i < 0) list.push({ id: pf.id, style: "", tag: "", link: "" });
      else list.splice(i, 1);
      renderCoverPreview(); buildPanel();
    };
    chips2.appendChild(c);
  }
  g2.appendChild(chips2);

  const r4 = el("div", "row");
  r4.appendChild(el("label", null, "默认样式"));
  const bs = el("select");
  for (const st of BADGE_STYLES) {
    const o = el("option", null, st.name); o.value = st.id; if (cover.badgeStyle === st.id) o.selected = true; bs.appendChild(o);
  }
  bs.onchange = () => { cover.badgeStyle = bs.value; renderCoverPreview(); };
  r4.appendChild(bs);
  g2.appendChild(r4);

  const r5 = el("div", "row");
  const bp = el("select");
  for (const [k, v] of [["tr", "右上"], ["tl", "左上"], ["br", "右下"], ["bl", "左下"]]) {
    const o = el("option", null, v); o.value = k; if (cover.badgePos === k) o.selected = true; bp.appendChild(o);
  }
  bp.onchange = () => { cover.badgePos = bp.value; renderCoverPreview(); };
  r5.appendChild(bp);
  const bl = el("select");
  for (const [k, v] of [["v", "竖排"], ["h", "横排"]]) {
    const o = el("option", null, v); o.value = k; if (cover.badgeLayout === k) o.selected = true; bl.appendChild(o);
  }
  bl.onchange = () => { cover.badgeLayout = bl.value; renderCoverPreview(); };
  r5.appendChild(bl);
  g2.appendChild(r5);
  g2.appendChild(wmSlider("标识大小", 0.03, 0.16, 0.005, cover.badgeSize, v => { cover.badgeSize = +v; renderCoverPreview(); }));

  list.forEach((b, idx) => {
    const pf = platformOf(b);
    if (!pf) return;
    const box = el("div", "pfbox");
    const hd = el("div", "hd");
    const dot = el("i"); dot.style.background = pf.color; hd.appendChild(dot);
    hd.appendChild(el("div", "nm", pf.name));
    const st = el("select");
    const o0 = el("option", null, "跟随默认"); o0.value = ""; if (!b.style) o0.selected = true; st.appendChild(o0);
    for (const sty of BADGE_STYLES) {
      const o = el("option", null, sty.name); o.value = sty.id; if (b.style === sty.id) o.selected = true; st.appendChild(o);
    }
    st.onchange = () => { b.style = st.value; renderCoverPreview(); };
    hd.appendChild(st);
    const del = el("button", "danger", "×");
    del.onclick = () => { list.splice(idx, 1); renderCoverPreview(); buildPanel(); };
    hd.appendChild(del);
    box.appendChild(hd);
    const tg = el("input"); tg.type = "text"; tg.placeholder = "ID（例：@hchimao）"; tg.value = b.tag || "";
    tg.oninput = () => { b.tag = tg.value; renderCoverPreview(); };
    box.appendChild(tg);
    const lk = el("input"); lk.type = "text"; lk.placeholder = "链接（可空，例：patreon.com/xxx）"; lk.value = b.link || "";
    lk.oninput = () => { b.link = lk.value; renderCoverPreview(); };
    box.appendChild(lk);
    g2.appendChild(box);
  });

  g2.appendChild(el("div", "hint", "整图链接行（单独显示在角落，可空）"));
  const lkAll = el("input"); lkAll.type = "text"; lkAll.placeholder = "例：pixiv.net/users/12345678";
  lkAll.value = cover.link || "";
  lkAll.oninput = () => { cover.link = lkAll.value; renderCoverPreview(); };
  g2.appendChild(lkAll);
  p.appendChild(g2);

  /* ---- 操作 ---- */
  const g3 = el("div", "grp");
  const row = el("div", "row");
  const bGo = el("button", "primary", "确认封面并复制一张");
  bGo.disabled = !cur;
  bGo.onclick = () => applyCover(bGo);
  const bExp = el("button", null, "单独导出已做更改");
  bExp.disabled = !store.items.length;
  bExp.onclick = () => exportEdited(bExp, "cover");
  row.append(bGo, bExp);
  g3.appendChild(row);
  const bRs = el("button", null, "重置封面设置");
  bRs.onclick = () => {
    Object.assign(cover, {
      title: "", sub: "", showTitle: true, titleStyle: "editorial", font: "hei",
      size: 0.075, color: "#ffffff", align: "left", pos: "bottom",
      filter: "none", dim: 0.28, badges: [], badgePos: "tr", badgeSize: 0.085,
      badgeLayout: "v", badgeStyle: "official", link: "",
    });
    renderCoverPreview(); buildPanel();
  };
  g3.appendChild(bRs);
  p.appendChild(g3);
  p.appendChild(el("div", "hint", "「单独导出已做更改」只导出画面里勾选的那些图，并套上现在的封面设置（勾选后再点）；点「确认封面并复制一张」则是把封面焊进一张新图。"));
}
function composeCover(it) {
  const w = Math.max(1, it.w || 800), h = Math.max(1, it.h || 600);
  const out = document.createElement("canvas"); out.width = w; out.height = h;
  const ox = out.getContext("2d");
  ox.fillStyle = "#000"; ox.fillRect(0, 0, w, h);
  return { out, ox, w, h };
}
async function applyCover(btn) {
  if (!cur) { toast("先在中间选一张图"); return; }
  const it = cur;
  const old = btn ? btn.textContent : "";
  if (btn) { btn.disabled = true; btn.textContent = "生成中…"; }
  try {
    const w = Math.max(1, it.w || 800), h = Math.max(1, it.h || 600);
    const out = document.createElement("canvas"); out.width = w; out.height = h;
    const ox = out.getContext("2d");
    const base = await composeCanvas(it);
    ox.drawImage(base, 0, 0, w, h);
    drawCover(ox, w, h);
    const blob = await new Promise(r => out.toBlob(r, "image/png"));
    const dir = dirOf(it);
    const dot = it.name.lastIndexOf(".");
    const stem = dot > 0 ? it.name.slice(0, dot) : it.name;
    const j = await api("/api/workspace/put?dir=" + encodeURIComponent(dir) + "&name=" + encodeURIComponent(stem + "_封面.png"),
      { method: "POST", body: blob });
    if (!j.ok) throw new Error(j.error || "写入失败");
    const nit = poolItem(j.item);
    if (!store.items.some(x => x.path === nit.path)) store.items.push(nit);
    store.curId = nit.id;
    renderGrid(); updateSelInfo();
    if (btn) btn.textContent = old;
    toast("已生成封面：" + j.item.name);
  } catch (e) { toast("封面生成失败：" + e.message, 3000); }
  finally { if (btn) { btn.disabled = false; btn.textContent = old; } }
}

/* ---------------- 排序工具 ---------------- */
let sortDrag = null;
function sortList() { return store.items; }
function bindSortCard(card, it) {
  card.addEventListener("dragstart", e => {
    const ids = store.sel.has(it.id) && store.sel.size ? [...store.sel] : [it.id];
    sortDrag = { ids, from: it.id };
    card.classList.add("dragging");
    try { e.dataTransfer.setData("text/plain", ids.join(",")); e.dataTransfer.effectAllowed = "move"; } catch (x) {}
  });
  card.addEventListener("dragend", () => {
    card.classList.remove("dragging");
    document.querySelectorAll(".dropbefore,.dropafter").forEach(n => n.classList.remove("dropbefore", "dropafter"));
    sortDrag = null;
  });
  card.addEventListener("dragover", e => {
    if (!sortDrag) return;
    e.preventDefault();
    const r = card.getBoundingClientRect();
    const before = (e.clientX - r.left) < r.width / 2;
    card.classList.toggle("dropbefore", before);
    card.classList.toggle("dropafter", !before);
  });
  card.addEventListener("dragleave", () => card.classList.remove("dropbefore", "dropafter"));
  card.addEventListener("drop", e => {
    if (!sortDrag) return;
    e.preventDefault(); e.stopPropagation();
    const r = card.getBoundingClientRect();
    const before = (e.clientX - r.left) < r.width / 2;
    card.classList.remove("dropbefore", "dropafter");
    dropOnCard(sortDrag.ids, it.id, before);
    sortDrag = null;
  });
}
function dropOnCard(ids, targetId, before) {
  const moving = store.items.filter(i => ids.includes(i.id));
  if (!moving.length) return;
  const rest = store.items.filter(i => !ids.includes(i.id));
  let ti = rest.findIndex(i => i.id === targetId);
  if (ti < 0) ti = rest.length;
  else if (!before) ti += 1;
  rest.splice(ti, 0, ...moving);
  const _sb = snap();
  store.items = rest;
  store.order = rest.map(i => i.rel);
  store.sorted = false;
  saveState(); renderGrid();
  if (store.mode === "sort") buildPanel();
  if (snapKey(_sb) !== snapKey(snap())) {
    pushW({ t: "snap", before: _sb, after: snap() });
  }
}
function moveSelected(where) {
  const ids = store.sel.size ? [...store.sel] : (store.curId ? [store.curId] : []);
  if (!ids.length) { toast("先在「全部」里勾选要移动的图"); return; }
  const moving = store.items.filter(i => ids.includes(i.id));
  let rest = store.items.filter(i => !ids.includes(i.id));
  if (where === "top") rest = [...moving, ...rest];
  else if (where === "bottom") rest = [...rest, ...moving];
  else if (where === "up" || where === "down") {
    const list = store.items.slice();
    if (where === "up") {
      for (let i = 1; i < list.length; i++) if (ids.includes(list[i].id) && !ids.includes(list[i - 1].id)) [list[i - 1], list[i]] = [list[i], list[i - 1]];
    } else {
      for (let i = list.length - 2; i >= 0; i--) if (ids.includes(list[i].id) && !ids.includes(list[i + 1].id)) [list[i + 1], list[i]] = [list[i], list[i + 1]];
    }
    rest = list;
  }
  const before = snap();
  store.items = rest; store.sorted = false;
  store.order = rest.map(i => i.rel);
  saveState(); renderGrid(); buildPanel();
  if (snapKey(before) !== snapKey(snap())) {
    const after = snap();
    pushW({ t: "snap", before, after });
  }
}

/* ===== 工作级撤回栈（排序 / 改名 / 删除） ===== */
const wU = { stack: [], redo: [] };
function pushW(op) { wU.stack.push(op); if (wU.stack.length > 80) wU.stack.shift(); wU.redo.length = 0; syncWU(); }
function syncWU() {
  const a = $("wUndo"), b = $("wRedo");
  if (a) a.disabled = !wU.stack.length;
  if (b) b.disabled = !wU.redo.length;
}
function snapKey(s) { return s && s.list ? s.list.map(x => x.name).join("|") : ""; }
function snap() {
  return { list: store.items.map(i => ({ uid: i.uid, rel: i.rel, name: i.name })) };
}
async function restoreSnap(s) {
  if (!s || !s.list) return;
  const pairs = [];
  for (const it of store.items) {
    let want = s.list.find(x => x.uid === it.uid);
    if (!want) want = s.list.find(x => x.rel === it.rel);
    if (!want) want = s.list.find(x => x.name === it.name);
    if (want && want.name && want.name !== it.name) pairs.push({ rel: it.rel, name: want.name });
  }
  if (pairs.length) await renameFiles(pairs, { push: false });
  const rank = new Map();
  s.list.forEach((x, i) => { rank.set(x.name, i); if (x.uid) rank.set("u:" + x.uid, i); });
  const key = i => {
    if (rank.has("u:" + i.uid)) return rank.get("u:" + i.uid);
    if (rank.has(i.name)) return rank.get(i.name);
    return 1e9;
  };
  store.items.sort((a, b) => key(a) - key(b));
  store.order = store.items.map(i => i.rel);
  store.sorted = false;
  saveState(); renderGrid(); buildPanel(); updateSelInfo();
}
async function renameFiles(pairs, opts) {
  const j = await api("/api/workspace/rename-file", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items: pairs }),
  });
  if (!j.ok) throw new Error(j.error || tf("改名失败"));
  const res = j.renamed || [];
  const byRel = new Map(store.items.map(i => [i.rel, i]));
  for (const r of res) {
    const it = byRel.get(r.from);
    if (!it) continue;
    const oldPath = it.path;
    store.pool.delete(oldPath);
    if (it.src === oldPath) it.src = r.path;
    if (it.baseSrc === oldPath) it.baseSrc = r.path;
    it.path = r.path; it.rel = r.rel; it.name = r.name;
    it.thumb = "/api/thumb?w=420&path=" + encodeURIComponent(r.path);
    it.thumbCache = null;
    store.pool.set(r.path, it);
  }
  store.order = store.items.map(i => i.rel);
  saveState();
  if (!opts || opts.push !== false) pushW({ t: "snap", before: null, after: snap() });
  renderGrid(); buildPanel(); updateSelInfo();
  return res;
}
async function undoWork() {
  const op = wU.stack.pop();
  if (!op) { toast(tf("没有可撤回的操作")); return; }
  try {
    if (op.t === "snap") {
      await restoreSnap(op.before);
    } else if (op.t === "del") {
      await api("/api/workspace/restore", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: op.token }),
      });
      await syncItems();
      await restoreSnap(op.before);
    }
    wU.redo.push(op);
    toast(tf("已撤回"));
  } catch (e) { toast(e.message, 3000); }
  syncWU();
}
async function redoWork() {
  const op = wU.redo.pop();
  if (!op) return;
  try {
    if (op.t === "snap") {
      await restoreSnap(op.after);
    } else if (op.t === "del") {
      const j = await api("/api/workspace/delete-files", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rels: op.rels }),
      });
      if (!j.ok) throw new Error(j.error || tf("删除失败"));
      const gone = new Set(op.rels);
      store.items.filter(i => gone.has(i.rel)).forEach(i => store.pool.delete(i.path));
      store.items = store.items.filter(i => !gone.has(i.rel));
      store.order = store.items.map(i => i.rel);
      saveState(); renderGrid(); buildPanel(); updateSelInfo();
    }
    wU.stack.push(op);
    toast(tf("已重做"));
  } catch (e) { toast(e.message, 3000); }
  syncWU();
}

function startRename(it, node) {
  node.contentEditable = "true";
  node.spellcheck = false;
  node.for;
  node.focus();
  try {
    const r = document.createRange(); r.selectNodeContents(node);
    const s = getSelection(); s.removeAllRanges(); s.addRange(r);
  } catch (e) {}
  let cancelled = false;
  node.onkeydown = ev => {
    ev.stopPropagation();
    if (ev.key === "Enter") { ev.preventDefault(); node.blur(); }
    else if (ev.key === "Escape") { ev.preventDefault(); cancelled = true; node.blur(); }
  };
  node.onblur = async () => {
    node.contentEditable = "false";
    const v = (node.textContent || "").trim();
    node.textContent = it.name;
    node.onkeydown = null; node.onblur = null;
    if (cancelled || !v || v === it.name) return;
    try {
      const before = snap();
      await renameFiles([{ rel: it.rel, name: v }], { push: false });
      pushW({ t: "snap", before, after: snap() });
      toast(tf("已改名") + "：" + v);
    } catch (e) { toast(e.message, 3000); }
  };
}

async function deleteChecked() {
  const sel = store.items.filter(i => store.sel.has(i.id));
  if (!sel.length) { toast(tf("先选中要加水印的图片")); return; }
  const before = snap(), rels = sel.map(i => i.rel);
  try {
    const j = await api("/api/workspace/delete-files", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rels }),
    });
    if (!j.ok) throw new Error(j.error || tf("删除失败"));
    sel.forEach(i => store.pool.delete(i.path));
    store.items = store.items.filter(i => !store.sel.has(i.id));
    store.sel.clear();
    store.order = store.items.map(i => i.rel);
    saveState();
    pushW({ t: "del", before, token: j.token, rels });
    renderGrid(); buildPanel(); updateSelInfo();
    toast(tf("已删除") + " ×" + (j.items || []).length);
  } catch (e) { toast(e.message, 3000); }
}

async function batchRename(scope) {
  const list = scope === "sel" ? store.items.filter(i => store.sel.has(i.id)) : store.items.slice();
  if (!list.length) { toast(tf("先选中要加水印的图片")); return; }
  const pre = ($("bnPre") || {}).value || "";
  const st = Math.max(0, parseInt(($("bnStart") || {}).value || "1", 10) || 1);
  const dg = Math.min(6, Math.max(1, parseInt(($("bnDigits") || {}).value || "2", 10) || 2));
  const before = snap();
  const pairs = list.map((it, k) => {
    const m = String(it.name).match(/\.[^.]+$/);
    const ext = m ? m[0] : ".png";
    return { rel: it.rel, name: pre + String(st + k).padStart(dg, "0") + ext };
  });
  try {
    await renameFiles(pairs, { push: false });
    pushW({ t: "snap", before, after: snap() });
    toast(tf("已改名") + " ×" + pairs.length);
  } catch (e) { toast(e.message, 3000); }
}

let _rnVal = "";
async function renameCheckedTo() {
  const sel = store.items.filter(i => store.sel.has(i.id));
  if (!sel.length) { toast(tf("先在「全部」里勾选要移动的图")); return; }
  const inp = $("rnName"), base = ((inp && inp.value) || "").trim();
  if (!base) { toast(tf("先输入新文件名")); return; }
  const before = snap();
  const pairs = sel.map((it, k) => ({ rel: it.rel, name: base + (sel.length > 1 ? "_" + (k + 1) : "") }));
  try {
    await renameFiles(pairs, { push: false });
    _rnVal = "";
    pushW({ t: "snap", before, after: snap() });
    toast(tf("已改名") + " ×" + pairs.length);
  } catch (e) { toast(e.message, 3000); }
}
function buildSortPanel(p) {
  p.innerHTML = "";
  p.appendChild(el("h3", null, "排序工具"));
  p.appendChild(el("div", "hint", "在「全部」界面拖动卡片排好顺序（勾选多张可整组拖动），双击卡片文件名能直接改名。确认后按顺序把图片改名成「文件名_01」这种格式，导出时顺序就会直接生效。"));
  const info = el("div", "hint strong",
    `${tf("当前")} ${store.items.length} ${tf("张")} · ${store.sorted ? tf("已确认排序") : tf("还未确认排序")}`);
  p.appendChild(info);

  const gp = el("div", "grp");
  gp.appendChild(el("div", "hint strong", "排序包"));
  gp.appendChild(el("div", "hint", "镜像目录里存的已经是排好的顺序，导出时直接按镜像文件打包，不再逐张重新合成，几秒就能出包。"));
  const pk = el("input", "inp");
  pk.id = "sortPkg";
  pk.placeholder = "排序包文件名（必填，例：小雅的夏天）";
  pk.value = store.pkg || "";
  pk.oninput = () => { store.pkg = pk.value; saveState(); };
  gp.appendChild(pk);
  const pe = el("button", "primary", "单独导出排序后的包");
  pe.disabled = !store.items.length;
  pe.onclick = () => exportSortedPackage(pe);
  gp.appendChild(pe);
  gp.appendChild(el("div", "hint", "包内文件按当前顺序命名为「文件名_1、_2 … _n」，序号由排列自动生成，不用你定位数。确认排序前必须填好上面的文件名。"));
  p.appendChild(gp);

  const g0 = el("div", "grp");
  const r0 = el("div", "row");
  for (const [k, v] of [["top", "置顶"], ["up", "上移"], ["down", "下移"], ["bottom", "置底"]]) {
    const b = el("button", null, v);
    b.onclick = () => moveSelected(k);
    r0.appendChild(b);
  }
  g0.appendChild(r0);
  const r0b = el("div", "row");
  const wu = el("button", null, "撤回"); wu.id = "wUndo"; wu.disabled = !wU.stack.length;
  wu.onclick = () => undoWork();
  const wr = el("button", null, "重做"); wr.id = "wRedo"; wr.disabled = !wU.redo.length;
  wr.onclick = () => redoWork();
  const dc = el("button", null, "删除勾选的文件");
  dc.className = "danger";
  dc.onclick = () => deleteChecked();
  r0b.append(wu, wr, dc);
  g0.appendChild(r0b);
  p.appendChild(g0);

  const g2 = el("div", "grp");
  const r2 = el("div", "row");
  const ok = el("button", "primary", "确认排序并改名");
  ok.disabled = !store.items.length;
  ok.onclick = () => confirmSort(ok);
  const rs = el("button", null, "恢复文件名顺序");
  rs.onclick = async () => {
    const before = snap();
    store.order = []; store.sorted = false; saveState();
    await syncItems();
    pushW({ t: "snap", before, after: snap() });
    renderGrid(); buildPanel();
  };
  r2.append(ok, rs);
  g2.appendChild(r2);
  p.appendChild(g2);

  const g3 = el("div", "grp");
  g3.appendChild(el("div", "hint strong", "批量命名"));
  const rp = el("div", "row3");
  const pre = el("input", "inp"); pre.id = "bnPre"; pre.placeholder = "前缀"; pre.value = "";
  const st = el("input", "inp"); st.id = "bnStart"; st.type = "number"; st.value = "1"; st.title = "起始号";
  const dg = el("input", "inp"); dg.id = "bnDigits"; dg.type = "number"; dg.value = "2"; dg.title = "位数";
  rp.append(pre, st, dg);
  g3.appendChild(rp);
  const rp2 = el("div", "row");
  const b1 = el("button", null, "应用到勾选"); b1.onclick = () => batchRename("sel");
  const b2 = el("button", null, "应用到全部"); b2.onclick = () => batchRename("all");
  rp2.append(b1, b2);
  g3.appendChild(rp2);
  g3.appendChild(el("div", "hint", "批量命名会按当前顺序把勾选（或全部）图片改名成「前缀 + 起始号」，位数决定补零个数。"));
  p.appendChild(g3);

  const g4 = el("div", "grp");
  g4.appendChild(el("div", "hint strong", "重命名勾选的文件"));
  const rn = el("input", "inp"); rn.id = "rnName"; rn.placeholder = "新文件名（不含扩展名）";
  rn.value = _rnVal; rn.oninput = () => { _rnVal = rn.value; };
  g4.appendChild(rn);
  const rb = el("button", null, "应用改名");
  rb.onclick = () => renameCheckedTo();
  g4.appendChild(rb);
  g4.appendChild(el("div", "hint", "勾选一张就是这个名字；勾选多张会自动加 _1、_2 后缀。改名会同步到磁盘，仓库/工作区里看到的都是新名字。"));
  p.appendChild(g4);

  const g1 = el("div", "grp");
  p.appendChild(g1);
  const order = el("div", "layers");
  store.items.forEach((it, i) => {
    order.appendChild(el("div", "layer", `${String(i + 1).padStart(2, "0")} · ${dirOf(it) || "根"} / ${it.name}`));
  });
  g1.appendChild(order);
  p.appendChild(el("div", "hint", "双击卡片上的文件名可以直接改名；确认排序会在磁盘上真的改名（可重复执行，会按当前顺序重新编号）。"));
}
async function confirmSort(btn) {
  if (!store.items.length) { toast("没有可排序的图片"); return; }
  const pkg = String(store.pkg || "").trim();
  if (!pkg) {
    toast("先填排序包文件名，再确认排序", 2800);
    const pin = $("sortPkg");
    if (pin) { pin.focus(); pin.classList.add("need"); setTimeout(() => pin.classList.remove("need"), 1600); }
    return;
  }
  const old = btn.textContent;
  btn.disabled = true; btn.textContent = "改名中…";
  try {
    const order = store.items.map(i => i.rel);
    const _before = snap();
    const j = await api("/api/workspace/sort-rename", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order }),
    });
    if (!j.ok) throw new Error(j.error || "改名失败");
    const byRel = new Map(store.items.map(i => [i.rel, i]));
    for (const r of j.renamed || []) {
      const it = byRel.get(r.from);
      if (!it) continue;
      const oldPath = it.path;
      store.pool.delete(oldPath);
      if (it.src === oldPath) it.src = r.path;
      if (it.baseSrc === oldPath) it.baseSrc = r.path;
      it.path = r.path; it.rel = r.rel; it.name = r.name;
      it.thumb = "/api/thumb?w=420&path=" + encodeURIComponent(r.path);
      it.thumbCache = null;
      store.pool.set(r.path, it);
    }
    store.order = (j.renamed || []).map(r => r.rel);
    store.sorted = true;
    saveState();
    await syncItems();
    renderGrid(); buildPanel(); updateSelInfo();
    pushW({ t: "snap", before: _before, after: snap() });
    toast(`已按新顺序改名 ${store.order.length} 张`);
  } catch (e) { toast("排序失败：" + e.message, 3000); }
  finally { btn.disabled = false; btn.textContent = old; }
}

/* ---------------- 导出 ---------------- */
async function exportSortedPackage(btn) {
  const pkg = String(store.pkg || "").trim();
  if (!pkg) {
    toast("先填排序包文件名", 2800);
    const pin = $("sortPkg");
    if (pin) { pin.focus(); pin.classList.add("need"); setTimeout(() => pin.classList.remove("need"), 1600); }
    return;
  }
  if (!store.items.length) { toast("工作区里还没有图片"); return; }
  const old = btn ? btn.textContent : "";
  if (btn) { btn.disabled = true; btn.textContent = "打包中…"; }
  try {
    const order = store.items.map(i => i.rel);
    const r = await fetch("/api/workspace/sort-export", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order, name: pkg }),
    });
    if (!r.ok) {
      let msg = "HTTP " + r.status;
      try { const j = await r.json(); if (j && j.error) msg = j.error; } catch (e) {}
      throw new Error(msg);
    }
    const blob = await r.blob();
    const nm = safeName(pkg) + ".zip";
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = nm; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    toast(`已导出 ${nm} · ${store.items.length} 张（直接用镜像里排好的文件，没有重新合成）`, 2600);
  } catch (e) { toast("导出失败：" + e.message, 3000); }
  finally { if (btn) { btn.disabled = false; btn.textContent = old || "单独导出排序后的包"; } }
}

function safeName(s) { return String(s).replace(/[\\/:*?"<>|]/g, "_"); }

async function composeCanvas(it, opts) {
  opts = opts || {};
  const w = Math.max(1, it.w || 800), h = Math.max(1, it.h || 600);
  // 底图单独一层，兼作马赛克/模糊的取样源
  const baseC = document.createElement("canvas"); baseC.width = w; baseC.height = h;
  await drawBaseOf(it, baseC.getContext("2d"), w, h);
  // 笔迹层
  const paintC = document.createElement("canvas"); paintC.width = w; paintC.height = h;
  const pc = paintC.getContext("2d");
  if (it.id === store.curId && cvPaint.width === w && cvPaint.height === h && !(it.sticks || []).length) {
    pc.drawImage(cvPaint, 0, 0);            // 兜底：只有画布没有笔迹记录时直接用画布
  } else {
    renderStrokes(pc, it.sticks, baseC, w, h);
  }
  const out = document.createElement("canvas"); out.width = w; out.height = h;
  const ox = out.getContext("2d");
  ox.drawImage(baseC, 0, 0);
  ox.drawImage(paintC, 0, 0);
  if (opts.wm !== false) {
    const layers = wmLayers(it);
    for (const cfg of layers) drawWatermark(ox, w, h, cfg, cfg.phase || 0);
    if (store.mode === "mark" && it.id === store.curId && wm.text.trim() && !wmEdit) {
      drawWatermark(ox, w, h, wm, wm.phase || 0);   // 预览用：还没固化的这一层
    }
  }
  if (opts.cover) drawCover(ox, w, h);
  return out;
}

function stripSeq(stem) { return String(stem).replace(/_\d{2,4}$/, ""); }
function zipFileName(suffix) {
  const w = store.work || {};
  let nm = w.name || w.folder || "";
  if (!nm) {
    nm = (store.root || "工作区").replace(/[\\/]+$/, "").split(/[\\/]/).pop() || "工作区";
  }
  nm = safeName(nm);
  return (suffix ? nm + safeName(suffix) : nm) + ".zip";
}

async function exportZip(list, label, btn, opts) {
  opts = opts || {};
  if (!list.length) { toast("没有可导出的图片"); return; }
  const old = btn ? btn.textContent : "";
  if (btn) btn.disabled = true;
  try {
    const files = [];
    for (let i = 0; i < list.length; i++) {
      if (btn) btn.textContent = `合成 ${i + 1}/${list.length}`;
      const c = await composeCanvas(list[i], opts);
      const blob = await new Promise(r => c.toBlob(r, "image/png"));
      const rel = list[i].rel;
      const dot = rel.lastIndexOf(".");
      let name;
      if (store.sorted) {
        // 排过序：所有文件夹的图压平到一个目录，靠序号保证顺序
        const base = rel.slice(rel.lastIndexOf("/") + 1);
        const bd = base.lastIndexOf(".");
        const stem = stripSeq(bd > 0 ? base.slice(0, bd) : base);
        name = String(i + 1).padStart(Math.max(2, String(list.length).length), "0") + "_" + stem + ".png";
      } else {
        name = (dot > 0 ? rel.slice(0, dot) : rel) + ".png";   // 未排序：保留批次子文件夹
      }
      const j = await api("/api/put?name=" + encodeURIComponent(name), { method: "POST", body: blob });
      files.push({ id: j.id, name });
    }
    if (btn) btn.textContent = "打包中…";
    const nm = zipFileName(opts.suffix);
    const r = await fetch("/api/export", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: nm, files }),
    });
    if (!r.ok) throw new Error("打包失败 HTTP " + r.status);
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = nm; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    toast(`已导出 ${files.length} 张 · ${nm}`);
  } catch (e) { toast("导出失败：" + e.message, 3000); }
  finally { if (btn) { btn.disabled = false; btn.textContent = old || label; } }
}

async function downloadOne() {
  if (!cur) { toast("先打开一张图"); return; }
  const btn = $("dlOneBtn");
  const old = btn ? btn.textContent : "";
  if (btn) { btn.disabled = true; btn.textContent = "生成中…"; }
  try {
    const c = await composeCanvas(cur);
    const blob = await new Promise(r => c.toBlob(r, "image/png"));
    const dot = cur.rel.lastIndexOf(".");
    const nm = safeName((dot > 0 ? cur.rel.slice(0, dot) : cur.rel).split("/").pop()) + ".png";
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = nm; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    toast("已下载 " + nm);
  } catch (e) { toast("下载失败：" + e.message, 3000); }
  finally { if (btn) { btn.disabled = false; btn.textContent = old || "下载这张"; } }
}
$("dlOneBtn").onclick = downloadOne;

function exportCensored(btn) {
  const list = store.items.filter(i => i.done);
  if (!list.length) { toast("还没有打码过的图片，可点「导出所有」"); return; }
  exportZip(list, "导出打码后文件", btn || $("exportBtn"));
}
function exportScope() {
  return store.items.filter(i => store.sel.has(i.id));
}
function exportEdited(btn, kind) {
  if (!store.items.length) { toast("工作区里还没有图片"); return; }
  const list = exportScope();
  if (!list.length) { toast("先勾选要导出的图片；整包导出用顶栏的「导出所有已做更改」", 2800); return; }
  if (kind === "cover") {
    exportZip(list, "导出已做更改（封面）", btn, { wm: true, cover: true, suffix: "_封面" });
  } else {
    exportZip(list, "导出已做更改（水印）", btn, { wm: true, cover: false, suffix: "_水印" });
  }
}
function exportAll(btn) {
  if (!store.items.length) { toast("工作区里还没有图片"); return; }
  exportZip(store.items.slice(), "导出所有已做更改", btn || $("exportAllBtn"));
}
$("exportBtn").onclick = () => exportCensored($("exportBtn"));
$("exportAllBtn").onclick = () => exportAll($("exportAllBtn"));

/* ---------------- 启动 ---------------- */
async function bootWorkspace() {
  try {
    const j = await api("/api/workspace");
    store.work = {
      name: j.name || "", folder: j.folder || "", count: j.count || 0,
      bytes: j.bytes || 0, path: j.path || "",
    };
    store.enabled = new Set(j.enabled || []);
    store.order = j.order || [];
    store.sorted = !!j.sorted;
    store.pkg = j.pkg || "";
  } catch (e) {}
}
let _started = false;
function startApp() {
  if (_started) return;
  _started = true;
  setMode("repo");
  setLayout("all");
  updateSelInfo();
  bootWorkspace().then(() => {
    if (store.mode === "work") buildPanel();
    const _saved = localStorage.getItem("cb-root") || "";
    if (_saved) {
      openDir(_saved).then(j => {
        if (j) { try { localStorage.setItem("cb-root", j.path); } catch (e) {} }
      });
    } else {
      renderBrowse(); updateSelInfo();
    }
  });
}

/* 首屏：先问账号，再谈语言 */
(async function bootAuth() {
  try {
    const j = await fetch("/api/auth/state", { credentials: "same-origin" }).then(r => r.json());
    AUTH.registered = !!j.registered;
    AUTH.user = j.user || null;
    if (j.authed) {
      hideGate();
      const done = !!localStorage.getItem("cb-lang-done");
      AUTH.registered = true;
      enterApp(!done && !(j.user && j.user.lang));
      return;
    }
  } catch (e) {}
  showGate();
})();
</script>
<footer style="position:fixed;right:14px;bottom:10px;z-index:60;display:flex;gap:10px;align-items:center;font-size:12px;color:#8b93a7;background:rgba(10,13,20,.72);border:1px solid rgba(255,255,255,.08);border-radius:999px;padding:5px 12px">
  <span>PolyForm SB · 授权与源码</span>
  <a href="https://github.com/CHANGE_ME/hchimao-studio" target="_blank" rel="noopener" style="color:#e8b25a;text-decoration:none">取源码 / 对应版本</a>
</footer>
</body>
</html>
"""
INDEX_HTML = (INDEX_HTML
              .replace("__ALL_CLASSES__", json.dumps(LABELS))
              .replace("__DEFAULT_CLASSES__", json.dumps(DEFAULT_CLASSES))
              .replace("__ZH__", json.dumps({
                  "FEMALE_GENITALIA_COVERED": "女性生殖器·遮挡",
                  "FACE_FEMALE": "女性面部",
                  "BUTTOCKS_EXPOSED": "臀部露出",
                  "FEMALE_BREAST_EXPOSED": "胸部露出",
                  "FEMALE_GENITALIA_EXPOSED": "女性生殖器露出",
                  "MALE_BREAST_EXPOSED": "男性胸部露出",
                  "ANUS_EXPOSED": "肛门露出",
                  "FEET_EXPOSED": "双脚露出",
                  "BELLY_COVERED": "腹部·遮挡",
                  "FEET_COVERED": "双脚·遮挡",
                  "ARMPITS_COVERED": "腋下·遮挡",
                  "ARMPITS_EXPOSED": "腋下露出",
                  "FACE_MALE": "男性面部",
                  "BELLY_EXPOSED": "腹部露出",
                  "MALE_GENITALIA_EXPOSED": "男性生殖器露出",
                  "ANUS_COVERED": "肛门·遮挡",
                  "FEMALE_BREAST_COVERED": "胸部·遮挡",
                  "BUTTOCKS_COVERED": "臀部·遮挡",
              })))
INDEX_HTML = INDEX_HTML.replace("__HASFFMPEG__", "true" if HAS_FFMPEG else "false")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype, extra_headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        ck = getattr(self, "_extra_cookie", None)
        if ck:
            self.send_header("Set-Cookie", ck)
            self._extra_cookie = None
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json")

    def _body_json(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) if n else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return {}

    def _token(self):
        c = self.headers.get("Cookie") or ""
        for part in c.split(";"):
            k, _, v = part.strip().partition("=")
            if k == "cb_token":
                return v
        return (self.headers.get("X-Auth-Token") or "").strip()

    def _authed_user(self):
        return _auth_user_by_token(self._token())

    def _set_token(self, tok):
        self._extra_cookie = ("cb_token=%s; Path=/; Max-Age=%d; SameSite=Lax"
                              % (tok, SESSION_TTL))

    def do_GET(self):
        path = urlparse(self.path).path
        q = parse_qs(urlparse(self.path).query)
        if path not in ("/", "/infer") and not path.startswith("/api/auth/") and not self._authed_user():
            self._json(401, {"error": "未登录", "need": "auth"})
            return
        if path == "/api/auth/state":
            users = _auth_users()["users"]
            u = self._authed_user()
            self._json(200, {"ok": True, "registered": bool(users),
                             "authed": bool(u),
                             "user": _auth_pub(u) if u else None})
            return
        if path in ("/", "/infer"):
            # 页面不缓存：改完代码刷新即生效，避免浏览器拿旧 HTML
            self._send(200, INDEX_HTML.encode("utf-8"),
                       "text/html; charset=utf-8",
                       extra_headers={"Cache-Control": "no-store, must-revalidate"})
            return
        if path == "/api/wsbrowse":
            with WS_LOCK:
                ws = _ws_read()
            root = os.path.abspath(ws["path"])
            target = os.path.abspath(q.get("path", [""])[0] or root)
            try:
                inside = os.path.commonpath([os.path.normcase(target), os.path.normcase(root)]) == os.path.normcase(root)
            except ValueError:
                inside = False
            if not inside:
                self._json(403, {"error": "只能浏览工作区内的目录"})
                return
            try:
                if not os.path.isdir(target):
                    os.makedirs(target, exist_ok=True)
                j = _list_dir(target)
            except Exception as e:
                self._json(404, {"error": str(e)})
                return
            rel = os.path.relpath(target, root).replace("\\", "/")
            j["root"] = root
            j["rel"] = "" if rel == "." else rel
            j["parent"] = "" if target == root else os.path.dirname(target)
            self._json(200, j)
            return

        if path == "/api/workspace":
            with WS_LOCK:
                ws = _ws_read()
                n, size = _ws_stats(ws)
                dirs = _ws_dirs(ws)
            self._json(200, {"id": ws["id"], "name": ws.get("name") or "",
                             "folder": os.path.basename(ws["path"]),
                             "path": ws["path"], "count": n, "bytes": size,
                             "dirs": dirs,
                             "enabled": ws.get("enabled") or [],
                             "order": ws.get("order") or [],
                             "pkg": ws.get("pkg") or "",
                             "sorted": bool(ws.get("sorted"))})
            return
        if path == "/api/wsfiles":
            raw = (q.get("dirs", [""])[0] or "").strip()
            names = [x for x in raw.split("|") if x.strip()] if raw else []
            with WS_LOCK:
                ws = _ws_read()
                items = _ws_items(ws, names)
            self._json(200, {"files": items, "count": len(items)})
            return
        if path.startswith("/asset/"):
            aid = path[len("/asset/"):]
            with MEDIA_LOCK:
                entry = ASSETS.get(aid)
            if entry is None:
                self._json(404, {"error": "素材不存在或已过期"})
                return
            ok, buf = cv2.imencode(".png", entry["img"])
            if not ok:
                self._json(500, {"error": "编码失败"})
                return
            self._send(200, buf.tobytes(), "image/png")
            return
        if path == "/video/status":
            jid = (q.get("id", [""])[0] or "")
            with MEDIA_LOCK:
                job = MEDIA.get(jid)
                snap = _media_snapshot(job) if job else None
            if snap is None:
                self._json(404, {"error": "任务不存在"})
            else:
                snap["id"] = jid
                self._json(200, snap)
            return
        if path == "/video/result":
            jid = (q.get("id", [""])[0] or "")
            with MEDIA_LOCK:
                job = MEDIA.get(jid)
                p = job.get("path") if job and job.get("st") == "done" else None
                ext = job.get("ext") if job else None
            if not p or not os.path.exists(p):
                self._json(404, {"error": "结果不存在或已过期"})
                return
            mime = MEDIA_MIME.get(ext, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(os.path.getsize(p)))
            self.send_header("Content-Disposition",
                             'attachment; filename="censored.%s"' % ext)
            self.end_headers()
            with open(p, "rb") as fh:
                self.wfile.write(fh.read())
            return
        if path == "/api/fs":
            try:
                self._json(200, _list_dir(q.get("path", [""])[0]))
            except FileNotFoundError as e:
                self._json(404, {"error": str(e)})
            except PermissionError as e:
                self._json(403, {"error": str(e)})
            except Exception as e:
                self._json(500, {"error": str(e)})
            return
        if path == "/api/thumb":
            try:
                w = max(80, min(1200, int(q.get("w", ["420"])[0])))
            except ValueError:
                w = 420
            try:
                data = _thumb(q.get("path", [""])[0], w)
            except Exception as e:
                self._json(404, {"error": str(e)})
                return
            self._send(200, data, "image/jpeg")
            return
        if path == "/api/file":
            fp = os.path.abspath(os.path.expanduser(q.get("path", [""])[0]))
            if not os.path.isfile(fp):
                self._json(404, {"error": "文件不存在"})
                return
            ext = os.path.splitext(fp)[1].lower()
            mime = {".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp",
                    ".gif": "image/gif", ".avif": "image/avif"}.get(ext, "image/jpeg")
            try:
                data = open(fp, "rb").read()
            except OSError as e:
                self._json(500, {"error": str(e)})
                return
            self._send(200, data, mime)
            return
        if path == "/lan":
            # 当前局域网访问状态
            self._json(200, {"lan": CURRENT_LAN, "port": CURRENT_PORT,
                             "ips": _lan_ips()})
            return
        self._send(404, b"not found", "text/plain")

    def _cfg_from_query(self, q):
        cfg = build_cfg(
            mode=q.get("mode", ["mosaic"])[0],
            strength=q.get("strength", ["35"])[0],
            margin=q.get("margin", ["15"])[0],
            color=q.get("color", ["#000000"])[0],
            classes=(q.get("classes", [None])[0] or "").split(",")
                    if "classes" in q else None,
            conf=q.get("conf", ["0.25"])[0],
            asset=q.get("asset", [None])[0],
            world_classes=q.get("world", [""])[0],
        )
        try:
            cfg["res"] = int(q.get("res", ["640"])[0])
        except ValueError:
            cfg["res"] = 640
        if cfg["res"] not in ALLOWED_RESOLUTIONS:
            cfg["res"] = 640
        # 输出格式（齿轮设置）：图片 jpg/png + 质量；视频 webm/mp4
        cfg["fmt"] = q.get("fmt", ["jpg"])[0]
        if cfg["fmt"] not in ("jpg", "png"):
            cfg["fmt"] = "jpg"
        try:
            cfg["quality"] = max(40, min(100, int(q.get("quality", ["92"])[0])))
        except ValueError:
            cfg["quality"] = 92
        cfg["vfmt"] = q.get("vfmt", ["webm"])[0]
        if cfg["vfmt"] not in ("webm", "mp4"):
            cfg["vfmt"] = "webm"
        return cfg

    def do_POST(self):
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)

        if not parsed.path.startswith("/api/auth/") and not self._authed_user():
            self._json(401, {"error": "未登录", "need": "auth"})
            return

        if parsed.path == "/api/auth/register":
            b = self._body_json()
            name = (b.get("name") or "").strip()
            pwd = (b.get("pwd") or "").strip()
            code = (b.get("code") or "").strip()
            if len(name) < 2 or len(name) > 24:
                self._json(400, {"ok": False, "error": "用户名 2-24 个字符"})
                return
            if len(pwd) < 4:
                self._json(400, {"ok": False, "error": "密码至少 4 位"})
                return
            if code != REG_CODE:
                self._json(400, {"ok": False, "error": "注册码不正确"})
                return
            with AUTH_LOCK:
                j = _auth_users()
                if any((u.get("name") or "").lower() == name.lower() for u in j["users"]):
                    self._json(400, {"ok": False, "error": "用户名已被占用"})
                    return
                u = {"id": uuid.uuid4().hex[:12], "name": name,
                     "pwd": _pwd_hash(pwd), "lang": (b.get("lang") or ""),
                     "created": time.time()}
                j["users"].append(u)
                _auth_save(USERS_PATH, j)
            tok = _sess_new(u["id"])
            self._set_token(tok)
            self._json(200, {"ok": True, "token": tok, "user": _auth_pub(u),
                             "first": len(j["users"]) == 1})
            return

        if parsed.path == "/api/auth/login":
            b = self._body_json()
            u = _auth_find(b.get("name") or "")
            if not u or not _pwd_ok((b.get("pwd") or "").strip(), u.get("pwd")):
                self._json(400, {"ok": False, "error": "用户名或密码不正确"})
                return
            tok = _sess_new(u["id"])
            self._set_token(tok)
            self._json(200, {"ok": True, "token": tok, "user": _auth_pub(u)})
            return

        if parsed.path == "/api/auth/logout":
            _sess_drop(self._token())
            self._extra_cookie = "cb_token=; Path=/; Max-Age=0; SameSite=Lax"
            self._json(200, {"ok": True})
            return

        if parsed.path == "/api/auth/lang":
            b = self._body_json()
            u = self._authed_user()
            if u:
                with AUTH_LOCK:
                    j = _auth_users()
                    for x in j["users"]:
                        if x.get("id") == u.get("id"):
                            x["lang"] = (b.get("lang") or "")[:12]
                    _auth_save(USERS_PATH, j)
            self._json(200, {"ok": True})
            return

        if parsed.path == "/api/auth/pwd":
            b = self._body_json()
            u = self._authed_user()
            if not u:
                self._json(401, {"ok": False, "error": "未登录"})
                return
            if not _pwd_ok((b.get("old") or "").strip(), u.get("pwd")):
                self._json(400, {"ok": False, "error": "原密码不正确"})
                return
            new_pwd = (b.get("pwd") or "").strip()
            if len(new_pwd) < 4:
                self._json(400, {"ok": False, "error": "新密码至少 4 位"})
                return
            with AUTH_LOCK:
                j = _auth_users()
                for x in j["users"]:
                    if x.get("id") == u.get("id"):
                        x["pwd"] = _pwd_hash(new_pwd)
                _auth_save(USERS_PATH, j)
            self._json(200, {"ok": True})
            return

        if parsed.path == "/api/workspace/rename-file":
            b = self._body_json()
            with WS_LOCK:
                ws = _ws_read()
            res = _ws_rename_files(ws, b.get("items") or [])
            with WS_LOCK:
                ws2 = _ws_read()
                order = list(ws2.get("order") or [])
                mp = {r["from"]: r["rel"] for r in res}
                ws2["order"] = [mp.get(x, x) for x in order]
                _ws_write(ws2)
            self._json(200, {"ok": True, "renamed": res})
            return

        if parsed.path == "/api/workspace/delete-files":
            b = self._body_json()
            with WS_LOCK:
                ws = _ws_read()
            token, moved = _ws_delete_files(ws, b.get("rels") or [])
            if not token:
                self._json(400, {"ok": False, "error": "没有可删除的文件"})
                return
            with WS_LOCK:
                ws2 = _ws_read()
                gone = {m["rel"] for m in moved}
                ws2["order"] = [x for x in (ws2.get("order") or []) if x not in gone]
                if not ws2["order"]:
                    ws2["sorted"] = False
                _ws_write(ws2)
            self._json(200, {"ok": True, "token": token, "items": moved})
            return

        if parsed.path == "/api/workspace/restore":
            b = self._body_json()
            back = _ws_restore(b.get("token") or "")
            self._json(200, {"ok": True, "restored": back})
            return

        if parsed.path == "/lan":
            # 切换局域网访问：on=允许, off=仅本机；os.execv 重启服务生效
            mode = q.get("mode", [""])[0]
            target = "lan" if mode == "on" else "local"
            cur = "lan" if CURRENT_LAN else "local"
            if target == cur:
                self._json(200, {"ok": True, "restarting": False,
                                 "lan": CURRENT_LAN})
                return
            # 先把"即将重启"响应发给前端，再替换进程
            body = json.dumps({"ok": True, "restarting": True,
                               "lan": target == "lan",
                               "port": CURRENT_PORT},
                              ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json")
            self.wfile.flush()
            try:
                _restart_with_mode("lan" if target == "lan" else "local")
            except Exception as e:
                # execv 失败：尽力报错
                try:
                    self._json(500, {"error": "重启失败: %s" % e})
                except Exception:
                    pass
            return

        if parsed.path == "/api/workspace/state":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:
                self._json(400, {"error": "请求格式错误"})
                return
            with WS_LOCK:
                ws = _ws_read()
                if "enabled" in body and isinstance(body["enabled"], list):
                    ws["enabled"] = [str(x) for x in body["enabled"]]
                if "order" in body and isinstance(body["order"], list):
                    ws["order"] = [str(x) for x in body["order"]]
                if "sorted" in body:
                    ws["sorted"] = bool(body["sorted"])
                if "pkg" in body:
                    ws["pkg"] = _ws_name_ok(body.get("pkg")) or ""
                _ws_write(ws)
                keep = {"enabled": ws["enabled"], "order": ws["order"],
                        "sorted": ws["sorted"], "pkg": ws.get("pkg") or ""}
            self._json(200, dict(ok=True, **keep))
            return

        if parsed.path == "/api/workspace/put":
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > 96 * 1024 * 1024:
                self._json(400, {"error": "内容为空或超过 96MB"})
                return
            data = self.rfile.read(length)
            dname = (q.get("dir", [""])[0] or "").strip("/")
            name = (q.get("name", [""])[0] or "").strip()
            with WS_LOCK:
                ws = _ws_read()
                root = os.path.abspath(ws["path"])
                target_dir = root
                if dname:
                    cand = _ws_safe_rel(ws, dname)
                    if not cand:
                        self._json(403, {"error": "只能写入工作区内"})
                        return
                    target_dir = cand
                os.makedirs(target_dir, exist_ok=True)
                safe = _ws_name_ok(name) or ("封面-" + uuid.uuid4().hex[:8] + ".png")
                if os.path.splitext(safe)[1].lower() not in IMG_EXTS:
                    safe += ".png"
                dst = os.path.join(target_dir, safe)
                k = 2
                while os.path.exists(dst):
                    st, ex = os.path.splitext(safe)
                    dst = os.path.join(target_dir, "%s(%d)%s" % (st, k, ex))
                    k += 1
                try:
                    with open(dst, "wb") as fh:
                        fh.write(data)
                except OSError as e:
                    self._json(500, {"error": str(e)})
                    return
                rel = os.path.relpath(dst, root).replace(os.sep, "/")
                w, h = _img_size(dst)
                item = {"name": os.path.basename(dst), "rel": rel, "path": dst,
                        "w": w, "h": h,
                        "thumb": "/api/thumb?w=420&path=" + quote(dst)}
                batch = rel.split("/")[0] if "/" in rel else ""
                if batch and batch not in (ws.get("enabled") or []):
                    ws["enabled"] = list(ws.get("enabled") or []) + [batch]
                _ws_write(ws)
            self._json(200, {"ok": True, "item": item, "rel": rel})
            return

        if parsed.path == "/api/workspace/sort-export":
            import zipfile
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:
                self._json(400, {"error": "请求格式错误"})
                return
            order = body.get("order") or []
            name = _ws_name_ok(body.get("name") or "")
            if not name:
                self._json(400, {"error": "先填排序包文件名"})
                return
            if not isinstance(order, list) or not order:
                self._json(400, {"error": "排序列表为空"})
                return
            with WS_LOCK:
                ws = _ws_read()
                picked = []
                for rel in order:
                    full = _ws_safe_rel(ws, rel)
                    if full and os.path.isfile(full):
                        picked.append(full)
                    if len(picked) >= 20000:
                        break
            if not picked:
                self._json(400, {"error": "镜像里没有可打包的图片"})
                return
            tmp = os.path.join(OUT_DIR, uuid.uuid4().hex + ".zip")
            try:
                with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
                    for i, full in enumerate(picked):
                        ext = os.path.splitext(full)[1] or ".png"
                        zf.write(full, "%s_%d%s" % (name, i + 1, ext))
                with open(tmp, "rb") as fh:
                    blob = fh.read()
            except Exception as e:
                self._json(500, {"error": "打包失败: %s" % e})
                return
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Content-Disposition",
                             "attachment; filename=\"result.zip\"; filename*=UTF-8''%s"
                             % quote(name + ".zip"))
            self.end_headers()
            self.wfile.write(blob)
            return

        if parsed.path == "/api/workspace/sort-rename":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:
                self._json(400, {"error": "请求格式错误"})
                return
            order = body.get("order") or []
            if not isinstance(order, list) or not order:
                self._json(400, {"error": "排序列表为空"})
                return
            with WS_LOCK:
                ws = _ws_read()
                try:
                    renamed = _ws_sort_rename(ws, order)
                except Exception as e:
                    self._json(500, {"error": "改名失败: %s" % e})
                    return
                ws["order"] = [r["rel"] for r in renamed]
                ws["sorted"] = bool(renamed)
                _ws_write(ws)
            self._json(200, {"ok": True, "renamed": renamed})
            return

        if parsed.path == "/api/workspace/rename":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:
                self._json(400, {"error": "请求格式错误"})
                return
            name = _ws_name_ok(body.get("name"))
            if not name:
                self._json(400, {"error": "名称不能为空，也不能包含 \\ / : * ? \" < > |"})
                return
            with WS_LOCK:
                ws = _ws_read()
                ws["name"] = name
                _ws_write(ws)
            self._json(200, {"ok": True, "name": name})
            return

        if parsed.path == "/api/workspace/delete":
            with WS_LOCK:
                ws = _ws_read()
                old = ws["path"]
                name = ws.get("name") or ""
                try:
                    for fn in os.listdir(old):
                        fp = os.path.join(old, fn)
                        if os.path.isdir(fp):
                            shutil.rmtree(fp, ignore_errors=True)
                        else:
                            os.remove(fp)
                except Exception as e:
                    self._json(500, {"error": "删除失败: %s" % e})
                    return
                _ws_write({"id": ws["id"], "name": name, "created": time.time(),
                           "enabled": [], "order": [], "sorted": False})
            self._json(200, {"ok": True})
            return

        if parsed.path == "/api/mirror":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:
                self._json(400, {"error": "请求格式错误"})
                return
            paths = body.get("paths") or []
            if not isinstance(paths, list) or not paths:
                self._json(400, {"error": "没有选中文件"})
                return
            if len(paths) > 2000:
                self._json(400, {"error": "一次最多导入 2000 张"})
                return
            try:
                res = _mirror_files(paths, body.get("root") or "")
            except Exception as e:
                self._json(500, {"error": "镜像失败: %s" % e})
                return
            res["count"] = len(res["items"])
            self._json(200, res)
            return

        if parsed.path == "/api/put":
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > 64 * 1024 * 1024:
                self._json(400, {"error": "内容为空或超过 64MB"})
                return
            data = self.rfile.read(length)
            fid = uuid.uuid4().hex
            fp = os.path.join(OUT_DIR, fid + ".png")
            try:
                with open(fp, "wb") as fh:
                    fh.write(data)
            except OSError as e:
                self._json(500, {"error": str(e)})
                return
            name = q.get("name", [fid + ".png"])[0]
            with OUT_LOCK:
                OUT_FILES[fid] = {"path": fp, "name": name, "ts": time.time()}
                if len(OUT_FILES) > 400:
                    for k in sorted(OUT_FILES, key=lambda k: OUT_FILES[k]["ts"])[:100]:
                        try:
                            os.remove(OUT_FILES[k]["path"])
                        except OSError:
                            pass
                        OUT_FILES.pop(k, None)
            self._json(200, {"id": fid, "name": name, "size": len(data)})
            return

        if parsed.path == "/api/export":
            import zipfile
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:
                self._json(400, {"error": "请求格式错误"})
                return
            files = body.get("files") or []
            name = (body.get("name") or "打码结果.zip").strip() or "打码结果.zip"
            if not name.lower().endswith(".zip"):
                name += ".zip"
            if not files:
                self._json(400, {"error": "没有可导出的文件"})
                return
            tmp = os.path.join(OUT_DIR, uuid.uuid4().hex + ".zip")
            used = set()
            try:
                with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
                    for item in files:
                        fid = item.get("id")
                        with OUT_LOCK:
                            meta = OUT_FILES.get(fid)
                        if not meta or not os.path.exists(meta["path"]):
                            continue
                        arc = (item.get("name") or meta["name"] or (fid + ".png")).replace("\\", "/")
                        arc = arc.lstrip("/") or (fid + ".png")
                        if not os.path.splitext(arc)[1]:
                            arc += ".png"
                        base, ext = os.path.splitext(arc)
                        n = 1
                        while arc in used:
                            arc = "%s(%d)%s" % (base, n, ext)
                            n += 1
                        used.add(arc)
                        zf.write(meta["path"], arc)
                blob = open(tmp, "rb").read()
            except Exception as e:
                self._json(500, {"error": "打包失败: %s" % e})
                return
            finally:
                for item in files:
                    with OUT_LOCK:
                        meta = OUT_FILES.pop(item.get("id"), None)
                    if meta:
                        try:
                            os.remove(meta["path"])
                        except OSError:
                            pass
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Content-Disposition",
                             "attachment; filename=\"result.zip\"; filename*=UTF-8''%s"
                             % quote(name))
            self.end_headers()
            self.wfile.write(blob)
            return

        if parsed.path == "/asset":
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > ASSET_LIMIT:
                self._json(400, {"error": "遮挡图片为空或超过 8MB 上限"})
                return
            data = self.rfile.read(length)
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                self._json(400, {"error": "无法解析遮挡图片内容"})
                return
            h, w = img.shape[:2]
            if max(h, w) > 1024:  # 控制内存，等比缩小超大图
                scale = 1024 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            aid = uuid.uuid4().hex
            with MEDIA_LOCK:
                ASSETS[aid] = {"img": img, "ts": time.time()}
                if len(ASSETS) > ASSET_KEEP:
                    for k in sorted(ASSETS, key=lambda k: ASSETS[k]["ts"])[:-ASSET_KEEP]:
                        ASSETS.pop(k, None)
            self._json(200, {"id": aid, "w": img.shape[1], "h": img.shape[0]})
            return

        cfg = self._cfg_from_query(q)

        if parsed.path == "/infer":
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                self._json(400, {"error": "无法解析图片内容"})
                return
            stamp, err = _resolve_stamp(cfg)
            if err:
                self._json(400, {"error": err})
                return
            t0 = time.time()
            try:
                detections, enabled_world = _detect_combined(img, cfg)
            except RuntimeError as e:
                self._json(400, {"error": str(e)})
                return
            if enabled_world:
                cfg["classes"] = list(cfg["classes"]) + enabled_world
            censored_count = censor_regions(img, detections, cfg, stamp=stamp)
            if cfg["fmt"] == "png":
                ok, buf = cv2.imencode(".png", img)
                mime = "image/png"
            else:
                ok, buf = cv2.imencode(".jpg", img,
                                       [cv2.IMWRITE_JPEG_QUALITY, cfg["quality"]])
                mime = "image/jpeg"
            if not ok:
                self._json(500, {"error": "结果编码失败"})
                return
            elapsed = int((time.time() - t0) * 1000)
            self._json(200, {
                "detections": detections,
                "censored_count": censored_count,
                "elapsed_ms": elapsed,
                "fmt": cfg["fmt"],
                "censored": "data:" + mime + ";base64," + base64.b64encode(buf).decode(),
            })
            return

        if parsed.path == "/video":
            kind = q.get("kind", ["video"])[0]
            if kind not in ("gif", "video"):
                self._json(400, {"error": "kind 仅支持 gif/video"})
                return
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MEDIA_LIMIT:
                self._json(400, {"error": "文件为空或超过 500MB 上限"})
                return
            stamp, err = _resolve_stamp(cfg)
            if err:
                self._json(400, {"error": err})
                return
            data = self.rfile.read(length)
            jid = uuid.uuid4().hex
            with MEDIA_LOCK:
                MEDIA[jid] = {"id": jid, "ts": time.time(), "st": "processing",
                              "pct": 0, "kind": kind}
            threading.Thread(target=_media_worker, args=(jid, data, kind, cfg, stamp),
                             daemon=True).start()
            self._json(200, {"id": jid, "kind": kind})
            return

        self._send(404, b"not found", "text/plain")

    def log_message(self, fmt, *args):
        pass  # 静默访问日志


if __name__ == "__main__":
    import argparse as _argparse
    ap = _argparse.ArgumentParser(description="%s V%s" % (APP_NAME, APP_VERSION))
    ap.add_argument("port", nargs="?", type=int, default=8080,
                    help="端口（默认 8080）")
    ap.add_argument("--lan", action="store_true",
                    help="允许局域网访问（绑定 0.0.0.0）")
    args = ap.parse_args()
    port = args.port
    bind_host = LAN_HOST if args.lan else LOOPBACK_HOST
    CURRENT_PORT = port       # 模块级变量，供 /lan 端点使用
    CURRENT_LAN = args.lan    # 是否允许局域网访问

    server = ThreadingHTTPServer((bind_host, port), Handler)
    mode = "局域网" if args.lan else "仅本机"
    print(f"NudeNet 自动打码网页版已启动 ({mode}) : http://localhost:{port}")
    if args.lan:
        for ip in _lan_ips():
            print(f"  局域网访问: http://{ip}:{port}")
    print("  Ctrl+C 退出")
    server.serve_forever()
