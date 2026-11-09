"""运行环境自检 + 缺件自动安装：python check_deps.py

检查四类东西，缺什么补什么：
  1. Python 版本（要求 3.9+，推荐 3.10 以上）
  2. pip 依赖包（对照下面的 PACKAGES 清单，缺的当场装）
  3. 打码模型权重（缺 640m 会下载；下载不了就用自带的 320n 顶着）
  4. YOLO-World 三件套（可选，只在用自定义类别时需要，缺了不阻塞）

安装走 python -m pip 的子进程调用，先默认源、失败自动换清华镜像。
要装的包名只来自本文件写死的清单，不接受任何外部输入，也不拼 shell 命令。
"""
import importlib
import os
import subprocess
import sys

MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

# (import 名, pip 包名)
PACKAGES = [
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("onnxruntime", "onnxruntime"),
    ("PIL", "Pillow"),              # GIF 逐帧需要
    ("imageio_ffmpeg", "imageio-ffmpeg"),   # 自带 ffmpeg，视频带音轨
    ("tokenizers", "tokenizers"),   # YOLO-World 文本编码
]

MIN_PYTHON = (3, 9)


def _console_utf8():
    """GBK 控制台里 emoji / 勾号不至于把脚本搞崩。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _have(module):
    try:
        importlib.import_module(module)
        return True, ""
    except Exception as exc:
        return False, exc.__class__.__name__


def install(pkg):
    """装一个包，默认源不行就换镜像。返回是否成功。"""
    for extra, tag in (([], "默认源"), (["-i", MIRROR], "清华镜像")):
        print(f"  安装 {pkg}（{tag}）…")
        cmd = [sys.executable, "-m", "pip", "install", "--quiet", pkg] + extra
        try:
            done = subprocess.run(cmd, check=False)
        except Exception as exc:
            print(f"    调用 pip 失败：{exc}")
            continue
        if done.returncode == 0:
            return True
        print(f"    pip 返回 {done.returncode}")
    return False


def check_python():
    ver = sys.version_info
    print(f"Python {sys.version.split()[0]}  ({sys.executable})")
    if ver[:2] < MIN_PYTHON:
        print(f"  [错误] 需要 Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} 或更高")
        return False
    return True


def check_packages():
    """返回仍然缺失的包名列表（空列表表示齐全）。"""
    print("\n== pip 依赖 ==")
    missing = []
    for mod, pkg in PACKAGES:
        ok, err = _have(mod)
        print(f"  [{'OK' if ok else '缺'}] {pkg}" + (f"  ({err})" if err else ""))
        if not ok:
            missing.append(pkg)
    if not missing:
        return []

    print("\n缺 " + "、".join(missing) + "，开始自动安装…")
    for pkg in missing:
        if not install(pkg):
            print(f"[错误] {pkg} 装不上，请手动执行：")
            print(f"  {sys.executable} -m pip install {pkg}")
            print(f"  {sys.executable} -m pip install {pkg} -i {MIRROR}")
            print("  提示权限不足时，命令末尾加 --user")

    print("\n== 复检 ==")
    still = []
    for mod, pkg in PACKAGES:
        ok, _ = _have(mod)
        print(f"  [{'OK' if ok else '仍缺'}] {pkg}")
        if not ok:
            still.append(pkg)
    return still


def check_model():
    print("\n== 打码模型 ==")
    try:
        from censor_core import ensure_model
        print(f"  [OK] {ensure_model()}")
    except Exception as exc:
        print(f"  [警告] 模型没准备好：{exc}")
        print("         启动服务后会自动重试下载")


def check_world():
    print("\n== YOLO-World 自定义类别（可选） ==")
    try:
        from yolo_world import DETECTOR_PATH, TEXT_MODEL_PATH, TOKENIZER_PATH
    except Exception as exc:
        print(f"  [跳过] {exc}")
        return
    files = [DETECTOR_PATH, TOKENIZER_PATH, TEXT_MODEL_PATH]
    if all(os.path.exists(p) for p in files):
        print("  [OK] 已就绪，自定义类别可直接用")
    else:
        print("  [提示] 未下载（约 300MB），首次用自定义类别时自动下载")


def main():
    _console_utf8()
    if not check_python():
        return 1
    missing = check_packages()
    check_model()
    check_world()
    if missing:
        print("\n[结果] 依赖仍不完整：" + "、".join(missing))
        return 1
    print("\n[结果] 环境检查通过，可以启动打码工作台")
    return 0


if __name__ == "__main__":
    sys.exit(main())
