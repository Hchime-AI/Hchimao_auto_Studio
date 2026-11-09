"""GIF / 视频逐帧打码。

对外入口（webui 的 /video 任务线程直接调用）：
    _find_ffmpeg()      探测 ffmpeg 可执行文件，找不到返回 None
    process_gif(...)    GIF：PIL 拆帧 -> 打码 -> 重新合成
    process_video(...)  视频：OpenCV 拆帧 -> 打码 -> ffmpeg/OpenCV 重编码

两个 process_* 的签名保持一致：
    (data, cfg, detector, stamp=None, detect_every=None, progress=None,
     world=None, world_embeds=None) -> (字节, 扩展名, 统计字典)

帧间复用策略：不是每帧都跑检测，而是按 _stride() 算出的间隔抽样检测，
中间的帧沿用最近一次检测框。这样 30fps 的片子真正推理的只有每秒若干帧，
画面上人物移动没那么快，观感上几乎看不出来。

文件路径全部由本模块用 tempfile 生成，客户端传来的文件名与扩展名
不参与任何路径拼接；输入统一写成固定名 in.bin，交给 OpenCV / ffmpeg
按内容嗅探容器格式。
"""
import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from censor_core import censor_regions

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:  # GIF 路径会明确报错，不影响视频
    HAVE_PIL = False


# ---------------------------------------------------------------- ffmpeg 探测

def _find_ffmpeg():
    """返回可用的 ffmpeg 路径。

    查找顺序：环境变量 CENSOR_FFMPEG（方便指定自带版本）->
    imageio-ffmpeg 的静态二进制（pip 装完即用，无需系统级安装）->
    系统 PATH。都找不到返回 None，上层退化为 OpenCV 编码。
    """
    forced = os.environ.get("CENSOR_FFMPEG")
    if forced and os.path.isfile(forced):
        return forced
    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.isfile(bundled):
            return bundled
    except Exception:
        pass
    return shutil.which("ffmpeg")


# ---------------------------------------------------------------- 抽样间隔

def _stride(kind, count):
    """检测抽样间隔：视频按约 6 次/秒折算，GIF 全程抽样约 10 次。"""
    count = max(1, int(count or 1))
    if kind == "gif":
        return max(1, count // 10)
    return max(1, min(15, round(count / 6.0)))


# 兼容早期调用名
_default_detect_every = _stride


# ---------------------------------------------------------------- 帧处理器

class _FrameCensor:
    """把"抽样检测 + 就地打码"包成可调用对象，GIF 与视频共用。"""

    def __init__(self, cfg, detector, stamp=None, step=1,
                 world=None, world_embeds=None):
        self.cfg = cfg
        self.detector = detector
        self.stamp = stamp
        self.step = max(1, int(step))
        self.world = world
        self.world_embeds = world_embeds
        self._boxes = []

    def _detect(self, frame):
        boxes = self.detector.detect(frame, conf=self.cfg["conf"])
        if self.world is not None and self.cfg.get("world_classes"):
            extra = self.world.detect(
                frame, self.cfg["world_classes"],
                embeds=self.world_embeds,
                conf=self.cfg.get("world_conf", 0.15),
            )
            boxes = list(boxes) + list(extra)
        self._boxes = boxes
        return boxes

    def __call__(self, frame, index):
        if index % self.step == 0:
            self._detect(frame)
        return censor_regions(frame, self._boxes, self.cfg, stamp=self.stamp)


# ---------------------------------------------------------------- GIF

def process_gif(data, cfg, detector, stamp=None, detect_every=None, progress=None,
                world=None, world_embeds=None):
    """逐帧处理 GIF。

    读：seek 遍历每一帧，保留原有每帧延时与 loop 次数；
    写：统一 disposal=2（还原背景），避免半透明帧叠影。
    """
    if not HAVE_PIL:
        raise RuntimeError("处理 GIF 需要 Pillow：pip install pillow")

    src = Image.open(io.BytesIO(data))
    total = int(getattr(src, "n_frames", 1) or 1)
    step = detect_every or _stride("gif", total)
    censor = _FrameCensor(cfg, detector, stamp, step, world, world_embeds)

    frames, delays = [], []
    for i in range(total):
        src.seek(i)
        delays.append(int(src.info.get("duration", 100) or 100))
        frame = cv2.cvtColor(np.array(src.convert("RGB")), cv2.COLOR_RGB2BGR)
        censor(frame, i)
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if progress:
            progress(i + 1, total)

    out = io.BytesIO()
    frames[0].save(
        out, format="GIF", save_all=True,
        append_images=frames[1:],
        duration=delays,
        loop=int(src.info.get("loop", 0) or 0),
        disposal=2,
    )
    return out.getvalue(), "gif", {"frames": total, "detect_every": step}


# ---------------------------------------------------------------- 视频

def _encoder_cmd(ffmpeg, src, dst, width, height, fps):
    """H.264 + AAC 的命令行。

    视频流从 stdin 吃原始 bgr24 帧（已解码、已打码的画面），
    音轨直接从原始文件读出来重编码，两条流用 -shortest 对齐。
    """
    return [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps:.4f}", "-i", "pipe:0",
        "-i", str(src),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest", str(dst),
    ]


def _open_cv_writer(folder, fps, width, height, prefer):
    """没有 ffmpeg 时的兜底编码器。

    prefer="webm" 先试 VP8（浏览器能直接预览，但没有音轨），打不开退 mp4v；
    prefer="mp4" 直接用 mp4v（体积小，多数浏览器不预览）。
    返回 (writer, 输出路径, 扩展名)。
    """
    order = [("webm", "VP80", "webm")] if prefer != "mp4" else []
    order += [("mp4", "mp4v", "mp4")]

    for _, fourcc, ext in order:
        path = folder / f"out.{ext}"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc),
                                 fps, (width, height))
        if writer.isOpened():
            return writer, path, ext
        writer.release()
    raise RuntimeError("无法初始化视频编码器（OpenCV 缺少可用编码后端）")


def process_video(data, cfg, detector, stamp=None, detect_every=None, progress=None,
                  prefer="webm", world=None, world_embeds=None):
    """逐帧处理视频，返回 (bytes, 扩展名, info)。

    有 ffmpeg 时一律输出 H.264 mp4（带音轨，兼容性最好）；
    没有 ffmpeg 时按 prefer 决定 webm 还是 mp4，且不含音轨。
    """
    ffmpeg = _find_ffmpeg()
    work = Path(tempfile.mkdtemp(prefix="censor_vid_"))
    cap = None
    proc = None
    writer = None

    try:
        src = work / "in.bin"
        src.write_bytes(data)

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError("无法解码该视频（容器或编码格式不受支持）")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps < 1e-3:                      # 少数容器读不到帧率，按 25 兜底
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("视频尺寸读取失败")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = detect_every or _stride("video", fps)

        if ffmpeg:
            dst = work / "out.mp4"
            ext = "mp4"
            proc = subprocess.Popen(
                _encoder_cmd(ffmpeg, src, dst, width, height, fps),
                stdin=subprocess.PIPE,
            )
        else:
            writer, dst, ext = _open_cv_writer(work, fps, width, height, prefer)

        censor = _FrameCensor(cfg, detector, stamp, step, world, world_embeds)
        done = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            censor(frame, done)
            if proc is not None:
                proc.stdin.write(frame.tobytes())
            else:
                writer.write(frame)
            done += 1
            if progress:
                progress(done, total or done)

        if proc is not None:
            proc.stdin.close()
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError("ffmpeg 编码失败（参数不受支持或磁盘不可写）")
        else:
            writer.release()

        return dst.read_bytes(), ext, {
            "frames": done,
            "fps": round(fps, 2),
            "detect_every": step,
            "audio": proc is not None,
        }
    finally:
        if cap is not None:
            cap.release()
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        shutil.rmtree(work, ignore_errors=True)
