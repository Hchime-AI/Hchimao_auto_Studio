"""命令行打码工具：批量处理图片，输出 *_censored.jpg / *.png。

用法：
    python auto_censor.py 图片1.jpg 图片2.png [选项]
    python auto_censor.py --list-classes        # 看所有可打码的类别

常用选项：
    --mode mosaic|blur|solid|img   打码方式（默认 mosaic）
    --strength 1-100               马赛克块大小 / 模糊核（默认 35）
    --margin 0-60                  检测框外扩百分比（默认 15）
    --color RRGGBB                 纯色填充色（默认 000000）
    --conf 0.04-0.9                置信度阈值，越低越敏感（默认 0.25）
    --res 320|640|960|1280         推理分辨率（默认 640）
    --classes A,B,...              打码类别；all=全部；缺省=隐私五类
    --world "gun,face"             YOLO-World 自定义类别（英文）
    --stamp 图片路径               mode=img 时的遮挡图
    --fmt jpg|png                  输出格式（默认 jpg）
    --quality 40-100               jpg 质量（默认 92）

命令行只负责参数解析与逐图循环，检测与打码都委托给 censor_core；
Web UI 与本工具共用同一份 cfg 与同一套模型。
"""
import argparse
import os
import sys

import cv2

from censor_core import (ALLOWED_RESOLUTIONS, DEFAULT_CLASSES, LABELS,
                         Detector, build_cfg, censor_regions)

MODE_LABELS = {"mosaic": "马赛克", "blur": "高斯模糊",
               "solid": "纯色", "img": "图片遮挡"}


def make_parser():
    p = argparse.ArgumentParser(description="自动打码工具（NudeNet 640m 模型）")
    p.add_argument("images", nargs="*", help="要处理的图片路径，可给多个")
    p.add_argument("--mode", default="mosaic",
                   help="打码方式：mosaic 马赛克 / blur 高斯模糊 / solid 纯色 / img 图片")
    p.add_argument("--strength", type=int, default=35,
                   help="强度 1-100：马赛克块大小 / 模糊核半径")
    p.add_argument("--margin", type=int, default=15, help="检测框外扩百分比 0-60")
    p.add_argument("--color", default="000000", help="纯色填充颜色，RRGGBB")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值 0.04-0.9")
    p.add_argument("--res", type=int, default=640, help="推理分辨率")
    p.add_argument("--classes", default=None,
                   help='打码类别，逗号分隔；all 表示全部；缺省为隐私部位')
    p.add_argument("--world", default=None,
                   help='YOLO-World 自定义类别（英文逗号分隔，如 "gun,face"）')
    p.add_argument("--stamp", default=None, help="mode=img 时使用的遮挡图片")
    p.add_argument("--fmt", default="jpg", choices=["jpg", "png"], help="输出格式")
    p.add_argument("--quality", type=int, default=92, help="jpg 质量 40-100")
    p.add_argument("--list-classes", action="store_true", help="列出全部类别后退出")
    return p


def parse_classes(text):
    """--classes：all -> ["all"]；否则按逗号切；缺省返回 None 交给 build_cfg。"""
    if text is None:
        return None
    text = text.strip()
    if text.lower() == "all":
        return ["all"]
    return [c.strip() for c in text.split(",") if c.strip()]


def list_classes():
    print("可用类别（* 为默认打码）：")
    for name in LABELS:
        mark = "*" if name in DEFAULT_CLASSES else " "
        print(f"  {mark} {name}")


def setup_world(cfg, res):
    """启用 YOLO-World 自定义类别，返回 (检测器, 文本嵌入)。"""
    from yolo_world import (WORLD_PREFIX, WORLD_RESOLUTION, expand_words,
                            get_world_detector)

    words = expand_words(cfg["world_classes"])
    print(f"自定义类别：{words}（首次使用会自动下载模型）")
    det = get_world_detector()
    det.resolution = max(WORLD_RESOLUTION, res)
    embeds = det.get_embeds(words)
    cfg["world_classes"] = words
    cfg["classes"] = list(cfg["classes"]) + [WORLD_PREFIX + w for w in words]
    return det, embeds


def out_path(src, fmt):
    stem, _ = os.path.splitext(src)
    return f"{stem}_censored.{fmt}"


def main(argv=None):
    args = make_parser().parse_args(argv)

    if args.list_classes:
        list_classes()
        return 0

    if not args.images:
        make_parser().print_help()
        return 1

    if args.res not in ALLOWED_RESOLUTIONS:
        print(f"无效分辨率 {args.res}，可选：{ALLOWED_RESOLUTIONS}")
        return 1

    if args.mode not in MODE_LABELS:
        print(f"未知打码方式 {args.mode}，可选：{list(MODE_LABELS)}")
        return 1

    stamp = None
    if args.mode == "img":
        if not args.stamp:
            print("mode=img 需要同时提供 --stamp 遮挡图片")
            return 1
        stamp = cv2.imread(args.stamp)
        if stamp is None:
            print(f"遮挡图片读不出来：{args.stamp}")
            return 1

    cfg = build_cfg(mode=args.mode, strength=args.strength, margin=args.margin,
                    color=args.color, classes=parse_classes(args.classes),
                    conf=args.conf, world_classes=args.world)

    detector = Detector(inference_resolution=args.res)
    world_det = world_embeds = None
    if cfg["world_classes"]:
        world_det, world_embeds = setup_world(cfg, args.res)

    quality = max(40, min(100, args.quality))
    written = 0
    for path in args.images:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: 读不出来，跳过")
            continue

        hits = detector.detect(img, conf=cfg["conf"])
        if world_det is not None:
            hits = list(hits) + list(world_det.detect(
                img, cfg["world_classes"], embeds=world_embeds, conf=cfg["conf"]))

        print(f"\n{path}: 命中 {len(hits)} 处")
        for d in sorted(hits, key=lambda v: -v["score"]):
            print(f"  {d['class']}  score={d['score']:.2f}  box={d['box']}")

        n = censor_regions(img, hits, cfg, stamp=stamp)
        dst = out_path(path, args.fmt)
        if args.fmt == "png":
            ok = cv2.imwrite(dst, img)
        else:
            ok = cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            print(f"  [写入失败] {dst}")
            continue
        written += 1
        print(f"  已处理 {n} 处（{MODE_LABELS[cfg['mode']]}）-> {dst}")

    print(f"\n完成：{written}/{len(args.images)} 张")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
