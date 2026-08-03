"""渲染结果的图像层校验。

只断言"接口返回了 PNG"是没用的——全黑图、纯色图、上下颠倒的图都能通过。
这里提供四类可断言的图像事实：

1. **可解码 & 尺寸正确**：PNG 头合法、宽高与请求一致；
2. **非空画面**：像素标准差高于阈值，挡住"渲染成功但全黑/纯色"这类静默失败；
3. **主色调**：平均色的主导通道，用来验证 palette 需求真的落到了画面上；
4. **图像回归**：感知哈希（aHash）+ 汉明距离，同输入必须稳定复现，
   同时能证明 `iTime` 不同确实产生了不同帧（动画生效）。
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@dataclass
class ImageStats:
    width: int
    height: int
    mean: tuple[float, float, float]
    std: float
    unique_colors: int

    @property
    def dominant_channel(self) -> str:
        names = ("r", "g", "b")
        return names[max(range(3), key=lambda i: self.mean[i])]


def decode_png(image_base64: str) -> bytes:
    raw = base64.b64decode(image_base64)
    assert raw[:8] == PNG_MAGIC, "返回的不是合法 PNG（magic 头不匹配）"
    return raw


def load_stats(image_base64: str) -> ImageStats:
    import numpy as np
    from PIL import Image

    raw = decode_png(image_base64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype="f4") / 255.0
    flat = arr.reshape(-1, 3)
    return ImageStats(
        width=img.width,
        height=img.height,
        mean=(float(flat[:, 0].mean()), float(flat[:, 1].mean()),
              float(flat[:, 2].mean())),
        std=float(arr.std()),
        unique_colors=int(len({tuple(x) for x in (flat[::37] * 255).astype("u1")})),
    )


def assert_image_ok(image_base64: str, *, width: int | None = None,
                    height: int | None = None, min_std: float = 0.02,
                    min_unique_colors: int = 8) -> ImageStats:
    st = load_stats(image_base64)
    if width is not None:
        assert st.width == width, f"图像宽度 {st.width} != 请求的 {width}"
    if height is not None:
        assert st.height == height, f"图像高度 {st.height} != 请求的 {height}"
    assert st.std >= min_std, (
        f"画面几乎无变化（std={st.std:.4f} < {min_std}），疑似全黑或纯色渲染失败")
    assert st.unique_colors >= min_unique_colors, (
        f"颜色数仅 {st.unique_colors}，疑似渲染退化")
    return st


def assert_dominant_channel(image_base64: str, channel: str,
                            *, margin: float = 0.02) -> None:
    st = load_stats(image_base64)
    idx = {"r": 0, "g": 1, "b": 2}[channel]
    others = [v for i, v in enumerate(st.mean) if i != idx]
    assert st.mean[idx] > max(others) + margin, (
        f"期望主色通道 {channel}，实际均值 r/g/b={st.mean}")


def ahash(image_base64: str, size: int = 8) -> int:
    """8×8 平均哈希：抗压缩噪声，能稳定表达"画面是否同一张"。"""
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(decode_png(image_base64))).convert("L")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype="f4")
    bits = (arr > arr.mean()).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(bool(b))
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def assert_same_image(a_b64: str, b_b64: str, *, max_distance: int = 2) -> None:
    d = hamming(ahash(a_b64), ahash(b_b64))
    assert d <= max_distance, f"两帧应当一致，感知哈希距离 ={d} > {max_distance}"


def assert_different_image(a_b64: str, b_b64: str, *, min_distance: int = 3) -> None:
    d = hamming(ahash(a_b64), ahash(b_b64))
    assert d >= min_distance, (
        f"两帧应当不同（如 iTime 变化后的动画帧），感知哈希距离仅 {d}")
