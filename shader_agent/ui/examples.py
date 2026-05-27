"""UI 预置示例：给三个标签页各放几个开箱即用的样例。

设计原则：
- Analyzer 示例：直接拿 v1 seeds 里最有教学性的几个，保证 mainImage 签名规范、能编译；
- Generator 示例：覆盖各主题的中文 prompt，让用户一键体验 5 种典型生成场景；
- Collaboration 示例：一段经典代码 + 一条改写指令的二元组，演示"先分析后改写"。
"""
from __future__ import annotations

from shader_agent.corpus.seed_shaders import get_seed_shaders


def analyzer_examples() -> list[list[str]]:
    """[[label, code], ...]，Gradio Examples 期望的二维列表。"""
    by_id = {s.shader_id: s for s in get_seed_shaders()}
    picks = [
        ("seed03", "Raymarched Sphere（经典 SDF 球 + 法线光照）"),
        ("seed14", "Smooth Min Blob（smin 软融合两个球）"),
        ("seed05", "Mandelbrot Set（escape-time 分形）"),
        ("seed10", "Domain Warping（IQ 域弯曲 FBM）"),
        ("seed08", "Polar Kaleidoscope（极坐标 6 折万花筒）"),
        ("seed28", "Orbiting Camera（lookAt 相机环绕）"),
    ]
    out: list[list[str]] = []
    for sid, label in picks:
        if sid in by_id:
            out.append([label, by_id[sid].code_image])
    return out


def generator_examples() -> list[list[str]]:
    """[[prompt, palette, complexity, dynamic], ...]"""
    return [
        ["画一个程序生成的霓虹蓝紫万花筒，6 折对称，带时间动画", "neon glowing", "simple", True],
        ["raymarching 一个软融合的两个球（smin），冷色调中等复杂", "cool blue cyan", "moderate", True],
        ["用 fbm 噪声画一个水波纹效果，蓝绿色，平静", "cool blue cyan", "simple", True],
        ["落日下的旋转方块，暖色，简单", "warm sunset orange red", "simple", True],
        ["黑白单色的扫描线 CRT 老电视效果", "monochrome grayscale", "simple", True],
        ["复杂的分形花朵图案，鲜艳", "vibrant saturated", "complex", False],
    ]


def collaborate_examples() -> list[list[str]]:
    """[[label, code, ask], ...]"""
    by_id = {s.shader_id: s for s in get_seed_shaders()}
    picks = [
        ("seed03", "Raymarched Sphere → 改成霓虹紫主题",
         "保留 raymarching 算法主干，但把光照换成紫色霓虹氛围，加一点 fresnel 边缘高光"),
        ("seed05", "Mandelbrot → 换成 Julia + 暖色",
         "把 Mandelbrot 改写成 Julia 集，c 参数随 iTime 缓慢游走，暖色橙红调色板"),
        ("seed08", "Kaleidoscope → 改成 8 折 + 慢速旋转",
         "把 6 折万花筒改成 8 折对称，再额外整体随 iTime 慢速旋转，调色板换成粉彩"),
        ("seed10", "Domain Warping → 加上发光中心",
         "保留 domain warping 噪声主干，在中心叠加一个柔和发光的橙色球状光斑"),
    ]
    out: list[list[str]] = []
    for sid, label, ask in picks:
        if sid in by_id:
            out.append([label, by_id[sid].code_image, ask])
    return out
