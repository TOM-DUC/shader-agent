"""内嵌种子 shader 库（阶段二 v2：8 → 33 条）。

为什么扩到 33 条？
- 首版 8 条仅覆盖 5 个主题，向量检索"相似样本"几乎只能命中自己；
- Shadertoy 等级不足无法申请官方 API key 时，本文件是知识库唯一来源；
- 33 条覆盖了 TOPIC_VOCAB 全部 8 个主题，每主题至少 3 条样本，
  保证 Analyzer 的 "对照参考样本" 和 Generator 的 "in-context example"
  在任一查询上都能命中 ≥1 条非自身样本。

所有 shader 都是：
1. 自己手写（无第三方版权）；
2. 仅用 Shadertoy 内置 uniform（iResolution / iTime / iMouse），无外部贴图；
3. 经 moderngl GLSL 330 编译通过的最小教学示例；
4. 行数 50~120，密度适合 LLM in-context（不会塞爆 prompt）。

兼容性约束（DO NOT BREAK）：
- seed01..seed08 名称、id、代码必须与 v1 完全一致，被多处测试与 verify_* 引用；
- 顺序可以追加，但 v1 段不要插入新条目。
"""
from __future__ import annotations

from .models import RenderPass, ShaderRecord


def _mk(
    shader_id: str,
    name: str,
    description: str,
    code: str,
    tags_raw: list[str] | None = None,
) -> ShaderRecord:
    return ShaderRecord(
        shader_id=shader_id,
        name=name,
        username="seed",
        description=description,
        likes=999,
        viewed=0,
        tags_raw=tags_raw or [],
        source="seed",
        passes=[RenderPass(name="Image", type="image", code=code)],
        code_image=code,
    )


# ========================================================================
# v1 段（seed01 ~ seed08）：与首版完全一致；被 tests/* 与 scripts/verify_* 引用。
# 严禁修改 id / name / code。
# ========================================================================

_V1_SEEDS: list[ShaderRecord] = [
    _mk(
        "seed01",
        "Horizontal Gradient",
        "Simple horizontal black-to-white gradient using UV coordinates.",
        """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    fragColor = vec4(vec3(uv.x), 1.0);
}
""".strip(),
        tags_raw=["gradient", "2d", "basic"],
    ),
    _mk(
        "seed02",
        "Animated Circle",
        "A circle that pulses with time, demonstrating distance-to-point and smoothstep.",
        """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float r = 0.3 + 0.1 * sin(iTime * 2.0);
    float d = length(uv) - r;
    float a = smoothstep(0.005, 0.0, d);
    fragColor = vec4(vec3(a), 1.0);
}
""".strip(),
        tags_raw=["2d", "animation", "circle"],
    ),
    _mk(
        "seed03",
        "Raymarched Sphere",
        "Classic raymarching of a signed-distance sphere with simple lambert shading.",
        """
float sdSphere(vec3 p, float r) { return length(p) - r; }

float map(vec3 p) { return sdSphere(p, 1.0); }

vec3 calcNormal(vec3 p) {
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
    ));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0.0, 0.0, 3.0);
    vec3 rd = normalize(vec3(uv, -1.5));
    float t = 0.0;
    for (int i = 0; i < 64; i++) {
        vec3 p = ro + rd * t;
        float d = map(p);
        if (d < 0.001) break;
        t += d;
        if (t > 20.0) { t = -1.0; break; }
    }
    vec3 col = vec3(0.0);
    if (t > 0.0) {
        vec3 p = ro + rd * t;
        vec3 n = calcNormal(p);
        vec3 l = normalize(vec3(0.7, 0.8, 0.6));
        col = vec3(0.3, 0.6, 0.9) * max(dot(n, l), 0.0);
    }
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["raymarching", "sdf", "3d", "sphere"],
    ),
    _mk(
        "seed04",
        "Value Noise",
        "Cheap value-noise based on hash function and bilinear interpolation.",
        """
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
        mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
        u.y
    );
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float n = noise(uv * 8.0 + iTime);
    fragColor = vec4(vec3(n), 1.0);
}
""".strip(),
        tags_raw=["noise", "hash", "procedural"],
    ),
    _mk(
        "seed05",
        "Mandelbrot",
        "Mandelbrot set escape-time fractal with simple smooth coloring.",
        """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec2 c = uv * 2.5 - vec2(0.5, 0.0);
    vec2 z = vec2(0.0);
    float i = 0.0;
    const float N = 128.0;
    for (float k = 0.0; k < N; k++) {
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        if (dot(z, z) > 4.0) break;
        i++;
    }
    float t = i / N;
    vec3 col = 0.5 + 0.5 * cos(6.2831 * (t + vec3(0.0, 0.33, 0.67)));
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["fractal", "mandelbrot", "escape-time"],
    ),
    _mk(
        "seed06",
        "Voronoi Cells",
        "2D voronoi diagram using floor/fract cell hashing and minimum-distance search.",
        """
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.y * 8.0;
    vec2 ip = floor(uv);
    vec2 fp = fract(uv);
    float minDist = 1e9;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 o = vec2(float(x), float(y));
            vec2 r = o + hash2(ip + o) - fp;
            minDist = min(minDist, dot(r, r));
        }
    }
    minDist = sqrt(minDist);
    fragColor = vec4(vec3(minDist), 1.0);
}
""".strip(),
        tags_raw=["voronoi", "2d", "pattern"],
    ),
    _mk(
        "seed07",
        "Vignette Post Process",
        "Demonstrates a vignette-style post-processing effect: uv-distance darkening.",
        """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 base = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0.0, 2.0, 4.0));
    vec2 cuv = uv - 0.5;
    float vig = smoothstep(0.8, 0.2, length(cuv));
    fragColor = vec4(base * vig, 1.0);
}
""".strip(),
        tags_raw=["post-processing", "vignette", "2d"],
    ),
    _mk(
        "seed08",
        "Polar Kaleidoscope",
        "Polar coordinate kaleidoscope: a simple 2D pattern symmetry effect.",
        """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float r = length(uv);
    float a = atan(uv.y, uv.x);
    float seg = 6.0;
    a = mod(a, 6.2831 / seg);
    a = abs(a - 3.1415 / seg);
    vec2 p = vec2(cos(a), sin(a)) * r;
    float v = sin(10.0 * p.x + iTime) * cos(10.0 * p.y);
    vec3 col = 0.5 + 0.5 * vec3(v, -v, sin(v + iTime));
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["kaleidoscope", "2d", "pattern"],
    ),
]


# ========================================================================
# v2 段（seed09 ~ seed33）：阶段二补齐扩容；新增 25 个高质量教学示例。
# 命名风格保持英文短语，便于检索匹配中文 query 的英文骨架。
# ========================================================================

_V2_SEEDS: list[ShaderRecord] = [
    # ---- 9: FBM 多层噪声 ----
    _mk(
        "seed09",
        "Fractal Brownian Motion",
        "FBM: layered value noise summed across 5 octaves with persistence.",
        """
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i),                hash(i + vec2(1,0)), u.x),
               mix(hash(i + vec2(0,1)),    hash(i + vec2(1,1)), u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) {
        v += a * noise(p);
        p *= 2.0; a *= 0.5;
    }
    return v;
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.y;
    float f = fbm(uv * 3.0 + iTime * 0.2);
    vec3 col = mix(vec3(0.1, 0.2, 0.4), vec3(0.9, 0.85, 0.7), f);
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["noise", "fbm", "procedural", "animation"],
    ),

    # ---- 10: Domain Warping ----
    _mk(
        "seed10",
        "Domain Warping",
        "IQ-style domain warping: feed FBM output back into FBM coordinates to bend space.",
        """
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1,0)), u.x),
               mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) { v += a * noise(p); p *= 2.0; a *= 0.5; }
    return v;
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.y * 2.0;
    vec2 q = vec2(fbm(uv + iTime * 0.1),
                  fbm(uv + vec2(5.2, 1.3)));
    vec2 r = vec2(fbm(uv + 4.0 * q + vec2(1.7, 9.2)),
                  fbm(uv + 4.0 * q + vec2(8.3, 2.8)));
    float f = fbm(uv + 4.0 * r);
    vec3 col = mix(vec3(0.1, 0.0, 0.3), vec3(1.0, 0.6, 0.3), f);
    col = mix(col, vec3(0.9, 0.9, 1.0), length(q) * 0.4);
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["noise", "fbm", "domain-warp", "procedural"],
    ),

    # ---- 11: IQ Cosine Palette ----
    _mk(
        "seed11",
        "IQ Cosine Palette",
        "Inigo Quilez's compact procedural palette via cosines; classic for shader rainbows.",
        """
vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(6.2831 * (c * t + d));
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float t = uv.x + 0.3 * sin(iTime + uv.y * 6.2831);
    vec3 col = palette(
        t,
        vec3(0.5, 0.5, 0.5),
        vec3(0.5, 0.5, 0.5),
        vec3(1.0, 1.0, 1.0),
        vec3(0.00, 0.10, 0.20)
    );
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["palette", "color", "2d", "animation"],
    ),

    # ---- 12: Plasma ----
    _mk(
        "seed12",
        "Plasma",
        "Demo-scene plasma: sum of sines in x/y/distance space with smooth color mapping.",
        """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y * 4.0;
    float v =
        sin(uv.x + iTime) +
        sin(uv.y + iTime * 0.5) +
        sin(uv.x + uv.y + iTime * 0.3) +
        sin(length(uv) * 2.0 - iTime);
    v *= 0.25;
    vec3 col = 0.5 + 0.5 * cos(6.2831 * (v + vec3(0.0, 0.33, 0.66)));
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["plasma", "2d", "animation", "sine"],
    ),

    # ---- 13: Truchet Tiles ----
    _mk(
        "seed13",
        "Truchet Arcs",
        "Quarter-circle Truchet tiles: each cell randomly draws one of two arc orientations.",
        """
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.y * 8.0;
    vec2 ip = floor(uv);
    vec2 fp = fract(uv);
    float h = hash(ip);
    // 一半的格子翻转 x，得到 2 种弧形
    if (h > 0.5) fp.x = 1.0 - fp.x;
    float d1 = length(fp - vec2(0.0, 0.0)) - 0.5;
    float d2 = length(fp - vec2(1.0, 1.0)) - 0.5;
    float d  = min(abs(d1), abs(d2)) - 0.08;
    float a  = smoothstep(0.005, -0.005, d);
    vec3 bg  = vec3(0.05, 0.1, 0.15);
    vec3 fg  = vec3(0.85, 0.95, 1.0);
    fragColor = vec4(mix(bg, fg, a), 1.0);
}
""".strip(),
        tags_raw=["truchet", "2d", "pattern", "tiles"],
    ),

    # ---- 14: Smooth Min Raymarching ----
    _mk(
        "seed14",
        "Smooth Min Blob",
        "Raymarch two animated spheres unioned with smooth-min (smin) to make a soft blob.",
        """
float sdSphere(vec3 p, float r) { return length(p) - r; }
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
float map(vec3 p) {
    vec3 c = vec3(sin(iTime) * 0.7, cos(iTime * 0.7) * 0.5, 0.0);
    float a = sdSphere(p - c, 0.6);
    float b = sdSphere(p + c, 0.5);
    return smin(a, b, 0.3);
}
vec3 nrm(vec3 p) {
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)));
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0.0, 0.0, 3.0);
    vec3 rd = normalize(vec3(uv, -1.5));
    float t = 0.0;
    for (int i = 0; i < 96; i++) {
        vec3 p = ro + rd * t;
        float d = map(p);
        if (d < 0.001) break;
        t += d;
        if (t > 20.0) { t = -1.0; break; }
    }
    vec3 col = vec3(0.05, 0.06, 0.08);
    if (t > 0.0) {
        vec3 p = ro + rd * t;
        vec3 n = nrm(p);
        float diff = max(dot(n, normalize(vec3(0.5, 0.8, 0.3))), 0.0);
        col = vec3(0.4, 0.6, 1.0) * diff + vec3(0.05);
    }
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["raymarching", "sdf", "smin", "blob", "lighting"],
    ),

    # ---- 15: Phong Shading ----
    _mk(
        "seed15",
        "Phong Shaded Sphere",
        "Raymarched sphere with full Phong shading (ambient + diffuse + specular).",
        """
float sdSphere(vec3 p, float r) { return length(p) - r; }
float map(vec3 p) { return sdSphere(p, 1.0); }
vec3 nrm(vec3 p) {
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)));
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0.0, 0.0, 3.0);
    vec3 rd = normalize(vec3(uv, -1.5));
    float t = 0.0;
    for (int i = 0; i < 64; i++) {
        vec3 p = ro + rd * t;
        float d = map(p);
        if (d < 0.001) break;
        t += d;
        if (t > 20.0) { t = -1.0; break; }
    }
    vec3 col = vec3(0.03, 0.04, 0.05);
    if (t > 0.0) {
        vec3 p = ro + rd * t;
        vec3 n = nrm(p);
        vec3 l = normalize(vec3(sin(iTime), 0.6, cos(iTime)));
        vec3 v = -rd;
        vec3 h = normalize(l + v);
        float diff = max(dot(n, l), 0.0);
        float spec = pow(max(dot(n, h), 0.0), 32.0);
        col = vec3(0.1) + vec3(0.3, 0.5, 0.9) * diff + vec3(1.0) * spec * 0.6;
    }
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["raymarching", "sdf", "phong", "lighting", "specular"],
    ),

    # ---- 16: Domain Repetition ----
    _mk(
        "seed16",
        "Infinite Spheres",
        "Domain repetition via mod(p, c) - 0.5*c → repeats one SDF infinitely in 3D.",
        """
float sdSphere(vec3 p, float r) { return length(p) - r; }
float map(vec3 p) {
    vec3 q = mod(p + 1.5, 3.0) - 1.5;  // 周期 3.0，居中
    return sdSphere(q, 0.7);
}
vec3 nrm(vec3 p) {
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)));
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0.0, 0.0, iTime);
    vec3 rd = normalize(vec3(uv, -1.5));
    float t = 0.0;
    for (int i = 0; i < 80; i++) {
        vec3 p = ro + rd * t;
        float d = map(p);
        if (d < 0.001) break;
        t += d;
        if (t > 30.0) { t = -1.0; break; }
    }
    vec3 col = vec3(0.02);
    if (t > 0.0) {
        vec3 p = ro + rd * t;
        vec3 n = nrm(p);
        float diff = max(dot(n, normalize(vec3(0.6, 0.8, 0.2))), 0.0);
        col = vec3(0.6, 0.7, 0.9) * diff;
        col *= exp(-t * 0.1);  // 雾化
    }
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["raymarching", "sdf", "repetition", "fog", "infinite"],
    ),

    # ---- 17: Box SDF ----
    _mk(
        "seed17",
        "Rotating Box",
        "Raymarch a box SDF rotated by time; demonstrates 3D rotation matrix + sdBox.",
        """
mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}
float map(vec3 p) {
    p.xz *= rot(iTime * 0.7);
    p.xy *= rot(iTime * 0.3);
    return sdBox(p, vec3(0.7));
}
vec3 nrm(vec3 p) {
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)));
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0.0, 0.0, 3.0);
    vec3 rd = normalize(vec3(uv, -1.5));
    float t = 0.0;
    for (int i = 0; i < 80; i++) {
        vec3 p = ro + rd * t;
        float d = map(p);
        if (d < 0.001) break;
        t += d;
        if (t > 20.0) { t = -1.0; break; }
    }
    vec3 col = vec3(0.05);
    if (t > 0.0) {
        vec3 p = ro + rd * t;
        vec3 n = nrm(p);
        float diff = max(dot(n, normalize(vec3(0.6, 0.8, 0.4))), 0.0);
        col = vec3(0.9, 0.7, 0.4) * diff + vec3(0.05);
    }
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["raymarching", "sdf", "box", "rotation", "3d"],
    ),

    # ---- 18: Caustics ----
    _mk(
        "seed18",
        "Water Caustics",
        "Fake caustics: layered sinusoidal distortions of uv summed into bright filaments.",
        """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.y;
    vec2 p = uv * 6.0;
    float c = 0.0;
    float t = iTime * 0.5;
    for (int i = 0; i < 4; i++) {
        float fi = float(i) + 1.0;
        p += vec2(sin(p.y * 1.3 + t * fi) / fi,
                  cos(p.x * 1.3 + t * fi) / fi);
        c += 1.0 / length(vec2(p.x / sin(t + fi) * 0.1,
                                p.y / cos(t + fi) * 0.1));
    }
    c /= 4.0;
    c = clamp(0.0, 1.0, c);
    vec3 col = mix(vec3(0.0, 0.1, 0.3), vec3(0.6, 0.95, 1.0), c);
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["caustics", "water", "2d", "animation"],
    ),

    # ---- 19: Hex Grid ----
    _mk(
        "seed19",
        "Hex Grid",
        "Axial hex grid using skewed coordinates; renders animated cell brightness.",
        """
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
vec4 hexCoords(vec2 p) {
    // 把 p 映射到 axial 六边形坐标
    vec2 q = vec2(p.x * 2.0 * 0.5773503, p.y + p.x * 0.5773503);
    vec2 pi = floor(q);
    vec2 pf = fract(q);
    float v = mod(pi.x + pi.y, 3.0);
    float ca = step(1.0, v);
    float cb = step(2.0, v);
    vec2 ma = step(pf.xy, pf.yx);
    return vec4(pi + ca - cb * ma, pf);
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y * 6.0;
    vec4 hc = hexCoords(uv);
    float h = hash(hc.xy);
    float v = 0.5 + 0.5 * sin(iTime * 2.0 + h * 12.0);
    vec3 col = mix(vec3(0.05, 0.07, 0.12), vec3(0.7, 0.95, 1.0), v) * h;
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["hex", "grid", "2d", "pattern", "animation"],
    ),

    # ---- 20: Starfield ----
    _mk(
        "seed20",
        "Starfield",
        "Procedural starfield: hash-based dot placement on grid cells with parallax.",
        """
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
float stars(vec2 p, float scale, float density) {
    vec2 ip = floor(p * scale);
    vec2 fp = fract(p * scale);
    float h = hash(ip);
    if (h < density) {
        float d = length(fp - 0.5);
        return smoothstep(0.1 * (1.0 - h), 0.0, d) * h;
    }
    return 0.0;
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.y;
    uv.x += iTime * 0.05;
    float s = 0.0;
    s += stars(uv, 30.0, 0.04);
    s += stars(uv + 100.0, 60.0, 0.02) * 0.6;
    s += stars(uv + 200.0, 120.0, 0.01) * 0.4;
    vec3 col = vec3(0.01, 0.01, 0.05) + vec3(s);
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["stars", "2d", "procedural", "animation"],
    ),

    # ---- 21: Tunnel ----
    _mk(
        "seed21",
        "Polar Tunnel",
        "Infinite tunnel illusion via polar coordinates with depth-based fog.",
        """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float r = length(uv);
    float a = atan(uv.y, uv.x);
    vec2 tp = vec2(0.5 / r + iTime, a / 3.1416 + 0.5);
    float pat = step(0.5, fract(tp.x * 4.0)) * step(0.5, fract(tp.y * 6.0));
    pat += step(0.5, fract(tp.x * 4.0 + 0.5)) * step(0.5, fract(tp.y * 6.0 + 0.5));
    vec3 col = mix(vec3(0.05, 0.1, 0.2), vec3(0.9, 0.7, 0.4), pat);
    col *= r;  // 距中心远更亮，模拟越走越亮
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["tunnel", "polar", "2d", "animation", "checker"],
    ),

    # ---- 22: Mandelbulb-lite ----
    _mk(
        "seed22",
        "Julia Set",
        "Julia set: companion to Mandelbrot but parameterized by a constant c; animated.",
        """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y * 1.7;
    vec2 c = vec2(0.7885 * cos(iTime * 0.3), 0.7885 * sin(iTime * 0.3));
    vec2 z = uv;
    float i = 0.0;
    const float N = 96.0;
    for (float k = 0.0; k < N; k++) {
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        if (dot(z, z) > 4.0) break;
        i++;
    }
    float t = i / N;
    vec3 col = 0.5 + 0.5 * cos(6.2831 * (t + vec3(0.5, 0.2, 0.0)));
    if (i >= N - 0.5) col = vec3(0.02);
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["fractal", "julia", "escape-time", "animation"],
    ),

    # ---- 23: Sierpinski 2D ----
    _mk(
        "seed23",
        "Sierpinski Triangle",
        "Sierpinski-like recursive triangles via bit-trick on integer coordinates.",
        """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.y * 8.0;
    vec2 ip = floor(uv);
    // 经典 bit-AND 等价：在 GLSL 里用 mod 模拟
    float m = 1.0;
    for (int k = 0; k < 6; k++) {
        if (mod(ip.x + ip.y, 2.0) >= 1.0) { m = 0.0; break; }
        ip = floor(ip * 0.5);
    }
    vec3 col = mix(vec3(0.02), vec3(0.95, 0.6, 0.3), m);
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["fractal", "sierpinski", "2d", "recursive"],
    ),

    # ---- 24: Spiral ----
    _mk(
        "seed24",
        "Spiral Arms",
        "Logarithmic spiral pattern via atan + log(r).",
        """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float r = length(uv);
    float a = atan(uv.y, uv.x);
    float s = sin(a * 5.0 + log(r) * 6.0 - iTime * 2.0);
    s = smoothstep(0.0, 0.4, s);
    vec3 col = mix(vec3(0.1, 0.0, 0.2), vec3(1.0, 0.8, 0.5), s);
    col *= smoothstep(1.0, 0.2, r);  // 外围淡出
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["spiral", "polar", "2d", "pattern", "animation"],
    ),

    # ---- 25: Bloom-style glow ----
    _mk(
        "seed25",
        "Glowing Orb",
        "Soft glowing orb with radial falloff and chromatic gradient; simple post-fx style.",
        """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float r = length(uv);
    float core = smoothstep(0.25, 0.0, r);
    float halo = pow(0.4 / (r + 0.02), 1.5) * 0.3;
    float pulse = 0.85 + 0.15 * sin(iTime * 3.0);
    vec3 inner = vec3(1.0, 0.9, 0.7) * core * pulse;
    vec3 outer = vec3(0.4, 0.6, 1.0) * halo;
    fragColor = vec4(inner + outer, 1.0);
}
""".strip(),
        tags_raw=["post-processing", "glow", "bloom", "2d", "animation"],
    ),

    # ---- 26: Scanlines / CRT ----
    _mk(
        "seed26",
        "CRT Scanlines",
        "Retro CRT effect: animated gradient base modulated by horizontal scanlines + vignette.",
        """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 base = 0.5 + 0.5 * cos(iTime + uv.xyx * 3.0 + vec3(0.0, 2.0, 4.0));
    // 扫描线
    float lines = 0.85 + 0.15 * sin(fragCoord.y * 3.14);
    // 渐变 vignette
    vec2 cv = uv - 0.5;
    float vig = smoothstep(0.9, 0.3, length(cv));
    vec3 col = base * lines * vig;
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["post-processing", "scanlines", "crt", "vignette", "2d"],
    ),

    # ---- 27: Distance Field Heart ----
    _mk(
        "seed27",
        "SDF Heart",
        "2D heart shape via analytic SDF; pulses with time.",
        """
float sdHeart(vec2 p) {
    p.x = abs(p.x);
    if (p.y + p.x > 1.0) {
        return sqrt(dot(p - vec2(0.25, 0.75), p - vec2(0.25, 0.75))) - sqrt(2.0) * 0.25;
    }
    return sqrt(min(
        dot(p - vec2(0.0, 1.0), p - vec2(0.0, 1.0)),
        dot(p - 0.5 * max(p.x + p.y, 0.0), p - 0.5 * max(p.x + p.y, 0.0))
    )) * sign(p.x - p.y);
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y * 2.5;
    uv.y += 0.3;
    float pulse = 1.0 + 0.07 * sin(iTime * 6.0);
    float d = sdHeart(uv / pulse) * pulse;
    float a = smoothstep(0.02, 0.0, d);
    vec3 col = mix(vec3(0.05, 0.0, 0.02), vec3(1.0, 0.15, 0.25), a);
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["sdf", "2d", "heart", "animation"],
    ),

    # ---- 28: Camera Animation Raymarching ----
    _mk(
        "seed28",
        "Orbiting Camera",
        "Raymarched scene with proper look-at camera orbiting around the origin.",
        """
mat3 lookAt(vec3 eye, vec3 tgt, vec3 up) {
    vec3 f = normalize(tgt - eye);
    vec3 r = normalize(cross(f, up));
    vec3 u = cross(r, f);
    return mat3(r, u, -f);
}
float sdSphere(vec3 p, float r) { return length(p) - r; }
float sdPlane(vec3 p) { return p.y + 1.0; }
float map(vec3 p) {
    return min(sdSphere(p, 1.0), sdPlane(p));
}
vec3 nrm(vec3 p) {
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)));
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float ang = iTime * 0.4;
    vec3 ro = vec3(3.5 * cos(ang), 1.5, 3.5 * sin(ang));
    mat3 R = lookAt(ro, vec3(0.0), vec3(0.0, 1.0, 0.0));
    vec3 rd = R * normalize(vec3(uv, -1.5));
    float t = 0.0;
    for (int i = 0; i < 80; i++) {
        vec3 p = ro + rd * t;
        float d = map(p);
        if (d < 0.001) break;
        t += d;
        if (t > 30.0) { t = -1.0; break; }
    }
    vec3 col = vec3(0.4, 0.6, 0.85);
    if (t > 0.0) {
        vec3 p = ro + rd * t;
        vec3 n = nrm(p);
        float diff = max(dot(n, normalize(vec3(0.5, 0.8, 0.3))), 0.0);
        col = vec3(0.7, 0.8, 0.9) * diff;
    }
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["raymarching", "camera", "orbit", "lookAt", "sdf"],
    ),

    # ---- 29: Worley + lines ----
    _mk(
        "seed29",
        "Cell Edges",
        "Voronoi variant: render only the edges between cells using F2-F1 (gap to second nearest).",
        """
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.y * 6.0;
    vec2 ip = floor(uv);
    vec2 fp = fract(uv);
    float d1 = 1e9, d2 = 1e9;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 o = vec2(float(x), float(y));
            vec2 r = o + 0.5 + 0.5 * sin(iTime + 6.2831 * hash2(ip + o)) - fp;
            float d = dot(r, r);
            if (d < d1) { d2 = d1; d1 = d; }
            else if (d < d2) { d2 = d; }
        }
    }
    float edge = smoothstep(0.0, 0.05, sqrt(d2) - sqrt(d1));
    vec3 col = mix(vec3(0.95, 0.4, 0.1), vec3(0.05, 0.07, 0.1), edge);
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["voronoi", "edges", "2d", "noise", "animation"],
    ),

    # ---- 30: Lissajous ----
    _mk(
        "seed30",
        "Lissajous Curve",
        "Plots a Lissajous curve trail with persistence; lines glow on dark.",
        """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec3 col = vec3(0.0);
    // 沿时间采样多个点，画轨迹
    for (int i = 0; i < 60; i++) {
        float t = iTime - float(i) * 0.03;
        vec2 q = vec2(sin(3.0 * t + 1.2), sin(2.0 * t)) * 0.7;
        float d = length(uv - q);
        col += vec3(0.4, 0.8, 1.0) * (0.005 / (d * d + 0.0005)) * (1.0 - float(i) / 60.0);
    }
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["lissajous", "trail", "2d", "animation", "glow"],
    ),

    # ---- 31: Wave Interference ----
    _mk(
        "seed31",
        "Wave Interference",
        "Two circular wave sources interfere; demonstrates length() to multiple anchors.",
        """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec2 a = vec2(-0.4, 0.0), b = vec2(0.4, 0.0);
    float wa = sin(length(uv - a) * 30.0 - iTime * 4.0);
    float wb = sin(length(uv - b) * 30.0 - iTime * 4.0);
    float v = (wa + wb) * 0.5;
    vec3 col = 0.5 + 0.5 * cos(6.2831 * (v + vec3(0.0, 0.33, 0.66)));
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["wave", "interference", "2d", "animation", "sine"],
    ),

    # ---- 32: Lava Lamp ----
    _mk(
        "seed32",
        "Lava Lamp",
        "Metaballs in 2D: sum of inverse squared distances to moving centers, thresholded.",
        """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float v = 0.0;
    for (int i = 0; i < 5; i++) {
        float fi = float(i);
        vec2 c = vec2(
            0.6 * sin(iTime * 0.7 + fi * 1.7),
            0.6 * cos(iTime * 0.9 + fi * 2.3)
        );
        v += 0.05 / (dot(uv - c, uv - c) + 0.01);
    }
    float k = smoothstep(2.0, 4.0, v);
    vec3 col = mix(vec3(0.02, 0.0, 0.05), vec3(1.0, 0.4, 0.1), k);
    col = mix(col, vec3(1.0, 0.9, 0.4), pow(k, 4.0));
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["metaballs", "2d", "animation", "lava"],
    ),

    # ---- 33: Bokeh Dots ----
    _mk(
        "seed33",
        "Bokeh Dots",
        "Layered animated colored dots; demonstrates accumulation + soft falloff.",
        """
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec3 col = vec3(0.02, 0.02, 0.04);
    for (int i = 0; i < 12; i++) {
        float fi = float(i);
        vec2 seed = vec2(fi * 1.7, fi * 3.3);
        vec2 c = vec2(hash(seed) - 0.5, hash(seed + 1.0) - 0.5) * 2.0;
        c.x += 0.4 * sin(iTime * 0.3 + fi);
        c.y += 0.4 * cos(iTime * 0.4 + fi);
        float r = 0.08 + 0.05 * hash(seed + 2.0);
        float d = length(uv - c);
        vec3 tint = 0.5 + 0.5 * cos(6.2831 * (fi / 12.0 + vec3(0.0, 0.33, 0.66)));
        col += tint * smoothstep(r, 0.0, d) * 0.5;
    }
    fragColor = vec4(col, 1.0);
}
""".strip(),
        tags_raw=["bokeh", "dots", "2d", "animation", "glow"],
    ),
]


SEED_SHADERS: list[ShaderRecord] = _V1_SEEDS + _V2_SEEDS


def get_seed_shaders() -> list[ShaderRecord]:
    """返回种子 shader 的可变拷贝，避免被外部修改污染原对象。"""
    return [ShaderRecord(**s.model_dump()) for s in SEED_SHADERS]
