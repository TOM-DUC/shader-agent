"""WebGL 实时预览：在浏览器里跑用户的 Shadertoy GLSL，真正"动起来"。

为什么用 iframe srcdoc（方案一）
================================
关键坑：Gradio 的 `gr.HTML` 通过 innerHTML 注入内容，而**浏览器规范规定：
用 innerHTML 插入的 <script> 标签不会执行**。这就是之前"所有 shader 都黑屏"的
根因——canvas 画出来了，但驱动 WebGL 的脚本从未运行。

修复方式：把整段 WebGL 页面塞进一个 <iframe srcdoc="...">。iframe 是一个独立
文档，其中的 <script> 会正常执行，彻底绕开 innerHTML 不跑脚本的限制。

后端 moderngl 仍保留，负责**编译校验**与生成时的修正循环；前端 iframe+WebGL
负责**动态展示**。两者互补。

注意
====
- 整页 HTML 先 base64，再放进 `srcdoc`，避免引号/换行/特殊字符破坏属性值，
  也避免用户代码里的字符干扰外层页面。
- WebGL 是 GLSL ES 1.0（#version 100），与桌面 GLSL 330 有差异；这里注入
  兼容前导（精度声明 + Shadertoy uniform），并把 mainImage 包进 main()。
- 多通道/纹理（iChannelN）不支持，调用方应在传入前过滤。
"""
from __future__ import annotations

import base64
import uuid


def _supports_webgl_preview(user_code: str) -> bool:
    """粗判是否能在 WebGL 单通道里预览（无外部纹理/多通道）。"""
    import re
    if not user_code or "mainImage" not in user_code:
        return False
    if re.search(r"\biChannel[0-9]\b|\bsampler(2D|Cube|3D)\b|"
                 r"\biChannelResolution\b|\biChannelTime\b", user_code):
        return False
    return True


# 完整的内层 HTML 文档模板（放进 iframe srcdoc）。
# 用户 GLSL 以 base64 注入到 JS 里，避免任何字符破坏文档结构。
_INNER_DOC = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  html,body{{margin:0;padding:0;background:#000;overflow:hidden;}}
  canvas{{display:block;width:100vw;height:100vh;}}
  #err{{position:fixed;left:0;right:0;bottom:0;display:none;
    background:rgba(179,38,30,.92);color:#fff;font:12px/1.5 monospace;
    padding:8px 10px;white-space:pre-wrap;}}
  #tag{{position:fixed;top:6px;right:8px;font:11px sans-serif;color:#9fb;
    background:rgba(0,0,0,.35);padding:2px 6px;border-radius:6px;}}
</style></head><body>
<canvas id="c"></canvas>
<div id="tag">WebGL 实时预览 · 拖动控制 iMouse</div>
<div id="err"></div>
<script>
(function(){{
  var canvas=document.getElementById("c");
  var errBox=document.getElementById("err");
  function showErr(m){{errBox.style.display="block";errBox.textContent=m;}}
  var gl=canvas.getContext("webgl")||canvas.getContext("experimental-webgl");
  if(!gl){{showErr("当前浏览器不支持 WebGL，无法实时预览。");return;}}

  var userCode=atob("{code_b64}");
  var vsSrc="attribute vec2 p;void main(){{gl_Position=vec4(p,0.0,1.0);}}";
  var fsHead=
    "#ifdef GL_FRAGMENT_PRECISION_HIGH\\n"+
    "precision highp float;\\n"+
    "#else\\n"+
    "precision mediump float;\\n"+
    "#endif\\n"+
    "uniform vec3 iResolution;\\n"+
    "uniform float iTime;\\n"+
    "uniform float iTimeDelta;\\n"+
    "uniform int iFrame;\\n"+
    "uniform vec4 iMouse;\\n"+
    "uniform vec4 iDate;\\n";
  var fsTail=
    "\\nvoid main(){{vec4 c=vec4(0.0);mainImage(c,gl_FragCoord.xy);"+
    "gl_FragColor=vec4(c.rgb,1.0);}}";
  var fsSrc=fsHead+userCode+fsTail;

  function compile(type,src){{
    var s=gl.createShader(type);
    gl.shaderSource(s,src);gl.compileShader(s);
    if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)){{
      throw new Error(gl.getShaderInfoLog(s)||"shader compile failed");
    }}
    return s;
  }}
  var prog;
  try{{
    var vs=compile(gl.VERTEX_SHADER,vsSrc);
    var fs=compile(gl.FRAGMENT_SHADER,fsSrc);
    prog=gl.createProgram();
    gl.attachShader(prog,vs);gl.attachShader(prog,fs);gl.linkProgram(prog);
    if(!gl.getProgramParameter(prog,gl.LINK_STATUS)){{
      throw new Error(gl.getProgramInfoLog(prog)||"link failed");
    }}
  }}catch(e){{
    showErr("WebGL 预览编译失败（不影响后端校验）：\\n"+e.message);return;
  }}
  gl.useProgram(prog);
  var buf=gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER,buf);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);
  var loc=gl.getAttribLocation(prog,"p");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);

  var uRes=gl.getUniformLocation(prog,"iResolution");
  var uTime=gl.getUniformLocation(prog,"iTime");
  var uDelta=gl.getUniformLocation(prog,"iTimeDelta");
  var uFrame=gl.getUniformLocation(prog,"iFrame");
  var uMouse=gl.getUniformLocation(prog,"iMouse");

  var mouse=[0,0,0,0];
  canvas.addEventListener("mousemove",function(ev){{
    var r=canvas.getBoundingClientRect();
    mouse[0]=(ev.clientX-r.left)*(canvas.width/r.width);
    mouse[1]=(r.height-(ev.clientY-r.top))*(canvas.height/r.height);
  }});
  canvas.addEventListener("mousedown",function(){{mouse[2]=mouse[0];mouse[3]=mouse[1];}});

  function resize(){{
    var dpr=Math.min(window.devicePixelRatio||1,2);
    var w=Math.max(1,Math.floor(canvas.clientWidth*dpr));
    var h=Math.max(1,Math.floor(canvas.clientHeight*dpr));
    if(canvas.width!==w||canvas.height!==h){{canvas.width=w;canvas.height=h;}}
  }}
  var start=performance.now(),last=start,frame=0;
  function render(now){{
    resize();
    gl.viewport(0,0,canvas.width,canvas.height);
    var t=(now-start)/1000.0;
    var dt=(now-last)/1000.0;last=now;
    if(uRes)gl.uniform3f(uRes,canvas.width,canvas.height,1.0);
    if(uTime)gl.uniform1f(uTime,t);
    if(uDelta)gl.uniform1f(uDelta,dt>0?dt:1.0/60.0);
    if(uFrame)gl.uniform1i(uFrame,frame++);
    if(uMouse)gl.uniform4f(uMouse,mouse[0],mouse[1],mouse[2],mouse[3]);
    gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
    requestAnimationFrame(render);
  }}
  requestAnimationFrame(render);
}})();
</script>
</body></html>"""


def webgl_preview_html(user_code: str, *, height: int = 300,
                       autoplay: bool = True) -> str:
    """构造一段 iframe（srcdoc 内含完整 WebGL 页面）。

    若代码不适合 WebGL 预览（多通道/无 mainImage），返回空字符串，
    调用方可回退到静态 PNG。

    实现：内层完整 HTML 文档先 base64，再用 `srcdoc` 以 data 形式注入，
    保证 iframe 里的 <script> 正常执行（绕开 gr.HTML innerHTML 不跑脚本的限制）。
    """
    if not _supports_webgl_preview(user_code):
        return ""

    code_b64 = base64.b64encode((user_code or "").encode("utf-8")).decode("ascii")
    inner_doc = _INNER_DOC.format(code_b64=code_b64)
    fid = "sgframe_" + uuid.uuid4().hex[:8]

    # 用 srcdoc 承载内层完整文档（iframe 内 <script> 会正常执行）。
    # srcdoc 是 HTML 属性值，需把 & 和 " 做实体转义；用户代码已 base64，
    # 内层文档本身不含裸引号问题，这里仅做标准属性转义保险。
    srcdoc = (inner_doc
              .replace("&", "&amp;")
              .replace('"', "&quot;"))
    return (
        f'<iframe id="{fid}" '
        f'srcdoc="{srcdoc}" '
        f'style="width:100%;height:{int(height)}px;border:0;'
        f'border-radius:8px;background:#000;display:block;" '
        f'sandbox="allow-scripts allow-same-origin"></iframe>'
    )
