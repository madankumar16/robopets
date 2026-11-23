"""engine3d.py - small offscreen OpenGL renderer used by RoboPets.

Everything is rendered into a multisampled framebuffer; the final pass hands back
premultiplied BGRA pixels in bottom-up row order, which is exactly what a Windows
per-pixel-alpha layered window (UpdateLayeredWindow) expects.
"""
import math

import moderngl
import numpy as np

# =================================================================== math helpers


def rx(d):
    a = math.radians(d)
    c, s = math.cos(a), math.sin(a)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def ry(d):
    a = math.radians(d)
    c, s = math.cos(a), math.sin(a)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def rz(d):
    a = math.radians(d)
    c, s = math.cos(a), math.sin(a)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def eul(x=0.0, y=0.0, z=0.0):
    """Rotate about X, then Y, then Z (degrees)."""
    return rz(z) @ ry(y) @ rx(x)


def m4(R=None, t=(0.0, 0.0, 0.0), s=1.0):
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R * s
    elif s != 1.0:
        M[:3, :3] *= s
    M[:3, 3] = t
    return M


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def smooth01(x):
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def quat_from_m3(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        q = (0.25 * S, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = math.sqrt(max(1e-12, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
        q = ((m[2, 1] - m[1, 2]) / S, 0.25 * S, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S)
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(max(1e-12, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
        q = ((m[0, 2] - m[2, 0]) / S, (m[0, 1] + m[1, 0]) / S, 0.25 * S, (m[1, 2] + m[2, 1]) / S)
    else:
        S = math.sqrt(max(1e-12, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
        q = ((m[1, 0] - m[0, 1]) / S, (m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, 0.25 * S)
    q = np.array(q)
    return q / np.linalg.norm(q)


def m3_from_quat(q):
    w, x, y, z = q
    return np.array((
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    s = math.sin(th)
    return (math.sin((1.0 - t) * th) * q0 + math.sin(t * th) * q1) / s


def perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy) / 2.0)
    M = np.zeros((4, 4))
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = 2.0 * far * near / (near - far)
    M[3, 2] = -1.0
    return M


def look_at(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def srgb(hexstr):
    h = hexstr.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def linear(c):
    return tuple(x ** 2.2 for x in c)

# =================================================================== primitives
# Every primitive returns flat triangle lists (P, N) with outward, counter-clockwise
# winding. `inf` builds the same shape grown outwards for the ink-outline pass.


def _fix(P, N):
    P = np.asarray(P, np.float64).reshape(-1, 3, 3).copy()
    N = np.asarray(N, np.float64).reshape(-1, 3, 3).copy()
    g = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    bad = np.einsum('ij,ij->i', g, N.sum(1)) < 0.0
    if bad.any():
        P[bad] = P[bad][:, [0, 2, 1]]
        N[bad] = N[bad][:, [0, 2, 1]]
    return P.reshape(-1, 3), N.reshape(-1, 3)


_HF = ((0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5))


def hexa(c, inf=0.0):
    """Convex 8-corner solid. Corner i = ix + 2*iy + 4*iz."""
    c = np.array(c, np.float64).reshape(8, 3)
    cen = c.mean(0)
    fn = []
    for f in _HF:
        q = c[list(f)]
        n = np.cross(q[2] - q[0], q[3] - q[1])
        d = q.mean(0) - cen
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            n = d / (np.linalg.norm(d) + 1e-12)
        else:
            n = n / ln
            if np.dot(n, d) < 0.0:
                n = -n
        fn.append(n)
    if inf:
        c2 = c.copy()
        for i in range(8):
            A = np.array([fn[k] for k, f in enumerate(_HF) if i in f])
            try:
                o = np.linalg.solve(A, np.full(3, inf))
            except np.linalg.LinAlgError:
                o = A.sum(0) * inf
            ln = np.linalg.norm(o)
            if not np.isfinite(ln):
                o, ln = A.sum(0) * inf, np.linalg.norm(A.sum(0) * inf)
            if ln > 2.5 * inf:
                o = o * (2.5 * inf / ln)
            c2[i] = c[i] + o
        c = c2
    P, N = [], []
    for k, f in enumerate(_HF):
        q = c[list(f)]
        for tri in ((0, 1, 2), (0, 2, 3)):
            t = q[list(tri)]
            n = np.cross(t[1] - t[0], t[2] - t[0])
            ln = np.linalg.norm(n)
            if ln < 1e-10:
                continue
            n = n / ln
            if np.dot(n, fn[k]) < 0.0:
                n = -n
            P.extend(t)
            N.extend((n, n, n))
    return _fix(P, N)


def box(w, h, d, inf=0.0):
    return hexa([((w / 2 if i & 1 else -w / 2), (h / 2 if i & 2 else -h / 2), (d / 2 if i & 4 else -d / 2))
                 for i in range(8)], inf)


def taper(wb, db, wt, dt, h, ox=0.0, oz=0.0, inf=0.0):
    """Box whose bottom is wb x db and top is wt x dt (top optionally shifted)."""
    c = []
    for i in range(8):
        top = bool(i & 2)
        w, d = (wt, dt) if top else (wb, db)
        x = (w / 2 if i & 1 else -w / 2) + (ox if top else 0.0)
        z = (d / 2 if i & 4 else -d / 2) + (oz if top else 0.0)
        c.append((x, h / 2 if top else -h / 2, z))
    return hexa(c, inf)


_PI = {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}


def ext_corners(axis, prof, a0, a1, prof1=None):
    """Corners of a 4-point profile (cyclic order) extruded along `axis` from a0 to a1.
    axis 'x': profile is (y, z); 'y': (z, x); 'z': (x, y)."""
    prof = np.asarray(prof, np.float64)
    prof1 = prof if prof1 is None else np.asarray(prof1, np.float64)
    c = []
    for i in range(8):
        bx, by, bz = i & 1, (i >> 1) & 1, (i >> 2) & 1
        if axis == 'x':
            ab, ub, vb = bx, by, bz
        elif axis == 'y':
            ab, ub, vb = by, bz, bx
        else:
            ab, ub, vb = bz, bx, by
        u, v = (prof1 if ab else prof)[_PI[(ub, vb)]]
        a = a1 if ab else a0
        c.append((a, u, v) if axis == 'x' else (v, a, u) if axis == 'y' else (u, v, a))
    return c


def plate_prof(A, B, t, side=1.0):
    """Profile of a strip from A to B with thickness t (2D points)."""
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    d = B - A
    n = np.array((-d[1], d[0])) / (np.linalg.norm(d) + 1e-12) * t * side
    return [A, B, B + n, A + n]


def bevel(w, h, d, b=0.03, inf=0.0):
    """Box with chamfered edges."""
    a = np.array((w / 2 + inf, h / 2 + inf, d / 2 + inf))
    b = min(b + inf * 0.4, float(a.min()) * 0.48)
    faces = []
    for ax in range(3):
        u, v = (ax + 1) % 3, (ax + 2) % 3
        for s in (-1, 1):
            q = []
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[ax], p[u], p[v] = s * a[ax], su * (a[u] - b), sv * (a[v] - b)
                q.append(p)
            faces.append(q)
    for ax in range(3):
        u, v = (ax + 1) % 3, (ax + 2) % 3
        for su in (-1, 1):
            for sv in (-1, 1):
                q = []
                for sa in (-1, 1):
                    p1 = np.zeros(3)
                    p1[ax], p1[u], p1[v] = sa * (a[ax] - b), su * a[u], sv * (a[v] - b)
                    p2 = np.zeros(3)
                    p2[ax], p2[u], p2[v] = sa * (a[ax] - b), su * (a[u] - b), sv * a[v]
                    q.append((p1, p2))
                faces.append([q[0][0], q[0][1], q[1][1], q[1][0]])
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                faces.append([np.array((sx * a[0], sy * (a[1] - b), sz * (a[2] - b))),
                              np.array((sx * (a[0] - b), sy * a[1], sz * (a[2] - b))),
                              np.array((sx * (a[0] - b), sy * (a[1] - b), sz * a[2]))])
    P, N = [], []
    for q in faces:
        cen = sum(q) / len(q)
        n = np.cross(q[1] - q[0], q[2] - q[0])
        if np.linalg.norm(n) < 1e-12 and len(q) > 3:
            n = np.cross(q[2] - q[0], q[3] - q[0])
        n = n / (np.linalg.norm(n) + 1e-12)
        if np.dot(n, cen) < 0.0:
            n = -n
        for i in range(1, len(q) - 1):
            P.extend((q[0], q[i], q[i + 1]))
            N.extend((n, n, n))
    return _fix(P, N)


def _to_axis(P, N, axis):
    if axis == 'y':
        return P, N
    R = rz(-90) if axis == 'x' else rx(90)
    return P @ R.T, N @ R.T


def cyl(r, h, axis='y', seg=16, r2=None, inf=0.0, caps=True):
    """Cylinder / frustum centred on the origin along `axis`."""
    r2 = r if r2 is None else r2
    r, r2, h = r + inf, r2 + inf, h + 2 * inf
    a = np.linspace(0.0, 2 * np.pi, seg + 1)
    ca, sa = np.cos(a), np.sin(a)
    sl = (r - r2) / h
    y0, y1 = -h / 2, h / 2
    P, N = [], []
    for i in range(seg):
        A0, A1 = (r * ca[i], y0, r * sa[i]), (r * ca[i + 1], y0, r * sa[i + 1])
        B0, B1 = (r2 * ca[i], y1, r2 * sa[i]), (r2 * ca[i + 1], y1, r2 * sa[i + 1])
        n0 = np.array((ca[i], sl, sa[i]))
        n0 /= np.linalg.norm(n0)
        n1 = np.array((ca[i + 1], sl, sa[i + 1]))
        n1 /= np.linalg.norm(n1)
        P += [A0, B0, B1, A0, B1, A1]
        N += [n0, n0, n1, n0, n1, n1]
        if caps:
            if r2 > 1e-6:
                P += [(0.0, y1, 0.0), B0, B1]
                N += [(0.0, 1.0, 0.0)] * 3
            if r > 1e-6:
                P += [(0.0, y0, 0.0), A0, A1]
                N += [(0.0, -1.0, 0.0)] * 3
    P, N = _fix(P, N)
    return _to_axis(P, N, axis)


def sphere(r, seg=14, rings=9, inf=0.0):
    r += inf
    P, N = [], []

    def pt(t, p):
        return np.array((math.sin(t) * math.cos(p), math.cos(t), math.sin(t) * math.sin(p)))
    for j in range(rings):
        t0, t1 = math.pi * j / rings, math.pi * (j + 1) / rings
        for i in range(seg):
            p0, p1 = 2 * math.pi * i / seg, 2 * math.pi * (i + 1) / seg
            a, b, c, d = pt(t0, p0), pt(t1, p0), pt(t1, p1), pt(t0, p1)
            for tri in ((a, b, c), (a, c, d)):
                P += [x * r for x in tri]
                N += list(tri)
    return _fix(P, N)


class Mesh:
    """Collects rigidly-skinned triangles: every vertex belongs to one bone."""

    def __init__(self):
        self.v = []
        self.o = []

    def add(self, bone, M, P, N, col, mat, O=None):
        R, t = M[:3, :3], M[:3, 3]
        Pw = P @ R.T + t
        a = np.empty((len(Pw), 14), np.float32)
        a[:, 0:3] = Pw
        a[:, 3:6] = N @ R.T
        a[:, 6:9] = col
        a[:, 9:13] = mat
        a[:, 13] = bone
        self.v.append(a)
        if O is not None:
            b = np.empty((len(O), 4), np.float32)
            b[:, 0:3] = O @ R.T + t
            b[:, 3] = bone
            self.o.append(b)

    def arrays(self):
        v = np.concatenate(self.v) if self.v else np.zeros((0, 14), np.float32)
        o = np.concatenate(self.o) if self.o else np.zeros((0, 4), np.float32)
        return v, o

# =================================================================== shaders


VS_MESH = """
#version 330
layout(std140) uniform Bones { mat4 bones[64]; };
uniform mat4 u_vp;
in vec3 in_pos; in vec3 in_nrm; in vec3 in_col; in vec4 in_mat; in float in_bone;
out vec3 v_w; out vec3 v_n; out vec3 v_c; out vec4 v_m;
void main() {
    mat4 B = bones[int(in_bone + 0.5)];
    vec4 w = B * vec4(in_pos, 1.0);
    v_w = w.xyz;
    v_n = mat3(B) * in_nrm;
    v_c = in_col;
    v_m = in_mat;
    gl_Position = u_vp * w;
}
"""

FS_MESH = """
#version 330
uniform vec3 u_cam;
uniform vec3 u_key;
uniform vec3 u_fill;
uniform vec3 u_back;
uniform vec4 u_flash;
uniform vec3 u_flashcol;
in vec3 v_w; in vec3 v_n; in vec3 v_c; in vec4 v_m;
out vec4 frag;
void main() {
    vec3 N = normalize(v_n);
    vec3 V = normalize(u_cam - v_w);
    float metal = v_m.x, gloss = v_m.y, emis = v_m.z;
    vec3 base = v_c;
    vec3 col;
    if (emis > 0.0) {
        col = base * (1.0 + 0.7 * emis);
    } else {
        float ndl = dot(N, u_key);
        float d1 = smoothstep(-0.12, 0.10, ndl);
        float d2 = smoothstep(0.55, 0.72, ndl);
        vec3 amb = mix(vec3(0.15, 0.13, 0.13), vec3(0.46, 0.50, 0.60), 0.5 + 0.5 * N.y);
        float fill = max(dot(N, u_fill), 0.0);
        vec3 diff = base * (amb * 0.90 + d1 * vec3(1.00, 0.95, 0.88) * 0.80 + d2 * 0.20
                            + fill * vec3(0.20, 0.26, 0.36));
        vec3 H = normalize(u_key + V);
        float sp = pow(max(dot(N, H), 0.0), mix(14.0, 110.0, gloss));
        float spec = smoothstep(0.20, 0.55, sp) * mix(0.10, 0.85, gloss);
        vec3 R = reflect(-V, N);
        vec3 env = mix(vec3(0.10, 0.10, 0.12), vec3(0.80, 0.84, 0.92), smoothstep(-0.30, 0.60, R.y));
        env += vec3(1.15) * (1.0 - smoothstep(0.0, 0.08, abs(R.y - 0.20)));
        vec3 metalCol = base * env * 1.20 + base * d1 * 0.25;
        col = mix(diff, metalCol, metal);
        col += spec * mix(vec3(1.0), base * 1.3 + 0.25, metal);
        float fres = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 3.0);
        float bk = clamp(dot(N, u_back) * 0.6 + 0.45, 0.0, 1.0);
        col += fres * bk * vec3(0.34, 0.42, 0.56) * 0.85;
    }
    vec3 L = u_flash.xyz - v_w;
    float dl = length(L);
    col += u_flashcol * u_flash.w * max(dot(N, L / max(dl, 1e-3)) * 0.8 + 0.2, 0.0) / (1.0 + dl * dl * 0.9);
    col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
    frag = vec4(col, 1.0);
}
"""

VS_LINE = """
#version 330
layout(std140) uniform Bones { mat4 bones[64]; };
uniform mat4 u_vp;
in vec3 in_pos; in float in_bone;
void main() { gl_Position = u_vp * (bones[int(in_bone + 0.5)] * vec4(in_pos, 1.0)); }
"""

FS_LINE = """
#version 330
uniform vec3 u_ocol;
out vec4 frag;
void main() { frag = vec4(u_ocol, 1.0); }
"""

VS_SPR = """
#version 330
uniform mat4 u_vp;
uniform vec3 u_right;
uniform vec3 u_up;
in vec2 in_corner; in vec3 i_pos; in float i_size; in vec4 i_col; in float i_glow;
out vec2 v_uv; out vec4 v_col; out float v_glow;
void main() {
    vec3 w = i_pos + (u_right * in_corner.x + u_up * in_corner.y) * i_size;
    v_uv = in_corner; v_col = i_col; v_glow = i_glow;
    gl_Position = u_vp * vec4(w, 1.0);
}
"""

FS_SPR = """
#version 330
in vec2 v_uv; in vec4 v_col; in float v_glow;
out vec4 frag;
void main() {
    float r = length(v_uv);
    if (r > 1.0) discard;
    float core = 1.0 - smoothstep(0.0, 1.0, r);
    float a = pow(core, 1.5) * v_col.a;
    vec3 c = mix(v_col.rgb, vec3(1.0), (1.0 - smoothstep(0.0, 0.45, r)) * 0.65 * v_glow);
    frag = vec4(c * a, a * (1.0 - 0.75 * v_glow));
}
"""

VS_SHD = """
#version 330
uniform mat4 u_vp;
in vec2 in_corner; in vec2 i_c; in vec2 i_r; in float i_ang; in float i_a;
out vec2 v_uv; out float v_a;
void main() {
    vec2 p = in_corner * i_r;
    float c = cos(i_ang), s = sin(i_ang);
    p = vec2(c * p.x + s * p.y, -s * p.x + c * p.y);
    v_uv = in_corner; v_a = i_a;
    gl_Position = u_vp * vec4(i_c.x + p.x, 0.003, i_c.y + p.y, 1.0);
}
"""

FS_SHD = """
#version 330
in vec2 v_uv; in float v_a;
out vec4 frag;
void main() {
    float r = length(v_uv);
    float a = v_a * (1.0 - smoothstep(0.30, 1.0, r));
    frag = vec4(0.0, 0.0, 0.0, a);
}
"""

VS_FIN = """
#version 330
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

FS_FIN = """
#version 330
uniform sampler2D tex;
out vec4 frag;
void main() {
    vec4 c = texelFetch(tex, ivec2(gl_FragCoord.xy), 0);
    c.rgb = min(c.rgb, vec3(1.0));
    float a = clamp(max(c.a, max(c.r, max(c.g, c.b))), 0.0, 1.0);
    frag = vec4(c.b, c.g, c.r, a);
}
"""


def _set(prog, name, value):
    try:
        prog[name].value = value
    except KeyError:
        pass


class Renderer:
    MAXB = 64
    SPRITE_MAX = 3000

    def __init__(self, w, h, samples=8):
        ctx = moderngl.create_standalone_context(require=330)
        self.ctx = ctx
        self.samples = int(max(0, min(samples, ctx.max_samples)))
        self.prog = ctx.program(vertex_shader=VS_MESH, fragment_shader=FS_MESH)
        self.prog_o = ctx.program(vertex_shader=VS_LINE, fragment_shader=FS_LINE)
        self.prog_s = ctx.program(vertex_shader=VS_SPR, fragment_shader=FS_SPR)
        self.prog_h = ctx.program(vertex_shader=VS_SHD, fragment_shader=FS_SHD)
        self.prog_f = ctx.program(vertex_shader=VS_FIN, fragment_shader=FS_FIN)
        self.prog['Bones'].binding = 0
        self.prog_o['Bones'].binding = 0
        self.ubo = ctx.buffer(reserve=self.MAXB * 64)
        self.ubo.bind_to_uniform_block(0)
        self.quad = ctx.buffer(np.array((-1, -1, 1, -1, -1, 1, 1, 1), 'f4').tobytes())
        self.sbuf = ctx.buffer(reserve=36 * self.SPRITE_MAX)
        self.svao = ctx.vertex_array(self.prog_s, [(self.quad, '2f', 'in_corner'),
                                                   (self.sbuf, '3f 1f 4f 1f/i', 'i_pos', 'i_size', 'i_col',
                                                    'i_glow')])
        self.hbuf = ctx.buffer(reserve=24 * 16)
        self.hvao = ctx.vertex_array(self.prog_h, [(self.quad, '2f', 'in_corner'),
                                                   (self.hbuf, '2f 2f 1f 1f/i', 'i_c', 'i_r', 'i_ang', 'i_a')])
        self.tri = ctx.buffer(np.array((-1, -1, 3, -1, -1, 3), 'f4').tobytes())
        self.fvao = ctx.vertex_array(self.prog_f, [(self.tri, '2f', 'in_pos')])
        _set(self.prog_f, 'tex', 0)
        self.size = None
        self._objs = []
        self.resize(w, h)

    def resize(self, w, h):
        if self.size == (w, h):
            return
        for o in self._objs:
            o.release()
        ctx = self.ctx
        self.rb_c = ctx.renderbuffer((w, h), 4, samples=self.samples)
        self.rb_d = ctx.depth_renderbuffer((w, h), samples=self.samples)
        self.fbo_ms = ctx.framebuffer([self.rb_c], self.rb_d)
        self.tex = ctx.texture((w, h), 4)
        self.fbo_res = ctx.framebuffer([self.tex])
        self.rb_f = ctx.renderbuffer((w, h), 4)
        self.fbo_fin = ctx.framebuffer([self.rb_f])
        self._objs = [self.fbo_ms, self.fbo_res, self.fbo_fin, self.rb_c, self.rb_d, self.tex, self.rb_f]
        self.size = (w, h)

    def upload(self, mesh):
        v, o = mesh.arrays()
        vb = self.ctx.buffer(v.tobytes())
        vao = self.ctx.vertex_array(self.prog, [(vb, '3f 3f 3f 4f 1f', 'in_pos', 'in_nrm', 'in_col', 'in_mat',
                                                 'in_bone')])
        oao = None
        if len(o):
            ob = self.ctx.buffer(o.tobytes())
            oao = self.ctx.vertex_array(self.prog_o, [(ob, '3f 1f', 'in_pos', 'in_bone')])
        return vao, oao

    def render(self, f):
        """f: dict with vp, eye, right, up, key, fill, back, flash, flashcol, bots, sprites, shadows."""
        ctx = self.ctx
        vp = np.ascontiguousarray(f['vp'].T, dtype='f4').tobytes()
        self.fbo_ms.use()
        self.fbo_ms.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.front_face = 'ccw'

        sh = f.get('shadows')
        if sh is not None and len(sh):
            sh = np.ascontiguousarray(sh[:16], dtype='f4')
            self.prog_h['u_vp'].write(vp)
            self.hbuf.orphan()
            self.hbuf.write(sh.tobytes())
            self.fbo_ms.depth_mask = False
            self.hvao.render(moderngl.TRIANGLE_STRIP, instances=len(sh))
            self.fbo_ms.depth_mask = True

        P = self.prog
        P['u_vp'].write(vp)
        _set(P, 'u_cam', tuple(f['eye']))
        _set(P, 'u_key', tuple(f['key']))
        _set(P, 'u_fill', tuple(f['fill']))
        _set(P, 'u_back', tuple(f['back']))
        _set(P, 'u_flash', tuple(f['flash']))
        _set(P, 'u_flashcol', tuple(f['flashcol']))
        self.prog_o['u_vp'].write(vp)
        ctx.enable(moderngl.CULL_FACE)
        for (vao, oao), bones, ocol in f['bots']:
            self.ubo.write(bones)
            ctx.front_face = 'ccw'
            vao.render(moderngl.TRIANGLES)
            if oao is not None:
                _set(self.prog_o, 'u_ocol', tuple(ocol))
                ctx.front_face = 'cw'      # cull the outside -> only the grown back faces show
                oao.render(moderngl.TRIANGLES)
        ctx.front_face = 'ccw'
        ctx.disable(moderngl.CULL_FACE)

        sp = f.get('sprites')
        if sp is not None and len(sp):
            sp = np.ascontiguousarray(sp[:self.SPRITE_MAX], dtype='f4')
            self.prog_s['u_vp'].write(vp)
            _set(self.prog_s, 'u_right', tuple(f['right']))
            _set(self.prog_s, 'u_up', tuple(f['up']))
            self.sbuf.orphan()
            self.sbuf.write(sp.tobytes())
            self.fbo_ms.depth_mask = False
            self.svao.render(moderngl.TRIANGLE_STRIP, instances=len(sp))
            self.fbo_ms.depth_mask = True

        ctx.copy_framebuffer(self.fbo_res, self.fbo_ms)
        self.fbo_fin.use()
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.tex.use(0)
        self.fvao.render(moderngl.TRIANGLES)

    def read_into(self, buf):
        self.fbo_fin.read_into(buf, components=4, alignment=1)

    def read_rgba(self, bg=None):
        """Top-down RGBA image (for snapshots). bg = RGB tuple to composite over."""
        w, h = self.size
        a = np.frombuffer(self.fbo_fin.read(components=4, alignment=1), np.uint8).reshape(h, w, 4)[::-1]
        rgb = a[:, :, [2, 1, 0]].astype(np.float32)
        al = a[:, :, 3:4].astype(np.float32) / 255.0
        if bg is None:
            un = np.where(al > 0, rgb / np.maximum(al, 1e-3), 0)
            return np.concatenate([np.clip(un, 0, 255), al * 255], 2).astype(np.uint8)
        out = rgb + np.array(bg, np.float32) * (1.0 - al)
        return np.concatenate([np.clip(out, 0, 255), np.full_like(al, 255)], 2).astype(np.uint8)
