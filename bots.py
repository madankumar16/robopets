"""bots.py - the two original transforming robots: BULWARK (6x6 armored truck) and VOLT (sports car).

Every visible piece is a node. In robot mode a node hangs off the skeleton (rest_t/rest_R relative to
its parent). In vehicle mode every node has an absolute pose (veh_t/veh_R/veh_s) in the vehicle frame
(forward = +z, ground = y 0). The transformation blends each node between the two, staggered by `delay`,
so the vehicle is built from exactly the same parts as the robot.

Car-panel pieces ("kibble") are authored directly in vehicle coordinates (addv/addc), core pieces in
joint-local robot coordinates (add).
"""
import math

import numpy as np

from engine3d import (Mesh, bevel, box, cyl, ext_corners, hexa, linear, m4, plate_prof, rx, ry, rz, sphere,
                      srgb, taper)

MATS = {
    # name: (srgb hex, metalness, gloss, emissive)
    'crimson': ('#b8141d', 0.15, 0.60, 0), 'crimson_dk': ('#6e0c12', 0.15, 0.45, 0),
    'graphite': ('#34383f', 0.35, 0.40, 0), 'gunmetal': ('#5d646f', 0.55, 0.55, 0),
    'steel': ('#aab2bc', 0.80, 0.75, 0), 'chrome': ('#d8dde3', 0.95, 0.90, 0),
    'black': ('#16181c', 0.20, 0.30, 0), 'rubber': ('#1c1d20', 0.00, 0.12, 0),
    'glass': ('#1b2836', 0.65, 0.95, 0), 'amber': ('#ffb627', 0, 0, 1), 'redlamp': ('#ff3030', 0, 0, 1),
    'blade': ('#ff7a1a', 0, 0, 1), 'blade_core': ('#fff1c4', 0, 0, 1),
    'teal': ('#12aac0', 0.25, 0.75, 0), 'teal_dk': ('#0b6c7b', 0.25, 0.60, 0),
    'white': ('#e6ecf0', 0.05, 0.60, 0), 'carbon': ('#24272d', 0.30, 0.45, 0),
    'magenta': ('#ff3dcf', 0, 0, 1), 'ice': ('#c9fbff', 0, 0, 1), 'tail': ('#ff2d5a', 0, 0, 1),
    'rim_v': ('#b8c4cc', 0.85, 0.85, 0), 'rim_b': ('#4a515b', 0.60, 0.60, 0),
}
_MC = {}


def material(name):
    if name not in _MC:
        h, m, g, e = MATS[name]
        _MC[name] = (np.array(linear(srgb(h)), np.float32), np.array((m, g, e, 0.0), np.float32))
    return _MC[name]


SIDES = ((1, 'L'), (-1, 'R'))
CORE = ('pelvis', 'torso', 'head', 'upper_L', 'fore_L', 'hand_L', 'thigh_L', 'shin_L', 'foot_L',
        'upper_R', 'fore_R', 'hand_R', 'thigh_R', 'shin_R', 'foot_R')


class Node:
    def __init__(self, idx, name, parent, at, R, kind):
        self.idx, self.name, self.parent, self.kind = idx, name, parent, kind
        self.rest_t = np.array(at, float)
        self.rest_R = np.eye(3) if R is None else np.array(R, float)
        self.veh_t = np.zeros(3)
        self.veh_R = np.eye(3)
        self.veh_s = 1.0
        self.delay = 0.0
        self.arc = 0.2
        self.center = np.zeros(3)


class Rig:
    def __init__(self, name, ol):
        self.name, self.ol = name, ol
        self.nodes, self.n = [], {}
        self.mesh = Mesh()

    def node(self, name, parent=None, at=(0, 0, 0), R=None, kind='core'):
        nd = Node(len(self.nodes), name, self.n[parent] if parent else None, at, R, kind)
        self.nodes.append(nd)
        self.n[name] = nd
        return nd

    def veh(self, name, t, R=None, s=1.0, delay=0.0, arc=0.15):
        nd = self.n[name]
        nd.veh_t = np.array(t, float)
        nd.veh_R = np.eye(3) if R is None else R
        nd.veh_s, nd.delay, nd.arc = s, delay, arc

    def kib(self, name, parent, center, at, R=None, delay=0.2, arc=0.35, kind='kib'):
        """Car panel authored in vehicle coordinates around `center`."""
        nd = self.node(name, parent, at, R, kind)
        nd.center = np.array(center, float)
        nd.veh_t = nd.center.copy()
        nd.delay, nd.arc = delay, arc
        return nd

    def add(self, name, prim, *a, at=(0, 0, 0), R=None, mat='graphite', ol=True, **kw):
        nd = self.n[name] if isinstance(name, str) else name
        P, N = prim(*a, **kw)
        O = prim(*a, inf=self.ol, **kw)[0] if ol else None
        col, m = material(mat)
        self.mesh.add(nd.idx, m4(R if R is not None else np.eye(3), at), P, N, col, m, O)

    def addv(self, name, prim, *a, at=(0, 0, 0), **kw):
        nd = self.n[name] if isinstance(name, str) else name
        self.add(nd, prim, *a, at=np.array(at, float) - nd.center, **kw)

    def addc(self, name, corners, mat, ol=True):
        nd = self.n[name] if isinstance(name, str) else name
        c = np.array(corners, float) - nd.center
        self.add(nd, hexa, c, mat=mat, ol=ol)

    def plate(self, name, A, B, t, x0, x1, mat, side=1, ol=True):
        """Side-profile strip (y,z points) extruded across x."""
        self.addc(name, ext_corners('x', plate_prof(A, B, t, side), x0, x1), mat, ol)


def skeleton(r, d):
    d['hip_y'] = d['ankle'] + d['shin'] + d['thigh'] + 0.06
    r.node('pelvis', None, (0, d['hip_y'], 0))
    r.node('torso', 'pelvis', (0, d['p2s'], 0))
    r.node('head', 'torso', (0, d['s2n'], 0.02))
    for s, S in SIDES:
        r.node('upper_' + S, 'torso', (s * d['shx'], d['shy'], 0))
        r.node('fore_' + S, 'upper_' + S, (0, -d['upper'], 0))
        r.node('hand_' + S, 'fore_' + S, (0, -d['fore'], 0))
        r.node('thigh_' + S, 'pelvis', (s * d['hipw'], -0.06, 0))
        r.node('shin_' + S, 'thigh_' + S, (0, -d['thigh'], 0))
        r.node('foot_' + S, 'shin_' + S, (0, -d['shin'], 0))


def wheel(r, name, rad, w, rim, offroad):
    r.add(name, cyl, rad, w, 'x', 24, mat='rubber')
    if offroad:
        n = 14
        for k in range(n):
            R = rx(360.0 * k / n)
            r.add(name, box, w * 1.05, 0.07, 2 * math.pi * rad / n * 0.5, at=R @ np.array((0, rad - 0.01, 0)),
                  R=R, mat='rubber', ol=False)
    r.add(name, cyl, rad * 0.62, w + 0.03, 'x', 18, mat=rim, ol=False)
    r.add(name, cyl, rad * 0.18, w + 0.08, 'x', 10, mat='chrome', ol=False)
    for sd in (-1, 1):
        for k in range(5):
            R = rx(72.0 * k + 36)
            r.add(name, box, 0.02, rad * 0.40, rad * 0.11, at=R @ np.array((sd * (w / 2 + 0.018), rad * 0.36, 0)),
                  R=R, mat='steel', ol=False)


def arch(r, n, zc, half, ytop, ybot, x0, x1, mat, t=0.07):
    """Wheel-arch fender seen from the side: flat top plus front/back slopes."""
    zf, zb = zc + half, zc - half
    r.plate(n, (ytop, zb), (ytop, zf), t, x0, x1, mat, side=-1)
    r.plate(n, (ytop + t, zf), (ybot, zf + half * 0.55), t, x0, x1, mat, side=-1)
    r.plate(n, (ybot, zb - half * 0.55), (ytop + t, zb), t, x0, x1, mat, side=-1)


# ============================================================================ BULWARK
def build_bulwark():
    d = dict(name='Bulwark', ankle=0.20, shin=0.84, thigh=0.78, hipw=0.31, p2s=0.22, s2n=0.98, shx=0.72,
             shy=0.80, upper=0.58, fore=0.60, ocol=(0.10, 0.03, 0.035), wheel_r=0.40,
             flame=((1.0, 0.86, 0.45), (1.0, 0.42, 0.08)), weapon='blade', length=4.2)
    r = Rig('Bulwark', 0.021)
    skeleton(r, d)
    A = r.add
    # --- pelvis
    A('pelvis', bevel, 0.74, 0.34, 0.50, 0.05, at=(0, 0.04, 0), mat='graphite')
    A('pelvis', bevel, 0.82, 0.10, 0.56, 0.03, at=(0, 0.22, 0), mat='gunmetal')
    A('pelvis', taper, 0.22, 0.10, 0.42, 0.10, 0.30, at=(0, -0.03, 0.25), R=rx(-8), mat='crimson')
    for s, S in SIDES:
        A('pelvis', sphere, 0.12, at=(s * 0.31, -0.06, 0), mat='gunmetal')
        A('pelvis', bevel, 0.08, 0.28, 0.36, 0.02, at=(s * 0.41, 0.03, 0), mat='crimson_dk')
    # --- torso
    A('torso', bevel, 0.52, 0.34, 0.40, 0.05, at=(0, 0.16, 0), mat='graphite')
    for k in range(3):
        A('torso', box, 0.42, 0.035, 0.05, at=(0, 0.06 + k * 0.09, 0.205), mat='gunmetal', ol=False)
    A('torso', bevel, 1.04, 0.60, 0.60, 0.07, at=(0, 0.62, -0.02), mat='graphite')
    A('torso', taper, 1.04, 0.56, 0.80, 0.44, 0.14, at=(0, 0.99, -0.04), mat='gunmetal')
    A('torso', box, 0.26, 0.32, 0.04, at=(0, 0.60, 0.285), mat='gunmetal')
    A('torso', box, 0.13, 0.20, 0.04, at=(0, 0.60, 0.302), mat='amber', ol=False)
    for s, S in SIDES:
        A('torso', cyl, 0.15, 0.22, 'x', at=(s * 0.60, 0.80, 0), mat='gunmetal')
    # --- head
    A('head', cyl, 0.09, 0.16, 'y', at=(0, 0.06, 0), mat='gunmetal')
    A('head', bevel, 0.34, 0.30, 0.36, 0.06, at=(0, 0.27, -0.01), mat='crimson')
    r.add('head', hexa, ext_corners('x', [(0.40, -0.22), (0.53, -0.22), (0.45, 0.13), (0.40, 0.16)], -0.035, 0.035),
          mat='graphite')
    A('head', bevel, 0.26, 0.20, 0.08, 0.025, at=(0, 0.20, 0.16), mat='steel')
    A('head', box, 0.30, 0.07, 0.03, at=(0, 0.275, 0.19), mat='black', ol=False)
    for s, S in SIDES:
        A('head', box, 0.12, 0.04, 0.02, at=(s * 0.065, 0.275, 0.207), R=rz(s * 12), mat='amber', ol=False)
        A('head', cyl, 0.065, 0.06, 'x', at=(s * 0.18, 0.26, 0), mat='steel')
        A('head', cyl, 0.025, 0.065, 'x', at=(s * 0.195, 0.26, 0), mat='amber', ol=False)
        A('head', bevel, 0.05, 0.16, 0.22, 0.015, at=(s * 0.155, 0.17, 0.05), mat='crimson_dk')
    for k in range(3):
        A('head', box, 0.12, 0.012, 0.01, at=(0, 0.15 + k * 0.03, 0.202), mat='graphite', ol=False)
    # --- limbs
    for s, S in SIDES:
        u, f, h = 'upper_' + S, 'fore_' + S, 'hand_' + S
        A(u, sphere, 0.16, at=(0, 0, 0), mat='gunmetal')
        A(u, bevel, 0.24, 0.46, 0.26, 0.04, at=(0, -0.30, 0), mat='gunmetal')
        A(u, cyl, 0.12, 0.26, 'x', at=(0, -0.58, 0), mat='graphite')
        A(f, bevel, 0.30, 0.52, 0.32, 0.05, at=(0, -0.30, 0), mat='graphite')
        A(f, bevel, 0.22, 0.38, 0.05, 0.015, at=(0, -0.30, 0.17), mat='crimson_dk')
        A(f, cyl, 0.12, 0.08, 'y', at=(0, -0.58, 0), mat='gunmetal')
        A(h, bevel, 0.22, 0.20, 0.22, 0.04, at=(0, -0.12, 0.01), mat='gunmetal')
        A(h, box, 0.23, 0.05, 0.07, at=(0, -0.08, 0.11), mat='steel', ol=False)
        A(h, bevel, 0.06, 0.11, 0.08, 0.015, at=(-s * 0.13, -0.09, 0.06), mat='gunmetal')
        t, sh, ft = 'thigh_' + S, 'shin_' + S, 'foot_' + S
        A(t, bevel, 0.36, 0.66, 0.38, 0.05, at=(0, -0.36, 0), mat='gunmetal')
        A(t, bevel, 0.30, 0.46, 0.06, 0.02, at=(0, -0.34, 0.20), mat='crimson')
        A(t, cyl, 0.13, 0.36, 'x', at=(0, -0.78, 0), mat='graphite')
        A(sh, bevel, 0.42, 0.78, 0.42, 0.05, at=(0, -0.42, 0), mat='graphite')
        A(sh, bevel, 0.34, 0.24, 0.12, 0.03, at=(0, -0.08, 0.23), R=rx(-12), mat='crimson')
        A(sh, bevel, 0.34, 0.44, 0.07, 0.02, at=(0, -0.48, 0.22), mat='crimson')
        A(sh, cyl, 0.10, 0.30, 'x', at=(0, -0.84, 0), mat='gunmetal')
        A(ft, bevel, 0.40, 0.18, 0.66, 0.04, at=(0, -0.11, 0.08), mat='graphite')
        r.add(ft, hexa, ext_corners('x', [(-0.20, 0.30), (-0.05, 0.30), (-0.12, 0.47), (-0.20, 0.47)], -0.21, 0.21),
              mat='crimson')
        A(ft, bevel, 0.36, 0.14, 0.16, 0.03, at=(0, -0.12, -0.26), mat='gunmetal')
        A(ft, cyl, 0.10, 0.03, 'y', at=(0, -0.205, 0.02), mat='black', ol=False)
        A(ft, cyl, 0.06, 0.032, 'y', at=(0, -0.206, 0.02), mat='amber', ol=False)

    # --- plasma blade (right forearm), hidden until a fight
    r.node('weapon', 'fore_R', (0, -0.52, 0.19), kind='weapon')
    A('weapon', bevel, 0.12, 0.16, 0.08, 0.02, at=(0, -0.02, 0), mat='gunmetal')
    r.add('weapon', hexa, [((0.06 if i & 1 else -0.06) * (0.25 if i & 2 else 1), (-1.40 if i & 2 else -0.08),
                            (0.025 if i & 4 else -0.025)) for i in range(8)], mat='blade', ol=False)
    A('weapon', box, 0.025, 1.0, 0.055, at=(0, -0.62, 0), mat='blade_core', ol=False)
    r.veh('weapon', (0, 1.0, 0.4), s=0.0)

    # --------------------------------------------------- vehicle panels (truck: +z forward)
    def ht(z):
        return 1.25 - (z - 1.0) * 0.093
    for s, S in SIDES:
        n = r.kib('hood_' + S, 'torso', (s * 0.33, 1.16, 1.43), at=(s * 0.37, 0.50, 0.40),
                  R=ry(s * 16) @ rx(90), delay=0.18, arc=0.5)
        r.addc(n, ext_corners('x', [(1.12, 1.00), (1.25, 1.00), (1.17, 1.86), (1.07, 1.86)], s * 0.04, s * 0.62),
               'crimson')
        r.addv(n, box, 0.10, 0.05, 0.74, at=(s * 0.20, ht(1.43) + 0.015, 1.43), R=rx(5.3), mat='crimson_dk')
        for k in range(3):
            z = 1.18 + k * 0.10
            r.addv(n, box, 0.24, 0.02, 0.05, at=(s * 0.42, ht(z) + 0.004, z), R=rx(5.3), mat='black', ol=False)
    n = r.kib('grille', 'torso', (0, 0.80, 1.95), at=(0, 0.30, -0.42), R=ry(180), delay=0.2, arc=0.4)
    r.addv(n, box, 1.20, 0.62, 0.10, at=(0, 0.84, 1.93), mat='graphite')
    for k in range(7):
        r.addv(n, box, 0.045, 0.50, 0.04, at=(-0.45 + k * 0.15, 0.84, 1.99), mat='steel', ol=False)
    r.addv(n, box, 1.26, 0.07, 0.12, at=(0, 1.15, 1.94), mat='crimson')
    for s, S in SIDES:
        r.addv(n, box, 0.24, 0.13, 0.04, at=(s * 0.46, 1.01, 2.00), mat='steel', ol=False)
        r.addv(n, box, 0.20, 0.09, 0.03, at=(s * 0.46, 1.01, 2.02), mat='amber', ol=False)
        r.addv(n, box, 0.08, 0.10, 0.08, at=(s * 0.40, 0.50, 2.13), mat='steel', ol=False)
    r.addv(n, bevel, 1.34, 0.22, 0.26, 0.04, at=(0, 0.50, 1.98), mat='gunmetal')

    # cab shell: windshield + roof + side windows + light bar -> backpack
    n = r.kib('cab', 'torso', (0, 1.58, 0.52), at=(0, 0.66, -0.86), R=rx(-90), delay=0.22, arc=0.6)
    r.plate(n, (1.25, 1.00), (1.92, 0.80), 0.05, -0.74, 0.74, 'glass', side=-1)
    for s, S in SIDES:
        r.plate(n, (1.25, 1.00), (1.92, 0.80), 0.07, s * 0.74, s * 0.87, 'crimson', side=-1)
        r.addc(n, ext_corners('x', [(1.62, 0.10), (1.90, 0.10), (1.90, 0.79), (1.62, 0.88)], s * 0.84, s * 0.87),
               'glass', ol=False)
        r.addv(n, box, 0.03, 0.30, 0.06, at=(s * 0.855, 1.76, 0.07), mat='crimson', ol=False)
    r.plate(n, (1.93, 0.86), (1.93, 0.02), 0.11, -0.88, 0.88, 'crimson', side=-1)
    r.addv(n, box, 1.74, 0.36, 0.06, at=(0, 1.77, 0.05), mat='graphite')
    r.addv(n, box, 1.20, 0.07, 0.12, at=(0, 2.08, 0.66), mat='graphite')
    for k in range(4):
        r.addv(n, box, 0.15, 0.045, 0.02, at=(-0.45 + k * 0.30, 2.08, 0.725), mat='amber', ol=False)
    r.addv(n, box, 0.40, 0.03, 0.30, at=(0, 2.055, 0.35), mat='graphite', ol=False)

    for s, S in SIDES:
        # front fenders -> shoulder pauldrons, front wheels on the shoulders
        n = r.kib('fender_' + S, 'upper_' + S, (s * 0.81, 0.86, 1.30), at=(s * 0.06, 0.06, 0.0), delay=0.08, arc=0.4)
        arch(r, n, 1.30, 0.30, 0.97, 0.64, s * 0.64, s * 0.98, 'crimson', t=0.08)
        r.addv(n, box, 0.03, 0.06, 0.12, at=(s * 0.985, 0.92, 1.66), mat='amber', ol=False)
        n = r.node('wf_' + S, 'upper_' + S, (s * 0.30, -0.16, -0.02), kind='wheel')
        wheel(r, n, 0.40, 0.30, 'rim_b', True)
        r.veh(n.name, (s * 0.80, 0.40, 1.30), delay=0.12, arc=0.3)
        # doors -> forearm shields
        n = r.kib('door_' + S, 'fore_' + S, (s * 0.88, 1.26, 0.53), at=(s * 0.20, -0.28, 0.0), R=rx(-90),
                  delay=0.12, arc=0.35)
        r.addv(n, bevel, 0.06, 0.68, 0.78, 0.02, at=(s * 0.88, 1.26, 0.53), mat='crimson')
        r.addv(n, box, 0.02, 0.06, 0.70, at=(s * 0.915, 1.32, 0.53), mat='graphite', ol=False)
        r.addv(n, box, 0.03, 0.03, 0.12, at=(s * 0.92, 1.45, 0.28), mat='steel', ol=False)
        # rear body side panels (thigh = front half, shin = back half)
        for part, par, zc, wz, dl in (('tside', 'thigh_', -0.53, -0.60, 0.30), ('sside', 'shin_', -1.47, -1.44, 0.34)):
            at = (s * 0.24, -0.36, 0.0) if par == 'thigh_' else (s * 0.25, -0.42, 0.0)
            n = r.kib(part + '_' + S, par + S, (s * 0.75, 1.12, zc), at=at, R=rx(90), delay=dl, arc=0.35)
            r.addv(n, bevel, 0.08, 0.52, 0.94, 0.02, at=(s * 0.75, 1.12, zc), mat='crimson')
            arch(r, n, wz, 0.28, 0.86, 0.66, s * 0.72, s * 0.95, 'graphite', t=0.06)
            r.addv(n, box, 0.02, 0.05, 0.10, at=(s * 0.795, 1.25, zc + 0.30), mat='amber', ol=False)
            wn = 'wm_' + S if par == 'thigh_' else 'wr_' + S
            wat = (s * 0.44, -0.42, 0.0) if par == 'thigh_' else (s * 0.46, -0.42, -0.02)
            w = r.node(wn, par + S, wat, kind='wheel')
            wheel(r, w, 0.40, 0.30, 'rim_b', True)
            r.veh(wn, (s * 0.80, 0.40, wz), delay=dl + 0.06, arc=0.3)
        # deck halves on the backs of thigh and calf
        for part, par, zc, at, dl in (('tdeck', 'thigh_', -0.53, (0, -0.36, -0.25), 0.38),
                                      ('cdeck', 'shin_', -1.47, (0, -0.44, -0.28), 0.42)):
            n = r.kib(part + '_' + S, par + S, (s * 0.32, 1.40, zc), at=at, R=rx(-90), delay=dl, arc=0.45)
            r.addv(n, bevel, 0.56, 0.06, 0.94, 0.02, at=(s * 0.32, 1.40, zc), mat='crimson')
            for xx in (0.15, 0.48):
                r.addv(n, box, 0.04, 0.03, 0.88, at=(s * xx, 1.445, zc), mat='graphite', ol=False)
        # rear face halves on the heels
        n = r.kib('heel_' + S, 'foot_' + S, (s * 0.39, 0.96, -2.00), at=(0, 0.34, -0.42), delay=0.44, arc=0.4)
        r.addv(n, bevel, 0.70, 0.84, 0.08, 0.02, at=(s * 0.39, 0.96, -1.98), mat='graphite')
        r.addv(n, box, 0.70, 0.08, 0.09, at=(s * 0.39, 1.34, -1.98), mat='crimson', ol=False)
        r.addv(n, box, 0.18, 0.09, 0.03, at=(s * 0.62, 1.22, -2.03), mat='redlamp', ol=False)
        r.addv(n, box, 0.09, 0.09, 0.03, at=(s * 0.44, 1.22, -2.03), mat='amber', ol=False)
        r.addv(n, bevel, 0.74, 0.14, 0.18, 0.03, at=(s * 0.39, 0.56, -2.02), mat='gunmetal')

    # --- where the robot core goes inside the truck
    V = r.veh
    V('pelvis', (0, 0.80, -0.20), rx(90), delay=0.25)
    V('torso', (0, 0.78, 0.05), rx(90), s=0.9, delay=0.15)
    V('head', (0, 0.86, 1.05), rx(90), s=0.85, delay=0.0)
    for s, S in SIDES:
        V('upper_' + S, (s * 0.48, 0.92, 0.66), rx(-90), s=0.9, delay=0.05)
        V('fore_' + S, (s * 0.62, 1.60, 0.55), s=0.95, delay=0.07)
        V('hand_' + S, (s * 0.55, 1.05, 0.30), s=0.8, delay=0.09)
        V('thigh_' + S, (s * 0.36, 0.80, -0.16), rx(90), delay=0.28)
        V('shin_' + S, (s * 0.36, 0.82, -0.98), rx(90), delay=0.32)
        V('foot_' + S, (s * 0.36, 0.84, -1.80), rx(90), delay=0.36)
    return r, d


# ============================================================================ VOLT
def build_volt():
    d = dict(name='Volt', ankle=0.15, shin=0.70, thigh=0.66, hipw=0.22, p2s=0.16, s2n=0.78, shx=0.52, shy=0.62,
             upper=0.48, fore=0.48, ocol=(0.02, 0.07, 0.09), wheel_r=0.30,
             flame=((0.75, 0.97, 1.0), (0.20, 0.55, 1.0)), weapon='blaster', length=3.0)
    r = Rig('Volt', 0.017)
    skeleton(r, d)
    A = r.add
    A('pelvis', bevel, 0.50, 0.24, 0.34, 0.04, at=(0, 0.03, 0), mat='carbon')
    A('pelvis', bevel, 0.54, 0.07, 0.36, 0.02, at=(0, 0.15, 0), mat='gunmetal')
    A('pelvis', taper, 0.14, 0.08, 0.30, 0.08, 0.22, at=(0, -0.02, 0.18), R=rx(-8), mat='white')
    for s, S in SIDES:
        A('pelvis', sphere, 0.085, at=(s * 0.22, -0.05, 0), mat='gunmetal')
        A('pelvis', bevel, 0.06, 0.20, 0.26, 0.015, at=(s * 0.29, 0.02, 0), mat='teal')
    A('torso', bevel, 0.34, 0.26, 0.28, 0.04, at=(0, 0.12, 0), mat='carbon')
    for k in range(3):
        A('torso', box, 0.28, 0.025, 0.04, at=(0, 0.05 + k * 0.07, 0.145), mat='gunmetal', ol=False)
    A('torso', bevel, 0.74, 0.46, 0.44, 0.06, at=(0, 0.48, -0.01), mat='carbon')
    A('torso', taper, 0.74, 0.40, 0.56, 0.32, 0.10, at=(0, 0.76, -0.03), mat='gunmetal')
    A('torso', box, 0.18, 0.22, 0.03, at=(0, 0.46, 0.21), mat='gunmetal')
    A('torso', box, 0.09, 0.13, 0.03, at=(0, 0.46, 0.225), mat='ice', ol=False)
    for s, S in SIDES:
        A('torso', cyl, 0.10, 0.16, 'x', at=(s * 0.44, 0.62, 0), mat='gunmetal')
    A('head', cyl, 0.07, 0.12, 'y', at=(0, 0.05, 0), mat='gunmetal')
    A('head', bevel, 0.27, 0.24, 0.28, 0.06, at=(0, 0.21, -0.01), mat='white')
    r.add('head', hexa, ext_corners('x', [(0.31, -0.12), (0.36, -0.15), (0.41, 0.17), (0.32, 0.09)], -0.025, 0.025),
          mat='teal')
    A('head', bevel, 0.20, 0.13, 0.07, 0.02, at=(0, 0.15, 0.12), mat='carbon')
    A('head', box, 0.25, 0.06, 0.03, at=(0, 0.215, 0.135), mat='black', ol=False)
    A('head', box, 0.22, 0.035, 0.02, at=(0, 0.215, 0.152), mat='magenta', ol=False)
    for s, S in SIDES:
        A('head', box, 0.025, 0.035, 0.09, at=(s * 0.128, 0.215, 0.10), mat='magenta', ol=False)
        A('head', bevel, 0.03, 0.12, 0.16, 0.01, at=(s * 0.145, 0.15, 0.02), R=rx(-15), mat='teal')
    for s, S in SIDES:
        u, f, h = 'upper_' + S, 'fore_' + S, 'hand_' + S
        A(u, sphere, 0.10, mat='gunmetal')
        A(u, bevel, 0.17, 0.38, 0.18, 0.03, at=(0, -0.24, 0), mat='gunmetal')
        A(u, cyl, 0.085, 0.20, 'x', at=(0, -0.48, 0), mat='carbon')
        A(f, bevel, 0.21, 0.42, 0.22, 0.035, at=(0, -0.24, 0), mat='carbon')
        A(f, box, 0.15, 0.30, 0.03, at=(0, -0.24, 0.115), mat='teal_dk', ol=False)
        A(f, cyl, 0.085, 0.06, 'y', at=(0, -0.47, 0), mat='gunmetal')
        A(h, bevel, 0.16, 0.15, 0.16, 0.03, at=(0, -0.09, 0.01), mat='gunmetal')
        A(h, box, 0.17, 0.035, 0.05, at=(0, -0.06, 0.08), mat='white', ol=False)
        t, sh, ft = 'thigh_' + S, 'shin_' + S, 'foot_' + S
        A(t, bevel, 0.26, 0.56, 0.28, 0.04, at=(0, -0.30, 0), mat='white')
        A(t, box, 0.06, 0.40, 0.02, at=(0, -0.30, 0.145), mat='teal', ol=False)
        A(t, cyl, 0.095, 0.26, 'x', at=(0, -0.66, 0), mat='carbon')
        A(sh, bevel, 0.30, 0.64, 0.30, 0.04, at=(0, -0.35, 0), mat='carbon')
        A(sh, bevel, 0.24, 0.18, 0.09, 0.02, at=(0, -0.06, 0.17), R=rx(-12), mat='teal')
        A(sh, bevel, 0.24, 0.38, 0.05, 0.015, at=(0, -0.38, 0.16), mat='teal')
        A(sh, cyl, 0.075, 0.22, 'x', at=(0, -0.70, 0), mat='gunmetal')
        A(ft, bevel, 0.28, 0.13, 0.50, 0.03, at=(0, -0.085, 0.07), mat='carbon')
        r.add(ft, hexa, ext_corners('x', [(-0.15, 0.22), (-0.04, 0.22), (-0.09, 0.37), (-0.15, 0.37)], -0.15, 0.15),
              mat='teal')
        A(ft, bevel, 0.24, 0.10, 0.12, 0.02, at=(0, -0.09, -0.19), mat='gunmetal')
        A(ft, cyl, 0.07, 0.03, 'y', at=(0, -0.152, 0.02), mat='black', ol=False)
        A(ft, cyl, 0.045, 0.032, 'y', at=(0, -0.153, 0.02), mat='ice', ol=False)

    # --- pulse blaster on the LEFT forearm (faces the camera when Volt fights)
    r.node('weapon', 'fore_L', (0, -0.30, 0.15), kind='weapon')
    A('weapon', bevel, 0.16, 0.30, 0.12, 0.025, at=(0, -0.05, 0), mat='gunmetal')
    for xx in (-0.04, 0.04):
        A('weapon', cyl, 0.032, 0.42, 'y', at=(xx, -0.36, 0.02), mat='carbon')
        A('weapon', cyl, 0.026, 0.02, 'y', at=(xx, -0.575, 0.02), mat='magenta', ol=False)
    A('weapon', box, 0.10, 0.18, 0.03, at=(0, -0.05, 0.065), mat='teal', ol=False)
    r.veh('weapon', (0, 0.4, 0.2), s=0.0)

    def hy(z):
        return 0.65 - (z - 0.46) * (0.12 / 0.72)
    for s, S in SIDES:
        n = r.kib('hood_' + S, 'torso', (s * 0.255, 0.56, 0.82), at=(s * 0.235, 0.38, 0.30),
                  R=ry(s * 18) @ rx(90), delay=0.18, arc=0.45)
        r.addc(n, ext_corners('x', [(0.58, 0.46), (0.65, 0.46), (0.53, 1.18), (0.47, 1.18)], s * 0.03, s * 0.48),
               'teal')
        r.addv(n, box, 0.18, 0.02, 0.16, at=(s * 0.28, hy(0.72) + 0.004, 0.72), R=rx(9.5), mat='black', ol=False)
    n = r.kib('nose', 'torso', (0, 0.33, 1.33), at=(0, 0.26, -0.34), R=ry(180), delay=0.2, arc=0.4)
    r.addc(n, ext_corners('x', [(0.14, 1.16), (0.53, 1.16), (0.44, 1.44), (0.16, 1.52)], -0.60, 0.60), 'teal')
    r.addv(n, box, 1.16, 0.04, 0.20, at=(0, 0.13, 1.44), mat='carbon')
    r.addv(n, box, 0.56, 0.12, 0.04, at=(0, 0.27, 1.495), R=rx(-16), mat='black', ol=False)
    for s, S in SIDES:
        r.addv(n, box, 0.32, 0.05, 0.06, at=(s * 0.38, 0.455, 1.425), R=rx(-17) @ rz(-s * 5), mat='black', ol=False)
        r.addv(n, box, 0.28, 0.025, 0.05, at=(s * 0.38, 0.47, 1.43), R=rx(-17) @ rz(-s * 5), mat='ice', ol=False)
    n = r.kib('canopy', 'torso', (0, 0.80, 0.0), at=(0, 0.50, -0.62), R=rx(-90), delay=0.22, arc=0.55)
    r.plate(n, (0.64, 0.48), (0.95, -0.05), 0.04, -0.46, 0.46, 'glass', side=-1)
    for s, S in SIDES:
        r.plate(n, (0.64, 0.48), (0.95, -0.05), 0.05, s * 0.46, s * 0.53, 'teal', side=-1)
        r.addc(n, ext_corners('x', [(0.62, -0.42), (0.92, -0.40), (0.93, -0.08), (0.64, 0.40)], s * 0.50, s * 0.53),
               'glass', ol=False)
    r.plate(n, (0.96, -0.03), (0.95, -0.52), 0.05, -0.50, 0.50, 'carbon', side=-1)
    for s, S in SIDES:
        n = r.kib('fender_' + S, 'upper_' + S, (s * 0.555, 0.52, 0.95), at=(s * 0.04, 0.05, 0.0), delay=0.08, arc=0.35)
        arch(r, n, 0.95, 0.23, 0.62, 0.42, s * 0.46, s * 0.66, 'teal', t=0.06)
        n = r.node('wf_' + S, 'upper_' + S, (s * 0.21, -0.12, -0.02), kind='wheel')
        wheel(r, n, 0.30, 0.24, 'rim_v', False)
        r.veh(n.name, (s * 0.57, 0.30, 0.95), delay=0.12, arc=0.25)
        n = r.kib('door_' + S, 'fore_' + S, (s * 0.62, 0.40, 0.05), at=(s * 0.135, -0.24, 0.0), R=rx(-90),
                  delay=0.12, arc=0.3)
        r.addv(n, bevel, 0.05, 0.46, 0.74, 0.02, at=(s * 0.62, 0.40, 0.05), mat='teal')
        r.addv(n, box, 0.012, 0.05, 0.66, at=(s * 0.648, 0.46, 0.05), mat='white', ol=False)
        r.addv(n, bevel, 0.10, 0.05, 0.08, 0.015, at=(s * 0.69, 0.60, 0.36), mat='teal')
        n = r.kib('skirt_' + S, 'thigh_' + S, (s * 0.62, 0.15, 0.10), at=(s * 0.155, -0.30, 0.04), R=rx(90),
                  delay=0.30, arc=0.3)
        r.addv(n, bevel, 0.08, 0.09, 0.62, 0.02, at=(s * 0.62, 0.15, 0.10), mat='carbon')
        n = r.kib('quarter_' + S, 'shin_' + S, (s * 0.60, 0.44, -0.80), at=(s * 0.19, -0.36, 0.0), R=rx(90),
                  delay=0.34, arc=0.35)
        arch(r, n, -0.92, 0.23, 0.62, 0.42, s * 0.50, s * 0.66, 'teal', t=0.06)
        r.addv(n, bevel, 0.09, 0.40, 0.22, 0.02, at=(s * 0.60, 0.40, -0.48), mat='teal')
        r.addv(n, box, 0.02, 0.24, 0.13, at=(s * 0.648, 0.44, -0.48), mat='black', ol=False)
        r.addv(n, bevel, 0.09, 0.30, 0.10, 0.02, at=(s * 0.60, 0.48, -1.22), mat='teal')
        n = r.node('wr_' + S, 'shin_' + S, (s * 0.32, -0.40, -0.03), kind='wheel')
        wheel(r, n, 0.30, 0.24, 'rim_v', False)
        r.veh(n.name, (s * 0.57, 0.30, -0.92), delay=0.40, arc=0.25)
        n = r.kib('engine_' + S, 'shin_' + S, (s * 0.25, 0.84, -0.87), at=(0, -0.36, -0.20), R=rx(-90),
                  delay=0.40, arc=0.45)
        r.plate(n, (0.95, -0.50), (0.74, -1.24), 0.05, s * 0.03, s * 0.48, 'teal', side=-1)
        for k in range(4):
            z = -0.66 - k * 0.15
            y = 0.95 + (z + 0.50) * (0.21 / 0.74) + 0.012
            r.addv(n, box, 0.36, 0.015, 0.05, at=(s * 0.25, y, z), R=rx(-16), mat='black', ol=False)
        n = r.kib('tail_' + S, 'foot_' + S, (s * 0.33, 0.48, -1.34), at=(0, 0.30, -0.30), delay=0.44, arc=0.4)
        r.addc(n, ext_corners('x', [(0.16, -1.22), (0.74, -1.22), (0.72, -1.36), (0.16, -1.44)], s * 0.03, s * 0.62),
               'teal')
        r.addv(n, box, 0.50, 0.04, 0.03, at=(s * 0.33, 0.66, -1.385), R=rx(-8), mat='tail', ol=False)
        r.addv(n, box, 0.56, 0.08, 0.16, at=(s * 0.33, 0.15, -1.36), mat='carbon')
        r.addv(n, bevel, 0.58, 0.035, 0.20, 0.01, at=(s * 0.33, 0.88, -1.36), R=rx(-6), mat='carbon')
        r.addv(n, box, 0.03, 0.14, 0.06, at=(s * 0.30, 0.80, -1.32), mat='carbon', ol=False)
        r.addv(n, box, 0.02, 0.12, 0.20, at=(s * 0.61, 0.86, -1.36), mat='teal')

    V = r.veh
    V('pelvis', (0, 0.40, -0.28), rx(90), delay=0.25)
    V('torso', (0, 0.34, 0.02), rx(90), s=0.88, delay=0.15)
    V('head', (0, 0.34, 0.85), rx(90), s=0.8, delay=0.0)
    for s, S in SIDES:
        V('upper_' + S, (s * 0.36, 0.36, 0.30), rx(-90), s=0.9, delay=0.05)
        V('fore_' + S, (s * 0.42, 0.62, 0.05), s=0.9, delay=0.07)
        V('hand_' + S, (s * 0.40, 0.26, -0.10), s=0.8, delay=0.09)
        V('thigh_' + S, (s * 0.20, 0.32, -0.38), rx(90), delay=0.28)
        V('shin_' + S, (s * 0.25, 0.56, -1.18), rx(-90), s=0.85, delay=0.32)
        V('foot_' + S, (s * 0.26, 0.32, -1.10), rx(90), s=0.8, delay=0.36)
    return r, d
