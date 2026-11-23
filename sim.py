"""sim.py - animation, behaviour, camera and effects for RoboPets (no Windows code here)."""
import math
import random

import numpy as np

import bots
from bots import CORE, SIDES
from engine3d import clamp, eul, look_at, m3_from_quat, m4, perspective, quat_from_m3, rx, ry, slerp, smooth01

SPAN = 0.55          # share of the transform each piece spends moving
HOME_A = (-2.15, 0.0, -0.75)
HOME_B = (1.95, 0.0, 0.75)
VIEWS = (('front', 'Front'), ('hero', '3/4 hero'), ('back', 'Back'), ('left', 'Left side'),
         ('right', 'Right side'), ('top', 'Top (looking down)'), ('bottom', 'Bottom (looking up)'))
PRESETS = {'front': (0, 7), 'hero': (30, 11), 'back': (180, 9), 'left': (-90, 8), 'right': (90, 8),
           'top': (0, 86), 'bottom': (0, -42)}


def v(x=0.0, y=0.0, z=0.0):
    return np.array((x, y, z), float)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def env(u, a, b):
    return smooth01((u - a) / (b - a)) if b > a else float(u >= a)


def kf(u, keys):
    if u <= keys[0][0]:
        return np.array(keys[0][1], float)
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if u <= t1:
            w = smooth01((u - t0) / (t1 - t0))
            return np.array(v0, float) * (1 - w) + np.array(v1, float) * w
    return np.array(keys[-1][1], float)


# =========================================================================== actor
class Actor:
    def __init__(self, rig, d, home, seed):
        self.rig, self.d, self.seed = rig, d, seed
        self.home = np.array(home, float)
        self.pos = self.home.copy()
        self.pos_t = self.home.copy()
        self.yaw = self.yaw_t = 0.0
        self.pitch = self.pitch_t = self.roll = self.roll_t = 0.0
        self.form = self.form_t = 0.0
        self.weapon = self.weapon_t = 0.0
        self.thrust = self.thrust_t = 0.0
        self.ang = {n: np.zeros(3) for n in CORE}
        self.tgt = {n: np.zeros(3) for n in CORE}
        self.rate = 12.0
        self.move_rate = 9.0
        self.air = False
        self.phase = 0.0
        self.spin = 0.0
        self.bounce = 0.0
        self.veh_yaw = 90.0
        self.land_t = 0.0
        self.gesture, self.gt = None, 0.0
        self.look = np.zeros(2)
        self.handle = None
        self.n = len(rig.nodes)
        self.M = [np.eye(4)] * self.n
        self.W = np.eye(4)
        self.bones = np.zeros((64, 4, 4), np.float32)
        self.vq = [quat_from_m3(nd.veh_R) for nd in rig.nodes]
        self.idx = {nd.name: nd.idx for nd in rig.nodes}
        self.stride = (d['thigh'] + d['shin']) * 1.25
        self.size = d['hip_y'] / 1.9

    # ---------------------------------------------------------------- poses
    def pose_rest(self):
        for nm in CORE:
            self.tgt[nm] = v()
        for s, S in SIDES:
            self.tgt['upper_' + S] = v(0, 0, s * 6)
            self.tgt['fore_' + S] = v(-8)

    def pose_idle(self, t):
        T, b = self.tgt, math.sin(t * 1.7 + self.seed)
        T['pelvis'] = v()
        T['torso'] = v(1.0 + 1.2 * b)
        T['head'] = v(self.look[1], self.look[0])
        for s, S in SIDES:
            T['upper_' + S] = v(3, 0, s * (8 + 1.5 * b))
            T['fore_' + S] = v(-14)
            T['hand_' + S] = v()
            T['thigh_' + S] = v(-2, 0, s * 3)
            T['shin_' + S] = v(4)
            T['foot_' + S] = v(-2, 0, -s * 3)

    def pose_walk(self, k=1.0):
        T, ph = self.tgt, self.phase
        for s, S, off in ((1, 'L', 0.0), (-1, 'R', math.pi)):
            p = ph + off
            th, sh = -26 * k * math.sin(p), 8 + 46 * k * max(0.0, math.cos(p))
            T['thigh_' + S] = v(th, 0, s * 2)
            T['shin_' + S] = v(sh)
            T['foot_' + S] = v(-(th + sh) * 0.85)
            T['upper_' + S] = v(22 * k * math.sin(p), 0, s * 8)
            T['fore_' + S] = v(-22 - 14 * k * (0.5 + 0.5 * math.sin(p)))
            T['hand_' + S] = v()
        T['torso'] = v(4 * k, 5 * k * math.sin(ph))
        T['pelvis'] = v(0, -4 * k * math.sin(ph))
        T['head'] = v(-2, -4 * k * math.sin(ph))

    def pose_air(self, t, rising):
        T = self.tgt
        for s, S in SIDES:
            if rising:
                T['thigh_' + S] = v(-45 + s * 8, 0, s * 8)
                T['shin_' + S] = v(80)
                T['foot_' + S] = v(25)
                T['upper_' + S] = v(-150 + 12 * math.sin(t * 11 + s), 0, s * 22)
                T['fore_' + S] = v(-20)
            else:
                T['thigh_' + S] = v(-12 + s * 10 * math.sin(t * 9), 0, s * 9)
                T['shin_' + S] = v(25)
                T['foot_' + S] = v(10)
                T['upper_' + S] = v(-20 + 25 * math.sin(t * 13 + s), 0, s * 70)
                T['fore_' + S] = v(-30)
        T['torso'] = v(-4)
        T['head'] = v(-8)
        T['pelvis'] = v()

    def pose_land(self, k):
        T = self.tgt
        for s, S in SIDES:
            T['thigh_' + S] = v(-42 * k, 0, s * 5)
            T['shin_' + S] = v(80 * k)
            T['foot_' + S] = v(-38 * k)
            T['upper_' + S] = v(-25 * k, 0, s * (8 + 25 * k))
            T['fore_' + S] = v(-20 - 30 * k)
        T['torso'] = v(18 * k)
        T['head'] = v(-10 * k)

    def pose_fly(self, t):
        T, w = self.tgt, math.sin(t * 2.3 + self.seed)
        T['thigh_L'], T['shin_L'], T['foot_L'] = v(-28 + 5 * w, 0, 4), v(38), v(30)
        T['thigh_R'], T['shin_R'], T['foot_R'] = v(6 - 5 * w, 0, -4), v(22), v(34)
        for s, S in SIDES:
            T['upper_' + S] = v(20 + 4 * w, 0, s * 26)
            T['fore_' + S] = v(-28)
        T['torso'] = v(8)
        T['head'] = v(-12, self.look[0] * 0.5)
        T['pelvis'] = v()

    def pose_stance(self, t):
        T = self.tgt
        bob = math.sin(t * 7 + self.seed)
        if self.d['weapon'] == 'blade':
            T['thigh_L'], T['shin_L'], T['foot_L'] = v(-24, 0, 6), v(34), v(-10)
            T['thigh_R'], T['shin_R'], T['foot_R'] = v(14, 0, -6), v(28), v(-40)
            T['torso'] = v(6, -10)
            T['upper_R'], T['fore_R'] = v(-55, 0, -10), v(-62)
            T['upper_L'], T['fore_L'] = v(-38, 0, 14), v(-95)
            T['head'] = v(-4, 10)
        else:
            T['thigh_L'], T['shin_L'], T['foot_L'] = v(-18 + 3 * bob, 0, 6), v(32 + 4 * bob), v(-14)
            T['thigh_R'], T['shin_R'], T['foot_R'] = v(16, 0, -6), v(30 + 4 * bob), v(-46)
            T['torso'] = v(5, 14)
            T['upper_L'], T['fore_L'] = v(-50, 0, 10), v(-40)
            T['upper_R'], T['fore_R'] = v(-32, 0, -10), v(-105)
            T['head'] = v(-4, -14)
        T['pelvis'] = v()
        T['hand_L'] = T['hand_R'] = v()

    def pose_gesture(self, t):
        g, u, T = self.gesture, self.gt, self.tgt
        k = env(u, 0, 0.3) * (1 - env(u, 1.5, 1.9))
        if g == 'wave':
            T['upper_R'] = v(-160 * k, 0, -18 * k - 8)
            T['fore_R'] = v(-25 * k + 28 * k * math.sin(t * 11))
            T['head'] = v(-6 * k, 0, 8 * k)
        elif g == 'flex':
            for s, S in SIDES:
                T['upper_' + S] = v(-10 * k, 0, s * (8 + 80 * k))
                T['fore_' + S] = v(-14 - 100 * k)
            T['torso'] = v(-8 * k)
        elif g == 'nod':
            T['head'] = v(18 * k * max(0.0, math.sin(t * 9)))
        elif g == 'salute':
            T['upper_R'] = v(-115 * k, 0, -40 * k - 8)
            T['fore_R'] = v(-14 - 120 * k)
            T['head'] = v(-5 * k)
        elif g == 'stretch':
            for s, S in SIDES:
                T['upper_' + S] = v(-170 * k, 0, s * (8 + 15 * k))
                T['fore_' + S] = v(-14 + 10 * k)
            T['torso'] = v(-10 * k)

    # ---------------------------------------------------------------- simulation
    def update(self, dt):
        self.form += clamp(self.form_t - self.form, -dt / 1.7, dt / 1.7)
        self.weapon += clamp(self.weapon_t - self.weapon, -dt * 3, dt * 3)
        self.thrust += (self.thrust_t - self.thrust) * min(1.0, dt * 6)
        k = 1 - math.exp(-self.rate * dt)
        for nm in CORE:
            self.ang[nm] += (self.tgt[nm] - self.ang[nm]) * k
        self.pos += (self.pos_t - self.pos) * min(1.0, dt * self.move_rate)
        ky = min(1.0, dt * 7)
        self.yaw += wrap(self.yaw_t - self.yaw) * ky
        self.pitch += (self.pitch_t - self.pitch) * ky
        self.roll += (self.roll_t - self.roll) * ky
        if self.land_t > 0:
            self.land_t -= dt
        if self.gesture:
            self.gt += dt
            if self.gt > 1.9:
                self.gesture = None

    def compute(self):
        rig, d = self.rig, self.d
        hip = d['hip_y']
        if not self.air:
            Rp = eul(*self.ang['pelvis'])
            ys = []
            for S in 'LR':
                th, sh, ft = rig.n['thigh_' + S], rig.n['shin_' + S], rig.n['foot_' + S]
                p = th.rest_t + eul(*self.ang['thigh_' + S]) @ (sh.rest_t + eul(*self.ang['shin_' + S]) @ ft.rest_t)
                ys.append((Rp @ p)[1])
            hip = d['ankle'] - min(ys)
        M = self.M = [None] * self.n
        for nd in rig.nodes:
            if nd.kind == 'core':
                t = (0.0, hip, 0.0) if nd.name == 'pelvis' else nd.rest_t
                L = m4(nd.rest_R @ eul(*self.ang[nd.name]), t)
            elif nd.kind == 'weapon':
                w = self.weapon
                L = m4(nd.rest_R, nd.rest_t + np.array((0.0, 0.22 * (1 - w), 0.0)), max(w, 1e-4))
            else:
                L = m4(nd.rest_R, nd.rest_t)
            M[nd.idx] = M[nd.parent.idx] @ L if nd.parent else L
        p = self.form
        if p > 0.001:
            bounce = m4(rx(self.bounce * 0.6), (0.0, self.bounce * 0.05, 0.0))
            for nd in rig.nodes:
                w = smooth01((p - nd.delay) / SPAN)
                if w <= 0.0:
                    continue
                if nd.kind == 'wheel':
                    Rv = nd.veh_R @ rx(self.spin)
                    V = m4(Rv, nd.veh_t, nd.veh_s)
                    qb = quat_from_m3(Rv)
                else:
                    V = bounce @ m4(nd.veh_R, nd.veh_t, nd.veh_s)
                    qb = self.vq[nd.idx]
                if w >= 1.0:
                    M[nd.idx] = V
                    continue
                A = M[nd.idx]
                sa = float(np.linalg.norm(A[:3, 0]))
                qa = quat_from_m3(A[:3, :3] / sa) if sa > 1e-5 else qb
                q = slerp(qa, qb, w)
                tr = A[:3, 3] * (1 - w) + V[:3, 3] * w
                tr[1] += nd.arc * math.sin(math.pi * w)
                M[nd.idx] = m4(m3_from_quat(q), tr, sa + (nd.veh_s - sa) * w)
        hc = hip
        W = (m4(ry(self.yaw), self.pos) @ m4(None, (0, hc, 0)) @ m4(eul(self.pitch, 0, self.roll))
             @ m4(None, (0, -hc, 0)))
        self.W = W
        for i in range(self.n):
            self.bones[i] = (W @ M[i]).T

    def point(self, name, local=(0, 0, 0)):
        i = self.idx[name]
        return (self.bones[i].T @ np.array((*local, 1.0)))[:3]

    def axis(self, name, ax):
        return self.bones[self.idx[name]].T[:3, :3] @ np.array(ax, float)

    def head_pos(self):
        return self.point('head', (0, 0.25, 0))


# =========================================================================== camera
class Camera:
    def __init__(self):
        self.fov = 30.0
        self.az, self.el = PRESETS['front']
        self.goal = list(PRESETS['front'])
        self.dist = self.dist_goal = 10.6
        self.ty = 1.6
        self.preset = 'front'

    def set(self, name):
        if name in PRESETS:
            self.preset = name
            self.goal = list(PRESETS[name])
            self.dist_goal = {'hero': 12.6, 'left': 11.5, 'right': 11.5, 'bottom': 11.5}.get(name, 10.6)

    def orbit(self, daz, delv):
        self.preset = 'custom'
        self.goal[0] = wrap(self.goal[0] + daz)
        self.goal[1] = clamp(self.goal[1] + delv, -80, 88)

    def update(self, dt):
        k = min(1.0, dt * 5)
        self.az += wrap(self.goal[0] - self.az) * k
        self.el += (self.goal[1] - self.el) * k
        self.dist += (self.dist_goal - self.dist) * k

    def matrices(self, aspect, az=None, el=None, shake=(0, 0, 0)):
        az = math.radians(self.az if az is None else az)
        el = math.radians(self.el if el is None else el)
        tgt = np.array((0.0, self.ty, 0.0)) + np.array(shake)
        eye = tgt + self.dist * np.array((math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)))
        up = np.array((-math.sin(az) * math.sin(el), math.cos(el), -math.cos(az) * math.sin(el)))
        V = look_at(eye, tgt, up)
        P = perspective(self.fov, aspect, 0.5, 60.0)
        return P @ V, V, eye

    def calibrate(self, aspect, row=0.89):
        lo, hi = -2.0, 6.0
        for _ in range(40):
            self.ty = (lo + hi) / 2
            VP, _, _ = self.matrices(aspect, *PRESETS['front'])
            p = VP @ np.array((0, 0, 0, 1.0))
            r = (1 - p[1] / p[3]) / 2
            if r < row:
                lo = self.ty
            else:
                hi = self.ty


# =========================================================================== world
class World:
    """Both robots plus everything they do."""

    def __init__(self, renderer=None):
        rb, db = bots.build_bulwark()
        rv, dv = bots.build_volt()
        self.A = Actor(rb, db, HOME_A, 0.0)
        self.B = Actor(rv, dv, HOME_B, 2.1)
        self.actors = (self.A, self.B)
        if renderer:
            for a in self.actors:
                a.handle = renderer.upload(a.rig.mesh)
        self.cam = Camera()
        self.parts = []
        self.t = 0.0
        self.mode, self.sub, self.mt = 'idle', '', 0.0
        self.flash, self.flash_pos, self.flash_col = 0.0, np.zeros(3), np.array((1.0, 0.8, 0.5))
        self.shake = 0.0
        self.sounds = []
        self.toast = None
        self.fight_hold = False
        self.fight_timer = 0.0
        self.flying = False
        self.travel = None
        self.drag = None
        self.falling = False
        self.on_ground = True
        self.wander = True
        self.next_auto = 9.0
        self.idle_time = 0.0
        self.ppu = 80.0
        self.cursor = None
        self.move = None
        self.fired = set()
        self.bolts = []

    # ---------------------------------------------------------------- user actions
    def say(self, text):
        self.toast = [text, 1.6]

    def sound(self, name):
        self.sounds.append(name)

    def busy(self):
        return self.mode != 'idle'

    def transform(self, a=None, user=True):
        if self.mode in ('fight', 'fly'):
            return
        for x in (self.actors if a is None else (a,)):
            x.form_t = 0.0 if x.form_t > 0.5 else 1.0
            x.weapon_t = 0.0
            if x.form_t > 0.5:
                x.veh_yaw = 90.0
        if user:
            self.sound('transform')
            self.idle_time = 0

    def toggle_fly(self):
        if self.mode == 'fly':
            if self.sub != 'landing':
                self.sub, self.mt = 'landing', 0.0
                self.flying = False
            return
        if self.mode == 'fight':
            return
        self.end_travel()
        self.mode, self.sub, self.mt = 'fly', 'prep', 0.0
        self.flying = True
        self.say('Taking off!')

    def start_fight(self, seconds=None):
        if self.mode == 'fight':
            self.fight_hold = True
            return
        self.end_travel()
        if self.mode == 'fly':
            self.flying = False
        self.mode, self.sub, self.mt = 'fight', 'prep', 0.0
        self.fight_hold = True
        self.fight_timer = seconds or 0.0
        self.move = None
        self.say('Fight!')

    def stop_fight(self):
        self.fight_hold = False

    def click(self, a):
        if self.mode != 'idle' or a.form_t > 0.5 or a.form > 0.02:
            return
        g = random.choice(('wave', 'flex', 'nod', 'salute', 'stretch' if a is self.A else 'wave'))
        a.gesture, a.gt = g, 0.0
        self.idle_time = 0

    def end_travel(self):
        if self.mode == 'travel':
            self.travel = None
            self.mode = 'idle'
            for a in self.actors:
                if a.form_t > 0.5:
                    a.form_t = 0.0

    # ---------------------------------------------------------------- effects
    def particle(self, pos, vel, life, s0, s1, c0, c1, glow=1.0, grav=0.0, drag=0.0):
        if len(self.parts) < 1500:
            self.parts.append([np.array(pos, float), np.array(vel, float), life, life, s0, s1, c0, c1, glow, grav, drag])

    def sparks(self, pos, n=24, col=((1, 0.95, 0.7), (1, 0.45, 0.1)), speed=4.0):
        for _ in range(n):
            d = np.random.normal(size=3)
            d /= np.linalg.norm(d) + 1e-9
            self.particle(pos, d * speed * random.uniform(0.3, 1.0) + (0, 1.0, 0), random.uniform(0.25, 0.6),
                          0.07, 0.02, (*col[0], 1.0), (*col[1], 0.0), 1.0, -9.0, 1.5)
        self.particle(pos, (0, 0, 0), 0.18, 0.55, 0.9, (1, 0.95, 0.8, 0.9), (*col[1], 0.0), 1.0)

    def impact(self, pos, col=(1.0, 0.75, 0.4), k=1.0, shake=0.12):
        self.flash, self.flash_pos, self.flash_col = 2.5 * k, np.array(pos), np.array(col)
        self.shake = max(self.shake, shake)

    # ---------------------------------------------------------------- per-frame
    def update(self, dt):
        self.t += dt
        t = self.t
        self.mt += dt
        if self.mode == 'fight':
            self.update_fight(dt)
        elif self.mode == 'fly':
            self.update_fly(dt)
        elif self.mode == 'travel':
            self.update_travel(dt)
        else:
            self.update_idle(dt)
        self.update_look()
        for a in self.actors:
            a.update(dt)
            if a.form > 0.001:
                a.bounce = math.sin(t * 23 + a.seed) * 0.15 + a.bounce * 0.0
            a.compute()
        self.update_effects(dt)
        self.cam.update(dt)
        if self.toast:
            self.toast[1] -= dt
            if self.toast[1] <= 0:
                self.toast = None

    def update_look(self):
        for a, o in ((self.A, self.B), (self.B, self.A)):
            hp = a.head_pos()
            tgt = self.cursor if self.cursor is not None else o.head_pos()
            dvec = tgt - hp
            yaw = math.degrees(math.atan2(dvec[0], dvec[2]))
            pitch = -math.degrees(math.atan2(dvec[1], math.hypot(dvec[0], dvec[2])))
            a.look[0] = clamp(wrap(yaw - a.yaw), -55, 55)
            a.look[1] = clamp(pitch, -25, 30)

    def update_idle(self, dt):
        t = self.t
        dragging = self.drag is not None
        for a in self.actors:
            a.air = False
            a.thrust_t = 0.0
            a.pitch_t = a.roll_t = 0.0
            a.pos_t = a.home.copy()
            a.weapon_t = 0.0
            if a.form_t > 0.5 or a.form > 0.02:
                a.pose_rest()
                a.yaw_t = a.veh_yaw if a.form_t > 0.5 else 0.0
                if a.form > 0.98 and dragging and abs(self.drag[0]) > 40:
                    a.spin += self.drag[0] / self.ppu * dt / a.d['wheel_r'] * 57.3 * (1 if a.veh_yaw > 0 else -1)
                continue
            if dragging:
                vx, vy, ground = self.drag
                if ground and abs(vx) > 40 and vy > -60:
                    a.yaw_t = 90.0 if vx > 0 else -90.0
                    a.phase += abs(vx) / self.ppu * dt / a.stride * 2 * math.pi
                    a.pose_walk(1.0 if abs(vx) < 500 else 1.5)
                elif not ground or vy < -60:
                    a.yaw_t = 0.0
                    a.pose_air(t, vy < 0)
                else:
                    a.yaw_t = 0.0
                    a.pose_idle(t)
            elif self.falling:
                a.yaw_t = 0.0
                a.pose_air(t, False)
            elif a.land_t > 0:
                a.pose_land(min(1.0, a.land_t / 0.35))
            elif a.gesture:
                a.yaw_t = 0.0
                a.pose_idle(t)
                a.pose_gesture(t)
            else:
                a.yaw_t = 0.0
                a.pose_idle(t)
        # autonomous fun
        if dragging or self.falling:
            self.idle_time = 0
        self.idle_time += dt
        if self.wander and self.idle_time > self.next_auto and self.cam.preset in ('front', 'hero'):
            self.idle_time = 0
            self.next_auto = random.uniform(9, 22)
            self.auto_event()

    def auto_event(self):
        r = random.random()
        if any(a.form_t > 0.5 for a in self.actors):
            self.transform(user=False)
        elif r < 0.30:
            a = random.choice(self.actors)
            a.gesture, a.gt = random.choice(('wave', 'nod', 'stretch', 'flex')), 0.0
        elif r < 0.62:
            self.start_travel('walk', random.choice((-1, 1)), random.uniform(160, 420))
        elif r < 0.85:
            self.start_travel('drive', random.choice((-1, 1)), random.uniform(350, 900))
        elif r < 0.93:
            a = random.choice(self.actors)
            self.transform(a, user=False)
        else:
            self.toggle_fly()
            self.fly_auto_end = self.t + random.uniform(6, 10)

    # ---------------------------------------------------------------- travel (window moves)
    def start_travel(self, kind, direction, dist_px):
        self.mode, self.sub, self.mt = 'travel', kind, 0.0
        self.travel = {'kind': kind, 'dir': direction, 'rem': dist_px, 'speed': 0.0, 'go': False}
        for a in self.actors:
            a.veh_yaw = 90.0 * direction
            if kind == 'drive':
                a.form_t = 1.0

    def update_travel(self, dt):
        tr = self.travel
        t = self.t
        for a in self.actors:
            a.pos_t = a.home.copy()
            a.air = False
        if tr['kind'] == 'walk':
            tr['go'] = tr['rem'] > 0
            tr['speed'] = 1.0 * self.ppu
            for a in self.actors:
                a.yaw_t = 90.0 * tr['dir']
                a.phase += 1.0 * dt / a.stride * 2 * math.pi
                a.pose_walk(1.0)
            if tr['rem'] <= 0:
                self.travel, self.mode = None, 'idle'
        else:
            for a in self.actors:
                a.pose_rest()
                a.yaw_t = a.veh_yaw if a.form_t > 0.5 else 0.0
            ready = all(a.form > 0.98 for a in self.actors)
            if tr['rem'] > 0 and ready:
                tr['go'] = True
                tr['speed'] = min(tr['speed'] + 5.0 * self.ppu * dt, 3.6 * self.ppu)
                for a in self.actors:
                    a.spin += tr['speed'] / self.ppu * dt / a.d['wheel_r'] * 57.3
            elif tr['rem'] <= 0:
                tr['go'] = False
                for a in self.actors:
                    a.form_t = 0.0
                if all(a.form < 0.02 for a in self.actors):
                    self.travel, self.mode = None, 'idle'
            if self.mt > 25:
                tr['rem'] = 0

    # ---------------------------------------------------------------- flying
    def update_fly(self, dt):
        t, A, B = self.t, self.A, self.B
        if self.sub == 'prep':
            for a in self.actors:
                a.form_t = 0.0
                a.pose_rest()
                a.yaw_t = 0.0
            if all(a.form < 0.02 for a in self.actors):
                self.sub, self.mt = 'takeoff', 0.0
                self.sound('thrust')
            return
        if getattr(self, 'fly_auto_end', None) and t > self.fly_auto_end and self.sub == 'hover':
            self.fly_auto_end = None
            self.toggle_fly()
        for i, a in enumerate(self.actors):
            ph = i * 2.3
            if self.sub == 'takeoff' and self.mt < 0.3:
                a.pose_land(0.8)
                a.thrust_t = 0.4
                continue
            if self.sub in ('takeoff', 'hover'):
                if self.sub == 'takeoff' and self.mt > 1.6:
                    self.sub = 'hover'
                hh = (1.05, 1.35)[i]
                sx = 0.45 * math.sin(0.42 * t + ph * 1.37)
                tgt = a.home + np.array((sx, hh + 0.10 * math.sin(1.7 * t + ph), 0.5 * math.sin(0.29 * t + ph)))
                vx = 0.45 * 0.42 * math.cos(0.42 * t + ph * 1.37)
                if self.drag:
                    vx += self.drag[0] / self.ppu * 0.3
                a.pos_t = tgt
                a.air = True
                a.thrust_t = 1.0
                a.yaw_t = clamp(vx * 70, -55, 55)
                a.pitch_t = 10 + abs(vx) * 8
                a.roll_t = -vx * 14
                a.move_rate = 3.0
                a.pose_fly(t)
            else:   # landing
                a.pos_t = a.home.copy()
                a.move_rate = 2.5
                a.thrust_t = 1.0 if a.pos[1] > 0.06 else 0.0
                a.pitch_t = a.roll_t = 0.0
                a.yaw_t = 0.0
                if a.pos[1] > 0.06:
                    a.air = True
                    a.pose_fly(t)
                elif a.air:
                    a.air = False
                    a.pos[1] = 0.0
                    a.land_t = 0.35
                    self.sparks(a.point('foot_L', (0, -0.2, 0)), 8, ((0.8, 0.8, 0.8), (0.4, 0.4, 0.4)), 2.0)
                else:
                    a.pose_land(max(0.0, a.land_t / 0.35))
        if self.sub == 'landing' and all(not a.air and a.land_t <= 0 for a in self.actors):
            for a in self.actors:
                a.move_rate = 9.0
            self.mode, self.sub = 'idle', ''
            self.idle_time = 0

    # ---------------------------------------------------------------- fighting
    MOVES = ('slash', 'volley', 'kick', 'clash', 'charge')
    DUR = {'slash': 1.5, 'volley': 1.7, 'kick': 1.5, 'clash': 1.45, 'charge': 1.5}

    def update_fight(self, dt):
        t, A, B = self.t, self.A, self.B
        if self.fight_timer > 0:
            self.fight_timer -= dt
            if self.fight_timer <= 0:
                self.fight_hold = False
        for a in self.actors:
            a.move_rate = 9.0
            a.thrust_t = 0.0
            a.pitch_t = a.roll_t = 0.0
        if self.sub == 'prep':
            for a in self.actors:
                a.form_t = 0.0
                if a.pos[1] > 0.05:
                    a.air = True
                    a.pos_t = a.home.copy()
                    a.thrust_t = 1.0
                    a.pose_fly(t)
                else:
                    a.air = False
                    a.pose_rest()
            if all(a.form < 0.02 and a.pos[1] < 0.06 for a in self.actors):
                self.sub, self.mt = 'approach', 0.0
                for a in self.actors:
                    a.air = False
            return
        spotA, spotB = np.array((-1.05, 0, 0.0)), np.array((1.05, 0, 0.0))
        if self.sub == 'approach':
            done = True
            for a, spot, yw in ((A, spotA, 90.0), (B, spotB, -90.0)):
                a.weapon_t = 1.0
                d = spot - a.pos_t
                L = float(np.linalg.norm(d))
                if L > 0.03:
                    done = False
                    step = min(L, 2.6 * dt)
                    a.pos_t = a.pos_t + d / L * step
                    a.yaw_t = math.degrees(math.atan2(d[0], d[2]))
                    a.phase += 2.6 * dt / a.stride * 2 * math.pi
                    a.pose_walk(1.5)
                else:
                    a.yaw_t = yw
                    a.pose_stance(t)
            if done and self.mt > 0.3:
                self.sub, self.mt = 'x', 0.0
                self.move = None
            if not self.fight_hold:
                self.sub, self.mt = 'out', 0.0
            return
        if self.sub == 'x':
            if self.move is None or self.mt >= self.DUR[self.move]:
                if not self.fight_hold:
                    self.sub, self.mt = 'out', 0.0
                    return
                choices = [m for m in self.MOVES if m != self.move]
                self.move, self.mt = random.choice(choices), 0.0
                self.fired = set()
            u = self.mt
            A.yaw_t, B.yaw_t = 90.0, -90.0
            A.pose_stance(t)
            B.pose_stance(t)
            fA, fB, yB = getattr(self, 'mv_' + self.move)(u)
            A.pos_t = spotA + np.array((fA, 0, 0))
            B.pos_t = spotB + np.array((-fB, yB, 0))
            B.air = yB > 0.02
            return
        # 'out': retreat home
        done = True
        for a in self.actors:
            a.weapon_t = 0.0
            a.air = False
            d = a.home - a.pos_t
            L = float(np.linalg.norm(d))
            if L > 0.03:
                done = False
                step = min(L, 2.0 * dt)
                a.pos_t = a.pos_t + d / L * step
                a.yaw_t = math.degrees(math.atan2(d[0], d[2]))
                a.phase += 2.0 * dt / a.stride * 2 * math.pi
                a.pose_walk(1.2)
            else:
                a.yaw_t = 0.0
                a.pose_idle(t)
        if done and self.mt > 0.4:
            self.mode, self.sub = 'idle', ''
            self.idle_time = 0
            if self.fight_hold:      # still holding: go again
                self.start_fight()

    def once(self, key, u, at):
        if u >= at and key not in self.fired:
            self.fired.add(key)
            return True
        return False

    def blade_tip(self):
        return self.A.point('weapon', (0, -1.30, 0))

    def chest(self, a):
        return a.point('torso', (0, 0.55 * a.size, 0.3 * a.size))

    def trail(self):
        A = self.A
        for k in (0.5, 0.8, 1.1, 1.35):
            self.particle(A.point('weapon', (0, -k, 0)), (0, 0, 0), 0.16, 0.13, 0.03, (1, 0.6, 0.2, 0.7),
                          (1, 0.3, 0.05, 0.0), 1.0)

    def mv_slash(self, u):
        A, B, T, S = self.A, self.B, self.A.tgt, self.B.tgt
        T['upper_R'] = kf(u, [(0, (-55, 0, -10)), (0.3, (-165, 0, -15)), (0.46, (-20, 0, -10)), (0.62, (-30, 0, -10)),
                              (0.78, (-135, 0, -25)), (1.1, (-55, 0, -10))])
        T['fore_R'] = kf(u, [(0, (-62, 0, 0)), (0.3, (-40, 0, 0)), (0.46, (-5, 0, 0)), (0.62, (-10, 0, 0)),
                             (0.78, (-60, 0, 0)), (1.1, (-62, 0, 0))])
        T['torso'] = kf(u, [(0, (6, -10, 0)), (0.3, (-4, 20, 0)), (0.46, (16, -22, 0)), (0.78, (4, 15, 0)),
                            (1.1, (6, -10, 0))])
        S['torso'] = kf(u, [(0.3, (5, 14, 0)), (0.45, (-18, 14, 0)), (0.85, (-10, 14, 0)), (1.1, (5, 14, 0))])
        S['upper_R'] = kf(u, [(0.3, (-32, 0, -10)), (0.42, (-100, 0, -25)), (0.85, (-100, 0, -25)), (1.1, (-32, 0, -10))])
        S['fore_R'] = kf(u, [(0.3, (-105, 0, 0)), (0.42, (-70, 0, 0)), (0.85, (-70, 0, 0)), (1.1, (-105, 0, 0))])
        if 0.3 < u < 0.5 or 0.62 < u < 0.8:
            self.trail()
        if self.once('w1', u, 0.32) or self.once('w2', u, 0.64):
            self.sound('swing')
        if self.once('hit', u, 0.46):
            p = self.blade_tip()
            self.sparks(p, 30)
            self.impact(p, k=0.7, shake=0.06)
            self.sound('clang')
        return (kf(u, [(0, 0), (0.3, 0.15), (0.46, 0.45), (0.8, 0.5), (1.2, 0)])[()],
                kf(u, [(0, 0), (0.3, 0), (0.45, -0.35), (0.8, -0.45), (1.3, 0)])[()], 0.0)

    def mv_volley(self, u):
        A, B, T, S = self.A, self.B, self.A.tgt, self.B.tgt
        S['upper_L'] = kf(u, [(0, (-50, 0, 10)), (0.25, (-88, 0, 4)), (1.3, (-88, 0, 4)), (1.6, (-50, 0, 10))])
        S['fore_L'] = kf(u, [(0, (-40, 0, 0)), (0.25, (0, 0, 0)), (1.3, (0, 0, 0)), (1.6, (-40, 0, 0))])
        S['torso'] = kf(u, [(0, (5, 14, 0)), (0.25, (0, 38, 0)), (1.3, (0, 38, 0)), (1.6, (5, 14, 0))])
        T['upper_R'] = kf(u, [(0, (-55, 0, -10)), (0.3, (-78, 0, -22)), (1.3, (-78, 0, -22)), (1.6, (-55, 0, -10))])
        T['fore_R'] = kf(u, [(0, (-62, 0, 0)), (0.3, (-98, 0, 0)), (1.3, (-98, 0, 0)), (1.6, (-62, 0, 0))])
        fA = 0.0
        for k, at in enumerate((0.40, 0.62, 0.84)):
            if self.once('f%d' % k, u, at):
                m = B.point('weapon', (0, -0.62, 0.02))
                self.bolts.append([m, self.blade_tip() * 0.5 + self.chest(A) * 0.5, 0.0])
                self.particle(m, (0, 0, 0), 0.12, 0.35, 0.5, (1, 0.5, 0.95, 0.9), (1, 0.2, 0.8, 0.0), 1.0)
                self.sound('blaster')
            fA -= 0.07 * math.exp(-((u - at - 0.12) * 14) ** 2)
        return fA, kf(u, [(0, 0), (0.3, -0.1), (1.3, -0.1), (1.6, 0)])[()], 0.0

    def mv_kick(self, u):
        A, B, T, S = self.A, self.B, self.A.tgt, self.B.tgt
        crouch = env(u, 0.0, 0.18) * (1 - env(u, 0.2, 0.3)) + env(u, 0.72, 0.8) * (1 - env(u, 0.9, 1.1))
        if crouch > 0.01:
            B.pose_land(crouch)
        if 0.3 < u < 0.72:
            k = env(u, 0.32, 0.42) * (1 - env(u, 0.58, 0.7))
            S['thigh_L'] = v(-20 - 75 * k, 0, 6)
            S['shin_L'] = v(60 * (1 - k))
            S['foot_L'] = v(-20 * k)
            S['thigh_R'] = v(5, 0, -6)
            S['shin_R'] = v(70 + 20 * k)
            S['torso'] = v(-14 * k, 20)
            S['upper_L'] = v(-60, 0, 30)
            S['upper_R'] = v(-30, 0, -45)
        hitk = env(u, 0.5, 0.56) * (1 - env(u, 0.85, 1.2))
        T['torso'] = T['torso'] + v(-22 * hitk)
        if self.once('jump', u, 0.25):
            self.sound('swing')
        if self.once('hit', u, 0.5):
            p = self.chest(A)
            self.sparks(p, 34, ((1, 1, 1), (0.4, 0.8, 1.0)), 5.0)
            self.impact(p, (0.6, 0.85, 1.0), 1.0, 0.16)
            self.sound('hit')
        fA = -0.35 * hitk
        fB = kf(u, [(0, 0), (0.2, -0.05), (0.5, 0.62), (0.75, 0.15), (1.2, 0)])[()]
        yB = kf(u, [(0, 0), (0.22, 0), (0.48, 0.75), (0.74, 0.0)])[()]
        return fA, fB, max(0.0, yB)

    def mv_clash(self, u):
        A, B, T, S = self.A, self.B, self.A.tgt, self.B.tgt
        T['upper_R'] = kf(u, [(0, (-55, 0, -10)), (0.25, (-125, 0, -40)), (0.4, (-82, 0, -20)), (0.8, (-82, 0, -20)),
                              (1.2, (-55, 0, -10))])
        T['fore_R'] = kf(u, [(0, (-62, 0, 0)), (0.4, (-48, 0, 0)), (0.8, (-48, 0, 0)), (1.2, (-62, 0, 0))])
        S['upper_R'] = kf(u, [(0, (-32, 0, -10)), (0.35, (-95, 0, -30)), (0.8, (-95, 0, -30)), (1.1, (-32, 0, -10))])
        S['fore_R'] = kf(u, [(0, (-105, 0, 0)), (0.35, (-60, 0, 0)), (0.8, (-60, 0, 0)), (1.1, (-105, 0, 0))])
        if 0.2 < u < 0.4:
            self.trail()
        mid = (self.blade_tip() + B.point('fore_R', (0, -0.3, 0))) * 0.5
        if self.once('hit', u, 0.40):
            self.sparks(mid, 50, speed=5.5)
            self.impact(mid, (1.0, 0.7, 0.35), 1.3, 0.2)
            self.sound('clang')
        if 0.42 < u < 0.78 and random.random() < 0.5:
            self.sparks(mid, 3, speed=3.0)
        j = 0.02 * math.sin(self.t * 60) if 0.4 < u < 0.78 else 0.0
        f = kf(u, [(0, 0), (0.35, 0.42), (0.78, 0.42), (0.95, -0.12), (1.3, 0)])[()]
        return f + j, f - j, 0.0

    def mv_charge(self, u):
        A, B, T, S = self.A, self.B, self.A.tgt, self.B.tgt
        T['torso'] = kf(u, [(0, (6, -10, 0)), (0.25, (15, 30, 0)), (0.45, (22, -38, 0)), (0.7, (10, -30, 0)),
                            (1.1, (6, -10, 0))])
        T['upper_L'] = kf(u, [(0, (-38, 0, 14)), (0.3, (-60, 0, 30)), (0.7, (-60, 0, 30)), (1.1, (-38, 0, 14))])
        if 0.25 < u < 0.45:
            A.phase += 0.5
            A.pose_walk(1.6)
        hk = env(u, 0.45, 0.55) * (1 - env(u, 0.9, 1.3))
        S['torso'] = S['torso'] + v(-25 * hk)
        S['upper_L'] = S['upper_L'] + v(-30 * hk, 0, 40 * hk)
        if self.once('go', u, 0.25):
            self.sound('swing')
        if self.once('hit', u, 0.45):
            p = self.chest(B)
            self.sparks(p, 36)
            self.impact(p, k=1.1, shake=0.2)
            self.sound('hit')
        fA = kf(u, [(0, 0), (0.25, -0.1), (0.45, 0.78), (0.7, 0.7), (1.2, 0)])[()]
        fB = -0.6 * hk
        return fA, fB, 0.0

    # ---------------------------------------------------------------- effects update
    def update_effects(self, dt):
        for b in self.bolts[:]:
            b[2] += dt / 0.16
            p = b[0] + (b[1] - b[0]) * min(1.0, b[2])
            for k in range(3):
                q = b[0] + (b[1] - b[0]) * max(0.0, min(1.0, b[2] - k * 0.05))
                self.particle(q, (0, 0, 0), 0.05, 0.16 - k * 0.03, 0.05, (1, 0.45, 0.95, 1), (1, 0.2, 0.8, 0), 1.0)
            if b[2] >= 1.0:
                self.bolts.remove(b)
                self.sparks(p, 18, ((1, 0.8, 1.0), (1, 0.25, 0.85)), 3.5)
                self.impact(p, (1.0, 0.4, 0.9), 0.6, 0.05)
        for a in self.actors:
            if a.thrust > 0.05:
                for S in 'LR':
                    sole = a.point('foot_' + S, (0, -a.d['ankle'] - 0.02, 0.02))
                    down = a.axis('foot_' + S, (0, -1, 0))
                    c0, c1 = a.d['flame']
                    for _ in range(3):
                        vel = down * random.uniform(3.5, 5.5) * a.thrust + np.random.normal(size=3) * 0.3
                        self.particle(sole, vel, random.uniform(0.18, 0.32), 0.34 * a.size * a.thrust, 0.06,
                                      (*c0, 0.95), (*c1, 0.0), 1.0)
            if a.weapon > 0.3 and a.d['weapon'] == 'blade':
                for k in (0.45, 0.8, 1.15):
                    self.particle(a.point('weapon', (0, -k, 0)), (0, 0, 0), 0.04, 0.22 * a.weapon, 0.22 * a.weapon,
                                  (1, 0.5, 0.12, 0.35), (1, 0.5, 0.12, 0.35), 1.0)
        alive = []
        for p in self.parts:
            p[2] -= dt
            if p[2] <= 0:
                continue
            p[1][1] += p[9] * dt
            if p[10]:
                p[1] *= max(0.0, 1 - p[10] * dt)
            p[0] = p[0] + p[1] * dt
            alive.append(p)
        self.parts = alive
        self.flash = max(0.0, self.flash - dt * 12)
        self.shake = max(0.0, self.shake - dt * 0.6)

    # ---------------------------------------------------------------- frame data for the renderer
    def frame(self, aspect, size):
        sh = tuple(np.random.normal(size=3) * self.shake * 0.5) if self.shake > 0 else (0, 0, 0)
        VP, V, eye = self.cam.matrices(aspect, shake=sh)
        Vi = np.linalg.inv(V)
        right, up, back = Vi[:3, 0], Vi[:3, 1], Vi[:3, 2]
        key = right * 0.55 + up * 0.75 + back * 0.45
        key /= np.linalg.norm(key)
        fill = -right * 0.8 + back * 0.4 + up * 0.1
        fill /= np.linalg.norm(fill)
        bots_ = [(a.handle, np.ascontiguousarray(a.bones, dtype='f4').tobytes(), a.d['ocol']) for a in self.actors]
        n = len(self.parts)
        sp = np.zeros((n, 9), np.float32)
        for i, p in enumerate(self.parts):
            k = 1.0 - p[2] / p[3]
            c0, c1 = p[6], p[7]
            sp[i, 0:3] = p[0]
            sp[i, 3] = p[4] + (p[5] - p[4]) * k
            sp[i, 4:8] = [c0[j] + (c1[j] - c0[j]) * k for j in range(4)]
            sp[i, 8] = p[8]
        shd = []
        if self.cam.el > 2:
            for a in self.actors:
                fade = clamp(1 - a.pos[1] / 2.6, 0.0, 1.0)
                if a.form > 0.5:
                    L = a.d['length'] * 0.55
                    shd.append((a.pos[0], a.pos[2], 0.95 * a.size, L, math.radians(a.yaw), 0.38 * fade))
                else:
                    sc = 0.8 + 0.4 * (1 - fade)
                    shd.append((a.pos[0], a.pos[2], 0.85 * a.size * sc, 0.5 * a.size * sc, 0.0, 0.36 * fade))
        f = dict(vp=VP, eye=eye, right=right, up=up, key=key, fill=fill, back=-back * 0.6 + up * 0.4,
                 flash=(*self.flash_pos, self.flash), flashcol=tuple(self.flash_col), bots=bots_, sprites=sp,
                 shadows=np.array(shd, np.float32) if shd else None)
        # screen boxes of each robot for click picking
        w, h = size
        self.boxes = []
        for a in self.actors:
            P = a.bones[:a.n, 3, :3] if False else np.array([a.bones[i].T[:3, 3] for i in range(a.n)])
            Ph = np.c_[P, np.ones(len(P))] @ VP.T
            xy = Ph[:, :2] / Ph[:, 3:4]
            px = (xy[:, 0] + 1) * 0.5 * w
            py = (1 - xy[:, 1]) * 0.5 * h
            m = 0.3 * self.ppu * (10.6 / self.cam.dist)
            self.boxes.append((px.min() - m, py.min() - m, px.max() + m, py.max() + m, float(np.mean(Ph[:, 3]))))
        self.vp = VP
        self.eye = eye
        return f

    def pick(self, x, y):
        best = None
        for a, b in zip(self.actors, getattr(self, 'boxes', [])):
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                if best is None or b[4] < best[1]:
                    best = (a, b[4])
        if best:
            return best[0]
        if getattr(self, 'boxes', None):
            cx = [(b[0] + b[2]) / 2 for b in self.boxes]
            return self.actors[int(np.argmin([abs(c - x) for c in cx]))]
        return self.A

    def unproject(self, x, y, size, dist=None):
        w, h = size
        nx, ny = x / w * 2 - 1, 1 - y / h * 2
        inv = np.linalg.inv(self.vp)
        p = inv @ np.array((nx, ny, 1.0, 1.0))
        p = p[:3] / p[3]
        d = p - self.eye
        d /= np.linalg.norm(d)
        return self.eye + d * (dist or self.cam.dist)
