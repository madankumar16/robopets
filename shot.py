"""shot.py - render test pictures offscreen: python shot.py out.png [scene] [view] [seconds]
scenes: idle, vehicle, fight, fly, transform"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image  # noqa: E402

from engine3d import Renderer  # noqa: E402
from sim import World  # noqa: E402


def main():
    out = sys.argv[1]
    scene = sys.argv[2] if len(sys.argv) > 2 else 'idle'
    view = sys.argv[3] if len(sys.argv) > 3 else 'front'
    secs = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    W, H = 1000, 566
    r = Renderer(W, H)
    w = World(r)
    w.wander = False
    w.cam.calibrate(W / H)
    w.cam.set(view)
    w.cam.az, w.cam.el = w.cam.goal
    w.cursor = None
    if scene == 'vehicle':
        w.transform()
    elif scene == 'transform':
        w.transform()
    elif scene == 'fight':
        w.start_fight()
    elif scene == 'fly':
        w.toggle_fly()
    dt = 1 / 30
    t0 = time.perf_counter()
    for _ in range(int(secs * 30)):
        w.update(dt)
    t1 = time.perf_counter()
    f = w.frame(W / H, (W, H))
    r.render(f)
    img = r.read_rgba(bg=(150, 160, 172))
    t2 = time.perf_counter()
    for _ in range(10):
        w.update(dt)
        r.render(w.frame(W / H, (W, H)))
        buf = bytearray(W * H * 4)
        r.read_into(buf)
    t3 = time.perf_counter()
    Image.fromarray(img, 'RGBA').convert('RGB').save(out)
    print('sim %.1f ms/frame, frame+render+read %.1f ms' % ((t1 - t0) / max(1, secs * 30) * 1000, (t3 - t2) / 10 * 1000))


if __name__ == '__main__':
    main()
