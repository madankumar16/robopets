"""RoboPets - two original transforming robots that live on your desktop.

  BULWARK  crimson heavy robot  <->  6-wheel armored truck   (plasma blade)
  VOLT     teal scout robot     <->  sports car              (wrist blaster)

Controls (on the robots)
  Hold RIGHT mouse ......... they fight (release to stop)
  Quick RIGHT click ........ take off / land (they fly)
  Double-click a robot ..... it transforms robot <-> vehicle      (T = both)
  Drag with LEFT mouse ..... move them; drag sideways = walk, drag up = jump, let go = fall
  Click a robot ............ it waves / flexes / salutes
  Mouse wheel .............. spin the camera around them (Shift = tilt, Ctrl = zoom)
  Keys 1-7 ................. Front, 3/4, Back, Left, Right, Top, Bottom views
  Middle click / tray icon . menu (views, size, wander, sound, start with Windows, quit)
"""
import ctypes
import json
import math
import os
import random
import sys
import time
import traceback
from ctypes import wintypes as wt

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
LOG = os.path.join(APP_DIR, 'robopets.log')
SETTINGS = os.path.join(APP_DIR, 'settings.json')


def log(msg):
    try:
        with open(LOG, 'a', encoding='utf-8') as fh:
            fh.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')
    except Exception:
        pass


user32 = ctypes.WinDLL('user32', use_last_error=True)
gdi32 = ctypes.WinDLL('gdi32')
kernel32 = ctypes.WinDLL('kernel32')
shell32 = ctypes.WinDLL('shell32')

_mutex = kernel32.CreateMutexW(None, False, 'RoboPets_SingleInstance_v1')
if kernel32.GetLastError() == 183 and '--shot' not in sys.argv:
    sys.exit(0)

try:
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
except Exception:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

import numpy as np  # noqa: E402

from engine3d import Renderer  # noqa: E402
from sim import VIEWS, World  # noqa: E402

# ------------------------------------------------------------------ win32 plumbing
LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [('cbSize', wt.UINT), ('style', wt.UINT), ('lpfnWndProc', WNDPROC), ('cbClsExtra', ctypes.c_int),
                ('cbWndExtra', ctypes.c_int), ('hInstance', wt.HINSTANCE), ('hIcon', wt.HICON),
                ('hCursor', wt.HANDLE), ('hbrBackground', wt.HBRUSH), ('lpszMenuName', wt.LPCWSTR),
                ('lpszClassName', wt.LPCWSTR), ('hIconSm', wt.HICON)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [('BlendOp', ctypes.c_ubyte), ('BlendFlags', ctypes.c_ubyte), ('SourceConstantAlpha', ctypes.c_ubyte),
                ('AlphaFormat', ctypes.c_ubyte)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [('biSize', wt.DWORD), ('biWidth', ctypes.c_long), ('biHeight', ctypes.c_long), ('biPlanes', wt.WORD),
                ('biBitCount', wt.WORD), ('biCompression', wt.DWORD), ('biSizeImage', wt.DWORD),
                ('biXPelsPerMeter', ctypes.c_long), ('biYPelsPerMeter', ctypes.c_long), ('biClrUsed', wt.DWORD),
                ('biClrImportant', wt.DWORD)]


class GUID(ctypes.Structure):
    _fields_ = [('Data1', wt.DWORD), ('Data2', wt.WORD), ('Data3', wt.WORD), ('Data4', ctypes.c_ubyte * 8)]


class NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [('cbSize', wt.DWORD), ('hWnd', wt.HWND), ('uID', wt.UINT), ('uFlags', wt.UINT),
                ('uCallbackMessage', wt.UINT), ('hIcon', wt.HICON), ('szTip', ctypes.c_wchar * 128),
                ('dwState', wt.DWORD), ('dwStateMask', wt.DWORD), ('szInfo', ctypes.c_wchar * 256),
                ('uVersion', wt.UINT), ('szInfoTitle', ctypes.c_wchar * 64), ('dwInfoFlags', wt.DWORD),
                ('guidItem', GUID), ('hBalloonIcon', wt.HICON)]


class ICONINFO(ctypes.Structure):
    _fields_ = [('fIcon', wt.BOOL), ('xHotspot', wt.DWORD), ('yHotspot', wt.DWORD), ('hbmMask', wt.HBITMAP),
                ('hbmColor', wt.HBITMAP)]


class RECT(ctypes.Structure):
    _fields_ = [('left', ctypes.c_long), ('top', ctypes.c_long), ('right', ctypes.c_long), ('bottom', ctypes.c_long)]


class MONITORINFO(ctypes.Structure):
    _fields_ = [('cbSize', wt.DWORD), ('rcMonitor', RECT), ('rcWork', RECT), ('dwFlags', wt.DWORD)]


def _sig(dll, name, res, args):
    f = getattr(dll, name)
    f.restype, f.argtypes = res, args
    return f


DefWindowProcW = _sig(user32, 'DefWindowProcW', LRESULT, [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM])
RegisterClassExW = _sig(user32, 'RegisterClassExW', wt.ATOM, [ctypes.POINTER(WNDCLASSEXW)])
CreateWindowExW = _sig(user32, 'CreateWindowExW', wt.HWND, [wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD, ctypes.c_int,
                                                             ctypes.c_int, ctypes.c_int, ctypes.c_int, wt.HWND,
                                                             wt.HMENU, wt.HINSTANCE, wt.LPVOID])
PeekMessageW = _sig(user32, 'PeekMessageW', wt.BOOL, [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT, wt.UINT])
TranslateMessage = _sig(user32, 'TranslateMessage', wt.BOOL, [ctypes.POINTER(wt.MSG)])
DispatchMessageW = _sig(user32, 'DispatchMessageW', LRESULT, [ctypes.POINTER(wt.MSG)])
UpdateLayeredWindow = _sig(user32, 'UpdateLayeredWindow', wt.BOOL,
                           [wt.HWND, wt.HDC, ctypes.POINTER(wt.POINT), ctypes.POINTER(wt.SIZE), wt.HDC,
                            ctypes.POINTER(wt.POINT), wt.DWORD, ctypes.POINTER(BLENDFUNCTION), wt.DWORD])
SetWindowPos = _sig(user32, 'SetWindowPos', wt.BOOL, [wt.HWND, wt.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                       ctypes.c_int, wt.UINT])
ShowWindow = _sig(user32, 'ShowWindow', wt.BOOL, [wt.HWND, ctypes.c_int])
GetCursorPos = _sig(user32, 'GetCursorPos', wt.BOOL, [ctypes.POINTER(wt.POINT)])
SetCapture = _sig(user32, 'SetCapture', wt.HWND, [wt.HWND])
ReleaseCapture = _sig(user32, 'ReleaseCapture', wt.BOOL, [])
GetKeyState = _sig(user32, 'GetKeyState', ctypes.c_short, [ctypes.c_int])
LoadCursorW = _sig(user32, 'LoadCursorW', wt.HANDLE, [wt.HINSTANCE, ctypes.c_void_p])
CreatePopupMenu = _sig(user32, 'CreatePopupMenu', wt.HMENU, [])
AppendMenuW = _sig(user32, 'AppendMenuW', wt.BOOL, [wt.HMENU, wt.UINT, ctypes.c_size_t, wt.LPCWSTR])
TrackPopupMenu = _sig(user32, 'TrackPopupMenu', ctypes.c_int, [wt.HMENU, wt.UINT, ctypes.c_int, ctypes.c_int,
                                                                ctypes.c_int, wt.HWND, ctypes.c_void_p])
DestroyMenu = _sig(user32, 'DestroyMenu', wt.BOOL, [wt.HMENU])
SetForegroundWindow = _sig(user32, 'SetForegroundWindow', wt.BOOL, [wt.HWND])
PostMessageW = _sig(user32, 'PostMessageW', wt.BOOL, [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM])
PostQuitMessage = _sig(user32, 'PostQuitMessage', None, [ctypes.c_int])
MsgWait = _sig(user32, 'MsgWaitForMultipleObjectsEx', wt.DWORD, [wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD,
                                                                  wt.DWORD])
RegisterWindowMessageW = _sig(user32, 'RegisterWindowMessageW', wt.UINT, [wt.LPCWSTR])
CreateIconIndirect = _sig(user32, 'CreateIconIndirect', wt.HICON, [ctypes.POINTER(ICONINFO)])
MessageBoxW = _sig(user32, 'MessageBoxW', ctypes.c_int, [wt.HWND, wt.LPCWSTR, wt.LPCWSTR, wt.UINT])
MonitorFromPoint = _sig(user32, 'MonitorFromPoint', ctypes.c_void_p, [wt.POINT, wt.DWORD])
GetMonitorInfoW = _sig(user32, 'GetMonitorInfoW', wt.BOOL, [ctypes.c_void_p, ctypes.POINTER(MONITORINFO)])
GetDC = _sig(user32, 'GetDC', wt.HDC, [wt.HWND])
CreateCompatibleDC = _sig(gdi32, 'CreateCompatibleDC', wt.HDC, [wt.HDC])
CreateDIBSection = _sig(gdi32, 'CreateDIBSection', wt.HBITMAP, [wt.HDC, ctypes.c_void_p, wt.UINT,
                                                                 ctypes.POINTER(ctypes.c_void_p), wt.HANDLE, wt.DWORD])
CreateBitmap = _sig(gdi32, 'CreateBitmap', wt.HBITMAP, [ctypes.c_int, ctypes.c_int, wt.UINT, wt.UINT, ctypes.c_void_p])
SelectObject = _sig(gdi32, 'SelectObject', wt.HGDIOBJ, [wt.HDC, wt.HGDIOBJ])
DeleteObject = _sig(gdi32, 'DeleteObject', wt.BOOL, [wt.HGDIOBJ])
GdiFlush = _sig(gdi32, 'GdiFlush', wt.BOOL, [])
GetModuleHandleW = _sig(kernel32, 'GetModuleHandleW', wt.HMODULE, [wt.LPCWSTR])
Shell_NotifyIconW = _sig(shell32, 'Shell_NotifyIconW', wt.BOOL, [wt.DWORD, ctypes.POINTER(NOTIFYICONDATAW)])

WM_DESTROY, WM_CLOSE, WM_KEYDOWN, WM_KEYUP = 0x2, 0x10, 0x100, 0x101
WM_MOUSEMOVE, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_LBUTTONDBLCLK = 0x200, 0x201, 0x202, 0x203
WM_RBUTTONDOWN, WM_RBUTTONUP, WM_MBUTTONUP, WM_MOUSEWHEEL, WM_CONTEXTMENU = 0x204, 0x205, 0x208, 0x20A, 0x7B
WM_CAPTURECHANGED, WM_APP, WM_NULL = 0x215, 0x8000, 0
WM_TRAY = WM_APP + 1
HWND_TOPMOST = wt.HWND(-1)


def work_area(x, y):
    try:
        h = MonitorFromPoint(wt.POINT(int(x), int(y)), 2)
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        GetMonitorInfoW(h, ctypes.byref(mi))
        r = mi.rcWork
        return r.left, r.top, r.right, r.bottom
    except Exception:
        return 0, 0, 1920, 1040


# ------------------------------------------------------------------ sound effects
def make_sounds(folder):
    import struct
    os.makedirs(folder, exist_ok=True)
    sr = 22050
    rng = np.random.default_rng(7)

    def save(name, x):
        p = os.path.join(folder, name + '.wav')
        if os.path.exists(p):
            return
        x = np.clip(x, -1, 1)
        data = (x * 32000).astype('<i2').tobytes()
        hdr = b'RIFF' + struct.pack('<I', 36 + len(data)) + b'WAVEfmt ' + struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2,
                                                                                         16)
        with open(p, 'wb') as fh:
            fh.write(hdr + b'data' + struct.pack('<I', len(data)) + data)

    def t(sec):
        return np.arange(int(sr * sec)) / sr

    def click(out, at, f, amp=0.5, dec=80):
        i, L = int(at * sr), int(0.06 * sr)
        k = np.arange(min(L, len(out) - i)) / sr
        out[i:i + len(k)] += (rng.standard_normal(len(k)) * 0.5 + np.sin(2 * np.pi * f * k)) * np.exp(-k * dec) * amp

    x = t(1.5)
    f = 300 + 650 * np.clip(x / 1.2, 0, 1) ** 0.7
    ph = 2 * np.pi * np.cumsum(f) / sr
    tr = (np.sin(ph) * 0.5 + np.sin(2 * ph) * 0.22 + np.sin(3.01 * ph) * 0.1) * np.sin(np.pi * np.clip(x / 1.5, 0, 1)) * 0.18
    for at in (0.04, 0.17, 0.29, 0.41, 0.58, 0.7, 0.84, 0.98, 1.1, 1.28):
        click(tr, at, rng.uniform(1500, 3200), 0.45)
    click(tr, 1.38, 140, 0.9, 25)
    save('transform', tr * 0.8)
    x = t(0.22)
    f = 1900 * np.exp(-x * 11) + 250
    save('blaster', np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-x * 9) * 0.5 + rng.standard_normal(len(x)) * 0.05 * np.exp(-x * 20))
    x = t(0.3)
    n = np.convolve(rng.standard_normal(len(x)), np.ones(12) / 12, 'same')
    save('swing', n * np.sin(np.pi * x / 0.3) ** 2 * 1.6)
    x = t(0.5)
    cl = sum(np.sin(2 * np.pi * fr * x) * a for fr, a in ((523, 0.4), (1187, 0.3), (2213, 0.2), (3407, 0.12)))
    save('clang', (cl * 0.5 + rng.standard_normal(len(x)) * 0.25 * np.exp(-x * 40)) * np.exp(-x * 7))
    x = t(0.35)
    save('hit', (np.sin(2 * np.pi * (90 + 60 * np.exp(-x * 20)) * x) * 0.8 + rng.standard_normal(len(x)) * 0.35)
         * np.exp(-x * 12))
    x = t(0.9)
    n = np.convolve(rng.standard_normal(len(x)), np.ones(30) / 30, 'same') * 3
    save('thrust', n * np.minimum(1, x * 6) * np.exp(-x * 2.2) * 0.9)


# ------------------------------------------------------------------ settings / autostart
RUN_KEY = r'Software\Microsoft\Windows\CurrentVersion\Run'


def autostart_cmd():
    exe = os.path.join(os.path.dirname(sys.executable), 'pythonw.exe')
    return '"%s" "%s"' % (exe, os.path.abspath(__file__))


def autostart_get():
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            winreg.QueryValueEx(k, 'RoboPets')
            return True
    except OSError:
        return False


def autostart_set(on):
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
        if on:
            winreg.SetValueEx(k, 'RoboPets', 0, winreg.REG_SZ, autostart_cmd())
        else:
            try:
                winreg.DeleteValue(k, 'RoboPets')
            except OSError:
                pass


# ------------------------------------------------------------------ the overlay window
class App:
    BASE = (760, 430)
    SIZES = (('Small', 0.65), ('Medium', 0.85), ('Large', 1.1), ('Huge', 1.4))

    def __init__(self):
        self.settings = {'size': 0.85, 'wander': True, 'sound': True, 'x': None}
        try:
            with open(SETTINGS, encoding='utf-8') as fh:
                self.settings.update(json.load(fh))
        except Exception:
            pass
        try:
            self.dpi = user32.GetDpiForSystem() / 96.0
        except Exception:
            self.dpi = 1.0
        self.sfx_dir = os.path.join(APP_DIR, 'sfx')
        try:
            make_sounds(self.sfx_dir)
        except Exception:
            log('sound gen failed\n' + traceback.format_exc())
        self.W, self.H = self.dims()
        self.r = Renderer(self.W, self.H, samples=8)
        self.world = World(self.r)
        self.world.wander = bool(self.settings['wander'])
        self.world.cam.calibrate(self.W / self.H)
        self.ground_row = self.H * 0.89
        self.world.ppu = self.ppu()
        self.hidden = False
        self.auto_hidden = False
        self.drag = None
        self.rdown = None
        self.pending_click = None
        self.vy = 0.0
        self.toast_cache = {}
        self.create_window()
        l, tp, rr, b = work_area(*self.primary_center())
        x = self.settings.get('x')
        if x is None or not (l - self.W < x < rr):
            x = rr - self.W - 30 * self.dpi
        self.wx = float(x)
        self.wy = float(b - self.ground_row) - 220 * self.dpi    # drop in from a little above
        self.world.falling = True
        self.make_dib()
        self.add_tray()
        self.world.say('Hello!  Right-click & hold to fight')

    def primary_center(self):
        return user32.GetSystemMetrics(0) // 2, user32.GetSystemMetrics(1) // 2

    def dims(self):
        k = float(self.settings['size']) * self.dpi
        return int(self.BASE[0] * k) // 2 * 2, int(self.BASE[1] * k) // 2 * 2

    def ppu(self):
        return self.H / (2 * self.world.cam.dist * math.tan(math.radians(self.world.cam.fov / 2)))

    # ---------------------------------------------------------- window
    def create_window(self):
        self.msg_taskbar = RegisterWindowMessageW('TaskbarCreated')
        hinst = GetModuleHandleW(None)
        self._proc = WNDPROC(self.wndproc)
        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.style = 0x0008  # CS_DBLCLKS
        wc.lpfnWndProc = self._proc
        wc.hInstance = hinst
        wc.hCursor = LoadCursorW(None, ctypes.c_void_p(32649))  # IDC_HAND
        wc.lpszClassName = 'RoboPetsWnd'
        RegisterClassExW(ctypes.byref(wc))
        ex = 0x00080000 | 0x00000008 | 0x00000080   # LAYERED | TOPMOST | TOOLWINDOW
        self.hwnd = CreateWindowExW(ex, 'RoboPetsWnd', 'RoboPets', 0x80000000, 0, 0, self.W, self.H, None, None,
                                    hinst, None)
        ShowWindow(self.hwnd, 4)  # SW_SHOWNOACTIVATE
        self.msg_taskbar = RegisterWindowMessageW('TaskbarCreated')

    def make_dib(self):
        if getattr(self, 'hbmp', None):
            DeleteObject(self.hbmp)
        self.hdc_screen = GetDC(None)
        if not getattr(self, 'hdc_mem', None):
            self.hdc_mem = CreateCompatibleDC(self.hdc_screen)
        bi = BITMAPINFOHEADER()
        bi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bi.biWidth, bi.biHeight, bi.biPlanes, bi.biBitCount = self.W, self.H, 1, 32
        bits = ctypes.c_void_p()
        self.hbmp = CreateDIBSection(self.hdc_mem, ctypes.byref(bi), 0, ctypes.byref(bits), None, 0)
        SelectObject(self.hdc_mem, self.hbmp)
        self.buf = (ctypes.c_ubyte * (self.W * self.H * 4)).from_address(bits.value)
        self.pix = np.frombuffer(self.buf, np.uint8).reshape(self.H, self.W, 4)

    def resize(self, k):
        feet = self.wy + self.ground_row
        cx = self.wx + self.W / 2
        self.settings['size'] = k
        self.W, self.H = self.dims()
        self.r.resize(self.W, self.H)
        self.world.cam.calibrate(self.W / self.H)
        self.ground_row = self.H * 0.89
        self.world.ppu = self.ppu()
        self.wx, self.wy = cx - self.W / 2, feet - self.ground_row
        self.make_dib()
        self.save()

    def save(self):
        self.settings['x'] = int(self.wx)
        try:
            with open(SETTINGS, 'w', encoding='utf-8') as fh:
                json.dump(self.settings, fh)
        except Exception:
            pass

    # ---------------------------------------------------------- tray + menu
    def make_icon(self):
        from PIL import Image, ImageDraw
        im = Image.new('RGBA', (32, 32), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((1, 1, 30, 30), 7, fill=(184, 20, 29, 255))
        d.polygon([(30, 1), (30, 30), (1, 30)], fill=(18, 170, 192, 255))
        d.rounded_rectangle((8, 9, 23, 22), 3, fill=(36, 39, 45, 255))
        d.rectangle((10, 13, 21, 15), fill=(255, 182, 39, 255))
        a = np.array(im)
        bgra = a[:, :, [2, 1, 0, 3]].copy()
        bgra[:, :, :3] = (bgra[:, :, :3].astype(np.uint16) * bgra[:, :, 3:4] // 255).astype(np.uint8)
        bi = BITMAPINFOHEADER()
        bi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bi.biWidth, bi.biHeight, bi.biPlanes, bi.biBitCount = 32, -32, 1, 32
        bits = ctypes.c_void_p()
        col = CreateDIBSection(self.hdc_screen, ctypes.byref(bi), 0, ctypes.byref(bits), None, 0)
        ctypes.memmove(bits.value, bgra.tobytes(), 32 * 32 * 4)
        mask = CreateBitmap(32, 32, 1, 1, (ctypes.c_ubyte * 128)())
        ii = ICONINFO(True, 0, 0, mask, col)
        return CreateIconIndirect(ctypes.byref(ii))

    def add_tray(self):
        try:
            if not getattr(self, 'hicon', None):
                self.hicon = self.make_icon()
            nid = NOTIFYICONDATAW()
            nid.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
            nid.hWnd, nid.uID, nid.uFlags = self.hwnd, 1, 1 | 2 | 4
            nid.uCallbackMessage, nid.hIcon = WM_TRAY, self.hicon
            nid.szTip = 'RoboPets - right-click for menu, left-click to show/hide'
            self.nid = nid
            Shell_NotifyIconW(0, ctypes.byref(nid))
        except Exception:
            log('tray failed\n' + traceback.format_exc())

    def show_menu(self):
        w = self.world
        m = CreatePopupMenu()

        def add(menu, i, text, checked=False):
            AppendMenuW(menu, 0x8 if checked else 0, i, text)
        add(m, 1, 'Transform both\tT')
        add(m, 2, ('Land' if w.mode == 'fly' else 'Fly') + '\tright-click')
        add(m, 3, 'Fight for 8 seconds\thold right mouse')
        AppendMenuW(m, 0x800, 0, None)
        cm = CreatePopupMenu()
        for i, (key, label) in enumerate(VIEWS):
            add(cm, 10 + i, '%s\t%d' % (label, i + 1), w.cam.preset == key)
        AppendMenuW(m, 0x10, cm, 'Camera view')
        sm = CreatePopupMenu()
        for i, (label, k) in enumerate(self.SIZES):
            add(sm, 20 + i, label, abs(self.settings['size'] - k) < 1e-6)
        AppendMenuW(m, 0x10, sm, 'Size')
        add(m, 30, 'Wander around on their own', w.wander)
        add(m, 31, 'Sound effects', self.settings['sound'])
        add(m, 32, 'Start with Windows', autostart_get())
        AppendMenuW(m, 0x800, 0, None)
        add(m, 40, 'Show pets' if self.hidden else 'Hide pets')
        add(m, 99, 'Quit RoboPets')
        pt = wt.POINT()
        GetCursorPos(ctypes.byref(pt))
        SetForegroundWindow(self.hwnd)
        cmd = TrackPopupMenu(m, 0x100 | 0x2 | 0x80, pt.x, pt.y, 0, self.hwnd, None)
        PostMessageW(self.hwnd, WM_NULL, 0, 0)
        DestroyMenu(m)
        self.command(cmd)

    def command(self, cmd):
        w = self.world
        if cmd == 1:
            w.transform()
        elif cmd == 2:
            w.toggle_fly()
        elif cmd == 3:
            w.start_fight(8.0)
        elif 10 <= cmd < 10 + len(VIEWS):
            self.set_view(VIEWS[cmd - 10][0])
        elif 20 <= cmd < 20 + len(self.SIZES):
            self.resize(self.SIZES[cmd - 20][1])
        elif cmd == 30:
            w.wander = self.settings['wander'] = not w.wander
            self.save()
        elif cmd == 31:
            self.settings['sound'] = not self.settings['sound']
            self.save()
        elif cmd == 32:
            try:
                autostart_set(not autostart_get())
            except Exception:
                log(traceback.format_exc())
        elif cmd == 40:
            self.set_hidden(not self.hidden)
        elif cmd == 99:
            self.quit()

    def set_view(self, key):
        self.world.cam.set(key)
        self.world.say(dict(VIEWS)[key] + ' view')

    def set_hidden(self, hide):
        self.hidden = hide
        ShowWindow(self.hwnd, 0 if hide else 4)

    def quit(self):
        self.save()
        try:
            Shell_NotifyIconW(2, ctypes.byref(self.nid))
        except Exception:
            pass
        self.running = False
        PostQuitMessage(0)

    # ---------------------------------------------------------- input
    def cursor(self):
        pt = wt.POINT()
        GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def wndproc(self, hwnd, msg, wp, lp):
        try:
            r = self.handle(msg, wp, lp)
            if r is not None:
                return r
        except Exception:
            log('input error\n' + traceback.format_exc())
        return DefWindowProcW(hwnd, msg, wp, lp)

    def handle(self, msg, wp, lp):
        w = self.world
        if msg == WM_LBUTTONDOWN:
            x, y = self.cursor()
            self.drag = {'x0': x, 'y0': y, 'wx0': self.wx, 'wy0': self.wy, 'moved': False, 'samples': [],
                         'lx': x - self.wx, 'ly': y - self.wy}
            SetCapture(self.hwnd)
            return 0
        if msg == WM_MOUSEMOVE and self.drag:
            x, y = self.cursor()
            d = self.drag
            if abs(x - d['x0']) + abs(y - d['y0']) > 5:
                d['moved'] = True
            if d['moved']:
                self.wx = d['wx0'] + x - d['x0']
                ny = d['wy0'] + y - d['y0']
                self.wy = min(ny, self.ground_y())
                w.falling = False
                self.vy = 0.0
            return 0
        if msg == WM_LBUTTONUP:
            d, self.drag = self.drag, None
            ReleaseCapture()
            w.drag = None
            if d and not d['moved']:
                a = w.pick(d['lx'], d['ly'])
                self.pending_click = (time.perf_counter() + 0.28, a)
            elif d:
                vx, vy = self.drag_vel(d)
                if self.wy < self.ground_y() - 2 and w.mode != 'fly':
                    w.falling = True
                    self.vy = max(-1800.0, min(vy, 1800.0))
            self.save()
            return 0
        if msg == WM_LBUTTONDBLCLK:
            self.pending_click = None
            x, y = self.cursor()
            w.transform(w.pick(x - self.wx, y - self.wy))
            return 0
        if msg == WM_RBUTTONDOWN:
            self.rdown = time.perf_counter()
            SetCapture(self.hwnd)
            return 0
        if msg == WM_RBUTTONUP:
            ReleaseCapture()
            if self.rdown is not None:
                held = time.perf_counter() - self.rdown
                self.rdown = None
                if w.mode == 'fight' and w.fight_hold and w.fight_timer <= 0:
                    w.stop_fight()
                elif held < 0.3:
                    w.toggle_fly()
            return 0
        if msg == WM_CONTEXTMENU:
            return 0
        if msg == WM_MBUTTONUP:
            self.show_menu()
            return 0
        if msg == WM_MOUSEWHEEL:
            delta = ctypes.c_short(wp >> 16 & 0xFFFF).value / 120.0
            keys = wp & 0xFFFF
            if keys & 0x8:
                c = w.cam
                c.dist_goal = max(6.0, min(20.0, c.dist_goal - delta * 0.8))
            elif keys & 0x4:
                w.cam.orbit(0, delta * 8)
            else:
                w.cam.orbit(delta * 15, 0)
            return 0
        if msg == WM_KEYDOWN:
            k = wp
            if ord('1') <= k <= ord('7'):
                self.set_view(VIEWS[k - ord('1')][0])
            elif k == ord('T'):
                w.transform()
            elif k == ord('F'):
                w.toggle_fly()
            elif k == 0x20 and w.mode != 'fight':
                w.start_fight()
            elif k == ord('Q') or k == 0x25:
                w.cam.orbit(-15, 0)
            elif k == ord('E') or k == 0x27:
                w.cam.orbit(15, 0)
            elif k == 0x26:
                w.cam.orbit(0, 8)
            elif k == 0x28:
                w.cam.orbit(0, -8)
            elif k == ord('R'):
                self.set_view('front')
            elif k == ord('M'):
                self.show_menu()
            return 0
        if msg == WM_KEYUP and wp == 0x20:
            w.stop_fight()
            return 0
        if msg == WM_TRAY:
            ev = lp & 0xFFFF
            if ev == WM_RBUTTONUP:
                self.show_menu()
            elif ev == WM_LBUTTONUP:
                self.set_hidden(not self.hidden)
            return 0
        if msg == self.msg_taskbar:
            self.add_tray()
            return 0
        if msg == WM_DESTROY:
            self.quit()
            return 0
        return None

    def drag_vel(self, d):
        s = d['samples']
        if len(s) >= 2 and s[-1][0] - s[0][0] > 0.01:
            return (s[-1][1] - s[0][1]) / (s[-1][0] - s[0][0]), (s[-1][2] - s[0][2]) / (s[-1][0] - s[0][0])
        return 0.0, 0.0

    def ground_y(self):
        _, _, _, b = work_area(self.wx + self.W / 2, self.wy + self.ground_row - 10)
        return b - self.ground_row

    # ---------------------------------------------------------- per frame
    def step(self, dt):
        w = self.world
        now = time.perf_counter()
        if self.pending_click and now > self.pending_click[0]:
            w.click(self.pending_click[1])
            self.pending_click = None
        if self.rdown is not None and now - self.rdown > 0.3 and w.mode != 'fight':
            w.start_fight()
            self.sound('clang')
        l, tp, rr, b = work_area(self.wx + self.W / 2, self.wy + self.ground_row - 10)
        ground = b - self.ground_row
        if self.drag and self.drag['moved']:
            d = self.drag
            d['samples'].append((now, self.wx, self.wy))
            d['samples'] = [s for s in d['samples'] if now - s[0] < 0.12]
            vx, vy = self.drag_vel(d)
            w.drag = (vx, vy, self.wy >= ground - 3)
        else:
            w.drag = None
        flying_hold = w.mode == 'fly' and w.sub in ('takeoff', 'hover', 'prep')
        if not self.drag and not flying_hold:
            if self.wy < ground - 1 or w.falling:
                self.vy += 2600 * self.dpi * dt
                self.wy += self.vy * dt
                w.falling = True
                if self.wy >= ground:
                    self.wy = ground
                    if self.vy > 700 * self.dpi:
                        for a in w.actors:
                            if a.form < 0.5:
                                a.land_t = 0.35
                        self.sound('hit')
                    self.vy = 0.0
                    w.falling = False
            else:
                self.wy = ground
        tr = w.travel
        if tr and tr.get('go') and not self.drag:
            step = tr['speed'] * dt
            nx = self.wx + tr['dir'] * step
            lo, hi = l - self.W * 0.25, rr - self.W * 0.75
            if nx < lo or nx > hi:
                tr['rem'] = 0
            else:
                self.wx = nx
                tr['rem'] -= step
        # head tracking: cursor in scene space when it is near
        cx, cy = self.cursor()
        lx, ly = cx - self.wx, cy - self.wy
        if -self.W * 0.6 < lx < self.W * 1.6 and -self.H < ly < self.H * 1.6 and hasattr(w, 'vp'):
            w.cursor = w.unproject(lx, ly, (self.W, self.H))
        else:
            w.cursor = None
        w.update(dt)
        while w.sounds:
            self.sound(w.sounds.pop(0))
        # hide during full-screen apps
        if int(now) % 2 == 0 and now - getattr(self, '_qcheck', 0) > 1.5:
            self._qcheck = now
            st = ctypes.c_int(0)
            try:
                shell32.SHQueryUserNotificationState(ctypes.byref(st))
                busy = st.value in (2, 3, 4)
            except Exception:
                busy = False
            if busy != self.auto_hidden:
                self.auto_hidden = busy
                if not self.hidden:
                    ShowWindow(self.hwnd, 0 if busy else 4)

    def sound(self, name):
        if not self.settings.get('sound', True):
            return
        p = os.path.join(self.sfx_dir, name + '.wav')
        if os.path.exists(p):
            try:
                import winsound
                winsound.PlaySound(p, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
            except Exception:
                pass

    def draw_toast(self):
        t = self.world.toast
        if not t:
            return
        text, life = t
        img = self.toast_cache.get((text, self.H))
        if img is None:
            from PIL import Image, ImageDraw, ImageFont
            sz = max(11, int(self.H * 0.045))
            try:
                font = ImageFont.truetype('segoeuib.ttf', sz)
            except Exception:
                font = ImageFont.load_default()
            tw = int(font.getlength(text)) + sz
            th = int(sz * 1.7)
            im = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.rounded_rectangle((0, 0, tw - 1, th - 1), th // 2, fill=(20, 22, 28, 200))
            d.text((sz // 2, int(sz * 0.3)), text, font=font, fill=(255, 255, 255, 255))
            a = np.array(im).astype(np.float32)
            pm = a[:, :, :3] * (a[:, :, 3:4] / 255.0)
            img = np.concatenate([pm[:, :, [2, 1, 0]], a[:, :, 3:4]], 2)[::-1].copy()   # BGRA bottom-up
            self.toast_cache = {(text, self.H): img}
        k = min(1.0, life / 0.4)
        h, wd = img.shape[:2]
        if wd >= self.W:
            return
        x0 = (self.W - wd) // 2
        y0 = int(self.H * 0.97) - h      # bottom-up rows => near the window top
        dst = self.pix[y0:y0 + h, x0:x0 + wd].astype(np.float32)
        src = img * k
        out = src + dst * (1 - src[:, :, 3:4] / 255.0)
        self.pix[y0:y0 + h, x0:x0 + wd] = np.clip(out, 0, 255).astype(np.uint8)

    def present(self):
        f = self.world.frame(self.W / self.H, (self.W, self.H))
        self.r.render(f)
        GdiFlush()
        self.r.read_into(self.buf)
        self.draw_toast()
        bl = BLENDFUNCTION(0, 0, 255, 1)
        UpdateLayeredWindow(self.hwnd, self.hdc_screen, ctypes.byref(wt.POINT(int(self.wx), int(self.wy))),
                            ctypes.byref(wt.SIZE(self.W, self.H)), self.hdc_mem, ctypes.byref(wt.POINT(0, 0)), 0,
                            ctypes.byref(bl), 2)

    def run(self):
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass
        self.running = True
        msg = wt.MSG()
        last = time.perf_counter()
        nxt = last
        errors = 0
        topmost_t = 0
        while self.running:
            while PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
                if msg.message == 0x12:   # WM_QUIT
                    self.running = False
                    break
                TranslateMessage(ctypes.byref(msg))
                DispatchMessageW(ctypes.byref(msg))
            if not self.running:
                break
            now = time.perf_counter()
            if now >= nxt:
                dt = min(0.05, now - last)
                last = now
                w = self.world
                active = w.mode != 'idle' or self.drag or w.falling or any(a.form not in (0.0, 1.0) for a in w.actors)
                period = 1 / 30 if active or w.parts else 1 / 20
                nxt = max(nxt + period, now + period * 0.5)
                try:
                    self.step(dt)
                    if not (self.hidden or self.auto_hidden):
                        self.present()
                    if now - topmost_t > 3:
                        topmost_t = now
                        SetWindowPos(self.hwnd, HWND_TOPMOST, 0, 0, 0, 0, 0x1 | 0x2 | 0x10)
                except Exception:
                    errors += 1
                    log('frame error\n' + traceback.format_exc())
                    if errors > 50:
                        raise
            wait = max(0.0, nxt - time.perf_counter())
            MsgWait(0, None, int(wait * 1000), 0x04FF, 0x0004)
        self.save()


def main():
    try:
        App().run()
    except Exception:
        log('fatal\n' + traceback.format_exc())
        MessageBoxW(None, 'RoboPets could not start:\n\n' + traceback.format_exc()[-900:], 'RoboPets', 0x10)


if __name__ == '__main__':
    main()
