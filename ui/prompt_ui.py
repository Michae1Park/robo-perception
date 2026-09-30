"""Prompt the robot from a web page: type a phrase or drop an example image, and the Panda picks that object and puts
it on the shelf.

  source ros2/install/setup.bash && .venv/bin/python ui/prompt_ui.py      (steps 1 + 2 of docs/ROS.md running)
  -> http://localhost:8088 (Cursor forwards the port) or http://192.168.33.118:8088

The prompt goes to detect_node ~/set_prompt, which finds the object in the live frame and names it; if it is a target,
the task (task.launch.py) runs for it, and its Detect uses the prompt. The page shows the fixed camera, what detect_node
saw (~/overlay), the wrist camera (it looks into the shelf before each pick), the task's steps and its log.
"""
import argparse
import base64
import glob
import json
import os
import re
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from ppp_interfaces.srv import SetPrompt

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(HERE)
SAMPLES = os.path.join(REPO, 'assets', 'prompts')   # <ycb name>.png + .json ({"bbox": ...}): example images to try
STEPS = ['Find it', 'Grasps', 'Shelf space', 'Pick', 'Place']
# the task_manager log line (logger task_manager.<tree node>) that completes each step
STEP_DONE = [r'task_manager\.EstimatePose', r'task_manager\.PlanGrasps', r'task_manager\.GetPlacements',
             r'task_manager\.CheckGrasp', r'===== \w+: PLACED']
LOG_LINE = re.compile(r'\[(task_manager(?:\.\w+)?)\]: (.*)')
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def image_to_bgr(msg):
    ch = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}[msg.encoding]
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
    a = a.reshape(msg.height, msg.width, ch)[..., :3]
    return np.ascontiguousarray(a[..., ::-1] if msg.encoding.startswith('rgb') else a)


class Bridge(Node):
    """The ROS side: latest camera + overlay frames (as JPEG), detect_node ~/set_prompt, /sim/reset."""
    LIVE = ('live', 'wrist')   # camera feeds, thinned to ~10 fps

    def __init__(self, camera_topic, wrist_topic):
        super().__init__('prompt_ui')
        self.jpeg = {'live': None, 'wrist': None, 'overlay': None}
        self.seq = {k: 0 for k in self.jpeg}
        self.last = {k: 0.0 for k in self.LIVE}
        self.cond = threading.Condition()
        q = QoSProfile(depth=2)
        self.create_subscription(Image, camera_topic, lambda m: self.on_image('live', m), q)
        self.create_subscription(Image, wrist_topic, lambda m: self.on_image('wrist', m), q)
        self.create_subscription(Image, '/detect_node/overlay', lambda m: self.on_image('overlay', m), q)
        self.set_prompt = self.create_client(SetPrompt, '/detect_node/set_prompt')
        self.reset = self.create_client(Trigger, '/sim/reset')

    def on_image(self, which, msg):
        now = time.monotonic()
        if which in self.LIVE:   # ~10 fps is plenty for the page
            if now - self.last[which] < 0.1:
                return
            self.last[which] = now
        ok, buf = cv2.imencode('.jpg', image_to_bgr(msg), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with self.cond:
                self.jpeg[which] = buf.tobytes()
                self.seq[which] += 1
                self.cond.notify_all()

    def wait_frame(self, which, seq, timeout=5.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq[which] != seq, timeout)
            return self.seq[which], self.jpeg[which]

    def call(self, client, req, timeout=60.0):
        if not client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError('%s is not running' % client.srv_name)
        done = threading.Event()
        fut = client.call_async(req)
        fut.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            client.remove_pending_request(fut)
            raise TimeoutError('%s timed out' % client.srv_name)
        return fut.result()


class Task:
    """One task_manager run (ros2 launch ppp_bringup task.launch.py) at a time; its steps and log lines."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.state = {'running': False, 'target': '', 'prompt': None, 'done': -1, 'failed': False, 'result': '',
                      'log': [], 'steps': STEPS}

    def start(self, target, prompt):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                raise RuntimeError('a task is already running')
            self.state.update(running=True, target=target, prompt=prompt, done=-1, failed=False, result='', log=[])
            self.proc = subprocess.Popen(
                ['ros2', 'launch', 'ppp_bringup', 'task.launch.py', 'targets:=[%s]' % target],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
                env={**os.environ, 'RCUTILS_COLORIZED_OUTPUT': '0'})
        threading.Thread(target=self.read, args=(self.proc,), daemon=True).start()

    def read(self, proc):
        for raw in proc.stdout:
            line = ANSI.sub('', raw.rstrip())
            m = LOG_LINE.search(line)
            if not m:
                continue
            who, msg = m.groups()
            with self.lock:
                s = self.state
                s['log'] = (s['log'] + ['%s: %s' % (who.split('.')[-1], msg)])[-200:]
                for i, pat in enumerate(STEP_DONE):
                    if i > s['done'] and re.search(pat, line):
                        s['done'] = i
                if re.search(r'===== \w+: FAILED', line):
                    s['failed'] = True
                if msg.startswith('done:'):
                    s['result'] = msg
        proc.wait()
        with self.lock:
            s = self.state
            s['running'] = False
            if not s['result']:
                s['result'] = 'stopped' if proc.returncode in (-signal.SIGINT, -signal.SIGTERM) else \
                    'task exited (%s)' % proc.returncode
                s['failed'] = s['done'] < len(STEPS) - 1

    def stop(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                os.killpg(self.proc.pid, signal.SIGINT)

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))


def samples():
    out = []
    for png in sorted(glob.glob(os.path.join(SAMPLES, '*.png'))):
        name = os.path.splitext(os.path.basename(png))[0]
        try:
            with open(os.path.splitext(png)[0] + '.json') as f:
                bbox = json.load(f)['bbox']
        except (OSError, KeyError, ValueError):
            bbox = None
        out.append({'name': name.split('_', 1)[-1].replace('_', ' '), 'url': '/samples/' + name + '.png',
                    'bbox': bbox})
    return out


def make_handler(bridge, task):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, body, ctype='application/json'):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/':
                with open(os.path.join(HERE, 'index.html'), 'rb') as f:
                    return self.send(200, f.read(), 'text/html; charset=utf-8')
            if path == '/api/samples':
                return self.send(200, samples())
            if path.startswith('/samples/'):
                f = os.path.join(SAMPLES, os.path.basename(path))
                if not os.path.isfile(f):
                    return self.send(404, {'error': 'no such sample'})
                with open(f, 'rb') as fh:
                    return self.send(200, fh.read(), 'image/png')
            if path == '/api/status':
                return self.send(200, task.snapshot())
            if path in ('/stream/live', '/stream/wrist', '/stream/overlay'):
                return self.stream(path.rsplit('/', 1)[-1])
            self.send(404, {'error': 'not found'})

        def stream(self, which):
            """MJPEG: the <img> tag shows each new frame as it arrives."""
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            seq = -1
            try:
                while True:
                    seq, jpg = bridge.wait_frame(which, seq)
                    if jpg is None:
                        continue
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(jpg))
                    self.wfile.write(jpg + b'\r\n')
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            n = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(n) or b'{}')
            try:
                if self.path == '/api/prompt':
                    return self.send(200, self.prompt(body))
                if self.path == '/api/stop':
                    task.stop()
                    return self.send(200, {'ok': True})
                if self.path == '/api/reset':
                    res = bridge.call(bridge.reset, Trigger.Request(), timeout=10.0)
                    return self.send(200, {'ok': res.success, 'message': res.message})
                if self.path == '/api/clear':
                    res = bridge.call(bridge.set_prompt, SetPrompt.Request())
                    return self.send(200, {'ok': True, 'message': res.message})
            except Exception as e:  # noqa: BLE001  (shown on the page)
                return self.send(500, {'error': str(e)})
            self.send(404, {'error': 'not found'})

        def prompt(self, body):
            """{text} or {image: data URL, bbox} -> detect_node ~/set_prompt -> start the task for the target."""
            req = SetPrompt.Request()
            if body.get('image'):
                data = base64.b64decode(body['image'].split(',', 1)[-1])
                bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError('could not read that image')
                req.image.height, req.image.width = bgr.shape[:2]
                req.image.encoding, req.image.step = 'bgr8', bgr.shape[1] * 3
                req.image.data = bgr.tobytes()
                if body.get('bbox'):
                    req.bbox = [float(v) for v in body['bbox']]
            else:
                req.text = body.get('text', '').strip()
                if not req.text:
                    raise ValueError('type what to pick, or give an example image')
            if task.snapshot()['running']:
                raise RuntimeError('a task is already running')
            res = bridge.call(bridge.set_prompt, req)
            out = {'found': res.found, 'target': res.target, 'label': res.label, 'score': res.score,
                   'message': res.message, 'started': False}
            if res.target and body.get('pick', True):
                task.start(res.target, {'text': req.text} if req.text else {'thumb': body.get('thumb', '')})
                out['started'] = True
            return out

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', type=int, default=8088)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--camera', default='/camera/camera/color/image_raw', help='fixed camera image topic')
    ap.add_argument('--wrist-camera', default='/wrist_camera/camera/color/image_raw', help='wrist camera image topic')
    args = ap.parse_args()

    rclpy.init()
    bridge = Bridge(args.camera, args.wrist_camera)
    threading.Thread(target=rclpy.spin, args=(bridge,), daemon=True).start()
    task = Task()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(bridge, task))
    server.daemon_threads = True
    print('[prompt_ui] http://localhost:%d' % args.port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        task.stop()
        server.server_close()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
