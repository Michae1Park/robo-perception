"""Record the demo: drive the prompt UI (an example image, then a text prompt; --order) in a headless Chrome and record
the page, while the sim records its own view. scripts/demo_gif.py then stacks the two and makes the GIFs (D-045).

  1  .venv-sim/bin/python sim/ros_cell.py --record output/rec/sim --record-size 1920 1080
  2  source ros2/install/setup.bash && ros2 launch ppp_bringup all.launch.py eval:=true
  3  source ros2/install/setup.bash && .venv/bin/python ui/prompt_ui.py
  4  .venv/bin/python scripts/record_demo.py              (needs: .venv/bin/pip install playwright; uses the system Chrome)
  5  .venv/bin/python scripts/demo_gif.py --motion-speed 8 --split

The page is captured with Chrome's screencast at --scale device pixels (Playwright's own video is a low-bitrate VP8)
into output/rec/ui/<wall time ms>.jpg, like the sim's frames. Also writes output/rec/events.json (wall times of the
page start and of each prompt, and the order) and output/rec/prompt_ui.svg (scripts/ui_svg.py, after the first pick).
"""
import argparse
import base64
import glob
import json
import os
import time
import urllib.request

from playwright.sync_api import sync_playwright

import ui_svg

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def status(url):
    with urllib.request.urlopen(url + '/api/status', timeout=5) as r:
        return json.load(r)


def wait_task(page, url, timeout):
    """Until the task the page just started has finished. -> its status. Waits in the page, so screencast frames
    keep being handled."""
    t0 = time.time()
    while not status(url)['running']:           # it starts after set_prompt has found the object
        if time.time() - t0 > 60:
            return status(url)
        page.wait_for_timeout(500)
    while status(url)['running']:
        if time.time() - t0 > timeout:
            raise TimeoutError('task still running after %d s' % timeout)
        page.wait_for_timeout(1000)
    return status(url)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--url', default='http://localhost:8088')
    ap.add_argument('--out', default=os.path.join(REPO, 'output', 'rec'))
    ap.add_argument('--order', default='image,text', help='the prompts in order: image,text or text,image')
    ap.add_argument('--text', default='the yellow bottle', help='the text prompt')
    ap.add_argument('--image', default='tomato soup can', help='the sample image to prompt with')
    ap.add_argument('--width', type=int, default=1280, help='viewport, CSS px (the whole page, no scrolling)')
    ap.add_argument('--height', type=int, default=780)
    ap.add_argument('--scale', type=float, default=2, help='device pixels per CSS px')
    ap.add_argument('--timeout', type=float, default=420, help='s per task')
    args = ap.parse_args()
    ui_dir = os.path.join(args.out, 'ui')
    os.makedirs(ui_dir, exist_ok=True)
    for f in glob.glob(os.path.join(ui_dir, '*')):
        os.remove(f)
    order = args.order.split(',')
    events = {'order': order}

    def mark(name):
        events[name] = time.time()
        print('[record_demo] %s' % name, flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=True)
        page = browser.new_page(viewport={'width': args.width, 'height': args.height}, device_scale_factor=args.scale)
        cdp = page.context.new_cdp_session(page)

        def frame(f):
            with open(os.path.join(ui_dir, '%d.jpg' % round(f['metadata']['timestamp'] * 1000)), 'wb') as out:
                out.write(base64.b64decode(f['data']))
            cdp.send('Page.screencastFrameAck', {'sessionId': f['sessionId']})

        cdp.on('Page.screencastFrame', frame)
        mark('page')
        page.goto(args.url, wait_until='domcontentloaded')   # 'load' never fires: the camera views are endless MJPEG streams
        cdp.send('Page.startScreencast', {'format': 'jpeg', 'quality': 92, 'maxWidth': round(args.width * args.scale),
                                          'maxHeight': round(args.height * args.scale)})
        page.wait_for_timeout(3000)

        for i, kind in enumerate(order):
            if kind == 'text':
                page.click('#tab-text')
                page.wait_for_timeout(800)
                page.fill('#text', '')
                page.click('#text')
                page.type('#text', args.text, delay=90)
                page.wait_for_timeout(800)
            else:
                page.click('#tab-image')
                page.wait_for_timeout(800)
                page.click('#thumbs img[alt="%s"]' % args.image)
                page.wait_for_timeout(2000)
            mark(kind + '_go')
            page.click('#go')
            s = wait_task(page, args.url, args.timeout)
            mark(kind + '_done')
            print('[record_demo] %s: %s' % (kind, s['result'] or s), flush=True)
            if i == 0:
                ui_svg.snapshot(page, os.path.join(args.out, 'prompt_ui.svg'))
            page.wait_for_timeout(3000)
        mark('end')
        cdp.send('Page.stopScreencast')
        browser.close()

    with open(os.path.join(args.out, 'events.json'), 'w') as f:
        json.dump(events, f, indent=1)
    print('[record_demo] %d frames in %s' % (len(os.listdir(ui_dir)), ui_dir), flush=True)


if __name__ == '__main__':
    main()
