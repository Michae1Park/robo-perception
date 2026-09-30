"""Record the demo: drive the prompt UI (a text prompt, then an example image) in a headless Chrome and record the page,
while the sim records its own view. scripts/demo_gif.py then stacks the two and makes the GIFs (D-045).

  1  .venv-sim/bin/python sim/ros_cell.py --record output/rec/sim
  2  source ros2/install/setup.bash && ros2 launch ppp_bringup all.launch.py eval:=true
  3  source ros2/install/setup.bash && .venv/bin/python ui/prompt_ui.py
  4  .venv/bin/python scripts/record_demo.py              (needs: .venv/bin/pip install playwright; uses the system Chrome)
  5  .venv/bin/python scripts/demo_gif.py

Writes output/rec/ui/*.webm and output/rec/events.json (wall times of the page start and of each prompt).
"""
import argparse
import glob
import json
import os
import time
import urllib.request

from playwright.sync_api import sync_playwright

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SIZE = {'width': 1600, 'height': 780}   # the whole page, no scrolling


def status(url):
    with urllib.request.urlopen(url + '/api/status', timeout=5) as r:
        return json.load(r)


def wait_task(url, timeout):
    """Until the task the page just started has finished. -> its status."""
    t0 = time.time()
    while not status(url)['running']:           # it starts after set_prompt has found the object
        if time.time() - t0 > 60:
            return status(url)
        time.sleep(0.5)
    while status(url)['running']:
        if time.time() - t0 > timeout:
            raise TimeoutError('task still running after %d s' % timeout)
        time.sleep(1.0)
    return status(url)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--url', default='http://localhost:8088')
    ap.add_argument('--out', default=os.path.join(REPO, 'output', 'rec'))
    ap.add_argument('--text', default='the yellow bottle', help='the text prompt (first pick)')
    ap.add_argument('--image', default='tomato soup can', help='the sample image to prompt with (second pick)')
    ap.add_argument('--timeout', type=float, default=420, help='s per task')
    args = ap.parse_args()
    ui_dir = os.path.join(args.out, 'ui')
    os.makedirs(ui_dir, exist_ok=True)
    for f in glob.glob(os.path.join(ui_dir, '*.webm')):
        os.remove(f)
    events = {}

    def mark(name):
        events[name] = time.time()
        print('[record_demo] %s' % name, flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=True)
        ctx = browser.new_context(viewport=SIZE, record_video_dir=ui_dir, record_video_size=SIZE)
        page = ctx.new_page()
        mark('page')                                   # the video starts here
        page.goto(args.url)
        page.wait_for_timeout(3000)

        # 1: text
        page.click('#tab-text')
        page.click('#text')
        page.type('#text', args.text, delay=90)
        page.wait_for_timeout(800)
        mark('text_go')
        page.click('#go')
        s = wait_task(args.url, args.timeout)
        mark('text_done')
        print('[record_demo] text: %s' % (s['result'] or s), flush=True)
        page.wait_for_timeout(3000)

        # 2: example image
        page.click('#tab-image')
        page.wait_for_timeout(800)
        page.click('#thumbs img[alt="%s"]' % args.image)
        page.wait_for_timeout(2000)
        mark('image_go')
        page.click('#go')
        s = wait_task(args.url, args.timeout)
        mark('image_done')
        print('[record_demo] image: %s' % (s['result'] or s), flush=True)
        page.wait_for_timeout(3000)
        mark('end')
        ctx.close()                                    # writes the video
        browser.close()

    with open(os.path.join(args.out, 'events.json'), 'w') as f:
        json.dump(events, f, indent=1)
    print('[record_demo] %s' % glob.glob(os.path.join(ui_dir, '*.webm')), flush=True)


if __name__ == '__main__':
    main()
