"""Stack the sim's view (top) and the prompt UI recording (bottom) on one timeline, speed it up, and make GIFs (D-045).

  .venv/bin/python scripts/demo_gif.py [--speeds 2 3 4] [--width 800] [--fps 10] [--colors 256] [--dither bayer]
  .venv/bin/python scripts/demo_gif.py --motion-speed 8 [--prompt-speed 1.5] [--hold 4] [--split]

--motion-speed makes one GIF with a variable speed instead (D-046): the prompting (page load, typing / choosing the
image, and --hold s after Go for the detection) at --prompt-speed, each task at --motion-speed. --split cuts it
after the first prompt's task into two GIFs, *_image.gif and *_text.gif.
For the README GIFs (~10 MB each, UI text readable): --split --width 1280 --colors 48 --fps 8 --dither none.

Inputs from scripts/record_demo.py and sim/ros_cell.py --record: output/rec/{ui/<wall ms>.jpg, sim/<wall ms>.jpg,
events.json} (ui/*.webm in older recordings). Both are named by wall time, so they line up. Writes output/rec/demo_<speed>x.gif.
"""
import argparse
import glob
import json
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def frames_concat(frame_dir, t0, t1, path):
    """ffmpeg concat list of the frames (<wall time ms>.jpg) from wall time t0 to t1, each shown until the next one."""
    frames = sorted((int(os.path.basename(f)[:-4]) / 1000.0, f) for f in glob.glob(os.path.join(frame_dir, '*.jpg')))
    before = [x for x in frames if x[0] <= t0]
    frames = before[-1:] + [x for x in frames if t0 < x[0] <= t1]
    if len(frames) < 2:
        raise SystemExit('no frames in %s between the page start and the end' % frame_dir)
    with open(path, 'w') as f:
        for i, (t, jpg) in enumerate(frames):
            start = max(t, t0)
            end = frames[i + 1][0] if i + 1 < len(frames) else t1
            f.write("file '%s'\nduration %.4f\n" % (jpg, max(end - start, 0.001)))
        f.write("file '%s'\n" % frames[-1][1])   # the concat demuxer ignores the last duration otherwise


def badge(speed):
    """drawtext filter for the "3×" badge in the bottom right corner."""
    if not os.path.isfile(FONT):
        return 'null'
    return ("drawtext=fontfile=%s:text='%g×':x=w-tw-14:y=h-th-12:fontsize=26:fontcolor=white:"
            "box=1:boxcolor=black@0.55:boxborderw=8" % (FONT, speed))


def segments(ev, t0, end, hold, prompt_speed, motion_speed):
    """[(start, end, speed)] in video time: the prompting at prompt_speed, the tasks at motion_speed."""
    a, b = ev.get('order', ['text', 'image'])
    marks = [0.0, ev[a + '_go'] - t0 + hold, ev[a + '_done'] - t0, ev[b + '_go'] - t0 + hold, ev[b + '_done'] - t0, end]
    for i in range(1, len(marks)):
        marks[i] = min(max(marks[i], marks[i - 1]), end)
    speeds = [prompt_speed, motion_speed, prompt_speed, motion_speed, prompt_speed]
    return [(a, b, v) for a, b, v in zip(marks, marks[1:], speeds) if b - a > 0.05]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rec', default=os.path.join(REPO, 'output', 'rec'))
    ap.add_argument('--speeds', type=float, nargs='+', default=[2, 3, 4])
    ap.add_argument('--width', type=int, default=800)
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--colors', type=int, default=256, help='GIF palette size (fewer: smaller file)')
    ap.add_argument('--dither', default='bayer', choices=['bayer', 'none'], help='none: smaller file, some banding')
    ap.add_argument('--lead', type=float, default=2.5, help='s after the page opened to start at (page loading)')
    ap.add_argument('--motion-speed', type=float, help='variable speed: the tasks at this speed (one GIF)')
    ap.add_argument('--prompt-speed', type=float, default=1.5, help='variable speed: the prompting at this speed')
    ap.add_argument('--hold', type=float, default=4, help='variable speed: s after Go still at --prompt-speed')
    ap.add_argument('--split', action='store_true', help='variable speed: one GIF per prompt (_text, _image)')
    args = ap.parse_args()
    with open(os.path.join(args.rec, 'events.json')) as f:
        ev = json.load(f)
    t0, t1 = ev['page'], ev['end']
    webm = sorted(glob.glob(os.path.join(args.rec, 'ui', '*.webm')))        # older recordings: Playwright's video
    if webm:
        ui_in = ['-i', webm[-1]]
    else:                                                                   # screencast frames
        ui_in = ['-f', 'concat', '-safe', '0', '-i', os.path.join(args.rec, 'ui_frames.txt')]
        frames_concat(os.path.join(args.rec, 'ui'), t0, t1, ui_in[-1])
    concat = os.path.join(args.rec, 'sim_frames.txt')
    frames_concat(os.path.join(args.rec, 'sim'), t0, t1, concat)

    stack = ('[0:v]fps=30,trim=start={lead},setpts=PTS-STARTPTS,scale={w}:-2:flags=lanczos[ui];'
             '[1:v]fps=30,trim=start={lead},setpts=PTS-STARTPTS,scale={w}:-2:flags=lanczos[sim];'
             '[sim][ui]vstack=shortest=1').format(lead=args.lead, w=args.width)
    dither = 'bayer:bayer_scale=5' if args.dither == 'bayer' else 'none'
    gif = ',fps={fps},split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];' \
          '[b][p]paletteuse=dither={dither}:diff_mode=rectangle'.format(fps=args.fps, colors=args.colors, dither=dither)
    jobs = []
    if args.motion_speed:
        start = t0 + args.lead
        segs = segments(ev, start, t1 - start, args.hold, args.prompt_speed, args.motion_speed)
        name = 'demo_%gx-%gx' % (args.prompt_speed, args.motion_speed)
        if args.split:
            order = ev.get('order', ['text', 'image'])
            cut = ev[order[0] + '_done'] - start
            parts = [('%s_%s.gif' % (name, order[0]), [s for s in segs if s[1] <= cut + 1e-3]),
                     ('%s_%s.gif' % (name, order[1]), [s for s in segs if s[0] >= cut - 1e-3])]
        else:
            parts = [(name + '.gif', segs)]
        for out, part in parts:
            graph = stack + ',split=%d%s;' % (len(part), ''.join('[s%d]' % i for i in range(len(part))))
            for i, (a, b, v) in enumerate(part):
                graph += '[s%d]trim=start=%.3f:end=%.3f,setpts=(PTS-STARTPTS)/%g,%s[c%d];' % (i, a, b, v, badge(v), i)
            graph += ''.join('[c%d]' % i for i in range(len(part))) + 'concat=n=%d:v=1:a=0' % len(part) + gif
            jobs.append((out, graph, sum((b - a) / v for a, b, v in part)))
    else:
        for speed in args.speeds:
            jobs.append(('demo_%gx.gif' % speed, stack + ',setpts=PTS/%g,%s' % (speed, badge(speed)) + gif,
                         (t1 - t0 - args.lead) / speed))

    for name, graph, secs in jobs:
        out = os.path.join(args.rec, name)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error'] + ui_in + [
                        '-f', 'concat', '-safe', '0', '-i', concat,
                        '-filter_complex', graph, out], check=True)
        print('%s  ~%.0f s  %.1f MB' % (out, secs, os.path.getsize(out) / 1e6), flush=True)


if __name__ == '__main__':
    main()
