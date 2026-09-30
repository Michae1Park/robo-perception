"""Stack the prompt UI recording (top) and the sim's view (bottom) on one timeline, speed it up, and make GIFs (D-045).

  .venv/bin/python scripts/demo_gif.py [--speeds 2 3 4] [--width 800] [--fps 10]

Inputs from scripts/record_demo.py and sim/ros_cell.py --record: output/rec/{ui/*.webm, sim/<wall ms>.jpg,
events.json}. The sim frames are named by wall time, so they line up with the page's video (which starts at
events.json "page"). Writes output/rec/demo_<speed>x.gif.
"""
import argparse
import glob
import json
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def sim_concat(sim_dir, t0, t1, path):
    """ffmpeg concat list of the sim frames from wall time t0 to t1, each shown until the next one."""
    frames = sorted((int(os.path.basename(f)[:-4]) / 1000.0, f) for f in glob.glob(os.path.join(sim_dir, '*.jpg')))
    before = [x for x in frames if x[0] <= t0]
    frames = before[-1:] + [x for x in frames if t0 < x[0] <= t1]
    if len(frames) < 2:
        raise SystemExit('no sim frames between the page start and the end: was ros_cell.py run with --record?')
    with open(path, 'w') as f:
        for i, (t, jpg) in enumerate(frames):
            start = max(t, t0)
            end = frames[i + 1][0] if i + 1 < len(frames) else t1
            f.write("file '%s'\nduration %.4f\n" % (jpg, max(end - start, 0.001)))
        f.write("file '%s'\n" % frames[-1][1])   # the concat demuxer ignores the last duration otherwise


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rec', default=os.path.join(REPO, 'output', 'rec'))
    ap.add_argument('--speeds', type=float, nargs='+', default=[2, 3, 4])
    ap.add_argument('--width', type=int, default=800)
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--lead', type=float, default=2.5, help='s after the page opened to start at (page loading)')
    args = ap.parse_args()
    with open(os.path.join(args.rec, 'events.json')) as f:
        ev = json.load(f)
    webm = sorted(glob.glob(os.path.join(args.rec, 'ui', '*.webm')))[-1]
    t0, t1 = ev['page'], ev['end']
    concat = os.path.join(args.rec, 'sim_frames.txt')
    sim_concat(os.path.join(args.rec, 'sim'), t0, t1, concat)

    for speed in args.speeds:
        out = os.path.join(args.rec, 'demo_%gx.gif' % speed)
        label = ("drawtext=fontfile=%s:text='%g×':x=w-tw-14:y=h-th-12:fontsize=26:fontcolor=white:"
                 "box=1:boxcolor=black@0.55:boxborderw=8," % (FONT, speed)) if os.path.isfile(FONT) else ''
        graph = (
            '[0:v]fps=30,trim=start={lead},setpts=PTS-STARTPTS,scale={w}:-2:flags=lanczos[ui];'
            '[1:v]fps=30,trim=start={lead},setpts=PTS-STARTPTS,scale={w}:-2:flags=lanczos[sim];'
            '[ui][sim]vstack=shortest=1,setpts=PTS/{speed},fps={fps},{label}'
            'split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle'
        ).format(lead=args.lead, w=args.width, speed=speed, fps=args.fps, label=label)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', webm,
                        '-f', 'concat', '-safe', '0', '-i', concat,
                        '-filter_complex', graph, out], check=True)
        print('%s  %.1f MB' % (out, os.path.getsize(out) / 1e6), flush=True)


if __name__ == '__main__':
    main()
