"""Re-summarize one or more trials.jsonl files: `ros2 run ppp_eval summarize results/trials.jsonl`."""
import argparse
import json

from .core.metrics import summarize, to_markdown


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trials', nargs='+', help='trials.jsonl file(s)')
    ap.add_argument('--by-object', action='store_true', help='also print one table per target object')
    args = ap.parse_args(argv)
    trials = []
    for path in args.trials:
        with open(path) as f:
            trials += [json.loads(line) for line in f if line.strip()]
    print(to_markdown(summarize(trials)))
    if args.by_object:
        for name in sorted({t.get('target') for t in trials if t.get('target')}):
            print('### %s\n' % name)
            print(to_markdown(summarize([t for t in trials if t.get('target') == name])))


if __name__ == '__main__':
    main()
