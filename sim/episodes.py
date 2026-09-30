#!/usr/bin/env python3
"""Evaluation harness (reads ground truth, never part of the pipeline): N randomised episodes of the whole task.
Each episode: /sim/reset (new object yaws), run the task manager once, ask eval_node where everything ended up.

  (sim/ros_cell.py running, then ros2 launch ppp_bringup all.launch.py eval:=true)
  source ros2/install/setup.bash
  .venv/bin/python sim/episodes.py -n 10            -> output/episodes.json, output/episodes/task_NN.log

An object counts as placed only if the ground truth has it on a shelf level, resting upright or on its side (tilt
within 15 deg of 0 or 90 deg): not tipped over at an angle, not fallen off.
"""
import argparse
import json
import os
import subprocess
import time

import rclpy
from std_srvs.srv import Trigger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def call(node, name, timeout=20.0):
    c = node.create_client(Trigger, name)
    if not c.wait_for_service(timeout_sec=10.0):
        raise RuntimeError(name + ' not available')
    f = c.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, f, timeout_sec=timeout)
    return f.result()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-n', type=int, default=10, help='episodes')
    ap.add_argument('--out', default=os.path.join(REPO, 'output', 'episodes.json'))
    a = ap.parse_args()
    rclpy.init()
    node = rclpy.create_node('episodes')
    episodes = []
    for k in range(a.n):
        call(node, '/sim/reset')
        time.sleep(3.0)                                   # objects settle
        t0 = time.time()
        log = subprocess.run(['ros2', 'launch', 'ppp_bringup', 'task.launch.py'], capture_output=True, text=True,
                             timeout=900).stdout
        os.makedirs(os.path.join(os.path.dirname(a.out), 'episodes'), exist_ok=True)
        with open(os.path.join(os.path.dirname(a.out), 'episodes', 'task_%02d.log' % k), 'w') as f:
            f.write(log)
        after = json.loads(call(node, '/eval_node/report').message)
        ep = {'episode': k, 'wall_s': round(time.time() - t0, 1), 'objects': {}}
        for t, g in after.items():
            said = 'PLACED' if ('%s: PLACED' % t) in log else 'FAILED'
            tilt = g['z_axis_tilt_deg']
            placed = g['location'].startswith('shelf') and min(abs(tilt), abs(tilt - 90)) < 15
            ep['objects'][t] = {'task_says': said, 'ground_truth': g['location'], 'tilt_deg': round(tilt, 1),
                                'placed': placed, 'pose_errors': g['pose_errors'][-1:] if g['pose_errors'] else []}
        episodes.append(ep)
        print('episode %d (%.0f s): %s' % (k, ep['wall_s'], ', '.join(
            '%s %s (task: %s)' % (t, 'PLACED' if o['placed'] else 'no - ' + o['ground_truth'], o['task_says'])
            for t, o in ep['objects'].items())), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(episodes, f, indent=1)
    targets = list(episodes[0]['objects'])
    print('\n%-18s %8s %14s' % ('target', 'placed', 'task agreed'))
    for t in targets:
        ok = sum(e['objects'][t]['placed'] for e in episodes)
        agree = sum(e['objects'][t]['placed'] == (e['objects'][t]['task_says'] == 'PLACED') for e in episodes)
        print('%-18s %5d/%-3d %11d/%d' % (t, ok, len(episodes), agree, len(episodes)))
    print('-> %s' % os.path.relpath(a.out, REPO))
    rclpy.shutdown()


if __name__ == '__main__':
    main()
