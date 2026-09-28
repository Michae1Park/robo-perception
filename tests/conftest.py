import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for pkg in ('ppp_common', 'ppp_perception', 'ppp_manipulation', 'ppp_eval'):
    sys.path.insert(0, os.path.join(ROOT, 'ros2', pkg))
