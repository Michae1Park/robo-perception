"""Blocking helpers for calling services/actions from inside callbacks.

The nodes run a MultiThreadedExecutor with reentrant callback groups, so a callback may block
on a future while other threads keep spinning.
"""
import threading
import time


class CallError(RuntimeError):
    pass


def wait_future(future, timeout):
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    if not done.wait(timeout):
        future.cancel()
        raise CallError('timed out after %.1fs' % timeout)
    if future.exception() is not None:
        raise CallError(str(future.exception()))
    return future.result()


def call_service(client, request, timeout=10.0, wait_ready=5.0):
    if not client.wait_for_service(timeout_sec=wait_ready):
        raise CallError('service %s not available' % client.srv_name)
    return wait_future(client.call_async(request), timeout)


def run_action(client, goal, timeout=60.0, wait_ready=5.0, feedback_cb=None):
    """Send an action goal and wait for its result. Returns the result message."""
    if not client.wait_for_server(timeout_sec=wait_ready):
        raise CallError('action server not available')
    handle = wait_future(client.send_goal_async(goal, feedback_callback=feedback_cb), wait_ready)
    if not handle.accepted:
        raise CallError('goal rejected')
    try:
        return wait_future(handle.get_result_async(), timeout).result
    except CallError:
        handle.cancel_goal_async()
        raise


class Stopwatch:
    def __init__(self):
        self.t0 = time.perf_counter()

    def ms(self):
        return (time.perf_counter() - self.t0) * 1000.0
