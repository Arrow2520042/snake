"""Runtime verifier for optional acceleration backends.

Prints interpreter and module availability diagnostics used by bootstrap scripts.
By default, exits with code 0 even if optional accelerators are unavailable.
Use --strict to enforce accelerator availability.
"""

import argparse
import importlib.util
import sys


def _has_module(name):
    return importlib.util.find_spec(name) is not None


def main(argv=None):
    parser = argparse.ArgumentParser(description='Verify Snake runtime environment')
    parser.add_argument('--strict', action='store_true',
                        help='Exit with non-zero status when optional accelerators are unavailable')
    args = parser.parse_args(argv)

    print(f"python_executable={sys.executable}")
    print(f"numba_spec={_has_module('numba')}")
    print(f"cython_spec={_has_module('Cython')}")
    print(f"per_backend_spec={_has_module('per_cython_backend')}")

    from game_model import SnakeGameAI
    from replay_buffer import PrioritizedReplayBuffer

    env = SnakeGameAI(render=False)
    rb = PrioritizedReplayBuffer(1024)

    print(f"numba_enabled={env.numba_enabled}")
    print(f"per_cython_enabled={rb.cython_enabled}")

    failed = False
    if not env.numba_enabled:
        print("WARNING: numba is disabled in current interpreter.")
        failed = True
    if not rb.cython_enabled:
        print("WARNING: per_cython backend is disabled in current interpreter.")
        failed = True

    if failed and args.strict:
        return 1

    if failed:
        print("Runtime check passed in non-strict mode (training can run with slower fallbacks).")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
