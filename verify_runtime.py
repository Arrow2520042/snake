"""Runtime verifier for optional acceleration backends.

Prints interpreter and module availability diagnostics used by bootstrap scripts.
Exits with code 1 when numba or Cython PER backend is unavailable.
"""

import importlib.util
import sys


def _has_module(name):
    return importlib.util.find_spec(name) is not None


def main():
    print(f"python_executable={sys.executable}")
    print(f"numba_spec={_has_module('numba')}")
    print(f"cython_spec={_has_module('Cython')}")
    print(f"per_backend_spec={_has_module('per_cython_backend')}")

    from game_model import SnakeGameAI
    from dqn_agent import PrioritizedReplayBuffer

    env = SnakeGameAI(render=False)
    rb = PrioritizedReplayBuffer(1024)

    print(f"numba_enabled={env.numba_enabled}")
    print(f"per_cython_enabled={rb.cython_enabled}")

    if not env.numba_enabled:
        print("ERROR: numba is disabled in current interpreter.")
        return 1
    if not rb.cython_enabled:
        print("ERROR: per_cython backend is disabled in current interpreter.")
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
