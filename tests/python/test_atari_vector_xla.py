import os

import ale_py
import numpy as np

os.environ["JAX_TRACEBACK_FILTERING"] = "off"


def test_xla():

    envs = ale_py.AtariVectorEnv("pong", num_envs=3)
    handle, reset, step = envs.xla()

    print(f"{handle=}")
    print(f"{reset=}")
    print(f"{step=}")

    handle, (obs, info) = reset(handle, np.array([0, 1, 2]), np.array([0, 1, 2]))
    print(f"{handle=}")
    print(f"{obs=}")
    print(f"{info=}")

    handle, (obs, rew, terminated, truncated, info) = step(
        handle, np.array([0, 1, 2]), np.array([1, 1, 1])
    )
    print(f"{handle=}")
    print(f"{obs=}")
    print(f"{rew=}")
    print(f"{terminated=}")
    print(f"{truncated=}")
    print(f"{info=}")
