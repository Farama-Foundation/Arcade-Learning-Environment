"""Tests for ``AtariVectorEnv(skip_discarded_frames=True)``.

With the flag on, the emulator skips drawing the frames the vector env throws
away. The emulation itself must be unchanged, and on most ROMs so are the
observations; on ``video_checkers`` and ``tic_tac_toe_3d`` they are not, which is
why the flag is off by default.
"""

import inspect

import numpy as np
import pytest
from ale_py import AtariVectorEnv

NUM_ENVS = 2
STEPS = 1500


def rollout(game, skip, **kwargs):
    """Step ``NUM_ENVS`` envs ``STEPS`` times with seeded random actions.

    Short episodes and sticky actions make the rollout cross many episode
    boundaries, so emulated resets, pooled resets and the noop starts after them
    are all exercised.
    """
    kwargs = (
        dict(
            num_envs=NUM_ENVS,
            max_num_frames_per_episode=2000,
            repeat_action_probability=0.25,
            reset_pool_size=4,
        )
        | kwargs
    )
    env = AtariVectorEnv(game, skip_discarded_frames=skip, **kwargs)
    obs, info = env.reset(seed=0)
    rng = np.random.default_rng(0)
    num_actions = int(env.single_action_space.n)

    observations = [obs]
    logic = [(info["lives"], info["frame_number"], info["episode_frame_number"])]
    for _ in range(STEPS):
        obs, reward, terminated, truncated, info = env.step(
            rng.integers(0, num_actions, size=NUM_ENVS)
        )
        observations.append(obs)
        logic.append(
            (
                reward,
                terminated,
                truncated,
                info["lives"],
                info["frame_number"],
                info["episode_frame_number"],
            )
        )
    env.close()
    return np.stack(observations), logic


def assert_logic_equal(expected, actual):
    assert len(expected) == len(actual)
    for step, (a, b) in enumerate(zip(expected, actual)):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y, err_msg=f"step {step}")


def test_default_is_off():
    """Observations must match earlier versions unless the caller opts in."""
    parameter = inspect.signature(AtariVectorEnv).parameters["skip_discarded_frames"]
    assert parameter.default is False


@pytest.mark.parametrize(
    "game", ["breakout", "pong", "video_checkers", "tic_tac_toe_3d"]
)
def test_game_logic_is_unchanged(game):
    """Rewards, terminations, truncations, lives and frame counters are identical.

    Includes the two ROMs whose observations differ: what differs there is which
    stale rows a short frame inherits, never the emulation.
    """
    _, drawn = rollout(game, skip=False)
    _, skipped = rollout(game, skip=True)
    assert_logic_equal(drawn, skipped)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        dict(maxpool=False),
        dict(grayscale=False, stack_num=1, maxpool=False),
        dict(episodic_life=True),
        dict(reset_pool_size=0),
    ],
    ids=["default", "no-maxpool", "rgb-no-maxpool", "episodic-life", "no-pool"],
)
def test_observations_are_unchanged_on_exact_rom(kwargs):
    """On breakout, skipping the discarded frames leaves every observation as it was."""
    drawn_obs, drawn = rollout("breakout", skip=False, **kwargs)
    skipped_obs, skipped = rollout("breakout", skip=True, **kwargs)
    assert_logic_equal(drawn, skipped)
    np.testing.assert_array_equal(drawn_obs, skipped_obs)


@pytest.mark.parametrize("maxpool", [True, False], ids=["maxpool", "no-maxpool"])
@pytest.mark.parametrize("game", ["video_checkers", "tic_tac_toe_3d"])
def test_observations_differ_on_inexact_roms(game, maxpool):
    """The documented exception, and evidence that the flag takes effect at all.

    Frames on these ROMs can end before the beam has covered the whole frame
    buffer. The rows they do not reach keep what the frame two before them drew,
    since the TIA alternates two buffers, and skipping changes which frame that
    was. Without this check the tests above would also pass if the flag did
    nothing.
    """
    drawn_obs, _ = rollout(game, skip=False, maxpool=maxpool)
    skipped_obs, _ = rollout(game, skip=True, maxpool=maxpool)
    assert not np.array_equal(drawn_obs, skipped_obs)
