"""Tests for the vector environment's pooled snapshot-restore reset."""

import hashlib

import numpy as np
import pytest
from ale_py import AtariVectorEnv
from gymnasium.error import Error
from gymnasium.utils.env_checker import data_equivalence

POOL_SIZE = 8
# Short enough that the pool leaves its fill phase quickly, long enough that two
# different start states have diverged on screen by the end of an episode.
EPISODE_FRAMES = 400


def rollout_envs(game, episodes, **kwargs):
    """Drive every env through ``episodes`` truncated episodes.

    Returns, per env, one hash per episode of every observation in it, and the
    ``frame_number`` of every step. Actions cycle through the action set and
    restart with each episode, so an episode is a pure function of its start
    state. Hashing the whole episode matters: the start state is often invisible
    in the first frame.
    """
    kwargs = (
        dict(
            num_envs=1,
            max_num_frames_per_episode=EPISODE_FRAMES,
            repeat_action_probability=0.0,
            reset_pool_size=POOL_SIZE,
        )
        | kwargs
    )
    env = AtariVectorEnv(game, **kwargs)
    num_envs = env.num_envs
    obs, info = env.reset(seed=0)
    num_actions = int(env.single_action_space.n)

    signatures = [[] for _ in range(num_envs)]
    frame_numbers = [[int(n)] for n in info["frame_number"]]
    episodes_obs = [[o.tobytes()] for o in obs]
    t = np.zeros(num_envs, dtype=np.int32)
    while min(len(s) for s in signatures) < episodes:
        obs, _, term, trunc, info = env.step(t % num_actions)
        t += 1
        for i in range(num_envs):
            frame_numbers[i].append(int(info["frame_number"][i]))
            if term[i] or trunc[i]:
                signatures[i].append(
                    hashlib.blake2b(b"".join(episodes_obs[i])).digest()
                )
                # NextStep autoreset: the next step only resets (its action is
                # ignored) and returns the new episode's first observation, so
                # the action cycle restarts on the step after it.
                episodes_obs[i], t[i] = [], -1
            else:
                episodes_obs[i].append(obs[i].tobytes())
    env.close()
    return [s[:episodes] for s in signatures], frame_numbers


def rollout(game, episodes, **kwargs):
    """``rollout_envs`` for a single env."""
    signatures, frame_numbers = rollout_envs(game, episodes, num_envs=1, **kwargs)
    return signatures[0], frame_numbers[0]


# noop_max=1 with no fire reset emulates no frames after the console reset, so
# the first observation can only come from the screen restored with the snapshot.
DETERMINISTIC = dict(noop_max=1, use_fire_reset=False)


@pytest.mark.parametrize("game", ["breakout", "pong"])
def test_restored_resets_match_emulated_resets(game):
    """On a ROM with one start state, restored episodes equal emulated ones."""
    signatures, _ = rollout(game, POOL_SIZE + 6, **DETERMINISTIC)
    assert len(set(signatures)) == 1


@pytest.mark.parametrize("game", ["solaris", "space_war"])
def test_drawn_start_states_were_produced_by_real_resets(game):
    """Episodes after the fill phase start from states the fill phase captured."""
    signatures, _ = rollout(game, POOL_SIZE + 16, **DETERMINISTIC)
    captured, drawn = set(signatures[:POOL_SIZE]), set(signatures[POOL_SIZE:])
    assert len(captured) > 1, "the ROM produced a single start state"
    assert drawn <= captured
    assert len(drawn) > 1, "the pool collapsed onto one start state"


@pytest.mark.parametrize("game", ["breakout", "air_raid", "pooyan"])
def test_frame_number_matches_emulated_resets(game):
    """``frame_number`` must not see the counter rewind on restore.

    air_raid and pooyan run starting actions inside ``reset_game()``, which a
    restore has to count too.
    """
    _, with_pool = rollout(game, POOL_SIZE + 4)
    _, without_pool = rollout(game, POOL_SIZE + 4, reset_pool_size=0)
    assert with_pool == without_pool
    assert all(b >= a for a, b in zip(with_pool, with_pool[1:]))


def test_sticky_actions_match_emulated_resets():
    """A restored start state repeats NOOP, as ``reset_game()`` leaves it.

    Otherwise the first frames of a restored episode could repeat the previous
    episode's final action.
    """
    # High enough that a carried-over action outlasts the frames the ROM ignores input.
    kwargs = dict(DETERMINISTIC, repeat_action_probability=0.9)
    with_pool = rollout("breakout", POOL_SIZE + 6, **kwargs)
    without_pool = rollout("breakout", POOL_SIZE + 6, reset_pool_size=0, **kwargs)
    assert with_pool == without_pool


def test_pool_replays_its_states():
    """Once full, a one-state pool replays that state on every reset.

    space_war produces a different start state on most real resets, so this
    fails if the pool is never drawn from.
    """
    pooled, _ = rollout("space_war", 8, reset_pool_size=1, **DETERMINISTIC)
    emulated, _ = rollout("space_war", 8, reset_pool_size=0, **DETERMINISTIC)
    assert len(set(pooled)) == 1
    assert len(set(emulated)) > 1


def test_pool_size_is_split_between_envs():
    """``reset_pool_size`` is a total: two envs sharing two states keep one each."""
    signatures, _ = rollout_envs(
        "space_war", 8, num_envs=2, reset_pool_size=2, **DETERMINISTIC
    )
    for env_signatures in signatures:
        assert len(set(env_signatures)) == 1


def test_pool_smaller_than_num_envs():
    """A total below ``num_envs`` still gives every env one state."""
    env = AtariVectorEnv(
        "breakout", num_envs=4, reset_pool_size=2, reset_pool_prefill=True
    )
    env.reset(seed=0)
    for _ in range(50):
        env.step(np.zeros(4, dtype=np.int32))
    env.close()


@pytest.mark.parametrize("prefill", [False, True])
def test_seeded_rollouts_reproduce(prefill):
    """The same seed reproduces a rollout across the fill-to-draw transition."""
    first = rollout("tennis", POOL_SIZE + 8, reset_pool_prefill=prefill)
    second = rollout("tennis", POOL_SIZE + 8, reset_pool_prefill=prefill)
    assert first == second


def test_reseeding_clears_the_pool():
    """A seeded reset reloads the ROM, so it must not draw from the old pool."""
    kwargs = dict(num_envs=2, max_num_frames_per_episode=EPISODE_FRAMES)
    env = AtariVectorEnv("tennis", reset_pool_prefill=True, **kwargs)
    reference = AtariVectorEnv("tennis", reset_pool_prefill=True, **kwargs)
    actions = np.zeros(2, dtype=np.int32)

    env.reset(seed=np.array([1, 2]))
    for _ in range(300):
        env.step(actions)

    assert data_equivalence(
        env.reset(seed=np.array([3, 4])), reference.reset(seed=np.array([3, 4]))
    )
    for _ in range(300):
        assert data_equivalence(env.step(actions), reference.step(actions))

    env.close()
    reference.close()


@pytest.mark.parametrize("reset_pool_size", [0, POOL_SIZE])
def test_sub_envs_start_from_different_states(reset_pool_size):
    """Differently seeded sub-environments do not replay one start-state sequence.

    The emulator RNG behind start-state variation (the RIOT timer, power-on RAM)
    is seeded from ``system_random_seed``, not ``random_seed``. If the vector env
    left it at ALE's constant default, every sub-environment would produce the
    same start state at each reset, and with one pool entry per env (``POOL_SIZE``
    envs, ``POOL_SIZE`` total) the whole batch would replay a single state.
    """
    signatures, _ = rollout_envs(
        "solaris",
        2,
        num_envs=POOL_SIZE,
        reset_pool_size=reset_pool_size,
        **DETERMINISTIC,
    )
    for episode in range(2):
        distinct = len({s[episode] for s in signatures})
        assert distinct >= POOL_SIZE // 2, f"{episode=}: {distinct=} of {POOL_SIZE}"


def test_negative_pool_size_is_rejected():
    with pytest.raises(Error, match="reset_pool_size"):
        AtariVectorEnv("breakout", num_envs=1, reset_pool_size=-1)
