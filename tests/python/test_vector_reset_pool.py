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


def rollout(game, episodes, **kwargs):
    """Drive one env through ``episodes`` truncated episodes.

    Returns one hash per episode, of every observation in it, and the
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
    obs, info = env.reset(seed=0)
    num_actions = int(env.single_action_space.n)

    signatures, frame_numbers = [], [int(info["frame_number"][0])]
    episode, t = [obs[0].tobytes()], 0
    while len(signatures) < episodes:
        action = np.array([t % num_actions], dtype=np.int32)
        t += 1
        obs, _, term, trunc, info = env.step(action)
        frame_numbers.append(int(info["frame_number"][0]))
        if term[0] or trunc[0]:
            signatures.append(hashlib.blake2b(b"".join(episode)).digest())
            # NextStep autoreset: the next step returns the new episode's first
            # observation.
            obs, _, _, _, info = env.step(action)
            frame_numbers.append(int(info["frame_number"][0]))
            episode, t = [], 0
        episode.append(obs[0].tobytes())
    env.close()
    return signatures, frame_numbers


# noop_max=1 with no fire reset emulates no frames after the console reset, so
# the first observation can only come from the screen restored with the snapshot.
DETERMINISTIC = dict(noop_max=1, use_fire_reset=False)


@pytest.mark.parametrize("game", ["breakout", "pong"])
def test_restored_resets_match_emulated_resets(game):
    """On a ROM with one start state, restored episodes equal emulated ones."""
    signatures, _ = rollout(game, POOL_SIZE + 6, **DETERMINISTIC)
    assert len(set(signatures)) == 1


@pytest.mark.parametrize("game", ["tennis", "space_war"])
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


def test_negative_pool_size_is_rejected():
    with pytest.raises(Error, match="reset_pool_size"):
        AtariVectorEnv("breakout", num_envs=1, reset_pool_size=-1)
