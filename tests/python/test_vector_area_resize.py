"""Check the vector env's downscaled observations against cv2.resize(INTER_AREA).

The vector env fuses the maxpool and the INTER_AREA downscale into one kernel
(src/ale/vector/area_resize.hpp) for grayscale observations. This runs the same
episode twice, once at the native 210x160 resolution (maxpooled, not resized)
and once at a smaller size, and checks that every downscaled frame equals
cv2.resize of the native one.
"""

import platform

import numpy as np
import pytest
from ale_py.vector_env import AtariVectorEnv

cv2 = pytest.importorskip("cv2")

# Bit-exactness is only expected where the opencv-python wheel and the OpenCV the
# extension is built against share the same float code path (x86-64 SSE2).
EXACT = platform.machine().lower() in ("x86_64", "amd64")


@pytest.mark.parametrize(
    "img_height, img_width",
    [
        (84, 84),  # default, fused
        (64, 64),  # fused
        (105, 105),  # fused (only the vertical scale is an integer)
        (100, 120),  # fused
        (105, 80),  # exact 2x: cv::resize's resizeAreaFast path, not fused
    ],
)
@pytest.mark.parametrize("maxpool", [True, False])
def test_vector_obs_match_cv2_inter_area(img_height, img_width, maxpool):
    """Downscaled observations equal cv2.resize of the native-resolution ones."""
    kwargs = dict(
        game="breakout",
        num_envs=2,
        grayscale=True,
        stack_num=1,
        maxpool=maxpool,
        max_num_frames_per_episode=400,  # so autoresets happen within the run
        repeat_action_probability=0.0,
    )
    native = AtariVectorEnv(img_height=210, img_width=160, **kwargs)
    resized = AtariVectorEnv(img_height=img_height, img_width=img_width, **kwargs)

    native_obs, _ = native.reset(seed=7)
    resized_obs, _ = resized.reset(seed=7)

    rng = np.random.default_rng(0)
    resets = 0
    for t in range(250):
        for i in range(native_obs.shape[0]):
            expected = cv2.resize(
                native_obs[i, 0],
                (img_width, img_height),
                interpolation=cv2.INTER_AREA,
            )
            actual = resized_obs[i, 0]
            if EXACT:
                np.testing.assert_array_equal(
                    actual, expected, err_msg=f"t={t} env={i}"
                )
            else:
                diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
                assert diff.max() <= 1, f"t={t} env={i}"

        actions = rng.integers(0, native.single_action_space.n, size=2)
        native_obs, _, term, trunc, _ = native.step(actions)
        resized_obs, _, rterm, rtrunc, _ = resized.step(actions)
        np.testing.assert_array_equal(term, rterm)
        np.testing.assert_array_equal(trunc, rtrunc)
        resets += int(np.sum(term | trunc))

    assert resets > 0
    native.close()
    resized.close()
