# Installation

## Python Interface

The Python interface `ale-py` supports the following configurations:

| Platform | Architechture | Python Version |
| :------: | :-----------: | :------------: |
| Linux ([manylinux2014](https://www.python.org/dev/peps/pep-0571/)) | x64 | 3.6+ |
| macOS | x64, arm64 | 3.6+ |
| Windows | AMD64 | 3.6+ |


To install the Python interface from PyPi simply run:

```bash
pip install ale-py
```

Once installed you can import the native ALE interface as `ale_py`

```python
from ale_py import ALEInterface
ale = ALEInterface()
```

### OpenAI Gym

The ALE is natively supported by OpenAI Gym. Anytime you create an Atari environment it invokes the ALE in the background. We do suggest that you use the `ALE` namespaced environments which introduced the `v5` version suffix. For example,

```py
import gym
import ale_py

env = gym.make('ALE/Breakout-v5')
```

## C++ Interface

The C++ library requires:

* A C++17 compiler
* CMake 3.14+
* zlib
* (Optional) SDL 2 for display/audio support

SDL support allows for displaying the console's screen and enabling audio output. For example, *without* SDL support you'll still be able to train your agents, but you won't be able to visualize the resulting policy. It might be preferable to disable SDL support when compiled on a cluster but enable SDL locally. Note: SDL support defaults to **OFF**.

You can use any package manager to install these dependencies but we recommend using [`vcpkg`](https://github.com/microsoft/vcpkg). Here's a minimal example of installing these dependencies and building/installing the C++ library.

```sh
vcpkg install zlib sdl2

mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release
cmake --build . --target install
```

These steps will work on any platform, just make sure to specify the environment variable `VCPKG_INSTALLATION_ROOT` to point to your vcpkg installation so we can find the required dependencies. If you install any vcpkg dependencies using non-standard triplets you can specify the environment variable `VCPKG_TARGET_TRIPLET`. For more info check out the [vcpkg docs](https://vcpkg.readthedocs.io/en/latest/users/config-environment/) on how to configure your environment.

Once the ALE is installed you can link agaisnt the library in your C++ project as follows

```cmake
find_package(ale REQUIRED)
target_link_libraries(YourTarget ale::ale-lib)
```
