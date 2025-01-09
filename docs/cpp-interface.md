# C++ Interface

The shared library interface is the simplest way to implement a C++ agent for the Arcade Learning Environment (ALE).
This interface allows agents to directly access ALE via a class called
`ALEInterface`, defined in `ale_interface.hpp`. Example code detailing a simple random agent is provided under `examples/cpp-agent`.

To instantiate the Arcade Learning Environment it is enough to write:
```cpp
ale::ALEInterface ale;
```

Once the environment is initialized, it is now possible to set its arguments. This is done with the
functions `setBool()`, `setInt()`, `setFloat()`. For example, to set the environment's seed we write:

```cpp
ale.setInt("random_seed", 123);
```

Finally, after setting the desired environment parameters we now load the game ROM by providing its filename to the `loadROM` method:

```cpp
ale.loadROM("asterix.bin");
```

There are two different action sets provided by ALE: the "legal" set and the "minimal"
set. Save for a few rare exceptions, the legal action set consists of all 18 actions for all games, including duplicates and actions with no effect. On the other hand, the minimal action set for a game contains only
the actions that have some effect on that game. The `getLegalActionSet` and `getMinimalActionSet` methods provide the desired action sets:

```cpp
ale::ActionVect legal_actions = ale.getLegalActionSet();
```

Taking an action is done by calling the function `act()` with a value from the `Action` enum:

```cpp
ale::Action a = legal_actions[rand() % legal_actions.size()];
float reward = ale.act(a);
```

An optional sound observation is provided. To enable, set the associated environment parameter:

```cpp
ale.setBool("sound_obs", True);
```

Once enabled, the sound observation may be obtained by calling:

```cpp
ale.getAudio()
```

Finally, one can check whether the episode has terminated using the function `ale.game_over()`. With these functions one can already implement a very simple agent that plays randomly for one episode:

```cpp
#include <iostream>
#include <ale_interface.hpp>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " rom_file" << std::endl;
        return 1;
    }

    ale::ALEInterface ale;
    ale.setInt("random_seed", 123);
    ale.loadROM(argv[1]);

    ale::ActionVect legal_actions = ale.getLegalActionSet();

    float totalReward = 0.0;
    while (!ale.game_over()) {
        Action a = legal_actions[std::rand() % legal_actions.size()];
        float reward = ale.act(a);
        totalReward += reward;

        std::cout << "The episode ended with score: " << totalReward
            << std::endl;
    }

    return 0;
}
```

Compiling with the shared library can be done by appending `-lale` or by using `find_package(ale)` and linking to the cmake target `ale::ale-lib`.

```
cmake_minimum_required(VERSION 3.14)

project(example-cpp-lib)

find_package(ale REQUIRED)

add_executable(sharedLibraryInterfaceExample sharedLibraryInterfaceExample.cpp)
target_link_libraries(sharedLibraryInterfaceExample ale::ale-lib)

add_executable(sharedLibraryInterfaceWithModesExample sharedLibraryInterfaceWithModesExample.cpp)
target_link_libraries(sharedLibraryInterfaceWithModesExample ale::ale-lib)
```
