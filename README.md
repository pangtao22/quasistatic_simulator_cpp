# Quasistatic Simulator C++

## Model paths
Until this repo and `quasistatic_simulator` are merged, the relative paths between this repo and `quasistatic_simulator` need to be manually maintained so that the model files in `quasistatic_simulator` can be found by the C++ backend in this repo. 

Specifically, it is assumed that 
- the `get_model_paths.cc` file of this repo can be found at 
```bash
${HOME}/ClionProjects/quasistatic_simulator_cpp/src
```
- The [`quasistatic_simualtor`](https://github.com/pangtao22/quasistatic_simulator) repo can be found at 
```bash
${HOME}/PycharmProjects/quasistatic_simulator
```

- The [`robotics_utilities`](https://github.com/pangtao22/robotics_utilities) repo can be found at
```bash
${HOME}/PycharmProjects/robotics_utilities
```

## Eigen 3.4
Eigen 3.4 supports useful syntax such as slicing, so we'd like to use it. However, drake by default uses the Eigen bundled with the OS. Mac's default Eigen is 3.4, so there is nothing we need to do. On ubuntu 20.04, however, the system's default Eigen is 3.3.*, which means we need to ask drake to use a different, user-installed Eigen.

To build drake with a use-installed Eigen, one needs to turn on `WITH_USER_EIGEN` when building drake, and set `Eigen3_DIR` to the folder containing cmake config files of the user-installed Eigen. If Eigen is installed using the default options with the .tar ball downloaded from Eigen's official website, `Eigen3_DIR` should be `/usr/local/share/eigen3/cmake`. In contrast, the system's Eigen can be found at `/usr/lib/cmake/eigen3` instead.

To build this project with the user installed Eigen, add this flag when running cmake:
```
-DEigen3_DIR=/usr/local/share/eigen3/cmake
```

## Running tests
At the root of this repo, run
```bash
mkdir build && cd build
cmake .. # with -DCMAKE_PREFIX_PATH=/path/to/drake and -DCMAKE_BUILD_TYPE=release, if necessary.
make test 
```
When built in `release` mode, the `TestGradient*` tests may fail due to a handful of gradients solved in single thread is different from the corresponding gradient solved in parallel. This might be caused by the different behaviors of Eigen's `BdcSvd` in debug and release modes: the `TestGradient*` tests never failed in debug mode based on my testings. Therefore, it should be fine (for practical purposes) as long as the failures consist of only a few gradient differences exceeding tolerance.  