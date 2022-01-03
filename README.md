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