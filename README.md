## Self balancing

### Steps to run a project

1. Activate the python environment
2. Install all packages from requirements file.
3. Additionally, you will need to install Mujoco in your system. Please following this link for
   installation: https://github.com/openai/mujoco-py
4. To train a model, first create empty directories `models` and `reports`.
5. Then run `training.py` file. Note that this will output model file and report containing rewards at each time step in
   the above directories.
6. Once trained, use the file from the report in `plotter.py` to see the convergence graph
7. To test a model or test pretrained model from `saved_models` directories, run any file starting with `testing_.py`.
8. It is to note that to run `testing_hardware_noise.py` a hardware will be necessary to connect to the serial port.


### Core files
| File                      | Description                                                                            |
|---------------------------|----------------------------------------------------------------------------------------|
| training.py               | Used for training a model                                                              |
| testing_no_noise.py       | Testing with saved model                                                               |
| testing_random_action.py  | Testing with saved model intervened by random action every x steps                     |
| testing_random_noise.py   | Testing with saved model with continuous randomly generated noise                      |
| testing_hardware_noise.py | Testing with saved model with incoming noise from human interaction with real hardware |
| self_balancer_env.py      | A wrapper class for abstraction of environment that connect with mujoco api            |
| sbp_model.xml             | A mujoco model for self balancing platform                                             |

### Helper files
| File                     | Description                                                                            |
|--------------------------|----------------------------------------------------------------------------------------|
| plotter.py               | Helper to plot data from reports                                                       |
| model_converter.ipynb      | Converts tensorflow model to lite version required for deployment                      |
| test-tflite.py  | Tests if lite version also works                                                       |
| frame_exporter.py  | Exports frame into desired graph used in the latex                                     |


### Project slides and report
- Slides: https://drive.google.com/file/d/1eFKYXUDg4m0IQ7lJoQ1lyAW53x_PREwU/view?usp=sharing
- Report: https://drive.google.com/file/d/15l1oiXiA7_jSbQ0DV2EgFzqf8g7XDd0w/view?usp=sharing

Want to learn more or contribute?
Contact me at bikcrum@gmail.com
