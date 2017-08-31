### GPS

This code is a reimplementation of the guided policy search algorithm and iterative LQG-based trajectory optimization and supervised policy learning method, meant to help others understand, reuse, and build upon existing work. For full documentation, see [rll.berkeley.edu/gps](http://rll.berkeley.edu/gps).

The code base is **a work in progress**. See the [FAQ](http://rll.berkeley.edu/gps/faq.html) for information on planned future additions to the code.

### iDG

This fork of the code implements the iterative Dynamic Game that was proposed in the paper:

* iDG: A Robust Zero-Sum, Two-Player Reinforcement Learning

For details of the algorithm, please see the paper on arxiv under the name: Olalekan Ogunmolu.

#### Running iDG

* First train a protagonist agent by following the instructions on the rll.berkeley.edu/gps page.

* Go to the experiments directory and run the [copy_gps](/experiments/copy_gps) executable. This will copy the learned policy for the original system into a new folder.

* We will then make a few modifications in the hyperparams directory of the new folder as follows:

For box2d experiments, we will import the MDGPS class like so at the top of the hyperparams file:

```python
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS # for new experiments
```

```python
EXP_DIR: change this to point to the new experiment directory

common:
	|
	|--'experiment_name': 'name_of_new_experiment'
	|--'costs_filename': EXP_DIR + 'costs.csv',
  |--'mode': 'antagonist',  # whether we are running in block-alternating ascent mode
  |--'gamma': 1e8,   # the magnitude of the additive disturbance
```

where a full common dict will for example look like so:

```python
common = {
    'experiment_name': 'box2d_badmm_example' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'costs_filename': EXP_DIR + 'costs.csv',
    'conditions': 4,
    'mode': 'antagonist',
    'gamma': 1e8,
}
```

* In the `action_cost` dict, we would want to add the gamma and mode terms as well e.g.

```python
action_cost = {
    'type': CostAction,
    'wu': np.array([1, 1]),
    'gamma': 1e8,
    'mode': 'antagonist',
}
```

* So also for `algorithm['cost']` e.g.,

```python
algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
    'weights': [1e-5, 1.0],
    'gamma': 1e8,
    'mode': 'antagonist',
}
```


### Docker Image

The docker image for the base gps codes is located at [lakehanne/gps/](https://hub.docker.com/r/lakehanne/gps/)
