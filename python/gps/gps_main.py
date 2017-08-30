import matplotlib as mpl
mpl.use('Qt4Agg')

import logging
import imp
import os
import os.path
import sys
import copy
import argparse
import threading
import time
import traceback

from random import shuffle

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList

class GPSMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config, closeloop, test=False, quit_on_end=False):
        """
        Initialize GPSMain
        Args:
            config: Hyperparameters for experiment
            quit_on_end: When true, quit automatically on completion
        """
        self._quit_on_end = quit_on_end
        self.closeloop = closeloop
        self.test = test
        self._hyperparams = config
        self._conditions = config['common']['conditions'] # will be 4
        if 'train_conditions' in config['common']:
            self._train_idx = config['common']['train_conditions']
            self._test_idx = config['common']['test_conditions']
        else:
            self._train_idx = range(self._conditions)
            config['common']['train_conditions'] = config['common']['conditions'] # is 4
            self._hyperparams=config
            self._test_idx = self._train_idx

        self._data_files_dir = config['common']['data_files_dir']

        self.agent = config['agent']['type'](config['agent']) #will be AgentMujoCo object

        self.data_logger = DataLogger()
        self.gui = GPSTrainingGUI(config['common']) if config['gui_on'] else None

        config['algorithm']['agent'] = self.agent

        self.algorithm = config['algorithm']['type'](config['algorithm'])

    def run(self, itr_load=None):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        try:
            itr_start = self._initialize(itr_load)

            for itr in range(itr_start, self._hyperparams['iterations']):
                for cond in self._train_idx:
                    for i in range(self._hyperparams['num_samples']):
                        self._take_sample(itr, cond, i)


                traj_sample_lists = [
                    self.agent.get_samples(cond, -self._hyperparams['num_samples'])
                    for cond in self._train_idx
                ]

                # Clear agent samples.
                self.agent.clear_samples()
                # print('close loop in run: ', self.closeloop)
                self._take_iteration(itr, traj_sample_lists)  #see impl in alg_mdgps.py#L36
                pol_sample_lists = self._take_policy_samples()
                self._log_data(itr, traj_sample_lists, pol_sample_lists)

        except Exception as e:
            traceback.print_exception(*sys.exc_info())
        finally:
            self._end()

    def run_cl(self, itr, itr_load=None):

        protag_algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr
        self.protag_algorithm = self.data_logger.unpickle(protag_algorithm_file)

        if self.protag_algorithm is None:
            print("Error: cannot find '%s.'" % protag_algorithm_file)
            os._exit(1)

        try:
            # itr_start = 1 #self._initialize(itr_load)
            itr_start, itr_load = 1, 40
            # self._initialize(itr_load)

            self.protag_traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                ('traj_sample_itr_%02d.pkl' % itr)) # we don't need this.

            for itr in range(itr_start, self._hyperparams['iterations']):
                self.closeloop = True
                for cond in self._train_idx:
                    for i in range(self._hyperparams['num_samples']):
                        #take protagonist and antagonist policy samples
                        self._take_sample(itr, cond, i)

                traj_sample_lists = [
                    self.agent.get_samples(cond, -self._hyperparams['num_samples'])
                    for cond in self._train_idx
                ]

                # Clear agent samples.
                self.agent.clear_samples()

                #take protagonist and antagonist iterations
                print("Taking RL and Supervised Learning iterations")
                # if isinstance(self.algorithm

                self._take_iteration_cl(itr, traj_sample_lists)  # see impl in alg_mdgps.py# L36
                # self._take_iteration(itr, traj_sample_lists)

                print("Taking pro/antagonist policy samples")
                pol_sample_lists = self._take_policy_samples()
                # protag_pol_sample_lists = self.protag_pol_samples
                self._log_data(itr, traj_sample_lists, pol_sample_lists)

        except Exception as e:
            traceback.print_exception(*sys.exc_info())
        finally:
            self._end()

    def test_policy(self, itr, N):
        """
        Take N policy samples of the algorithm state at iteration itr,
        for testing the policy to see how it is behaving.
        (Called directly from the command line --policy flag).
        Args:
            itr: the iteration from which to take policy samples
            N: the number of policy samples to take
        Returns: None
        """
        # print('itr: ', itr) # is 41
        protag_algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr
        self.protag_algorithm = self.data_logger.unpickle(protag_algorithm_file)

        if self.protag_algorithm is None:
            print("Error: cannot find '%s.'" % protag_algorithm_file)
            os._exit(1)

        algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr
        self.algorithm = self.data_logger.unpickle(algorithm_file)
        if self.algorithm is None:
            print("Error: cannot find '%s.'" % algorithm_file)
            os._exit(1) # called instead of sys.exit(), since t
        traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
            ('traj_sample_itr_%02d.pkl' % itr))

        pol_sample_lists = self._take_policy_samples(N)
        self.data_logger.pickle(
            self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
            copy.copy(pol_sample_lists)
        )

        if self.gui:
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists, protag_pol_samples=self.protag_pol_samples)
            self.gui.set_status_text(('Took %d policy sample(s) from ' +
                'algorithm state at iteration %d.\n' +
                'Saved to: data_files/pol_sample_itr_%02d.pkl.\n') % (N, itr, itr))

    def _initialize(self, itr_load):
        """
        Initialize from the specified iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns:
            itr_start: Iteration to start from.
        """
        if itr_load is None:
            if self.gui:
                self.gui.set_status_text('Press \'go\' to begin.')
            return 0
        else:
            algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr_load
            self.algorithm = self.data_logger.unpickle(algorithm_file)
            if self.algorithm is None:
                print("Error: cannot find '%s.'" % algorithm_file)
                os._exit(1) # called instead of sys.exit(), since this is in a thread

            if self.gui:
                traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                    ('traj_sample_itr_%02d.pkl' % itr_load))
                if self.algorithm.cur[0].pol_info:
                    pol_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                        ('pol_sample_itr_%02d.pkl' % itr_load))
                else:
                    pol_sample_lists = None
                self.gui.set_status_text(
                    ('Resuming training from algorithm state at iteration %d.\n' +
                    'Press \'go\' to begin.') % itr_load)
            return itr_load + 1

    def _take_sample(self, itr, cond, i):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            i: Sample number.
        Returns: None
        """
        if self.algorithm._hyperparams['sample_on_policy'] \
                and self.algorithm.iteration_count > 0:
            pol = self.algorithm.policy_opt.policy  # calling alg_mdgps.py.policy_opt
            if self.closeloop:
                protag_pol = self.protag_algorithm.policy_opt.policy
        else:
            pol = self.algorithm.cur[cond].traj_distr # cur is every var in iteration data
            if self.closeloop:
                protag_pol = self.protag_algorithm.cur[cond].traj_distr
        if self.gui:
            self.gui.set_image_overlays(cond)   # Must call for each new cond.
            redo = True
            while redo:
                while self.gui.mode in ('wait', 'request', 'process'):
                    if self.gui.mode in ('wait', 'process'):
                        time.sleep(0.01)
                        continue
                    # 'request' mode.
                    if self.gui.request == 'reset':
                        try:
                            self.agent.reset(cond)
                        except NotImplementedError:
                            self.gui.err_msg = 'Agent reset unimplemented.'
                    elif self.gui.request == 'fail':
                        self.gui.err_msg = 'Cannot fail before sampling.'
                    self.gui.process_mode()  # Complete request.

                self.gui.set_status_text(
                    'Sampling: iteration %d, condition %d, sample %d.' %
                    (itr, cond, i)
                )
                self.agent.sample(
                    pol, cond,
                    verbose=(i < self._hyperparams['verbose_trials'])
                )
                #take adversary samples if we are in close loop
                if self.closeloop:
                    self.agent.sample(
                        protag_pol, cond,
                        verbose=(i < self._hyperparams['verbose_trials'])
                    )

                if self.gui.mode == 'request' and self.gui.request == 'fail':
                    redo = True
                    self.gui.process_mode()
                    self.agent.delete_last_sample(cond)
                else:
                    redo = False
        else:
            self.agent.sample(
                pol, cond,
                verbose=(i < self._hyperparams['verbose_trials'])
            )
            if self.closeloop:
                self.agent.sample(
                    protag_pol, cond,
                    verbose=(i < self._hyperparams['verbose_trials'])
                )

    def _take_iteration(self, itr, sample_lists):
        """
        Take an iteration of the algorithm.
        Args:
            itr: Iteration number.
        Returns: None
        """
        if self.gui:
            self.gui.set_status_text('Calculating.')
            self.gui.start_display_calculating()
        # see corresponding alg being used e.g. algorithm_mdgps
        self.algorithm.iteration(sample_lists)

        if self.closeloop:
            self.protag_algorithm.iteration(sample_lists)

        if self.gui:
            self.gui.stop_display_calculating()

    def _take_iteration_cl(self, itr, sample_lists):
        """
        Take an iteration of the algorithm.
        Args:
            itr: Iteration number.
        Returns: None
        """

        if self.gui:
            self.gui.set_status_text('Calculating.')
            self.gui.start_display_calculating()

        self.algorithm.iteration_cl(self.protag_traj_sample_lists, sample_lists)
        # self.protag_algorithm.iteration_cl(self.protag_traj_sample_lists, sample_lists)

        if self.gui:
            self.gui.stop_display_calculating()

    def _take_policy_samples(self, N=None):
        """
        Take samples from the policy to see how it's doing.
        Args:
            N  : number of policy samples to take per condition
        Returns: None
        """
        if 'verbose_policy_trials' not in self._hyperparams:
            # AlgorithmTrajOpt
            return None
        verbose = self._hyperparams['verbose_policy_trials']
        if self.gui:
            self.gui.set_status_text('Taking policy samples.')
        pol_samples = [[None] for _ in range(len(self._test_idx))]
        protag_pol_samples = pol_samples if self.closeloop or self.test else None
        # Since this isn't noisy, just take one sample.
        # TODO: Make this noisy? Add hyperparam?
        # TODO: Take at all conditions for GUI?
        for cond in range(len(self._test_idx)):
            pol_samples[cond][0] = self.agent.sample(
                self.algorithm.policy_opt.policy, self._test_idx[cond],
                verbose=verbose, save=False, noisy=False)

        if self.closeloop or self.test:
            # print 'self.protag_algorithm: ', self.agent.sample(self.protag_algorithm.policy_opt.policy, self._test_idx[0],verbose=verbose, save=False, noisy=False)
            for cond in range(len(self._test_idx)):
                protag_pol_samples[cond][0] = self.agent.sample(
                    self.protag_algorithm.policy_opt.policy, self._test_idx[cond],
                    verbose=verbose, save=False, noisy=False)
            self.protag_pol_samples = [SampleList(samples) for samples in protag_pol_samples]

        return [SampleList(samples) for samples in pol_samples]

    def _log_data(self, itr, traj_sample_lists, pol_sample_lists=None):
        """
        Log data and algorithm, and update the GUI.
        Args:
            itr: Iteration number.
            traj_sample_lists: trajectory samples as SampleList object
            pol_sample_lists: policy samples as SampleList object
        Returns: None
        """
        if self.gui:
            self.gui.set_status_text('Logging data and updating GUI.')
            if self.closeloop:
                self.gui.update(itr, self.algorithm, self.agent,
                    traj_sample_lists, pol_sample_lists, protag_pol_samples=self.protag_pol_samples)
            else:
                self.gui.update(itr, self.algorithm, self.agent,
                    traj_sample_lists, pol_sample_lists)
            self.gui.save_figure(
                self._data_files_dir + ('figure_itr_%02d.png' % itr)
            )
        if 'no_sample_logging' in self._hyperparams['common']:
            return
        self.data_logger.pickle(
            self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
            copy.copy(self.algorithm)
        )
        self.data_logger.pickle(
            self._data_files_dir + ('traj_sample_itr_%02d.pkl' % itr),
            copy.copy(traj_sample_lists)
        )
        if pol_sample_lists:
            self.data_logger.pickle(
                self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists)
            )

    def _end(self):
        """ Finish running and exit. """
        if self.gui:
            self.gui.set_status_text('Training complete.')
            self.gui.end_mode()
            if self._quit_on_end:
                # Quit automatically (for running sequential expts)
                os._exit(1)

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str,
                        help='experiment name')
    parser.add_argument('-n', '--new', action='store_true',
                        help='create new experiment')
    parser.add_argument('-t', '--targetsetup', action='store_true',
                        help='run target setup')
    parser.add_argument('-r', '--resume', metavar='N', type=int,
                        help='resume training from iter N')
    parser.add_argument('-p', '--policy', metavar='N', type=int,
                        help='take N policy samples (for BADMM/MDGPS only)')
    parser.add_argument('-c', '--closeloop', action='store_true',
                        help='run in close loo[]')  #to train the antagonist and protagonist
    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')
    parser.add_argument('-q', '--quit', action='store_true',
                        help='quit GUI automatically when finished')
    args = parser.parse_args()

    exp_name = args.experiment
    resume_training_itr = args.resume
    test_policy_N = args.policy
    run_cl_policy = args.closeloop

    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'

    print(args)

    if args.silent:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    if args.new:
        from shutil import copy

        if os.path.exists(exp_dir):
            sys.exit("Experiment '%s' already exists.\nPlease remove '%s'." %
                     (exp_name, exp_dir))
        os.makedirs(exp_dir)

        prev_exp_file = '.previous_experiment'
        prev_exp_dir = None
        try:
            with open(prev_exp_file, 'r') as f:
                prev_exp_dir = f.readline()
            copy(prev_exp_dir + 'hyperparams.py', exp_dir)
            if os.path.exists(prev_exp_dir + 'targets.npz'):
                copy(prev_exp_dir + 'targets.npz', exp_dir)
        except IOError as e:
            with open(hyperparams_file, 'w') as f:
                f.write('# To get started, copy over hyperparams from another experiment.\n' +
                        '# Visit rll.berkeley.edu/gps/hyperparams.html for documentation.')
        with open(prev_exp_file, 'w') as f:
            f.write(exp_dir)

        exit_msg = ("Experiment '%s' created.\nhyperparams file: '%s'" %
                    (exp_name, hyperparams_file))
        if prev_exp_dir and os.path.exists(prev_exp_dir):
            exit_msg += "\ncopied from     : '%shyperparams.py'" % prev_exp_dir
        sys.exit(exit_msg)

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" %
                 (exp_name, hyperparams_file))

    hyperparams = imp.load_source('hyperparams', hyperparams_file)

    if args.targetsetup:
        try:
            import matplotlib.pyplot as plt
            from gps.agent.ros.agent_ros import AgentROS
            from gps.gui.target_setup_gui import TargetSetupGUI

            agent = AgentROS(hyperparams.config['agent'])
            TargetSetupGUI(hyperparams.config['common'], agent)

            plt.ioff()
            plt.show()
        except ImportError:
            sys.exit('ROS required for target setup.')
    elif test_policy_N:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        seed = hyperparams.config.get('random_seed', 0)

        random.seed(seed)
        np.random.seed(seed)

        data_files_dir = exp_dir + 'data_files/'
        data_filenames = os.listdir(data_files_dir)
        algorithm_prefix = 'algorithm_itr_'
        algorithm_filenames = [f for f in data_filenames if f.startswith(algorithm_prefix)]
        current_algorithm = sorted(algorithm_filenames, reverse=True)[0]
        current_itr = int(current_algorithm[len(algorithm_prefix):len(algorithm_prefix)+2])

        gps = GPSMain(hyperparams.config, args.closeloop, test_policy_N)
        if hyperparams.config['gui_on']:
            test_policy = threading.Thread(
                target=lambda: gps.test_policy(itr=current_itr, N=test_policy_N)
            )
            test_policy.daemon = True
            test_policy.start()

            plt.ioff()
            plt.show()
        else:
            gps.test_policy(itr=current_itr, N=test_policy_N)
    elif run_cl_policy:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)

        closeloop = True

        data_files_dir = exp_dir + 'data_files/'
        data_filenames = os.listdir(data_files_dir)
        algorithm_prefix = 'algorithm_itr_'
        algorithm_filenames = [f for f in data_filenames if f.startswith(algorithm_prefix)]
        current_algorithm = sorted(algorithm_filenames, reverse=True)[0]
        current_itr = int(current_algorithm[len(algorithm_prefix):len(algorithm_prefix)+2])

        gps = GPSMain(hyperparams.config, args.quit, closeloop, False)
        if hyperparams.config['gui_on']:
            run_gps = threading.Thread(
                target=lambda: gps.run_cl(itr=current_itr, itr_load=resume_training_itr)
            )
            run_gps.daemon = True
            run_gps.start()

            plt.ioff()
            plt.show()
    else:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)

        gps = GPSMain(hyperparams.config, args.quit)
        if hyperparams.config['gui_on']:
            run_gps = threading.Thread(
                target=lambda: gps.run(itr_load=resume_training_itr)
            )
            run_gps.daemon = True
            run_gps.start()

            plt.ioff()
            plt.show()
        else:
            gps.run(itr_load=resume_training_itr)

if __name__ == "__main__":
    main()
