import os, sys
import argparse

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Parse log.txt file')
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
                        help='run closed loop robust adversarial example?')  #to train the antagonist and protagonist
    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')
    parser.add_argument('-q', '--quit', action='store_true',
                        help='quit GUI automatically when finished')
    args = parser.parse_args()

    exp_name = args.experiment

    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    log_file = exp_dir + 'log.txt'

    lines = None
    with open(log_file, 'rb') as f:
        read_data = f.read()
        lines = f.readlines()
        # print(read_data)
        # print(f.readline())
        # help(f)
    for i,line in enumerate(lines):
        print(line)

if __name__=="__main__":
    main()
