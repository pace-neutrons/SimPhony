import json
import numpy as np
import matplotlib.pyplot as plt


def plot_walltimes_for_file(filename: str):
    """
    Plot a figure of walltimes for each seedname/threads in filename

    Parameters
    ----------
    filename : str
        The file to get the walltimes from
    """
    data = json.load(open(filename))
    if 'benchmarks' in data:
        fig, subplots = plt.subplots()
        x_data = {}
        y_data = {}
        for per_test_data in data['benchmarks']:
            try:
                if per_test_data['params']['use_c']:
                    seedname = per_test_data['params']['seedname']
                    if not seedname in x_data.keys():
                        x_data[seedname] = []
                    if not seedname in y_data.keys():
                        y_data[seedname] = []
                    x_data[seedname].append(
                        int(per_test_data['params']['n_threads']))
                    y_data[seedname].append(
                        per_test_data['stats']['median'])
            except KeyError as e:
                print(f'Error for {per_test_data["fullname"]}\n {e}')
        for seedname in x_data.keys():
            subplots.plot(x_data[seedname], y_data[seedname], label=seedname)
        x_data = subplots.get_lines()[0].get_data()[0]
        subplots.set_xlim(x_data[0], x_data[-1])
        subplots.set_ylim(0)
        subplots.set_xlabel('Number of threads')
        subplots.set_ylabel('Walltime (s)')
        subplots.set_title(f'Walltimes for {filename}')
        subplots.legend(loc='upper right', fontsize='small')
