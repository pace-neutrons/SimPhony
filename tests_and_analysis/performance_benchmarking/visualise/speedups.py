import json
import numpy as np
import matplotlib.pyplot as plt


def plot_speedups_for_file(filename: str):
    """
    Plot a figure for each test that has had speedups calculated for it in
    filename. There is a trace for each seedname used in the test.

    Parameters
    ----------
    filename : str
        The file to get the calculated speedups from
    """
    data = json.load(open(filename))
    if 'speedups' in data:
        for test in data['speedups']:
            fig, subplots = plt.subplots()
            for seedname in data['speedups'][test]:
                subplots.plot(
                    [int(x) for x in data['speedups'][test][seedname].keys()],
                    list(data['speedups'][test][seedname].values()),
                    label=seedname
                )
            x_data = subplots.get_lines()[0].get_data()[0]
            # Plot perfect speedup
            subplots.plot(x_data, x_data, color='k', linestyle='--')
            subplots.set_xlim(x_data[0], x_data[-1])
            subplots.set_ylim(x_data[0], x_data[-1])
            subplots.set_xlabel('Number of threads')
            subplots.set_ylabel('Speedup')
            subplots.set_title(f'Speedups for {filename}\n {test}')
            subplots.legend(
                loc='upper left',
                fontsize='small'
            )
