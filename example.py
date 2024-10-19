from hjmodel import HJModel
import sys
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    res_path = os.path.join(dir_path, 'data', 'exp_data_dynamic.pq')
    model = HJModel(res_path=res_path)
    model.run_dynamic(time=12000, num_systems=200000)

    """
    model.plot_outcomes()

    outcome_probs = model.get_outcome_probabilities()
    print(outcome_probs)

    stats = model.get_statistics_for_outcome(['TD'], 'r')
    plt.hist(stats)
    plt.show()
    """



