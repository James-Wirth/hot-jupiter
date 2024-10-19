from hjmodel import HJModel
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model = HJModel(res_path=sys.argv[0], res_name='exp_data_dynamic')
    model.run(time=12000, num_systems=200000)
    
    """
    model.plot_outcomes()

    outcome_probs = model.get_outcome_probabilities()
    print(outcome_probs)

    stats = model.get_statistics_for_outcome(['TD'], 'r')
    plt.hist(stats)
    plt.show()
    """

