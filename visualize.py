import tools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    args = tools.parse_args_visualize()
    model = tools.load_model(args['load_path'])
    data, target = tools.read_data(args['data_path'])
    sns.set_style('darkgrid')
    sns.scatterplot(x=data, y=target, label='Data')
    reg_x = np.arange(min(data), max(data), 5)
    reg_y = [model.predict(i) for i in reg_x]
    plt.plot(reg_x, reg_y, color='red', label='Regression')
    plt.legend(loc='best')
    plt.savefig('plot.png')
    print('Plot saved to plot.png')
