'''
Contains functions for logistic regression through gradient descent.

Running module directly graphs visualizations of logistic regression.
Default parameters can be modified within main for configuration of 
training and graphing results. Example of function usage within main.
'''

import numpy as np
from random_data import random3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def main(): 

    # Config for main 
    plot_correct = False
    size_data_set = 20
    plot_plane = False

    # Set up generators and create random data 
    data = list()
    gen1 = random3d(points=size_data_set, s=2, b=70, v=True, noise=0)
    gen2 = random3d(points=size_data_set, s=2, b=-70, v=True, noise=0)
    for _ in range(int(size_data_set/2)):
        data.append([np.array(next(gen1)), 1])
    for _ in range(int(size_data_set/2)):
        data.append([np.array(next(gen2)), 0])
    np.random.shuffle(data)
    dataset = [data[i][0] for i in range(len(data))]
    category = [data[i][1] for i in range(len(data))]

    # Perform logistic regression for 20 epochs
    weights = logistic_trainer(dataset, category, 20, 0.0001, True)

    # Plot scatter of training set
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    x = [dataset[i][0] for i in range(len(dataset)) if category[i] == 1]
    y = [dataset[i][1] for i in range(len(dataset)) if category[i] == 1]
    z = [dataset[i][2] for i in range(len(dataset)) if category[i] == 1]
    x1 = [dataset[i][0] for i in range(len(dataset)) if category[i] == 0]
    y1 = [dataset[i][1] for i in range(len(dataset)) if category[i] == 0]
    z1= [dataset[i][2] for i in range(len(dataset)) if category[i] == 0]
    ax.scatter(x, y, z, color='orange', label="Class 1")
    ax.scatter(x1, y1, z1, color='blue', label="Class 2")

    # Test on remaining data in generators, plot incorrect classifications
    correct = 0
    for point1, point2 in zip(gen1, gen2):
        if logistic_classifier(point1, weights) == 1:
            correct += 1
            if plot_correct: 
                ax.plot([point1[0]], [point1[1]], [point1[2]], 'go')
        else: 
            ax.plot([point1[0]], [point1[1]], [point1[2]], 'ro')
        if logistic_classifier(point2, weights) == 0:
            correct += 1
            if plot_correct: 
                ax.plot([point2[0]], [point2[1]], [point2[2]], 'go')
        else:
            ax.plot([point2[0]], [point2[1]], [point2[2]], 'ro')

    # Print performance
    print(f'Percent correct over {int(size_data_set/2)}: '
      f'{round(100 * correct / size_data_set, 2)}')

    if plot_plane: 
      x, y = np.meshgrid(range(-70, 71, 10), range(-70, 71, 10))
      z = (0.5 -weights[1] * x - weights[2] * y - weights[0]) / weights[3]
      ax.plot_surface(x, y, z, alpha=0.3)

    ax.legend()
    plt.show(block=True)


def logistic_func(a): 
    return 1 / (1 + np.exp(-a))


def logistic_trainer(dataset, category, epoch=10, alpha=0.0001, v=False): 
    '''Takes a parallel dataset / category array, returns weights''' 
    weights = np.zeros(dataset[0].size)
    bias = 0
    cross_entropy = np.zeros(epoch)

    # Train for number of epochs
    for i in range(epoch):
        # Shuffle training set 
        state = np.random.get_state()
        np.random.shuffle(dataset)
        np.random.set_state(state)
        np.random.shuffle(category)

        # Gradient descent over each training example
        for j in range(len(dataset)): 
            y = logistic_func(np.dot(weights, dataset[j]) + bias)
            weights = weights + alpha * (category[j] - y) * dataset[j]
            bias = bias + alpha * (category[j] - y)
            cross_entropy[i] -= (category[j] * np.log(y) + \
                (1 - category[j]) * np.log(1 - y))

    # If v specified, print cost over epochs
    if v: 
        print('Cross Entropy cost function over ' + str(epoch) + ' epochs:')
        for i in range(epoch):
            print(round(cross_entropy[i], 2))

    # Return array of weights with bias weight attached 
    return np.insert(weights, 0, bias)  


def logistic_classifier(datapoint, weights):
    '''Given a data point, returns category based on weights'''
    # Insert constant bias
    datapoint = np.insert(datapoint, 0, 1)
    # Threshold of 0.5
    return 1 if logistic_func(np.dot(datapoint, weights)) > 0.5 else 0


if __name__ == '__main__':
    main()
