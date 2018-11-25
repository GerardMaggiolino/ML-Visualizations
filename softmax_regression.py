'''
Contains SoftmaxRegression class for multiclass classification.

Running module as main generates visualizations of classification.
Default parameters can be modified directly from main - examples of
usage in main. 
'''

import numpy as np
from random_data import random3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def main(): 
  # Config options
  size_data_set = 20
  num_cat = 3
  plot_correct = True

  # dataset contains rows of data points 
  dataset = []
  # categories contains rows of one-hot categories for data points
  categories = []
  # Set up generators, create random data, store training data
  generators = [random3d(size_data_set, v=True, b=i*50, s=1, noise=0) \
    for i in range(-num_cat//2, int((num_cat+0.5)//2))]
  for _ in range(int(size_data_set/2)):
    for category, gen in enumerate(generators, 0): 
      dataset.append(np.array(next(gen)))
      categories.append(category)

  dataset = np.array(dataset)
  # Convert to one-hot
  categories = np.array(categories)
  tmp = np.zeros((len(categories), num_cat))
  tmp[np.arange(len(categories)), categories] = 1
  categories = tmp

  # Shuffle data
  state = np.random.get_state()
  np.random.shuffle(dataset)
  np.random.set_state(state)
  np.random.shuffle(categories)

  # Plot training data
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d') 
  col = lambda x : \
    ((x+3)%num_cat/num_cat, (x+5)%num_cat/num_cat, (x+7)%num_cat/num_cat)
  for g in range(len(generators)):
    x = [dataset[i][0] for i in range(len(categories)) if \
      np.argmax(categories[i]) == g]
    y = [dataset[i][1] for i in range(len(categories)) if \
      np.argmax(categories[i]) == g]
    z = [dataset[i][2] for i in range(len(categories)) if \
      np.argmax(categories[i]) == g]
    ax.scatter(x, y, z, color=col(g), label=f'Class {g+1}')

  # Train model on data 
  model = SoftmaxRegression()
  model.trainer(dataset, categories, 200, alpha=0.0005)

  # Test on remaining data in generators, plot incorrect classifications
  correct = 0
  for cat, gen in enumerate(generators, 0): 
    for point in gen: 
      if model.classifier(np.array(point)) == cat: 
        correct += 1
        if plot_correct:
          ax.plot([point[0]], [point[1]], [point[2]], 'go')
      else:
        ax.plot([point[0]], [point[1]], [point[2]], 'ro')
  # Print performance
  print(f'Percent correct over {int(num_cat *size_data_set/2)}: \
    {round(100*correct/(size_data_set * num_cat/2), 2)}')

  ax.legend()
  plt.show(block=True)


class SoftmaxRegression: 
  def __init__(self): 
    # Stores training mean, std, allows for online classification of 
    # single data points.
    self.mu = None
    self.sigma = None
    # Stores weights of model 
    self.weights = None


  def trainer(self, dataset, categories, epoch=20, alpha=0.001):
    ''' 
    Performs full batch gradient descent using cross entropy cost.

    Expects dataset, categories to be np array with rows of training 
    examples. 
    '''
    # Set up weights with bias 
    weights = np.zeros((len(dataset[0]) + 1, len(categories[0])))

    # Z-score the dataset, append bias
    self.mu = np.mean(dataset, axis=0)
    self.sigma = np.std(dataset, axis=0)
    dataset = (dataset - self.mu) / self.sigma
    dataset = np.append(dataset, np.ones((len(dataset), 1)), axis=1)

    for _ in range(epoch):
      # Shuffle data
      state = np.random.get_state()
      np.random.shuffle(dataset)
      np.random.set_state(state)
      np.random.shuffle(categories)

      output = self._softmax_function(np.matmul(dataset, weights))
      grad = alpha * np.matmul(dataset.T, (categories - output))
      weights = weights + grad
    self.weights = weights


  def classifier(self, datapoint): 
    '''
    Performs classification of a single data point based on training.
    '''

    # Z-score, append bias
    datapoint = (datapoint - self.mu) / self.sigma
    datapoint = np.append(datapoint, [1]).reshape(1, len(datapoint) + 1)
    return np.argmax(self._softmax_function(np.matmul(datapoint, self.weights)))


  @staticmethod
  def _softmax_function(x):
    ''' 
    Takes in data with example per row. Returns softmax over all examples.
    '''
    for i in range(len(x)):
      x[i] = np.exp(x[i] - np.amax(x[i])) / np.sum(np.exp(x[i] - np.amax(x[i])))
    return x 

if __name__ == '__main__': 
  main()
