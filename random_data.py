'''
Contains functions for generation of uniform randomly distributed data.

Data is generated from an underlying linear function, optionally 
printed to stdout. Some control over noise and placement of data is 
obtained through function parameters - see random3d and random2d 
documentation. Both return generators to the random data.
'''

import numpy as np 

def random3d(points=50, noise=-1, b=None, r=50, s=5, v=False) -> tuple: 
    '''
    Returns generator yielding 3d noisy planar data around origin
    
    Param:
        points: Number of data points to return
        noise: Level of noise in data (Sigmoid calculated) 
        b: Offset from origin 
        r: Maximum radius from point closest to origin
        s: Maximum slopes
    ''' 
    
    # Coefficients for general form equation  
    coeff = np.random.uniform(-s, s, 3)

    # If no offset from origin given, cacluate one randomly
    if b is None: 
      b = np.random.uniform(-50, 50) 

    # Print information if verbose 
    if v: 
        print('Generating around equation: ', end='')
        sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        for i in range(coeff.size):
            print(str(round(coeff[i], 2)) + 'x' + str(i).translate(sub)\
                + ' +  ', end='')
        print(str(b) + ' = 0')

    # Calculate translation vector from b
    point = np.array([0, 0, b/coeff[2]])
    trans = np.dot(point, coeff) / np.dot(coeff, coeff) * coeff 

    # Calculate noise 
    noise = (r/2) * (1 / (1 + np.exp(-noise)))

    while points > 0: 
        # Unit vector parallel to plane 
        vec = np.random.uniform(-1, 1, 3)
        vec = np.cross(coeff, vec)
        vec = vec / np.sqrt(np.dot(vec, vec))

        # Scale and translate to plane
        vec = np.random.uniform(r) * vec 
        vec = vec + trans
        # Return point on plane with noise 
        yield tuple(vec + np.random.uniform(-noise, noise, 3))

        points -= 1


def random2d(points=50, noise=-1, b=None, s=None, v=False, r=100) -> tuple:
    '''
    Returns generator yielding noisy linear data around origin
    
    Param:
        points: Number of data points to return
        noise: Level of noise in data (Sigmoid calculated) 
        b: Intercept
        s: Slope
        r: Range of x to generate data for 
    ''' 
    
    # Set nonspecified data 
    if b is None:
        b = np.random.uniform(-50, 50) 
    if s is None: 
        s = np.random.uniform(-5, 5)

    # Print function if verbose 
    print('Generating around equation: ', end='')
    print(f'{round(s, 2)}x + {round(b, 2)} = y')

    # Set x values, noise 
    x = np.random.uniform(-r/2, r/2, points)
    noise = (r/2) * (1 / (1 + np.exp(-noise))) 

    while points > 0: 
        # Return tuple with point  
        yield x[points -1], \
            s * x[points - 1] + b + np.random.uniform(-noise, noise) 
        points -= 1
