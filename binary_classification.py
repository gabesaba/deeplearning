import numpy as np
from multi_layer_perceptron import *

def quadrant():
    
    rand_val = np.random.randint(1,5)
    
    if 1 == rand_val:
        #quadrant 1 positive
        x1 = 1
        x2 = 1
        y = 1
        
    elif 2 == rand_val:
        #quadrant 2 negative
        x1 = -1
        x2 = 1
        y = -1
        
    elif 3 == rand_val:
        #quadrant 3 positive
        x1 = -1
        x2 = -1
        y = 1
        
    elif 4 == rand_val:
        #quadrant 4 negative
        x1 = 1
        x2 = -1
        y = -1

    #add gaussian noise
    x1 += np.random.normal(0, 0.25)
    x2 += np.random.normal(0, 0.5)
    return {'data':np.array([x1,x2]),'label':y}


def circle():
    
    radius_small = 1
    radius_big = 2

    rand_val = np.random.randint(0,2)
    
    if 0 == rand_val:
        radius = radius_small
        label = -1
    else:
        radius = radius_big
        label = 1

    x1 = (np.random.random() * 2 - 1) * radius #point between -1 and 1 * radius
    x2 = np.math.sqrt(radius**2-x1**2) #solve for x2 
    if np.random.randint(0,2): #flip sign of x2 with probability 0.5
        x2 *= -1
        
    return {'data':np.array([x1,x2]),'label':label}


def generate(num_points, dist='quadrant'):
    
    points = []
    
    if 'quadrant' == dist:
        create_point = quadrant
    else:
        create_point = circle
        
    for _ in range(num_points):

        point = create_point()
        
        points.append(point)
        
    return np.array(points)


def train(model, D, iterations = 1000, print_freq = 100):
    
    correct = 0.0
    alpha = 0.01
    for i in range(iterations):
        if not i % print_freq:
            print('Training accuracy on iteration %d: %s'%(i,'{:.1%}'.format(evaluate(model, D))))
            
        d = np.random.choice(D)
        ret = model.forward(np.array(d['data']))


        if ret[0].value > 0.0:
            predicted_label = 1
        else:
            predicted_label = -1
            
        label = d['label']


        if 1 == label and ret[0].value < 0.0:
            grad = 1.0
        elif -1 == label and ret[0].value > 0.0:
            grad = -1.0
        else:
            grad = 0.0

        model.backward(np.array([grad]))
        model.update()
        model.reset_grads()

        if predicted_label == label:
            correct += 1

def evaluate(model, D):
    
    correct = 0.0
    
    for d in D:
        
        ret = model.forward(np.array(d['data']))

        if ret[0].value > 0.0:
            predicted_label = 1
        else:
            predicted_label = -1

        if predicted_label == d['label']:
            correct += 1

    accuracy = correct/len(D)
    return accuracy


def test(dist = 'circle', unit = 'sigmoid', alpha = 0.01, num_train = 100, num_test = 20, iterations = 2700):

    print('\nTesting binary classification on %s distribution with unit type %s\n'%(dist,unit))
    training_data = generate(num_train, dist)
    test_data = generate(num_test, dist)

    model = MLP(2, 1, alpha = alpha)
    model.add_layer(unit, 10)
    model.add_layer(unit, 10)
    model.compile()
    train(model, training_data, iterations = iterations, print_freq = iterations/10)

    test_accuracy = evaluate(model, test_data)
    print('Test Accuracy: %s'%('{:.1%}'.format(test_accuracy)))


if __name__ == '__main__':

    test(dist = 'circle', unit = 'sigmoid', alpha = 0.1)
    test(dist = 'quadrant', unit = 'sigmoid', alpha = 0.1)

    test(dist = 'circle', unit = 'relu', alpha = 0.01)
    test(dist = 'quadrant', unit = 'relu', alpha = 0.01)
