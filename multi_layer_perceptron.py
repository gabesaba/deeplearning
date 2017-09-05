from units import *


class MLP():

    def __init__(self, input_dim, output_dim, alpha = 0.01):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.layers = []
        
    def construct_unit(self, unit_type, dim_input):
        if 'sigmoid' == unit_type:
            return Sigmoid(dim_input)
        elif 'relu' == unit_type:
             return Relu(dim_input)
        elif 'output' == unit_type:
            return Output(dim_input)
        else:
            #by default choose relu :)
            return Relu(dim_input)

    def add_layer(self, unit_type, num_units):

        if 0 == len(self.layers):
            #if first layer, use input dimensionality
            dim_in = self.input_dim
        else:
            #else, use dimensionality of previous unit
            dim_in = len(self.layers[-1])
        self.layers.append([self.construct_unit(unit_type, dim_in) for _ in range(num_units)])
        

    def compile(self):
        self.add_layer('output', self.output_dim)
        
    def forward(self, inputs):

        assert type(np.array([])) == type(inputs)
        
        # convert input values into input units
        inp = [Unit(value) for value in inputs]
        
        for layer in self.layers:
            out = []
            for unit in layer:
                unit.forward(inp)
                out.append(unit.out)
            inp = out #last layers output is next layer's input
        self.out = out

        return out

    def backward(self, losses):

        assert type(np.array([])) == type(losses)
        
        for out, loss in zip(self.out, losses):
            out.grad = loss
        
        for layer in reversed(self.layers):
            for unit in layer:
                unit.backward()
                
    def update(self):
        
        for layer in reversed(self.layers):
            for unit in layer:
                unit.update(self.alpha) 

    def reset_grads(self):
        
        for layer in self.layers:
            for unit in layer:
                unit.reset_grad()
                

if __name__ == '__main__':
    
    inp = np.array([1, 2, 3]) #input

    target_value = 27.0
    
    two_layer_net = MLP(len(inp), 1, alpha = 0.01)
    two_layer_net.add_layer('relu', 10)
    two_layer_net.compile()

    import time
    converged = False
    print('\nAttempting to output %f\n'%target_value)
    
    i = 0 #anti infinite loop
    
    while not converged and i < 100:
        time.sleep(0.1)
        out = two_layer_net.forward(inp)
        for unit in out:
            print unit.value
        loss = target_value - unit.value
        
        #if we get within 2.5% of target value, stop
        if abs(loss) < abs(target_value * 0.025):
            converged = True

        #too large of gradient causing issue with relu (killing units), so set to -1 or 1
        loss = min(1, loss)
        loss = max(-1, loss)
        
        two_layer_net.backward(np.array([loss]))
        two_layer_net.update()
        two_layer_net.reset_grads()
    print 'converged :)'
