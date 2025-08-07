# Machine Learning Fundamental tutorials

This is code that I've following along various tutorials on the fundamentals of machine learning, which mostly follow tutorials by Andrej Karpathy.

Below I have descriptions of the different functionality I've created while coding along.

## Multi layer perceptron (mlp.ipynb)

Here I will attempt to explain how the neural network works by walking through my code.

There are 4 main classes in the multi layer perceptron. Pytorch defines its multi layer perceptron, or neural network using the nn.Module class. Here we will try to emulate the key features of this famous library's class. 

At a high level, a multi layer perceptron is a system with 4 key levels of abstraction. From the MLP itself, it moves down to individual layers (as is suggested in the name). Each of these layers are sets of neurons that have connections to layers before and after, and each neuron is composed of weights, biases, and activation functions which all mathematically define its determined output.

This system is then responsible for sending a signal based on the weights and biases of each neuron in the structure (forward propagation) and then correcting those weights based on the difference it calculates between its guess and the actual value we got (backward propagation).

We've defined 4 classes and those 2 core functionalities in the code.

### 1. Value

To start we've created a new value class which will represent any number we use in our multi layer perceptron, so that it can easily perform forward and backward propagation. These key attributes are:
a. data - the value of the number
b. children - the numbers it was connected to mathematically; f.e. in 3 * 2 = 6 the child of value 6 would be 3 * 2
c. backward - the derivative equation we get based on the connected children
d. gradient - the calculated value of the backward function

The way we tie all of these values together to perform forward and backward propagation is by using Python's magic methods (f.e. __mul__, __add__) to define what happens when you perform any mathematical operation using our Value class.

Two key things happen in our magic method:
a. the result and children of the operation is saved onto a new Value class with data and children (relevant to forward prop)
b. create a function which will define derivative functions for both children that calculate change of output with respect to both children.
  - f.e. 2 * 4 = 8; 2 will be gradient of 8 * 4 and 4 will be gradient of 8 * 2. Intuitively, if you add 1 to 2, the overall output will change from 8 to 12 so the results gradient is affected 
  - Think of these as building blocks that track each individual step of the overall derivative function connected by the chain rule (this explains why we multiply the output's gradient value to the children's gradients)
  - The cumulative nature of the derivatives takes into account for when the value is used twice. The amount that this value's derivative functions for both contexts will be the derivative values added together.

Now that we have this we can implement a backwards function on the Value which starts from itself (meant to be used on the final output node which we initialize as a gradient of 1, which is how much the output changes with respect to its own changes),and then walk backwards to each child in the tree to calculate all gradients. We use a topological sort to create the correct order.

We include numberical operations required for all the releavant equations in loss, activation, forward prop: addition, power, subtraction, division, and power function. 

For clarity, these are all the functions we are using in training following the order:
- forward propagation (w1x1 + w2x2 + ... + b = output)
- tanh activation function ((e^x - e^(-x)) / (e^x + e^(-x)))
- loss function (sum((y_guess - y_pred) ** 2))
  - yout = our guesses; ygt = y ground truths
  - we sum because we are using multiple training examples (not averaging for simplicity)

Here is the full code for the Value object which we've just outlined.

```
class Value:
  def __init__(self, data, _children = (), label = ""):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self.label = label

  def __repr__(self):
    return f"Value(data = {self.data})"
    
  def __add__(self, other):
    other = self.get_other(other)
    out = Value(self.data + other.data, (self, other), "+")
  
    def _backward():
      self.grad += out.grad
      other.grad += out.grad

    out._backward = _backward
    return out
  
  def __neg__(self):
    return self * -1
  
  def __sub__(self, other):
    return self + (-other)

  def __pow__(self, other):
    assert isinstance(other, (int, float))
    out = Value(self.data ** other, (self, ), f'**{other}')

    def _backward():
      self.grad += other * (self.data ** (other - 1)) * out.grad

    out._backward = _backward

    return out
  
  def __truediv__(self, other):
    return self * other**-1
  
  def __mul__(self, other):
    other = self.get_other(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
      self.grad += out.grad * other.data
      other.grad += out.grad * self.data

    out._backward = _backward

    return out
  
  def __rdiv__(self, other):
    return self / other
  
  def __rmul__(self, other):
    return self * other
  
  def __radd__(self, other):
    return self + other
  
  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(t, (self, ), "tanh")

    def _backward():
      self.grad += (1 - t ** 2) * out.grad

    out._backward = _backward
    
    return out
  
  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), "exp")

    def _backward():
      self.grad += out.data * out.grad
      
    out._backward = _backward
    
    return out
  
  def get_other(self, value):
    return value if isinstance(value, Value) else Value(value)

  def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    
    build_topo(self)

    self.grad = 1
    
    for node in reversed(topo):
      node._backward()
```

### 2. Neuron
Now that each Value is built to perform mathematical operations, store its connections, and the gradient, we can create a neuron. The neuron consists of 3 main parts:

a. the weights - an array of normalized numbers between -1 and 1
b. the bias (one for each neuron) - normalized number between -1 and 1
c. the inputs (raw inputs or results of activation)

To use:
a. We determine how many weights we want to initalize with. Bias gets created by default
b. call it with inputs which correspond to weights we've created(w1x1 + w2x2 + ... + b = output). The call should also handle activation (tanh in our case).

### 3. Layer
We now can create layer which is defined by two factors:

a. how many neurons do we have in the previous layer (number of inputs or neurons)
b. how many neurons are in the layer

With this information we:
a. initialize list of specified number of neurons, with specified number of inputs for each neuron
b. for each neuron call the designated array of inputs, which gives you an array of outputs one for each neuron

### 4. MLP
We can now create our full multi layered perceptron object, using number values which correspond to each layers input and output numbers. 

We take two numbers as inputs:
1. Number of inputs (actual x values fed into first layer)
2. An array representing number of nodes in each layer, with the length of the array representing the number of layers

Maintaining the same variable to keep outputs, we can use whatever output we get from the first layer and feed them into the second layer and so on until the last layer.


```
class Neuron:
  def __init__(self, nin):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1, 1))

  def __call__(self, x):
    act = sum((wi * xi for wi, xi in list(zip(self.w, x))), self.b)
    out = act.tanh()
    return out
  
  def parameters(self):
    return self.w + [self.b]
  
class Layer:
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x)for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    return[p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
```

### Training
To start I'd like to note that above we've created a parameters function for each of the classes, so that we can easily nest into each layer, neuron and grab all the weights and biases, which we'll need to tweak during training.

Then the steps to training are as follows:
1. Calculate the loss function
2. Zero all parameter gradients
3. Run the backward function to recalculate all parameter gradients
4. Adjust all the weights of each parameter based on the new gradient and learning rate

With this you should slowly see the loss function getting lower with each training and get parameters that accurately predict the expected y values.

