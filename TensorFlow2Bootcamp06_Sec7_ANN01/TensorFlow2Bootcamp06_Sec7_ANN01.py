"""
Tensorflow 2 and Keras Deep Learning Bootcamp
https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp

16 September 2023

Section 7: Basic Artificial Neural Networks - ANNs

Remember to have Num Lock on
Alt + 228 = Σ
Alt + 0178 = ²
Alt + 251 = √
remember |x| = abs(x)

"""
print("========= 32. Introduction to ANN Section =========")
"""
First Theory:
- Perceptron model to Neural Networks
- Activation Functions
- Cost Functions
- Feed Forward Networks
- BackPropagation

Then Coding Topics:
- TensorFlow 2.0 Keras Syntax API - which is the main API for TensorFlow 2.0
- ANN with Karas
    Regression
    Classification
- Exercises for Keras ANN
- Tensorboard Visualizations
"""
print("========= 33. Perceptron Model =========")
"""
- Dentrites (inputs), Axon (output)
- Frank Rosenblatt, 1958 - Perceptron
- However, in 1969, Marvin Minsky and Seymore Parpet published Perceptrons
    and screwed everying up by saying they couldn't solve the XOR problem.
    An AI winter ensued and funding dried up throughout the 1970s.

Imagine a simple perceptron with two inputs

X1-----w1----->
                f(X)--------> y
X2-----w2----->

    y = X1 * w1 + X2 * w2

- Weight can be +ve or -ve

- The Perceptron "learns" through the changing of the weights.
    But there is a basic problem here. If X1 and X2 are zero, y will also be zero and,
    if this is wrong, there is NOTHING WE CAN DO, to change the output. 
    However much we change the weights, y will still be zero.
- This is why we add a "bias"

    y = (X1 * w1 + b1) + (X2 * w2 + b2)
- Biases can be +ve or -ve

- Think of it like this: 
    The input X1 * w1 has to overcome (basicaly become bigger) than the bias value 
    before it has a significant effect on the output value, y.

- We can also extract the bias mathematically, so it is not associated with
individual inputs, but exists at a layer level. Theoreticaly, for any number of
biases at an input level, there exists a single bias that is the sum of all these
So we can say

    y = (X1 * w1) + (X2 * w2) + B

Remember, technicaly, B = b1 + b2
TensorFlow will deal with biases by having them as a single input into a layer
"""
print("================ 34. Neural Networks ================")
"""
- The multi-layer percetron model
Terminology
- Fully-connected layer == Dense Layer
- Neuron == Perceptron
- "Deep" neural network - two or more hidden layers
- The "width" of a network is how many neurons there are in a layer
- The "depth" of a network is how many layers there are in total

X                               y label(s)
Input layer--->hidden layers--->output layer

- What is incredible about the neural network framework 
    is that it can be used to approximate any continuous function
- Zhou Lu and later on Boris Hanin proved mathermaticaly that
    Neural Networks can approximate and convex continuous function.
                                        (can continualy integrate over)
- See Wiki page "Universal Approximation Theorem"
------------------------------------------------------
- Previously in our simple model, we used a very simple sum function.
For most use cases, however, we will want to set
    constraints on what values are output - especialy in classification tasks.
- It would be useful, for instance, to have all outputs fall between 0 and 1.
- These values can then be presented as probobility assignments 
    for each output class/label
- We use "activation functions" to set the boudaries to output values.
"""
print("================ 35. Activation Functions ================")
"""
- So for an input we have x * w + b
- We can think of b as an offset value,
    and x*w having to reach a certain threshold before having an effect.
    
e.g. b = -10
- Then x * w won't really matter, or have a majority effect, 
    until their product surpasses 10.
- After that, the majority of the effect will be solely based on w.
----------------------------------------------------------------------
- Next we want to set boundaries on the overall output value of x * w + b
- We can state: z = x * w + b 
- And then pass z through an activation function to limit its value.

================== Some Common Activation Functions ==================

- If we had a binary classification problem, we would want an output of 0 or 1
- In this context z = wx + b, z is defined as 
    the total input to the activation function - f(z)
- Keep in mind, you will often see variables capitalized, f(X) or Z 
    to denote a tensor input consisting of multiple values.
    
=== Step Function ===
- z = wx + z
- Input z = wx + b < output 0
- Input z = wx + b > output 1 

- Useful for classification
- However, this is a very "strong" function, 
    since small changes arn't reflected
    
=== Sigmoid Function aka Logistic Function ===   
    
- It would be nice, however, if we were to have a more "dynamic" function
- The Sigmoid function has the same upper and lower bound (useful for classification)
- z = wx + z

    f(z) = 1/(1 + exp(-z))

- Will output values between 0 and 1 and we can grab this, 
    and interpret it as a probability

=== Hyperbolic Tangent: tanh(z) === 
- Output between 1 and -1 - looks similar to Sigmoid

=== Rectified Linear Unit (ReLU) === 
A relativly simple function:

    y = max(0, z)
- Input < 0, output = 0
- Input > 0, output = z

- ReLU, good performance, especialy when dealing with 
    the issue of "vanishing gradient" (discussed later)
- We will often fall back on ReLU due to its overall good performance
-------------------------------------------------------------
Check out
    https://en.wikipedia.org/wiki/Activation_function
"""
print("===== 36. Multi-Class Classification Considerations =====")
"""
17 September 2023
PyCharm Settings: General > Appearences > Uncheck Show intention bulb 

- All activations in the previous lesson are good for true or false classifications
    or for continouse values between 0 and 1
    
In a multi-class situation, the output layer is going to have multiple neurons

- There are two main types of muli-class situation
    1) Non-Exclusive Classes 
        - a data point can have multiple classes/categories/labels assigned to it
    2) Mutualy Exclusive Classes
        - only one class per data point

====== Non-Exclusive Classes ======
e.g. A Photo - tagged beach + family + swimming + etc.
Several of many neurons turned "on"

====== Mutualy Exclusive Classes ======
One of many neurons turned "on"
e.g. A Photo (again) - greyscale or full colour - can't be both

How to organize multiple classes?
The easyiest way is to simple have one output node per class.

outputs
    ---> Class 1
    ---> Class 2

    ---> Class N

We can't pass around values like "red", "blue", "green" - can't multiply/divide them
So how do we transform our network correctly, 
to be able to use multiclass situations?

Instead we use "one-hot encoding" (one means "on") aka "dummy variables"

== Mutualy Exclusive Classes ==
                Red     Green   Blue
Data Point 1    1       0       0     
Data Point 2    0       1       0 
Data Point 3    0       0       1 

Data Point N    1       0       0 
- We use the Softmax activation function in the last layer
    The Softmax function calculates the probobility distribution
- Softmax calculates the probopilities of each target class over all
    target classes - the sum of all classes' probabilities = 1
- The target class chosen will have the highest probability

== Non-Exclusive Classes ==
                A       B       C
Data Point 1    1       1       0     
Data Point 2    1       0       0 
Data Point 3    0       1       1 

Data Point N    0       1       0 
- We can use the Sigmoid function here - each neuron will output
    a value between 0 and 1 indicting the probability of 
    having that class assigned to it.
"""
print("===== 37. Cost Functions and Gradient Decent =====")
"""
Cost functions aka loss functions aka error functions

Alt ? ŷ 
Alt 229 σ
Alt 228 Σ
Alt 0178 ²
- The last output layer ŷ is the model's estimation of
    what it predicts the label to be.
- So, after the network has created a prediction,
    how do we evaluate it?
- And, after the evaluation, how can we update the network's weights and biases?

- We need to take the estimated outputs and 
    compare them to the required/real values of the label
    Remember, this is the training data set
- We need the cost function to be an average (over all output neurons????)
    so it can output a single value.
- We will keep track of this loss/cost during training to monitor the network performance
Here we introduce some variable names

    "y" represents the true/required value
    "a" represents the neuron's prediction/actual value
    aL = actual output value of layer L - the last layer
    a(L-1) is next to last layer etc.
    We say "a of L" or "a of L minus 1"

In terms of weights and biases
    w*x + b = z
    Pass z into activation (Sigmoid for example) function σ(z) = a

=== The Quadratic Cost Function ===
Very common

    Cost = C = (1/2n) * Σ||y(x) - aL(x)||²
So

    C(W, B, Sr, Er)

W is the neural networks weights
B is the neural network biases
Sr is the input of a single training sample
Er is the desired output of that training sample

- If we have - even quite a modest - network, C will still be very complex
with huge vectors/tensors of weights and biases.
- Imagine labeling all the weights and biases in a network uniquly
    it is very quickly overwelming.

- So how do we deal with this problem?
- In the real world this means we have some cost function C 
    that is dependent on lots of weights
    C(w1, w2, w3,...wn)
- How do we figure out which weights lead us to the lowest cost?
- Imagine a very simple netweok with only one weight w
- We want to minimize our cost/loss (overall error)
- Which means we need to figure out, what value of w
    results in the minimum of C(w)
- Imagine a 2D plot of C(w) - w along the x-axis, C(w) on the y-axis
- Imagin a parabola
- We are looking for the minimum of the function C(w)
- Students of calculus will know, we could just take 
    the derivative of the cost function and solve for 0
- But remember our real cost function will be very complex (imagine a squiggel on the plot)
    possibly with many local minima. And it will not be 2-dimensional. It will be
    as many dimensions as there are ws. 1000s of weights - we can't take that derivative.
- Instead, what we need to do, is take a stochastic approch.
- We can use gradient descent to solve this type of problem. 
 
=== Gradient Decent === 
- Go back to our simple 2D parabela and imgine how gradient decent
    would work with that.
i)      Calculate a slope at a point on the C curve
ii)     Move in the downward direction from that point on the slop
iii)    Repeat process until we converge to gradient 0

Think about the step size - how far you move down the curve at each iteration
- If you take smaller step sizes, it takes longer to find the minimum
- If you take longer step sizes, its faster, but you might overshoot the minimum altogether.
- That step size is kown as the "learing rate"
- We can improved the process by reducing our step size as we go along 
    - this is known as "adaptive gradient decent"

- In 2025, Kingma ad Ba published their paper:
    "Adam: A Method for Stochastic Optimization".
- Adam is a much more effcient way of searching for these minima
    You'll see it used in our code.
- Adam out performs other gradient decent algoritmhs 
    such as AdaGrad, RMSProp, SGDNesterov and AdaDelta  

=== N-Dimensional Space ===
∇ - called "Del" - inverted delta
Alt ? ŷ 
Alt 229 σ
Alt 228 Σ
Alt 0178 ²

- But remember, we are calculating this descent in an n-dimensional space
    for all our weights
- When dealing with these N-dimensional vectors (tensors),
    the notation changes from "derivative" to "gradient"
- This means we calculate ∇C(w1, w2,...wn) - we say "Del C"

=== Cross Entropy Cost Function ===
- For classification problems, we often use the "cross entropy" loss function.
- The assumption here is that the model predicts 
    a probability distribution p(y=i) for each output class, i=1,2...
    Some complicated formula, that we are told not the worry about.

- So we are told, that for multiclass classification, where num classes  > 2
    we will call on cross entropy as our go-to cost function

Q: So once we get that cost/lost value, how do actualy go back and
    adust our weights and biases?
A: backpropogation
"""
print("========================= 38. Backpropogation =========================")
"""
∇ - called "Del" - inverted delta
Alt ? ŷ 
Alt 229 σ
Alt 228 Σ
Alt 0178 ²
⊙

- This is the hardest part of the entire theoretical deep learing process.
- Imagine a very simple network where one layer only has one neuron

- This means C(w1,b1,w2,b2,w3,b3)

            w1+b1   w2+b2   w3+b3   
        N---------N-------N--------N
        L-n      L-2     L-1       L     

L-1 = L minus 1

- Focussing on the last two layers (L-1 and L), let's define z = wx+b
    Then applying the activation function activation, a = σ(Z)

- x is only really valid, as the inital input to the leftermost neuron.
    As we move rightward through the network, x is technicaly the output
    of the neuron in that layer = a, so as you move further 
    into the network z = wa+b

    zL = wL * a(L-1) + bL - sorry, the Ls should all be superscripted

to make clear

    z at layer L = weight at layer L * a at layer (L-1) + b at layer L
    
So z at the last layer is defined by the weights and biases at that layer 
    and the activation at the previous layer

So

    aL = σ(zL)

    C = (aL - y)²

- Now partial derivative and the chain rule ensue

- The main take-away, here, is that we can use the gradient to go back trough
    the network and adjust weights and biases to minimize the output of
    the error vector at the last output layer.
    
- Using some calculus notation, we can expand this idea from networks with
    one neuron per layer, to networks with muliple neurons per layer.
- Hadamard Product 
          
    [1]  ⊙  [3]  = [1*3] =  [3]
    [2]     [4]    [2*4]    [8]

- Given this notation and backpropogation, we have a few main steps
    to train the neural network.
- Note! You don't have to fully understand this 
    to continue with the coding portions

=== Let us now Review the learning process for a network ===
Feedforward:
Step 1: Using input x set the activation function a or the input layer.
    z = wx + b
    a = σ(Z)
This a then feeds into the next layer (its z = wa + b) and so on.

Step2: For each layer, compute
    zL = wL * a(L-1) + bL
    aL = σ(zL)
Step3: We compute our error vector:
- some nice vector notiation I can't begin to duplicate here
- DON'T TAKE TOO SERIOUSLY - THIS IS THE BEST I COULD DO

    δL = (aL - y) ⊙ σ(ZL) the error vector for the last layer

    so δL is the error vector of the last layer

- Now let's write out our error term for a
    layer in terms of the error of the next layer (since we are moving backwards)

Step4: Backpropergate the error:

    lower case "l" = any layer l
    upper case "L" = the very last layer
    Transpose of a matrix = ᵀ
    
    - For each layer: L-1, L-2,... we compute:
    
    δl = w(l+1)ᵀ * δ(l+1) ⊙ σ(Zl) - the error vector of any layer l

    - w(l+1)ᵀ is the transpose of the weight matrix of layer l+1

- When we apply the transpose weight matrix, w(l+1)ᵀ we can think
    intuitivly of this as moving the error backward through the network,
    going us somesort of measure of the error at the output of the lth layer

- We then take the Hadamard product ⊙σ(Zl). This moves the error backward
    through the activation function in layer l, giving us the error δl in
    the weighted input to layer l.

- The gradient of the cost function is given by:
    -For each layer L-1, L-2,... we compute
    
    - Some partial   <--- something like this
    
        ∂C/ ∂wl = a(l-1) * δl
        That's the partial derivative of C with regards to wl
        
        ∂C/ ∂bl = δl
        That's the partial derivative of C with regards to bl

- This then allows us to adjust the weights and biases to help
    minimize the cost function
- Some external links

Onto next file 39. TensorFlow vs. Keras Explained
TensorFlow2Bootcamp06_Sec7_ANN02
"""




























