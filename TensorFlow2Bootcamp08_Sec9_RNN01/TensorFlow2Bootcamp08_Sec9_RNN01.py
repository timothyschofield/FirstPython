"""
Tensorflow 2 and Keras Deep Learning Bootcamp
https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp

05 October 2023

Section 9: Recurrent Neural Networks

"""
print("========== 76. Section Overview ==========")
"""
These are good at dealing with sequence information
Sequences in time like sales data or heart beat data (Time Series)
Sequences in space/time like text

Specialised types of neurons called LSTMs for Long Short Term Memory Units
and Gated Recurrent Units or GRUs

One of the first things we will do is create a network that can predict a sin wave
"""
print("========== 77. RNN Basic Theory ==========")
"""
Examples of sequences
    Time Series Data (Sales)
    Sentences
    Audio
    Car Trajectories
    Music

So, consider the sequence 
                [1,2,3,4,5,6]
    Can you predict a similar sequence shifted one time step into the future?
                [2,3,4,5,6,7]

- Can we build a network that can learn from the history?
- To do this properly, we need to somehow let the neuron "know"
    about its previous history of outputs
- One way to do this is to feed the neurons output back into itself as an input.

Recurrent Neuron

             Next layer/Output
              ↑
            Output-----------------→ 
              ↑                     |
             f(x)                   | Feeds back to itself
              Σ                     |
              ↑                     ↓
            Input ←------------------
              ↑
            Previous layer/input

How does this look over time?
This is the "unfolded/unrolled view" of a single recurrent neuron through time

Output                  Output                     Output
at t-1                  at t                       at t+1
  ↑                      ↑                           ↑
 f(x)→----------↘       f(x)→----------↘            f(x)→----------↘ 
  Σ              ↘ -----→ Σ              ↘ ---------→ Σ              ↘ ----------→
  ↑                      ↑                            ↑              
Input                  Input                        Input
at t-1                  at t                       at t+1

=============================== TIME =====================================>

- So we retain historical information
- Cells that are a function of inputs from previous time steps 
    are also know as "memory cells"
- RNNs are also flexible in their inputs and outputs, for both sequences and single
    vector values - not quite sure what this means
----------------------------------------------------
- We can create entire layers of recurrent neurons

- An unrolled RNN layer

Output                  Output                     Output
at t-1                  at t                       at t+1
  ↑                      ↑                           ↑
N1 N2...Nn→-----------→N1 N2...Nn→----------------→N1 N2...Nn→---------→
  ↑                      ↑                           ↑              
Input                  Input                        Input
at t-1                  at t                       at t+1

=============================== TIME =====================================>

----------------------------------------------------
- RNNs (as we have mysteriously mentioned earlier)
    are also very flexible in their inputs and outputs. 
    There are different types of architectures we can use.
Examples:

1) Sequence to sequence (Many to Many)
    Y0      Y1      Y2      Y3      Y4
    ↑       ↑       ↑       ↑       ↑ 
    N------→N------→N------→N------→N
    ↑       ↑       ↑       ↑       ↑ 
    X0      X1      X2      X3      X4

OK, imagine the Xs representing a sequence [X0, X1, X2, X3, X4]
and the Xs all being passes into the network. 
What we want is a sequence out [Y0, Y1, Y2, Y3, Y4]
e.g. If you have a five word question, predict the answer of five words.

2) Sequence to Vector (Many to One)
                                    Y4
                                    ↑ 
    N------→N------→N------→N------→N
    ↑       ↑       ↑       ↑       ↑ 
    X0      X1      X2      X3      X4

Pass in the sequence [X0, X1, X2, X3, X4] and get a single output Y4
e.g. Given five previous words, go ahead and predict the next word.

3) Vector to Sequence (One to Many)
    Y0      Y1      Y2      Y3      Y4
    ↑       ↑       ↑       ↑       ↑ 
    N------→N------→N------→N------→N
    ↑
    X0

Given one word, go ahead and predict the sequence of the next five words.
----------------------------------------------------------
- The basic RNN has a major disadvantage, 
    we only "remember" the previous output - we are only feeding in 
    the output of one timestep into the past.
- It would be great if we could keep track of longer histories,
    onto just short term history.

- Another issue that arises during training is the "vanishing gradient"
"""
print("========== 78. Vanishing Gradients and Exploding ==========")
"""
Consider a regular ANN

- Recall that the gradient is used in our training to adjust weights and
    biases in the network

- Thinks about a basic feed forward network.
    For complex data we have a multiple layer network with hidden layers.
- Now, issues can arise during backpropagation
- Remember we calculate some loss metric in the output layer and then propagate
    it back all the way to the input layer on each epoch.
    If we have lots of hidden layers, then the update to the weights and biases at a given layer,
    is a function of many, many other derivatives that we are calculating along the way back.
    So (to say it again) the error gradient gets propagated backwards.
- As we go backwards to the "lower" layers,
    the gradients often get smaller, to the point where weights never change at lower levels.
- The opposite can occur on the way back to the output layer, where gradients explode.

================== WHY? ================== 

Consider at a common activation function such a sigmoid: (diagram needed)

    f(x) = 1/(1 + exp(-x))

Input z = wx + b
Output between 0 and 1
The further away the input is from zero, the smaller the gradient. You don't need to
have z get very far from 0 (in either direction) before the gradient is really small.
And we know that backpropagation and the gradient calculation is essentially just 
calculating this derivative in multiple dimensions as you go back through the hidden layers.

If you plot a graph of the sigmoid function, on top of its derivative, and to scale, you
can see that the sigmoid dwarfs the derivative. Sigmoid maximum value is 1, but the max value
of the derivative is only about 0.2

So when you are dealing with n hidden layers, and you have an activation function like sigmoid, 
you have n small derivatives being multiplied together.
By the time we get down towards the input layer, the gradients are so small 
they cannot significantly effect the weights and biases.
This is bad.

====================== How to Fix? ========================

1) Different Activation Functions
- Perhaps use a different activation function to sigmoid - one where the derivative doesn't tend to zero
    as the magnitude of the input becomes large.

- The ReLu: f(x) = max(0, x)
- ReLu doesn't saturate positive values - whatever that means.

OpenAI explains:
The statement "ReLu doesn't saturate positive values" means that 
the ReLu activation function does not impose an upper bound on the output values when the input is positive. 
In other words, for positive input values, the ReLu function allows the output to grow without bound. 

Unlike the sigmoid where the derivative tends to zero for larger input values.

- Or the Exponential linear Unit, ELU - again gradient does not flatten out for large z

2) Perform batch normalization
Perform batch normalization, where the model will normalize each batch
using the batch mean and standard deviation.
- This alleviates the issue of vanishing gradient - but how? - I think this is explained in lesson 80.

3) Choose different initialization of weights (Xavier Initialization)

==============================================

There is also the issue of exploding gradients
    We can clip gradients (dirty trick), limiting them from -1 to 1

==============================================
The RNN for Time series present their own gradient challengers.
Let's explore a special neuron called LSTM (Long Short Term Memory)
to help fix these issues.
"""
print("========== 79. LSTM and GRU Units ==========")
"""
- Many of the solutions presented above for vanishing gradients in ANNs 
    can also be applied to RNNs: different activation functions, batch normalization etc.
- However because of the typical "length of time series input",
    (In time or...what?), what can happen, using these initializations, or activation functions,
    the training may slow down.

- A possible solution is just to shorten the time steps used for prediction,
    but this makes the model worse at predicting trends over longer periods.
- RNNs tend to forget quite quickly as information from the past is lost (diluted?)
    at each step going through the RNN.
- We need some sort of long term memory.

============ The LSTM ============
- The LSTM was created to help address RNN's forgetfulness
- So we long term and short term memory going in and out of the neuron
- OK, before going onto LSTMs let's review what happens in a regular RNN

- An unrolled RNN layer/network

Output                  Output                     Output
at t-1                  at t                       at t+1
  ↑                      ↑                           ↑
N1 N2...Nn→-----------→N1 N2...Nn→----------------→N1 N2...Nn→---------→
  ↑                      ↑                           ↑              
Input                  Input                        Input
at t-1                  at t                       at t+1


So a lets look at a single recurrent neuron:
                
    Output                  
   N(t-1)-------→N-------→ Output(t)
                 ↑
               Input (t)

It takes in the previous Output(t-1) and the current input(t)
and producing the next Output(t)

Now, for the sake of equation labeling, we change the nomenclature to:
     
   H(t-1)-------→N-------→ H(t)
                 ↑
               X(t)
               
Where "H" means Hidden.

But what is happening inside the neuron?
Well in a standard RNN neuron - a single hyperbolic tangent function

 H(t-1)-------→ tanh(W[H(t-1), X(t)]+b) = -------→ H(t)
                          ↑
                        X(t)

In a LSTM the "repeating module" (New term) has a slight difference to it.
Instead of having a single neural network layer, there are going to be 4 layers
working and interacting in a special way.

                        Output at time t 
                                 ↑
        Long Term      ╔ ═ ═ ═ ═ ═ ╗        New Long
        Memory-------→ ║           ║-------→Term Memory
                       ║           ║
        Sort Term      ║           ║        New Short
        Memory-------→ ║           ║-------→Term Memory
                       ╚═ ═ ═ ═ ═  ╝ 
                        ↑
                Input at time t 

More detail 

                                     Output at time t 
                                            ↑
        Long Term      ╔ ═ ═                 ═ ═ ═ ╗        New Long
        Memory-------→ ║  Forget           Output  ║-------→Term Memory
                       ║  Gateway          Gateway ║
                       ║                           ║                      
                       ║                           ║                       
        Sort Term      ║  Input            Update  ║        New Short
        Memory-------→ ║  Gateway          Gateway ║-------→Term Memory
                       ╚═ ═ ═                 ═ ═  ╝ 
                         ↑
                 Input at time t 
 
07 October 2023
Σ	Alt 228
σ	Alt 229
============================================
--- Input ---
Forget Gateway from Long Term Memory
Decides what to forget from the previous memory units

Input Gateway from Short Term Memory
Decides what to accept into the neuron from the previous memory units

--------------
--- Output ---
Output Gateway to New Long Term Memory
Outputs the new long therm memory

Update Gateway to New Short Term Memory
Updates the memories
============================================
- A gate, optionally lets information through in a way similar
    to sigma activation. Input information is squeezed to between 0 and 1
    and if the result is 0, we don't let it through, 
    if the result it 1 we do let it through

    ----→ σ ----→ x
    
============================================
He starts to give up here, it's too difficult to explain at this simple level.
- Where did this "Long Term" and "Short Term" memory come from - what's the difference?
Not even English
- "We can kind of think of this as being passed in through conveyor belts inside of this neuron"!

- "And what ends up happening, is it just ends up, running down, straight the entire chain
    and has some linear interactions with a few functions, inside of the cell"

For the purpose of mathematical notation we are going to relabel some of the above

                                        Output at time t
                                            h(t)
                                            ↑
         (LTMem)       ╔ ═ ═                 ═ ═ ═ ╗        (NewLTMem)
        C(t-1)-------→ ║  Forget           Output  ║-------→C(t)
                       ║  Gateway          Gateway ║
                       ║                           ║                      
                       ║                           ║                       
         (STMem)       ║  Input            Update  ║        (NewSTMem)
        h(t-1)-------→ ║  Gateway          Gateway ║-------→h(t)
                       ╚═ ═ ═                 ═ ═  ╝ 
                         ↑
                        x(t)
                    Input at time t

Here, diagrammatically and explanationally, we drop off a cliff - 5min 51sec
Need to look up a detailed explanation of the internal workings of a LSTM

There is a detailed description of the internals of a LSTM and
LSTM with peepholes.

Internals of GRU

GRU - getting more popular

5min 51sec -> end at 11min ->30sec

END of lesson -  next lesson 80. RNN Batches
"""






































