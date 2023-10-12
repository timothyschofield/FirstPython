"""
Tensorflow 2 and Keras Deep Learning Bootcamp
https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp

07 October 2023

Section 9: Recurrent Neural Networks

"""
print("========== 80. RNN Batches ==========")
"""
- How to use RNN on a basic time sequence such as a sine wave
- Before we do that, what do RNN sequence batches look like?

Consider a simple time series
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Let us separate this into two parts: here 9 is the "next sequence value"

            training sequence  predicted label/training point
    [0, 1, 2, 3, 4, 5, 6, 7, 8] → [9]
    
- We feed the training sequence into our network, the we say
    try to predict a value, and then we compare it to the next sequence value - 9 in this case
    
- Now remember, we can usually decide how long the training sequence
should be and how long the predicted label should be    
e.g.
    training sequence  predicted label/training point
    [0, 1, 2, 3, 4] → [5, 6, 7, 8, 9]
    
- Also remember, we are going to feed in batches of these "data points" aka. "training points"
"training point" - refers to [9] or [5, 6, 7, 8, 9] above examples  
    
- So not only we can vary the size of the training point, 
    but also the number of sequences to feed in per batch
    
Here is a single batch of time series sequences 
Here we have three data point entries, where we have a training sequence per data point of 4 numbers 

            training sequence  label
data point 1    [0, 1, 2, 3] → [4]
data point 2    [1, 2, 3, 4] → [5]   
data point 3    [2, 3, 4, 5] → [6]   
    
So we have three variables
1) How many time series sequence per batch
2) How long the training sequence/ portion should be
3) How long the label should be   
    
Q: How do we decide how long the training sequence should be?
A: No definitive answer, but it should be at least long enough to 
    capture any "useful" trend information.

"Useful"
This often takes domain knowledge.
Say we have a graph of sales of skying gear per month. 
We will see a yearly trend - more gear bought in November and December,
and also that sales are generally increasing year on year.

To predict future trends our training sequence should include at least 12 months.

This would give you a chance of predicting yearly trends, 
    but what of the year on year increase of sales - I don't know?!

============================================
How do we forecast with RNNs?

- Typically a good starting choice for the label is just one data point into the future.

Imagine our time series data is:
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
and we train from sequences such as:

  training sequence  label
    [0, 1, 2, 3] → [4]
    [1, 2, 3, 4] → [5]   
    [2, 3, 4, 5] → [6] 

We want to predict into the future, i.e. beyond 9
- Out forcasting technique is to predict a time step ahead and then
    incorporate our prediction into the next sequence we predict from

Example:
       [6, 7, 8, 9] → [10] is the forecast prediction
    
We are forcasting into the unknown future and 10 is predicted
but we don't know how far off 10 is. This isn't like training with a train/test split.
We can get an idea, 6...7...8...9, the pattern looks like +1 at each step, so 10 looks good, but we don't know.
    
- Say now we want to predict 4 steps into the future, we keep predicting further by
    including our forecast prediction into our batch and essentially dropping the 6 from [6, 7, 8, 9]
    
    [7, 8, 9, 10] → [11.2]

rinse and repeat 
    
    [8, 9, 10, 11.2] → [12.4]   
    [9, 10, 11.2, 12.4] → [14]    
    [10, 11.2, 12.4, 14] → Completed Forecast   
    
So, obviously, the further you forecast out, 
    the greater likelihood the forecast will deviate from the (what! - the ideal we don't know!)
"""
print("========== 81. RNN on a Sine Wave - The Data ==========")
"""


"""

































