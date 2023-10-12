"""
Tensorflow 2 and Keras Deep Learning Bootcamp
https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp

24 September 2023

Section 8: Convolutional Neural Networks - CNNs

"""
print("========== 59. CNN Overview ==========")
"""
CNNs are good with image data
- Image Kernels and Filters
- Convolutions
- Pooling Layers
MNIST       - Grayscale
CIFAR-10    - Color
"""
print("========== 60. Image Kernels and Filters ==========")
"""
- How do filters work?
- They are image kernals/filters 3 x 3 matrix (they can be 4 x 4, 5 x 5 etc.)
- What does "apply" a filter to an image mean?
- The nine elements of a filter are called "the filter weights"
- You multiply a filter weight by the pixel value it is over,
    sum the 9 results and that is the new pixel value at the center of the 3 x 3 matix.
- Note, you miss out the border pixels of the image, because the center of
    the kernal "can't reach them". Get around this by padding the image with 0s.
- The "stride distance", is the distance the matix/kernal moves on each iteration.

- Good iteractive example at https://setosa.io/ev/image-kernels/
- The process of dragging a convolution kernal across an image is called "a convolution".

- The big idea is that we get the network to choose the kernal weights 
    in order to do feature extraction - sounds like science fiction.

- So, what is the architecture of a CNN that allows the network
    to come up with the best weights for a filter in the
    "convolutional layer"?
"""
print("========== 61. Convolutional Layers ==========")
"""
25 September 2023
- Recall, we got quite good results with the MNIST dataset and an ANN
- However, there are some issues using ANNs with images
    - Huge amount of parameters, even for a 28 x 28 bit image 100,000 weights
    - With an ANN, we had to "flatten out" the data, so we loose all 2D infomation.
        What this means is, the input was 784 neurons in a straight line so infomation
        that might be constituted by the adacency of one pixel, to its eight neigbours is lost.
    - An ANN only performs well, where images are very similar and centered in the image.

- A CNN can use convolutional layer to alleviate the above issues.
- A convolutional layer is created, when we apply multiple image filters to the input image.
- The layer will then be trained to figure out the best filter weight values.

- A CNN also helps reduce parameters/weights by focusing on "local connectivity".
    In a CNN's convolutional layer, not all neurons are fully connected.
    Instead neurons are only connected to a subset of "local neurons" in the next layer.
    And these "local neurons" in the next layer, end up becoming the filters.

------------------------------------------------------------------------------------
- So, Let's use a simplified 1D example to understand this "local connectivity" and 
    its relation to filters. We'll extend this to 2D greyscale and 
    3D tensors for colour input in time.

- Fully connected ANNs have lots of parameters, so that's a problem for any real image.
- CNNs focuses on "local connections" - the neurons in one layer, 
    only connect to "local" neurons in the next layer. Diagram needed, but I guess
    we can imagine, a central neuron, and its eight nearest neighbours only connecting
    to the neuron directly adjacent to the central neuron in the next layer.
- So in the situation depicted above, we have 9 neurons in one layer, 
    connected to the one central neuron in the next - i.e. a 3 x 3 matrix, 
    outputing - via weights - to a single neuron.
    That does sound awfully like a filter, dosn't it?
    And the idea is, that the aforementioned weights, get trained to do feature detection.

- Apparently, as you are training this CNN, 
    you get to decide how many filters you apply to the image. How?
- There's a lot of hand waving going on about "some filter" and "some component" 
    and the CNN "figuring out what's the best"
- There is an idea, slipped in here, that each subsequent layer - moving from left to right -, 
    detects "higher" and "higher" level features. So line and edge features in one layer,
    create the foundations for "higher" eyebrow and nose features in the next.
  
=== 2D greyscale ===
- Well, they use a 2 x 2 matrix, with a stride of 2, but the above is correct.
- He explains how you can have more than one filter on an input image.
- Essentialy - image the 3 x 3 filter above, but one set of 9 connections/weights going
    to neuronA in the next layer and another set of 9 connections/weights going to neuronB 
    in the next layer. These constitute two seperate filters.
- In the example above 9 neurons connected to a single neuron in the next layer.
    This is not always the case. Appart from the fact that we might start off with 9, 16, 25...
    neurons in the first layer, these might connect down to 1, 4, 9 etc., in the next layer.
- It is not unusual to see 10s or 100s of filters, depending on the complexity
    of what we are trying to classify.
- So this "stack" of filters is what constitutes our convolutional layer.

=== 3D Tensor - colour images ===
- Colour images can be throught at 3D Tensors, consisting of RGB colour channels.
- So the shape of an image might be (1280, 720, 3)
- So a single images is represented by a 3D Tensor
- How do we perform a covolution on a 3D Tensor (colour image)
- We end up with a 3D filter, with weights/values for each colour channel.
- So if we use the 3 x 3 -> 1 neuron example, each neuron now becomes three
    So  3 x 3 x 3 -> 1 x 3
but it might be
        4 x 4 x 3 -> 2 x 2 x 3
So, everything 3D

------------------------------------------------------------------------------
- Often convolutional layers are fed into other convolutional layers
- This allows (or so the story goes) the network to discover patterns within patterns.

- We will now go onto "Pooling layers" - also known as downsampling layers
"""
print("========== 62. Pooling/Downsampeling Layers ==========")
"""
- Even using local connectivity, when dealing with 10s or 100s of
    filters we still have a very large number of perameters/weights
    So we use "Pooling Layers"
- Pooling layers accept convolutional layers as input
- Normaly our convolutional layer will have many filters, a filter per colour channel etc.
- There are different types of Pooling layer, 
    which use different sorts of downsampeling techniques
- Imagine a single 4 x 4 filter in a convolutional layer
    We are going to take a 2 x 2 window, and move it across the 4 x 4 matix with a stride of 2.
    At each sample we take the greatest value in 
    the window and that is our output to the Pooling layer

        4 x 4 filter            2 x 2 Pooling layer downsample
        1   3   6   8
        4   2   8   7           4       9
        8   7   3   2   ------> 9       3
        7   9   1   1

- This keeps the basic information <<<<< this is a big deal, does it?

- With "Average Pooling" you do the exact same process exept you 
    average the values un the window

        4 x 4 filter            2 x 2 Pooling layer downsample
        1   3   6   8
        4   2   8   7           2.5     7.5
        8   7   3   2   ------> 7.75    1.75
        7   9   1   1
--------------------------------------------------------------------------------
=== Dropout Layer ===
- We have already come across Dropout layers - during training, 
    a certain percentage of random neurons are dropped (along with their weights) 
    in each batch/epoch.
- Dropout can be thought of as a form of "Regularization" to prevent over fitting.
- This prevents units from "co-adapting" too much - this is when one neuron, is too
    dependent on another.
--------------------------------------------------------------------------------
=== Famouse CNN architectures ===

LeNet-5 by Yann LeCun
AlexNet by Alex Krizhevsky et al.
GoogLeNet by Szegedy at Google Research
ResNet by Kaiming He et al.

- Then there are some diagrams, describing the layer structure of AlexNet

- Keep in mind, CNN can have all sorts of architectures

e.g.
    i/p -> Conv -> Conv -> Conv -> Pool
or
    i/p -> Conv -> Pool -> Conv -> Pool

It's kind of an art - try things out and use the metrics

- But in the end you are going to have to have a Fully Connected layer (FC), 
    that connects the results to an output

    - About as simple as it gets
    i/p -> Conv -> Pool ->FC -> o/p 

o/p - same number of neurons as classes, in the case of a classification task.

"""






























































































