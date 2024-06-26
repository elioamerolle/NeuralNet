# Neural Net from scratch

This project aims to illustrate how neural networks function using the visual analogue (even at the cost of worse performance). Additionally it does not use any Machine Learning specific libraries like pytorch or TensorFlow (does use sklearn but only for the dataset). The project was originally inspired by 3Blue1Brown's youtube series on [neural networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&ab_channel=3Blue1Brown).
 

## How To Run

To run the project, it's required to use python version `3.0` or higher. Open the terminal or an external code editor and run the command `python main.py`. This project consists of the following dependencies with their respective versions. From the list below install the following packages using `pip` if you don't already have them installed.

* matplotlib (3.0.3): Tool for graphing cost function over time and showing image of hand drawn number.
* numpy (1.14.2): Used to optimize dot product.
* sklearn (0.0): Used to get MNIST data set.
* tqdm (4.64.1): Imports loading bar to track progress
* termcolor (1.1.0): Gives color to some terminal text

*Note: To stop the running the program, run `Ctrl+C`.*

## Documentation of Code

The following code contains contains 4 classes that allow the program to run.

* `Perceptron`: The smallest unit inside the neural network which handles the different functions a perceptron needs. This includes storing how many different sources does it take input from how many sources does it output to, and can store things like pre-sigmoid activation and can activate using pre sigmoid activation and applying the sigmoid function. 

* `NeuralNetwork`: This class is of type list and is the actual neural network thus storing all perceptron.

* `BackPropagation`: Deals with the finding the actual gradient, also  probably the most math heavy/interesting part.

* `Funcs`: Stores a wide range of useful functions that deal with dot product optimization, random initialization for perceptron weights, exponential decay of learning rate, user interface, etc.

## How To Use

Inside of main.py you should be able to see this section

```python
#                       =====================    Personal Preferences     =====================

see_success_as_they_train = False

#                       =====================    Personal Preferences     =====================


#                       =====================    Hyperparameter     =====================

# Size of each layer (input and output are determined)
layerDimensions  = [64, 29, 10]

# Number of images we will use
nImages = 1600

miniSize = 20

# Number of Epochs
nEpochs = 4

# learning rate
learnR = 0.1

# decay rate (exponential loss)
decay_rate = 2

#If we repeat
tryMulti = 1

#                       =====================     Hyperparameter    =====================
            

```

This allows you to tune how you want the neural net to train and how many neural nets you want (system will automatically choose the best). Additionally you can choose to see extra print statements which gives more information as the neural nets train. If you elect to only run one neural net the loss function over time is also given.

*Note: Its useful to look that the git log to see which commits had what success rate (success rate given in commit message), from the commit you can retrieve the hyperparameter*


### Basic Setup

As previously stated, once all the packages are installed, you can directly run it through the following command.

```sh
python main.py
```

The program will ask you if you want to train a new neural net (using the hyperparameter in main.py) or if you would like to test an old neural network stored in a pkl file with the following prompt (enter s for saved and t for train and return).

```sh
Would you like to test saved version or or train a new one s/t
```

If you delete the pkl file the program will automatically train and the prompt will not appear. Once finished training or if you are checking out an old version you will be asked to test your new neural network with the following prompt. If you are training only one neural net you will see the loss data at this stage and will be allowed to just hit enter to continue to the prompt below. 

```sh
indexes bigger than 1600 are unseen, stay below 1750
WHAT INDEX FROM MNIST WOULD YOU LIKE TO SAMPLE ON OUR NET
```

From here you can enter an integer between 0 and 1750 inclusive to test the neural network. Also the number 1600 is simply because of the hyperparameter nImages, we chose to train on that many images. At this point the program will produce an image of the image being tested on and give the confidences in the command line. 

