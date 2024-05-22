import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = nn.as_scalar(self.run(x))
        if score >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
                correctPredictions = True
                for x, y in dataset.iterate_once(1):
                    prediction = self.get_prediction(x)
                    if nn.as_scalar(y) != prediction:
                        self.w.update(x, nn.as_scalar(y))
                        correctPredictions = False
                if correctPredictions:
                    break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 200
        self.num_neurons_hidden_layer = 512
        self.w_1 = nn.Parameter(1, self.num_neurons_hidden_layer)
        self.b_1 = nn.Parameter(1, self.num_neurons_hidden_layer)
        self.output_w = nn.Parameter(self.num_neurons_hidden_layer, 1)
        self.output_b = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        transformation1 = nn.Linear(x, self.w_1)
        transformationBias1 = nn.AddBias(transformation1, self.b_1)
        layer = nn.ReLU(transformationBias1)
        transformation2 = nn.Linear(layer, self.output_w)
        return nn.AddBias(transformation2, self.output_b)
    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        node = self.run(x)
        return nn.SquareLoss(node, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                parameters = [self.w_1, self.output_w, self.b_1, self.output_b]
                gradients = nn.gradients(loss, parameters)
                self.w_1.update(gradients[0], -0.05)
                self.output_w.update(gradients[1], -0.05)
                self.b_1.update(gradients[2], -0.05)
                self.output_b.update(gradients[3], -0.05)

            loss = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            if nn.as_scalar(loss) < 0.02:
                return

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 100
        self.hidden_layer_size = 200
        self.num_labels = 10

        self.w_1 = nn.Parameter(784, self.hidden_layer_size)
        self.b_1 = nn.Parameter(1, self.hidden_layer_size)

        self.output_wt = nn.Parameter(self.hidden_layer_size, self.num_labels)
        self.output_bias = nn.Parameter(1, self.num_labels)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        transformation1 = nn.Linear(x, self.w_1)
        transformationBias1 = nn.AddBias(transformation1, self.b_1)
        layer = nn.ReLU(transformationBias1)

        transformation2 = nn.Linear(layer, self.output_wt)
        return nn.AddBias(transformation2, self.output_bias)


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        node = self.run(x)
        return nn.SoftmaxLoss(node, y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.w_1, self.b_1, self.output_wt, self.output_bias])

                self.w_1.update(gradients[0], -0.2)
                self.b_1.update(gradients[1], -0.2)
                self.output_wt.update(gradients[2], -0.2)
                self.output_bias.update(gradients[3], -0.2)

            if dataset.get_validation_accuracy() >= 0.975:
                return

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 100
        self.hidden_size = 200

        self.w = nn.Parameter(self.num_chars, self.hidden_size)
        self.w_hidden = nn.Parameter(self.hidden_size, self.hidden_size)
        self.output_w = nn.Parameter(self.hidden_size, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        z = nn.ReLU(nn.Linear(xs[0], self.w))
        for _, x in enumerate(xs[1:]):
            z = nn.Add(nn.ReLU(nn.Linear(x, self.w)), nn.ReLU(nn.Linear(z, self.w_hidden)))

        return nn.Linear(z, self.output_w)
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        node = self.run(xs)
        return nn.SoftmaxLoss(node, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.w, self.w_hidden, self.output_w])

                self.w.update(gradients[0], -0.05)
                self.w_hidden.update(gradients[1], -0.05)
                self.output_w.update(gradients[2], -0.05)

            if dataset.get_validation_accuracy() >= 0.87: # 87% so that it's more accurate for the 81% requirement
                return
