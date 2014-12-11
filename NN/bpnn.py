# Based on bpnn.py written by Neil Schemenauer <nas@arctrix.com>

import random
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def initialize_matrix(m, n, fill = 999):
    # Initialize a m * n matrix with random values
    matrix = []
    for i in range(m):
        row = []
        for j in range(n):
            if fill != 999:
                row.append(fill)
            else:
                row.append(random.uniform(-0.2, 0.2))
        matrix.append(row)
    logger.debug("Initialized Weight Matrix")
    logger.debug(str(matrix))
    return matrix

def sigmoid(x):
    return math.tanh(x)

def dsigmoid(y):
    return 1.0 - y ** 2

class NN:
    def __init__(self, ni, nh, no):
        # Structure of Neural Network
        self.ni = ni + 1
        self.nh = nh
        self.no = no

        # Activation of each layer
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # Initialize all Weights
        self.w_ih = initialize_matrix(self.ni, self.nh)
        self.w_ho = initialize_matrix(self.nh, self.no)

        # Last change in Weight for Momentum
        self.last_change_h = initialize_matrix(self.ni, self.nh, fill = 0.0)
        self.last_change_o = initialize_matrix(self.nh, self.no, fill = 0.0)

    def feed_forward(self, lst_input):
        if len(lst_input) != self.ni - 1:
            raise "ERROR, input size inconsistent!!"
        
        # Input
        self.ai[:-1] = lst_input
        
        # Input -> Hidden
        for j in range(self.nh):
            z = 0.0
            for i in range(self.ni):
                z += self.ai[i] * self.w_ih[i][j]
            self.ah[j] = sigmoid(z)

        # Hidden -> Output
        for k in range(self.no):
            z = 0.0
            for j in range(self.nh):
                z += self.ah[j] * self.w_ho[j][k]
            self.ao[k] = sigmoid(z)

        return self.ao[:]

    def back_propogation(self, lst_target, N, M):
        if len(lst_target) != self.no:
            raise "ERROR, output size inconsistent!!"
        
        # Calculate Error Term for Output
        delta_output = []
        for k in range(self.no):
            error = self.ao[k] - lst_target[k]
            delta_output.append(error * dsigmoid(self.ao[k]))
        
        # Calculate Error Term for Hidden
        delta_hidden = []
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += self.w_ho[j][k] * delta_output[k]
            delta_hidden.append(error * dsigmoid(self.ah[j]))

        # Update Hidden -> Output Weights
        for j in range(self.nh):
            for k in range(self.no):
                change = delta_output[k] * self.ah[j]
                self.w_ho[j][k] = self.w_ho[j][k] - N * change - M * self.last_change_o[j][k]
                self.last_change_o[j][k] = change

        # Update Input -> Hidden Weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = delta_hidden[j] * self.ai[i]
                self.w_ih[i][j] = self.w_ih[i][j] - N * change - M * self.last_change_h[i][j]
                self.last_change_h[i][j] = change

        # Calculate Error
        error = 0.0
        for k in range(len(lst_target)):
            error += 0.5 * (lst_target[k] - self.ao[k]) ** 2
        return error

    def train(self, lst_data, iterations = 1000, N = 0.5, M = 0.1):
        for i in range(iterations):
            error = 0.0
            for data in lst_data:
                lst_input = data[0]
                lst_target = data[1]
                self.feed_forward(lst_input)
                error += self.back_propogation(lst_target, N, M)
            if i % 100 == 0:
                logger.info("Error" + str (error))

    def test(self, lst_data):
        for data in lst_data:
            logger.info(str(self.feed_forward(data[0])))

    def weights(self):
        logger.info('input -> hidden')
        for i in range(self.ni):
            logger.info(str(self.w_ih[i]))

        logger.info('hidden -> output')
        for j in range(self.nh):
            logger.info(str(self.w_ho[j]))

def demo():
    lst_data = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
    ]

    nn =  NN(2, 2, 1)
    nn.train(lst_data)
    nn.weights()
    nn.test(lst_data)

if __name__ == '__main__':
    demo()
