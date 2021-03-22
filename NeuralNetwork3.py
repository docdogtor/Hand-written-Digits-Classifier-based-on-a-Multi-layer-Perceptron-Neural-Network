import csv
import numpy as np

def read_files():
    with open("train_image.csv", 'r') as train_img_f:
        train_img_data = list(csv.reader(train_img_f, delimiter=","))
    with open("train_label.csv", 'r') as train_lbl_f:
        train_lbl_data = list(csv.reader(train_lbl_f, delimiter=","))
    with open("test_image.csv", 'r') as test_img_f:
        test_img_data = list(csv.reader(test_img_f, delimiter=","))
    with open("test_label.csv", 'r') as test_lbl_f:
        test_lbl_data = list(csv.reader(test_lbl_f, delimiter=","))
    train_x = np.array(train_img_data, dtype=np.int)/255
    train_y = np.array(train_lbl_data, dtype=np.int)
    test_x = np.array(test_img_data, dtype=np.int)/255
    test_y = np.array(test_lbl_data, dtype=np.int)
    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = read_files()

fig, pixel = train_x.shape
test_fig, test_pixel = test_x.shape

def ini_weight(n):
    w = np.random.randn(n) / np.sqrt(n)
    return w

def sigmoid(s):
    """
    sigmoid activation function.
    inputs: s
    outputs: sigmoid(s)
    """
    sf = 1. / (1. + np.exp(-s))
    return sf

def sigmoid_dev(s):
    dev = s * (1 - s)
    return dev

def softmax(s):
    """
    Compute softmax values for each sets of scores in x.
    """
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    sf = exps / np.sum(exps, axis=1, keepdims=True)
    return sf

class update_model:
    def __init__(self, pixel, h1_unit, h2_unit, h3_unit, lr, size):
        self.lr = lr
        self.size = size
        self.w1 = np.zeros(shape=(pixel, h1_unit))
        self.dcdw1 = np.zeros(shape=(pixel, h1_unit))
        self.b1 = np.zeros(shape=(1, h1_unit))
        self.dcdb1 = np.zeros(shape=(1, h1_unit))
        self.s1 = np.zeros(shape=(1, h1_unit))
        self.o1 = np.zeros(shape=(1, h1_unit))

        self.w2 = np.zeros(shape=(h1_unit, h2_unit))
        self.dcdw2 = np.zeros(shape=(h1_unit, h2_unit))
        self.b2 = np.zeros(shape=(1, h2_unit))
        self.dcdb2 = np.zeros(shape=(1, h2_unit))
        self.s2 = np.zeros(shape=(1, h2_unit))
        self.o2 = np.zeros(shape=(1, h2_unit))

        self.w3 = np.zeros(shape=(h2_unit, h3_unit))
        self.dcdw3 = np.zeros(shape=(h2_unit, h3_unit))
        self.b3 = np.zeros(shape=(1, h3_unit))
        self.dcdb3 = np.zeros(shape=(1, h3_unit))
        self.s3 = np.zeros(shape=(1, h3_unit))
        self.o3 = np.zeros(shape=(1, h3_unit))

        self.x = np.zeros(shape=(1, pixel))
        self.y_real = np.zeros(shape=(1, h3_unit))

    def initial(self, w1, w2, w3, b1, b2, b3):
        self.w1 = w1.T
        self.b1 = b1

        self.w2 = w2.T
        self.b2 = b2

        self.w3 = w3.T
        self.b3 = b3

    def cross_entropy(self, pred, real):
        res = pred - real
        return res

    # def error(self, real, pred):
    #     pred = pred
    #     real = real
    #     n_samples = real.shape[0]
    #     logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    #     loss = np.sum(logp) / n_samples
    #     print(loss)

    def feedforward(self):
        self.s1 = np.dot(self.x, self.w1) + self.b1
        self.s1 = self.s1.reshape((1, h1_unit))
        self.o1 = sigmoid(self.s1)

        self.s2 = self.o1.dot(self.w2) + self.b2
        self.s2 = self.s2.reshape((1, h2_unit))
        self.o2 = sigmoid(self.s2)

        self.s3 = self.o2.dot(self.w3) + self.b3
        self.s3 = self.s3.reshape((1, h3_unit))
        self.o3 = softmax(self.s3)

    def backprop(self, x, y):
        self.x = x.reshape((1, pixel))
        self.y = y.reshape((1, h3_unit))

        self.feedforward()
        # self.error(self.y, self.o3)

        a3_delta = self. cross_entropy(self.o3, self.y)
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_dev(self.o2)
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_dev(self.o1)

        self.dcdw3 += np.dot(self.o2.T, a3_delta)
        self.dcdb3 += np.sum(a3_delta, axis=0, keepdims=True)

        self.dcdw2 += np.dot(self.o1.T, a2_delta)
        self.dcdb2 += np.sum(a2_delta, axis=0, keepdims=True)

        self.dcdw1 += np.dot(self.x.T, a1_delta)
        self.dcdb1 += np.sum(a1_delta, axis=0, keepdims=True)

    def predict(self, x):
        self.x = x.reshape((1, pixel))
        self.feedforward()
        y_pred = np.argmax(self.o3)
        return y_pred

    def update(self):
        alfa = self.lr / self.size
        self.w1 -= self.dcdw1 * alfa
        self.b1 -= self.dcdb1 * alfa

        self.w2 -= self.dcdw2 * alfa
        self.b2 -= self.dcdb2 * alfa

        self.w3 -= self.dcdw3 * alfa
        self.b3 -= self.dcdb3 * alfa

    def clear(self):
        self.dcdw1 = np.zeros(shape=(pixel, h1_unit))
        self.dcdb1 = np.zeros(shape=(1, h1_unit))

        self.dcdw2 = np.zeros(shape=(h1_unit, h2_unit))
        self.dcdb2 = np.zeros(shape=(1, h2_unit))

        self.dcdw3 = np.zeros(shape=(h2_unit, h3_unit))
        self.dcdb3 = np.zeros(shape=(1, h3_unit))

# number of epoch and batch
num_epoch = 12
batch_size = 20
num_batch = fig//batch_size

# number of units in each hidden layer
h1_unit = 256
h2_unit = 128
h3_unit = 10
lr = 0.5

Model = update_model(pixel, h1_unit, h2_unit, h3_unit, lr, batch_size)

# initial weights
def def_weight():
    w1 = np.zeros(shape=(h1_unit, pixel))
    b1 = np.zeros(shape=(1, h1_unit))
    w2 = np.zeros(shape=(h2_unit, h1_unit))
    b2 = np.zeros(shape=(1, h2_unit))
    w3 = np.zeros(shape=(h3_unit, h2_unit))
    b3 = np.zeros(shape=(1, h3_unit))

    for i in range(h1_unit):
        w1[i] = ini_weight(pixel)
    for j in range(h2_unit):
        w2[j] = ini_weight(h1_unit)
    for j in range(h3_unit):
        w3[j] = ini_weight(h2_unit)
    return w1, w2, w3, b1, b2, b3

w1, w2, w3, b1, b2, b3 = def_weight()

Model.initial(w1, w2, w3, b1, b2, b3)

for i in range(num_epoch):
    shuffle_index = np.random.permutation(fig)
    train_x, train_y = train_x[shuffle_index, :], train_y[shuffle_index, :]
    for j in range(num_batch):
        '''get mini-batch'''
        begin = j*batch_size
        # end = (j+1)*batch_size
        x = np.zeros(shape=(batch_size, pixel))
        y_real_ini = np.zeros(shape=(batch_size, h3_unit))
        for k in range(batch_size):
            x[k] = train_x[k+begin]
            y_k = train_y[k+begin]
            y_real = y_real_ini
            y_real[k][y_k] = 1
            Model.backprop(x[k], y_real[k])
        Model.update()
        Model.clear()

with open('test_predictions.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    right = 0
    total = 0
    for it in range(test_fig):
        x = test_x[it]
        y_pred = Model.predict(x)
        employee_writer.writerow([y_pred])


