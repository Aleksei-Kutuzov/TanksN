import threading

import numpy

import json
from scipy import special


class neuralNetwork:
    # инициализировать нейронную сеть
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate=0.3):
    # задать количество узлов во входном, скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.loss = 0
    # Матрицы весовых коэффициентов связей, wih и who.
    # Весовые коэффициенты связей между узлом i и узлом j
    # следующего слоя обозначены как w_i_j:
    # wll w21
    # wl2 w22 и т.д.
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5),
                                       (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5),
                                       (self.onodes, self.hnodes))
        # print(self.wih)
        # print()
        # # коэффициент обучения
        self.lr = learningrate

        # использование сигмоиды в качестве функции активации
        self.activation_function = lambda x: special.expit(x)
        pass

        # тренировка нейронной сети
    def train(self, inputs_list, targets_list):
        # преобразование списка входных значений
        # в двухмерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T


        # print(inputs)
        # print(
        #     'hfp ldf nhb'
        # )
        # print(targets)
        # print('vivod close')
        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        print(hidden_inputs.shape)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        print(hidden_outputs.shape)
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        print(final_inputs.shape)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs.shape)
        # ошибки выходного слоя =
        # (целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        # ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        # output_errors = numpy.resize(32, 4)
        print(output_errors.shape, self.who.T.shape)
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # обновить весовые коэффициенты для связей между
        # скрытым и выходным слоями
        self.who = self.who + self.lr * numpy.dot((output_errors *
                                         final_outputs * (1.0 - final_outputs)),
                numpy.transpose(hidden_outputs))
        # обновить весовые коэффициенты для связей между
        # входным и скрытым слоями
        self.wih = self.wih + self.lr * numpy.dot((hidden_errors *
                                      hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))


            # опрос нейронной сети
    def query(self, *inputs_list):

        # преобразовать список входных значений
        # в двухмерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def save_model(self, filename="model"):
        json_dict = dict()
        print(1)
        print(self.wih)
        print(self.who)
        json_dict["wih"] = self.wih.tolist()
        json_dict["who"] = self.who.tolist()
        with open(f'models/{filename}.json', 'w') as f:
            json_dict = json.dumps(json_dict, indent=0, separators=(",", ":"))
            f.write(json_dict)
        print(json_dict)

        # f = open(f'models/{filename}.json', 'w')
        # print(2)
        # f.write(json_dict)
        # print(3)
        # f.close()
        # f = open('model.txt', 'w')
        # f.write(str(self.wih) + '\n\n' + str(self.who))
        # f.close()

    def download_model(self, filename="model.json"):
        f = open(f"models/{filename}", "r")
        try:
            json_dict = json.load(f)
        except ValueError as e:
            json_dict = {"wih": "", "who": ""}
        if len(json_dict["wih"]) == self.inodes and len(json_dict["who"]) == self.hnodes:
            self.wih = numpy.array(json_dict["wih"])
            self.who = numpy.array(json_dict["who"])
        else:
            self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5),
                                           (self.hnodes, self.inodes))
            self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5),
                                           (self.onodes, self.hnodes))


# # количество входных, скрытых и выходных узлов
# input_nodes = 784
# hidden_nodes = 100
# output_nodes = 1 0
# # коэффициент обучения равен 0,3
# learning_rate =0.3
# # создать экземпляр нейронной сети
# n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
# # загрузить в список тестовый набор данных CSV-файла набора MNIST
# training_data_file = open(,,mnist_dataset/mnist_train_100.csv,,, 'г1)
# training_data_list = training_data_file.readlines ()
# training_data_file.close()
# # тренировка нейронной сети
# # перебрать все записи в тренировочном наборе данных
# for record in training_data_list:
# # получить список значений, используя символы запятой (1,1)
# # в качестве разделителей
# all_values = record.split(',')
# # масштабировать и сместить входные значения
# inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
# # создать целевые выходные значения (все равны 0,01, за исключением
# # желаемого маркерного значения, равного 0,99)
# targets = numpy.zeros(output_nodes) + 0.01
# # all_values[0] - целевое маркерное значение для данной записи
# targets[int(all_values[0])] =0.99
# n.train(inputs, targets)
# pass