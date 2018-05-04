import numpy as np

class HandleData(object):

    def __init__(self,batch_size,num_examples,num_ex_per_angle,num_angle):
        self.num_examples=num_examples
        self.num_ex_per_angle=num_ex_per_angle
        self.num_angle = num_angle
        self.current_point = 0
        self.data_set = np.zeros((self.num_examples,4),dtype=np.float32)
        self.label_set = np.zeros((self.num_examples,self.num_angle),dtype=np.float32)

    def onehot_encode(self,number):
        encoded_no = np.zeros(self.num_angle, dtype=np.float32)
        if number < self.num_angle:
            encoded_no[number] = 1
        return encoded_no

    def next_batch(self,batch_size):
        # print("start : " + str(self.current_point))
        if self.current_point == self.num_examples:
            self.current_point = 0
        start = self.current_point
        end = start + batch_size
        return_data = self.data_set[start:end]
        return_label = self.label_set[start:end]
        self.current_point=end
        # print(return_data)
        # print("end : " + str(self.current_point))
        return return_data,return_label

    def get_synthatic_data(self):
        _1 = np.random.randint(low=1, high=10, size=self.num_ex_per_angle)
        _2 = np.random.randint(low=15, high=25, size=self.num_ex_per_angle)
        _3 = np.random.randint(low=30, high=40, size=self.num_ex_per_angle)

        l_0 = np.array([_3,_2,_1,_2], np.float32)
        l_1 = np.array([_2,_2,_1,_1], np.float32)
        l_2 = np.array([_2,_3,_2,_1], np.float32)
        l_3 = np.array([_1,_2,_2,_1], np.float32)
        l_4 = np.array([_1,_2,_3,_2], np.float32)
        l_5 = np.array([_1,_1,_2,_2], np.float32)
        l_6 = np.array([_2,_1,_2,_3], np.float32)
        l_7 = np.array([_2,_1,_1,_2], np.float32)

        data_matrix = np.array([l_0, l_1, l_2, l_3, l_4, l_5, l_6, l_7], np.float32)
        tmp = np.zeros((self.num_angle, self.num_ex_per_angle, 4), dtype=np.float32)
        for j in range(0,self.num_angle):
            for i in range(0, self.num_ex_per_angle):
                tmp[j][i][0] = data_matrix[j][0][i]
                tmp[j][i][1] = data_matrix[j][1][i]
                tmp[j][i][2] = data_matrix[j][2][i]
                tmp[j][i][3] = data_matrix[j][3][i]

        # l_sel = np.random.randint(low=0, high=self.num_angle, size=1)[0]
        # index_sel = np.random.randint(low=0, high=self.num_ex_per_angle, size=1)[0]

        # data_set = np.zeros((self.num_examples,4),dtype=np.float32)
        # label_set = np.zeros((self.num_examples,self.num_angle),dtype=np.float32)

        for i in range(0,self.num_angle):
            for j in range(0, self.num_ex_per_angle):
                "add one hot"
                "add data"
                self.label_set[i * self.num_ex_per_angle + j] = self.onehot_encode(i)
                self.data_set[i * self.num_ex_per_angle + j] = tmp[i][j]
        return self.data_set ,self.label_set
# print(onehot_encode(0).shape)
# print(label_set)
# print(data_set)
# print(data_set.shape)
# print(label_set.shape)

# print(getSynthaticData()['data'])

# data = HandleData(batch_size=10,num_examples=800,num_ex_per_angle=100,num_angle=8)
# a,b = data.get_synthatic_data()
# _a,_b = data.next_batch(50)
# print(_a)
# print(data.current_point)
# _a,_b = data.next_batch(51)
# print(_a)
# print(data.current_point)