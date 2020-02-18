import csv 
import numpy as np
import matplotlib.pyplot as plt
import math

# 建立存18个影响因素的矩阵
data = []
for i in range(18):
    data.append([])

# 读入训练数据
with open(r'E:\VScode\Machine_Learning\HW1\train.csv', 'r', encoding='big5') as text:
    row = csv.reader(text, delimiter=',')
    n_row = 0
    for r in row:
        #print(r)
        if n_row != 0:    # 原始数据从第二行开始
            for i in range(3,27):    # 第3-27列为24小时数据
                if r[i] != 'NR':    # RAINFALL项值均为NR,视为0
                    data[(n_row-1) % 18].append(float(r[i]))
                else:
                    data[(n_row-1) % 18].append(float(0))
        n_row = n_row + 1
text.close()
#print(np.array(data).shape)  #(18, 5760)


# 切分数据
x = []
y = []
for i in range(12):   # 12个月
    for j in range(471):   # 每9个小时一组,一个月20天,即20*24-9
        x.append([])   # x每列存9个小时内18种变量数据
        for t in range(18):
            for s in range(9):
                x[471 * i + j].append(data[t][471 * i + j + s])
        y.append(data[9][480 * i + j + 9])   # 每组第十个小时的数据作结果
x = np.array(x)
y = np.array(y)
#print(x.shape)   #(5652, 162)
#print(y.shape)   #(5652,)


#x = np.concatenate((x, x ** 2), axis=1)
# 加入bias
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)  # 合并两个矩阵,将第一个放在第一列
#print(x.shape)   #(5652, 163)


#训练数据
w = np.zeros(len(x[0]))
#print(w.shape)   #(163,)
l_rate = 10   # 学习率
iteration = 10000   # 迭代次数

x_t = x.transpose()   # 矩阵转置
s_gra = np.zeros(len(x[0]))
#print(len(x))

for i in range(iteration):
    hypo = np.dot(x, w)  # 矩陈点乘
    loss = hypo - y  # 与真实数据差值
    cost = np.sum(loss ** 2) / len(x)   # loss函数
    cost_a = math.sqrt(cost)
    #print(gra[0])
    # 使用Adagrad
    gra = 2*np.dot(x_t, loss)   # Gradient Descent
    s_gra += gra ** 2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra / (ada + 0.0005)
    #print('iteration : %d | Cost: %f' % (i+1, cost_a))

"""
for i in range(iteration):
    hypo = np.dot(x, w)  # 矩陈点乘
    loss = hypo - y  # 与真实数据差值
    cost = np.sum(abs(loss)) / len(x)   # loss函数
    cost_a = math.sqrt(cost)
    gra = 2*np.dot(x_t, loss)   # Gradient Descent 
    w = w - l_rate * gra
    #print('iteration : %d | Cost: %f' % (i+1, cost_a))
"""

#保存Model   
np.save(r'E:\VScode\Machine_Learning\HW1\model.npy', w)
w = np.load(r'E:\VScode\Machine_Learning\HW1\model.npy')
#print(w.shape)   #(163,)


#读取test data
test_x = []
test = open(r'E:\VScode\Machine_Learning\HW1\test.csv', 'r')
row = csv.reader(test, delimiter=',')

n_row = 0
for r in row:
    if n_row % 18 == 0:   # 每18行为一组数据
        test_x.append([])
        for i in range(2, 11):
            test_x[n_row // 18].append(float(r[i]))
    else:
        for i in range(2, 11):
            if r[i] != 'NR':
                test_x[n_row // 18].append(float(r[i]))
            else:
                test_x[n_row // 18].append(0)
    n_row = n_row + 1
test.close()
test_x = np.array(test_x)
#print(test_x.shape)   #(240, 162)

# 加入bias
test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis=1)  # 合并两个矩阵,将第一个放在第一列
#print(test_x.shape)   #(240, 163)


# 生成预测数据
predict = []
y_p = []
for i in range(len(test_x)):
    predict.append(['id_' + str(i)])
    a = np.dot(w, test_x[i])   # 计算预测数值
    predict[i].append(a)
    y_p.append(a)

filename = r'E:\VScode\Machine_Learning\HW1\predict.csv'
text = open(filename, 'w+')
s = csv.writer(text, delimiter = ',', lineterminator = '\n')
s.writerow(['id', 'value']) 
for i in range(len(predict)):
    s.writerow(predict[i])
text.close()


# 读取答案
row_n = 0
y = []
with open(r'E:\VScode\Machine_Learning\HW1\ans.csv', 'r', encoding='big5') as text:
    row = csv.reader(text, delimiter=',')
    for i in row:
        if row_n != 0:
            y.append(i[1])
        row_n = row_n + 1
y = np.array(y).astype(int)   # 转int类型
y = y.tolist()   # 转list类型


# 绘图
plt.figure(figsize=(13, 7))
plt.plot(np.arange(0, 240, 1), y_p, 'r', label = 'predict PM2.5')
plt.plot(np.arange(0, 240, 1), y, 'b', label = 'ans PM2.5')
plt.legend()
plt.show()
