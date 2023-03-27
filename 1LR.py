import random

import numpy as np
from PIL import Image
import math
import re


class Point:
    def __init__(self, x, y, z=0.0):
        self._x = x
        self._y = y
        self._z = z
        self.color3 = 255

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @x.setter
    def x(self, x):
        self._x = x

    @y.setter
    def y(self, y):
        self._y = y

    @z.setter
    def z(self, z):
        self._z = z

    def color(self):
        return self.color3

    def __str__(self):
        return f"x= {self.x} y = {self.y} z={self.z}"

    def swap(self):
        self._x, self._y = self._y, self._x

    def to_vec(self):
        return np.array([self._x, self._y, self._z]).reshape(3, 1)

    def update_point(self, new_point):
        self._x = float(new_point[0])
        self._y = float(new_point[1])
        # self._z = 1.0
        # self._z = float(new_point[2])


class Image_class:
    def __init__(self, H=1000, W=1000, color=(0, 0, 0)):
        self.H = H
        self.W = W
        self.matrix = np.ones((H, W, 3), dtype=np.uint8)
        self.matrix[:] = np.array(color)

    def all_update_matrix(self, matrix):
        self.matrix = matrix

    def show(self, str):
        image = Image.fromarray(self.matrix, 'RGB')
        image.show()
        image.save(f"{str}.jpg")

    def update_matrix(self, Points):
        for point in Points:
            self.matrix[int(point.x)][int(point.y)] = 255


def task_1_1(H=1000, W=1000):
    my_image = Image_class(H, W)
    my_image.show("task_1_1")


def task_1_2(H: int, W: int):
    my_image = Image_class(H, W, 255)
    my_image.show("task_1_2")


def task_1_3(H: int, W: int):
    my_image = Image_class(H, W, (255, 0, 0))
    my_image.show("task_1_3")


def task_1_4(H: int, W: int):
    my_image = Image_class(H, W)
    matrix = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            for k in range(3):
                matrix[i, j, k] = (i + j + k) % 256
    my_image.all_update_matrix(matrix)
    my_image.show("task_1_4")


def task_3_1(H: int, W: int):
    print("Простейшая прямая: ")
    my_image = Image_class(H, W)
    middle = Point(100, 100)
    Points = []
    Points.append(middle)
    for i in range(13):
        last_point = Point(int(middle.x + 95 * math.cos(2 * i * math.pi / 13)),
                           int(middle.y + 95 * math.sin(2 * i * math.pi / 13)))
        Points.append(last_point)
        for j in range(100):
            x = round(middle.x * (1.0 - 0.01 * j) + last_point.x * 0.01 * j)
            y = round(middle.y * (1.0 - 0.01 * j) + last_point.y * 0.01 * j)
            new_point = Point(x, y)
            Points.append(new_point)
        #     print(str(x) + " " + str(y))
    my_image.update_matrix(Points)
    my_image.show("task_3_1")


def task_3_2(H: int, W: int):
    print("Второй вариант: ")
    my_image = Image_class(H, W)
    middle = Point(100, 100)
    Points = []
    Points.append(middle)
    for i in range(13):
        last_point = Point(int(middle.x + 95 * math.cos(2 * i * math.pi / 13)),
                           int(middle.y + 95 * math.sin(2 * i * math.pi / 13)))
        Points.append(last_point)
        x_list = [round(x) for x in np.linspace(middle.x, last_point.x)]
        for x in x_list:
            t = (x - middle.x) / (last_point.x - middle.x)
            y = int(middle.y * (1.0 - t) + last_point.y * t)
            new_point = Point(x, y)
            Points.append(new_point)
    my_image.update_matrix(Points)
    my_image.show("task_3_2")


def task_3_3(H: int, W: int):
    print("Третий вариант: ")
    my_image = Image_class(H, W)
    middle = Point(100, 100)
    Points = []
    Points.append(middle)
    for i in range(13):
        steep = False
        last_point = Point(int(middle.x + 95 * math.cos(2 * i * math.pi / 13)),
                           int(middle.y + 95 * math.sin(2 * i * math.pi / 13)))
        if abs(middle.x - last_point.x) < abs(middle.y - last_point.y):
            middle.x, middle.y = middle.y, middle.x
            last_point.x, last_point.y = last_point.y, last_point.x
            steep = True
        if middle.x > last_point.x:
            middle.x, last_point.x = last_point.x, middle.x
            middle.y, last_point.y = last_point.y, middle.y
        x_list = [round(x) for x in np.linspace(middle.x, last_point.x)]
        for x in x_list:
            t = (x - middle.x) / (last_point.x - middle.x)
            y = int(middle.y * (1.0 - t) + last_point.y * t)
            if steep:
                new_point = Point(y, x)
            else:
                new_point = Point(x, y)
            print(new_point)
            Points.append(new_point)
        middle.x, middle.y = 100, 100
    my_image.update_matrix(Points)
    my_image.show("task_3_3")


def task_3_4(H: int, W: int):
    print("Брезенхема вариант: ")
    my_image = Image_class(H, W)
    middle = Point(100, 100)
    Points = []
    Points.append(middle)
    for i in range(13):
        steep = False
        last_point = Point(int(middle.x + 95 * math.cos(2 * i * math.pi / 13)),
                           int(middle.y + 95 * math.sin(2 * i * math.pi / 13)))
        if abs(middle.x - last_point.x) < abs(middle.y - last_point.y):
            middle.x, middle.y = middle.y, middle.x
            last_point.x, last_point.y = last_point.y, last_point.x
            steep = True
        if middle.x > last_point.x:
            middle.x, last_point.x = last_point.x, middle.x
            middle.y, last_point.y = last_point.y, middle.y
        derror = abs((last_point.y - middle.y) / (last_point.x - middle.x))
        error = 0
        y = middle.y
        for x in range(middle.x, last_point.x):
            if steep:
                new_point = Point(y, x)
            else:
                new_point = Point(x, y)
            error += derror
            if error > 0.5:
                if last_point.y > middle.y:
                    y += 1
                else:
                    y -= 1
                error -= 1.0
            Points.append(new_point)
        middle.x, middle.y = 100, 100
    my_image.update_matrix(Points)
    my_image.show("task_3_4")


def task_4():
    f = open('african_head.obj', 'r')
    lines = f.read()
    Points = []
    for line in lines.split('\n'):
        try:
            v, x, y, z = re.split('\s+', line)
        except:
            continue
        if v == 'v':
            x = int((float(x) + 1) * 500)
            y = 1000 - int((float(y) + 1) * 500)
            z = int((float(z) + 1) * 400)
            Points.append(Point(y, x, z))
    return Points


def task_4_1():
    f = open('african_head.obj', 'r')
    lines = f.read()
    Points = []
    for line in lines.split('\n'):
        try:
            v, x, y, z = re.split('\s+', line)
        except:
            continue
        if v == 'v':
            x = (float(x) + 1) * 500
            y = 1000 - (float(y) + 1) * 500
            z = (float(z) + 1) * 400
            Points.append(Point(y, x, z))
    return Points


def task_5(H: int, W: int):
    print("Отрисовка вершин трёхмерной модели")
    points = task_4()
    my_image = Image_class(H + 1, W + 1)
    # my_image.update_matrix(points)
    # my_image.show()
    return my_image, points


def task_6():
    f = open('african_head.obj', 'r')
    lines = f.read()
    array = np.empty((0, 3), int)
    for line in lines.split('\n'):
        try:
            f = re.split('\s+', line)[0]
        except:
            continue
        if f == 'f':
            row = np.array([int(elem.split('/')[0]) - 1 for elem in re.split('\s+', line[2:])])
            array = np.append(array, [row], axis=0)

    return array


def task_7_1(x0, x1, y0, y1, my_image):
    steep = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    if x1 - x0 != 0:
        derror = abs((y1 - y0) / (x1 - x0))
    else:
        derror = abs((y1 - y0) / (x1 - x0 + 1))
    error = 0
    y = y0
    for x in range(int(x0), int(x1)):
        if steep:
            my_image.matrix[int(y), int(x)] = 255
        else:
            my_image.matrix[int(x), int(y)] = 255
        error += derror
        if error > 0.5:
            if y1 > y0:
                y += 1
            else:
                y -= 1
            error -= 1.0


def task_7():
    my_image, points = task_5(1000, 1000)
    array = task_6()
    for i in array:
        for j in range(len(i) - 1):
            for k in range(j, len(i)):
                if i[j] != i[k]:
                    task_7_1(points[i[j]].x, points[i[k]].x, points[i[j]].y, points[i[k]].y, my_image)
    my_image.show("task_7")


def task_8(Points, x, y, b=True):
    # points = task_4_1()// для вызова 8
    # array = task_6()
    #
    # for i in array:
    #     triangle = []
    #     for j in i:
    #         triangle.append(points[j])
    #     task_8(triangle, 575, 0)
    lambda_0 = ((Points[1].x - Points[2].x) * (y - Points[2].y) - (Points[1].y - Points[2].y) * (x - Points[2].x)) / (
            (Points[1].x - Points[2].x) * (Points[0].y - Points[2].y) - (Points[1].y - Points[2].y) *
            (Points[0].x - Points[2].x))
    lambda_1 = ((Points[2].x - Points[0].x) * (y - Points[0].y) - (Points[2].y - Points[0].y) * (x - Points[0].x)) / (
            (Points[2].x - Points[0].x) * (Points[1].y - Points[0].y) - (Points[2].y - Points[0].y) *
            (Points[1].x - Points[0].x))
    lambda_2 = ((Points[0].x - Points[1].x) * (y - Points[1].y) - (Points[0].y - Points[1].y) * (x - Points[1].x)) / (
            (Points[0].x - Points[1].x) * (Points[2].y - Points[1].y) - (Points[0].y - Points[1].y) *
            (Points[2].x - Points[1].x))
    # print(lambda_0, " ", lambda_1, " ", lambda_2)
    # print(lambda_0 + lambda_1 + lambda_2)
    if b:
        return lambda_0 >= 0 and lambda_1 >= 0 and lambda_2 >= 0
    return lambda_0, lambda_1, lambda_2


def task_9(H: int, W: int, Points):
    print("Отрисовка треугольника")
    my_image = Image_class(H, W)
    matrix = np.zeros((H, W, 3), dtype=np.uint8)

    xmin = min(Points[0].x, Points[1].x, Points[2].x)
    xmax = max(Points[0].x, Points[1].x, Points[2].x)
    ymin = min(Points[0].y, Points[1].y, Points[2].y)
    ymax = max(Points[0].y, Points[1].y, Points[2].y)
    # print(xmin, " ", xmax," ",ymax," ",ymin)

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            if task_8(Points, x, y):
                print("Подходит")
                matrix[x][y] = 255
    my_image.all_update_matrix(matrix)
    my_image.show("task_9")


def task_9_1(Points, matrix):
    xmin = min(Points[0].x, Points[1].x, Points[2].x)
    xmax = max(Points[0].x, Points[1].x, Points[2].x)
    ymin = min(Points[0].y, Points[1].y, Points[2].y)
    ymax = max(Points[0].y, Points[1].y, Points[2].y)
    # print(xmin, " ", xmax," ",ymax," ",ymin)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    for x in range(int(xmin), int(xmax) + 1):
        for y in range(int(ymin), int(ymax) + 1):
            if task_8(Points, x, y):
                # print("Подходит")
                matrix[x][y] = [r, g, b]


def task_9_2(Points, matrix, n):
    xmin = min(Points[0].x, Points[1].x, Points[2].x)
    xmax = max(Points[0].x, Points[1].x, Points[2].x)
    ymin = min(Points[0].y, Points[1].y, Points[2].y)
    ymax = max(Points[0].y, Points[1].y, Points[2].y)
    # print(xmin, " ", xmax," ",ymax," ",ymin)

    for x in range(int(xmin), int(xmax) + 1):
        for y in range(int(ymin), int(ymax) + 1):
            if task_8(Points, x, y):
                # print("Подходит")
                matrix[x][y] = [255 * n, 0, 0]


def task_9_3(Points, matrix, n, z, l):
    xmin = min(Points[0].x, Points[1].x, Points[2].x)
    xmax = max(Points[0].x, Points[1].x, Points[2].x)
    ymin = min(Points[0].y, Points[1].y, Points[2].y)
    ymax = max(Points[0].y, Points[1].y, Points[2].y)
    # print(xmin, " ", xmax," ",ymax," ",ymin)

    for x in range(int(xmin), int(xmax) + 1):
        for y in range(int(ymin), int(ymax) + 1):
            l1, l2, l3 = task_8(Points, x, y, False)
            if task_8(Points, x, y) and 0 <= x < 1000 and 0 <= y < 1000:
                cur_z = Points[0].z * l1 + Points[1].z * l2 + Points[2].z * l3
                if cur_z < z[x, y]:
                    z[x, y] = cur_z
                    n = -np.dot(task_12(Points), l)
                    #     # print("Подходит")
                    matrix[1000 - x][y] = [255 * n, 0, 0]


def task_11(H=1000, W=1000):
    my_image, points = task_5(H, W)
    points = task_4_1()
    array = task_6()
    # for i in array:
    #     for j in range(len(i) - 1):
    #         for k in range(j, len(i)):
    #             if i[j] != i[k]:
    #                 task_7_1(points[i[j]].x, points[i[k]].x, points[i[j]].y, points[i[k]].y, my_image)
    for i in array:
        triangle = []
        for j in i:
            triangle.append(points[j])
        task_9_1(triangle, my_image.matrix)
    my_image.show("task_11")


def task_12(Points):
    a = [Points[1].x - Points[0].x, Points[1].y - Points[0].y, Points[1].z - Points[0].z]
    b = [Points[1].x - Points[2].x, Points[1].y - Points[2].y, Points[1].z - Points[2].z]
    c = a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - b[0] * a[1]
    # print(c)
    return (c / np.linalg.norm(c))
    # points = task_4_1()
    # for point in points:
    #     print(point)


def task_13(H=1000, W=1000):
    l = [0, 0, 1]
    my_image, points = task_5(H, W)
    points = task_4_1()
    array = task_6()
    # for i in array:
    #     for j in range(len(i) - 1):
    #         for k in range(j, len(i)):
    #             if i[j] != i[k]:
    #                 task_7_1(points[i[j]].x, points[i[k]].x, points[i[j]].y, points[i[k]].y, my_image)

    for i in array:
        triangle = []
        for j in i:
            triangle.append(points[j])
        if np.dot(task_12(triangle), l) < 0.0:
            task_9_1(triangle, my_image.matrix)

    my_image.show("task_13")


def task_14(H=1000, W=1000):
    l = [0, 0, 1]
    my_image, points = task_5(H, W)
    points = task_4_1()
    array = task_6()
    # for i in array:
    #     for j in range(len(i) - 1):
    #         for k in range(j, len(i)):
    #             if i[j] != i[k]:
    #                 task_7_1(points[i[j]].x, points[i[k]].x, points[i[j]].y, points[i[k]].y, my_image)
    for i in array:
        triangle = []
        for j in i:
            triangle.append(points[j])
        n = np.dot(task_12(triangle), l)
        if n < 0:
            task_9_2(triangle, my_image.matrix, n)

    my_image.show("task_14")


def task_15(H=1000, W=1000):
    l = [0, 0, 1]
    my_image, points = task_5(H, W)
    z = np.full((1000, 1000), 10 ** 10)
    points = task_4_1()
    array = task_6()

    for i in array:
        triangle = []
        for j in i:
            triangle.append(points[j])
        n = np.dot(task_12(triangle), l)
        if n < 0:
            task_9_3(triangle, my_image.matrix, n, z, l)
    my_image.show("task_15")


def task_16_1():
    f = open('african_head.obj', 'r')
    lines = f.read()
    Points = []
    for line in lines.split('\n'):
        try:
            v, x, y, z = re.split('\s+', line)
        except:
            continue
        if v == 'v':
            x = float(x) + 1
            y = float(y) + 1
            z = float(z) + 21
            Points.append(Point(y, x, z))
    mat = np.array([[13000, 0, -100],
                    [0, 13000, -100],
                    [0, 0, 1]])

    for point in Points:
        point.update_point((mat @ point.to_vec()) / point.z)
        # if point.x > 999 or point.y > 999:
        print(point)
        point.z *= 400
        # print(point)
    return Points


def task_16(H=1000, W=1000):
    print("Отрисовка вершин трёхмерной модели")
    points = task_16_1()
    my_image = Image_class(H + 1, W + 1)
    array = task_6()
    l = [0, 0, -1]
    z = np.full((1000, 1000), 10 ** 10)
    # my_image.update_matrix(points)
    for i in array:
        triangle = []
        for j in i:
            triangle.append(points[j])
        n = np.dot(task_12(triangle), l)
        if n < 0:
            task_9_3(triangle, my_image.matrix, n, z, l)
    my_image.show("task_16")


def task_16_2():
    f = open('african_head.obj', 'r')
    lines = f.read()
    Points = []
    for line in lines.split('\n'):
        try:
            v, x, y, z = re.split('\s+', line)
        except:
            continue
        if v == 'v':
            x = float(x) + 1
            y = float(y) + 1
            z = float(z) + 1
            Points.append(Point(y, x, z))
    mat = np.array([[10000, 0, -200],
                    [0, 10000, 0],
                    [0, 0, 1]])
    R = np.array([[math.cos(math.pi / 7), 0, math.sin(math.pi / 7)], [0, 1, 0],
                  [-math.sin(math.pi / 7), 0, math.cos(math.pi / 7)]])
    for point in Points:
        point.update_point((R @ point.to_vec()))
        point.z += 20
        point.update_point((mat @ point.to_vec()) / point.z)
        # if point.x > 999 or point.y > 999:
        print(point)
        point.z *= 400
        # print(point)
    return Points


def task_17(H=1000, W=1000):
    points = task_16_2()
    my_image = Image_class(H + 1, W + 1)
    array = task_6()
    l = [0, 0, -1]
    z = np.full((1000, 1000), 10 ** 10)
    for i in array:
        triangle = []
        for j in i:
            triangle.append(points[j])
        n = np.dot(task_12(triangle), l)
        if n < 0:
            task_9_3(triangle, my_image.matrix, n, z, l)
    # my_image.update_matrix(points)
    my_image.show("task_17")


if __name__ == "__main__":
    task_16()
