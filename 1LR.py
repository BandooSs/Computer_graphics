import numpy as np
from PIL import Image
import math
import re


class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self.color3 = 255

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @x.setter
    def x(self, x):
        self._x = x

    @y.setter
    def y(self, y):
        self._y = y

    def color(self):
        return self.color3

    def __str__(self):
        return f"x= {self.x} y = {self.y}"

    def swap(self):
        self._x, self._y = self._y, self._x


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
            self.matrix[point.x][point.y] = 255


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
            Points.append(Point(y, x))
    return Points


def task_5(H: int, W: int):
    print("Отрисовка вершин трёхмерной модели")
    points = task_4()
    my_image = Image_class(H + 1, W + 1)
    my_image.update_matrix(points)
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
    for x in range(x0, x1):
        if steep:
            my_image.matrix[y, x] = 255
        else:
            my_image.matrix[x, y] = 255
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


if __name__ == "__main__":
    task_3_4(200,200)
