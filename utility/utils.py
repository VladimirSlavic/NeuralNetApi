import math
import sys
import os
import numpy as np


class Utiliy:
    @staticmethod
    def aproximate_distance(points_normalized):
        D = 0.0
        for i in range(1, len(points_normalized)):
            D += math.sqrt(math.pow(points_normalized[i][0] - points_normalized[i - 1][0], 2) + math.pow(
                points_normalized[i][1] - points_normalized[i - 1][1], 2))
        return D  # optimiraj sa list comprehension sa lambdom da se koristi c_inter

    @staticmethod
    def calc_distance_two_points(point1, point2):
        return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

    @staticmethod
    def get_representative_points(D, points_normalized, M=20):
        representive_points = [points_normalized[0]]
        first_p = points_normalized[0]

        for k in range(1, M):
            current_closest_point = sys.float_info.max
            best_point = None
            desired_distance = k * D / (M - 1)

            for p in points_normalized:
                p_dist = Utiliy.calc_distance_two_points(first_p, p)
                if abs(p_dist - desired_distance) < current_closest_point:
                    current_closest_point = abs(p_dist - desired_distance)
                    best_point = p
            representive_points.append(best_point)
        return representive_points

    @staticmethod
    def write_to_file(points, label):
        filename = '/home/elrond/PycharmProjects/NENR_LABOS_5/data/examples.txt'

        if os.path.exists(filename):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        with open(filename, append_write) as f:
            f.write(str(points)[1:-1] + ':' + label + "\n")

    @staticmethod
    def retrieve_greek_alphabet_data():
        lines_list = []
        with open(r'/home/elrond/PycharmProjects/NENR_LABOS_5/data/examples.txt', 'r') as f:
            for line in f:
                lines_list.append(line.split(':')[0])

        converted_result = []
        for elem in lines_list:
            tmp_result = []
            for num in elem.split(','):
                tmp_result.append(float(num.strip()))
            converted_result.append(tmp_result)

        X = np.asarray(converted_result)
        y = np.zeros(shape=(100, 5), dtype=np.float128)
        y[0:20, 0] = 1.0
        y[20:40, 1] = 1.0
        y[40:60, 2] = 1.0
        y[60:80, 3] = 1.0
        y[80:100, 4] = 1.0
        return X, y