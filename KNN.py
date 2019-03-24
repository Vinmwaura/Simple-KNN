import numpy as np
import csv

"""
K-Nearest Neighbour (KNN)
KNN is a non-parametric, lazy learning algorithm

Non-parametric means it doesn't make any assumptions on
the underlying data distribution

Lazy learning means it doesn't use the training data points
to do any generalization
"""


# Loads entire IRIS dataset as csv
def get_iris_data():
    with open('Iris_Dataset.csv') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)

        data = []
        for row in reader:
            data.append(np.array(row[:]))
        return data


# Computes Euclidean distance between all training data points
# and testing datapoint
def euclidean_distance(train_data, test_data):
    return np.sqrt(
        np.sum(
            np.square(
                np.subtract(
                    np.float16(train_data[:-1]),
                    np.float16(test_data[:-1])
                )
            )
        )
    )


# Computes the frequent class category of all K data points
# closest to the testing data point
def frequency_class(data):
    count_frequency = {}
    for data_point in data:
        if data_point[0] in count_frequency:
            count_frequency[data_point[0]] += 1
        else:
            count_frequency[data_point[0]] = 1

    sorted_data = sorted(count_frequency, key=lambda tup: tup[1])
    return sorted_data[-1]


# Performs KNN operation
def compute_KNN(train_data, test_data, k=5):
    for test in test_data:
        data = []
        for train in train_data:
            distance = euclidean_distance(train, test)
            data.append([str(train[-1]), float(distance)])
        # sorted_by_second = sorted(data, key=lambda tup: tup[1])
        sorted_data = sorted(data, key=lambda tup: tup[1])

        predicted_class = frequency_class(sorted_data[:k])
        print("Data: ", test[:-1], "| Predicted Class: ", predicted_class, " | Actual Class: ", test[-1])


def main():
    iris_dataset = get_iris_data()
    np.random.shuffle(iris_dataset)
    data_len = int(len(iris_dataset) * 0.8)

    train_data = iris_dataset[:data_len]
    test_data = iris_dataset[data_len:]

    compute_KNN(train_data, test_data, k=3)


if __name__ == '__main__':
    main()
