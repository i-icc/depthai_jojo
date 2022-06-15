import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pickle

def main():
    targets_data = pd.read_csv("./train_data/y_classified.csv")

    n = len(targets_data["id"])
    datas = np.empty((n, 10), float)
    for i in targets_data["id"]:
        filename = f"./train_data/angle/{i}.txt"
        f = open(filename, "r")
        line = f.readline().split(",")
        try:
            datas[i] = np.array(list(map(float, line)))
        except:
            print(i)


    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(datas, targets_data["judge"])

    with open('./models/jojo_model.pickle', mode='wb') as fp:
        pickle.dump(knn, fp)

if __name__ == "__main__":
    main()