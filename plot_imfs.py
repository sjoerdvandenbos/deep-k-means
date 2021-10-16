import numpy as np

from pathlib import Path


def read_csv(path):
    rows = []
    labels = []
    file = path.open("r")
    for index, line in enumerate(file.readlines()):
        try:
            composed = line.split(",")
            label = composed[-1]
            labels.append(label)
            feature_row = np.array(composed[:-1]).astype(float)
            rows.append(feature_row)
        except ValueError:
            print(index)
    return np.stack(rows), np.array(labels)


data_folder = Path.cwd() / "split" / "catted-imfs"

imf1_train, labels_train = read_csv(data_folder / "IMF1_train.csv")
imf2_train, _ = read_csv(data_folder / "IMF2_train.csv")
imf3_train, _ = read_csv(data_folder / "IMF3_train.csv")
denoised_train = imf1_train + imf2_train + imf3_train
print(imf1_train)
print(imf2_train)
print(imf3_train)
print(denoised_train.shape)
exit()

imf1_test, labels_test = read_csv(data_folder / "IMF1_test.csv")
imf2_test, _ = read_csv(data_folder / "IMF2_test.csv")
imf3_test, _ = read_csv(data_folder / "IMF3_test.csv")
denoised_test = imf1_test + imf2_test + imf3_test
print(denoised_test.shape)



