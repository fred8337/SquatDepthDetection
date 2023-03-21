import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd

data_directories_dict = {"D:/SquatData/LabelledSquats/Squats/Depth/": 1,
                         "D:/SquatData/LabelledSquats/Squats/NotDepth/": 0}

mean_image = None
file_names = []
labels = []

for data_directory in data_directories_dict.keys():
    for filename in os.listdir(data_directory):
        f = os.path.join(data_directory, filename)

        image = cv2.imread(data_directory+filename, cv2.IMREAD_GRAYSCALE)

        image = image.astype('float')
        image = image[150:300, 150:400]
        if mean_image is not None:
            image = image.reshape(np.shape(image)[0],np.shape(image)[1], 1)
            mean_image = mean_image.reshape(np.shape(mean_image)[0],np.shape(mean_image)[1], 1)
            mean_image = np.mean(np.concatenate((image, mean_image), axis=2), axis=2)
        else:
            mean_image = image
        cv2.imwrite("Data/Squats/"+filename, image)
        file_names.append(filename)
        labels.append(data_directories_dict[data_directory])
plt.imshow(mean_image, cmap='gray', vmin=0, vmax=255)
plt.show()

label_dataframe = pd.DataFrame({"Filename": file_names, "Depth": labels})
label_dataframe.to_csv("Data/Squats/annotations.csv")