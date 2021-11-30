import os
import cv2
import pandas
import sys
import pickle 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from analisador_classe import pre_processador, contours_and_hulls, generate_metrics, draw_poly, desenhar_classses
from metrics import drop_numerical_outliers
from tools import display_image
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# arquivo de entrada
input = sys.argv[1]

# arquivo de saída
output = "saida.png"

# used_metrics
use_metrics = []

columns = ['Eccentricity', 'Convexicity', 'Aspect Ratio', 'RatioHilumStarch']

# segmentação
original, pre_processada = pre_processador(input)
cnts, hulls, childs = contours_and_hulls(pre_processada)
metrics, cnts = generate_metrics(cnts, childs)
draw_poly(original, cnts, hulls)

# input dimensions
height, width, channels = original.shape 

for m in metrics:

    # get the metrics of every individual grain 
    eccentricity = m['eccentricity']
    convexity = m['convexity']
    roundness = m['roundness']
    ratio_child_contour = m['ratio_child_contour']
    area_in_pixels = m['area_in_pixels']
    aspect_ratio = m['aspect_ratio']

    area_proportional = (area_in_pixels/(width*height))*100

    use_metrics.append([eccentricity, convexity, aspect_ratio, ratio_child_contour])

    # area_in_pixels = m['area_in_pixels']

# dataframe
df = pandas.DataFrame(use_metrics, columns=columns)
print(len(df))
#drop_numerical_outliers(df)
print(len(df))

# carrega modelo treinado
loaded_model = pickle.load(open('modelos/knnpickle_file', 'rb'))
result = loaded_model.predict(df) 

final = desenhar_classses(original, cnts, result)
display_image(final)

cv2.imwrite(os.path.join("output", output), final)

df[['Eccentricity']].plot(kind='hist', rwidth=0.8)
#plt.show()

import tikzplotlib
tikzplotlib.save("test.tex")

for r in result:
    print(r)