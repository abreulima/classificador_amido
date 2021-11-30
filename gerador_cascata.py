import cv2
import pandas as pd
from pandas.core.frame import DataFrame
from analisador_classe import pre_processador, contours_and_hulls, generate_metrics, draw_poly, draw_hilos
from metrics import drop_numerical_outliers
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from scipy import stats
import numpy as np
import os
from os import listdir
from os.path import isfile, join

images_path = "simples"
output_file = "saida.csv"

appended_data = []
df_per_class = []

columns = ['Eccentricity', 'Convexicity', 'Aspect Ratio', 'RatioHilumStarch', 'Class']

#appended_data = pd.DataFrame(columns = columns)
classes = [name for name in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, name))]

for classe in classes:
    
    samples = []

    folder = join(images_path, classe)
    files = [f for f in listdir(folder) if isfile(join(folder, f))]

    for file in files:

        # current input file
        input =  os.path.join(images_path, classe, file)
        print(input)

        # segmentação
        original, pre_processada = pre_processador(input)
        cnts, hulls, childs = contours_and_hulls(pre_processada)

        # input dimensions
        height, width, channels = original.shape 

        print("Grânulos identificados:", len(cnts))
        draw_poly(original, cnts, hulls, ellipse=False, bbox=False, display=True)
        draw_hilos(original, childs)

        metrics, cnts = generate_metrics(cnts, childs)

        for m in metrics:

            # get the metrics of every individual grain 
            eccentricity = m['eccentricity']
            convexity = m['convexity']
            roundness = m['roundness']
            ratio_child_contour = m['ratio_child_contour']
            area_in_pixels = m['area_in_pixels']
            aspect_ratio = m['aspect_ratio']

            # new metrics
            area_proportional = (area_in_pixels/(width*height))*100

            # values
            samples.append([eccentricity, convexity, aspect_ratio, ratio_child_contour, classe])        
            
    print("Tam:", len(samples))

    class_df = pd.DataFrame(samples, columns=columns)
    df_per_class.append(class_df)


# outliers
for df in df_per_class:
    print(len(df))
    drop_numerical_outliers(df)
    print(len(df))


df = pd.concat(df_per_class)

print(df.head(15))
df.to_csv(join("dados_numericos", output_file))