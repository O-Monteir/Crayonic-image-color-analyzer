# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:37:46 2022

@author: Malaika Monteiro
"""

import streamlit as st
from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

st.title("Crayonic - Image Color Analyzer")

st.write('''
Image color analyzer 
''')

def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        i = int(i)
        hex_color += ("{:02x}".format(i))
    return hex_color

def prep_image(raw_img):
    modified_img = cv2.resize(raw_img, (900, 600), interpolation = cv2.INTER_AREA)
    modified_img = modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
    return modified_img

def color_analysis(image, img, clusters):
    clf = KMeans(n_clusters = clusters)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    fig.suptitle('Color analyzer - '+str(clusters)+' clusters', size = 'xx-large', weight = 800)
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Original image')
    ax2.pie(counts.values(), labels = hex_colors, colors = hex_colors)
    ax2.set_title('Distribution of top 5 HEX colors')
    
    return fig

def full_analysis(clusters = 5):

    uploaded_file = st.file_uploader("Upload an image to analyze:", type=["jpg","png", "jpeg"])

    if uploaded_file is not None:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        modified_image = prep_image(image)
        return color_analysis(image, modified_image, clusters)

k = st.slider('Number of clusters to analyze:', 1, 10, 5)

figure = full_analysis(k)

if figure is not None:

    st.pyplot(figure, dpi=300)