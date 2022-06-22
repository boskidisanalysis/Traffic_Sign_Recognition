#from nbformat import write
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import streamlit as st 

st.title('Recognize Traffic Signs from Photos')

st.write(''' # Data 
''') 
st.image("https://storage.googleapis.com/kaggle-datasets-images/82373/191501/4f9f4f59d288718705ff67449bbc8e66/dataset-cover.jpg?t=2018-11-25-18-48-30")
st.write('''
We used the [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) dataset to train our Model, which contains 43 different traffic signs and over 50000 photos. 


''')
st.write('''# Model
We build a Classification Model using the MobileNet's feature vectors and a Dense Layer for our Classification Purpose.

''')
st.image('Photos\Model_MobileNet.png')

st.write('''# Predictions 
We tested the model with an image we took with a mobile phone. These are the Results: 
 ''')

st.image('Photos/big.png') 
st.image("Photos/medium.png")
st.image("Photos/small.png")
st.write('''## Upload your own 
Upload a well-cropped photo of a road sign to get your own predictions!!!''')
uploaded_file = st.file_uploader("Select an Image",type=["PNG","JPEG","JPG"])
if uploaded_file is not None:

    #from tensorflow.keras.utils import plot_model

    # Python program to store list to JSON file
    import json

    def write_list(a_list,filename):
        print("Started writing list data into a json file")
        with open(filename, "w") as fp:
            json.dump(a_list, fp)
            print("Done writing JSON data into .json file")

    # Read list to memory
    def read_list(filename):
        # for reading also binary mode is important
        with open(f'{filename}', 'rb') as fp:
            n_list = json.load(fp)
            return n_list



    import tensorflow as tf 
    class_names= read_list('./class_names.json')

    m = tf.keras.models.load_model('Traffic_sign_mobilenet')
    #st.write(plot_model(m))
    #plot_model(m)
    #st.write(m.summary())

    def predict_sign(image_file):
        ''' Provide an image file of a road sign and outputs the plots of the Image and the predicted road sign '''
        image = tf.keras.preprocessing.image.load_img(image_file)
        #Turning to array
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])

        prediction = m.predict(input_arr)

        path_to_meta = 'meta/'
        predicted_image = path_to_meta + class_names[np.argmax(prediction)]+'.png'
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        img = mpimg.imread(predicted_image)

        fig,ax = plt.subplots(1,2)
        fig.patch.set_facecolor('white')
        ax[0].imshow(image)
        ax[0].set_title('Image to Predict')
        ax[0].axis('off')
        ax[1].imshow(img)
        ax[1].set_title('Predicted Sign')
        ax[1].axis('off')
        st.pyplot(fig)
    st.image(uploaded_file)
    st.write('## Your Result')
    predict_sign(uploaded_file)