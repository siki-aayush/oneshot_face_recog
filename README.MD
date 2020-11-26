# Face Recognition using Oneshot Algorithm

## Required Frameworks
- Tensorflow
- Keras

## Required Libraries:
- Numpy
- OpenCV


## FOR OLIVETTI DATASET FACE RECOGNITION
- Download your api key file from kaggle.
- Open the notebook in colab.
- Run all the cells (upload the api file when it asks to upload the file).

## FOR CUSTOM FACE RECOGNITION
- Run `python get_faces.py` and get the image of the faces for at least two persons.
- You can manually insert images inside datasetfolder under train directory (Foldername should be the name of the persons image).
- Open the custom_face_recog notebook and run all the cells.

By default the images will be converted to grayscale and be trained, if you want to train the images in rgb replace the arguments in the functions from False to True
