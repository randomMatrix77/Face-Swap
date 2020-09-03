# Face-Swap

Think of Face-Swap it as a privacy mode filter. If the neural network recognises your face in your webcam feed, it will be overlayed with an image present in the assets folder. The specific image to be overlayed on the video feed is selected based on the expression on your face.

The script uses a pretrained VGG16 model for facial recognition and a Resnet based Facial Emotion recognition model (https://github.com/TumAro/FER-Pytorch). The accuracy of FER model is still quite low.
