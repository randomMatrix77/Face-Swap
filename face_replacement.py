import os
import cv2
import time
import torch
import numpy as np
from FER_Resnet import FERModel
from torchvision import models
from scipy.spatial import distance

os.environ['TORCH_HOME'] = "D:/Softwares/miniconda/torch_models"
print('current location : {}'.format(os.getenv("TORCH_HOME",os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'),
                                                                         'torch'))))
classes = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

no_assets = os.listdir('./assets')


print('Setting up VGG16 model')
vgg = models.vgg16(pretrained = True)
face_cascade = cv2.CascadeClassifier('haar_face.xml')

def face_recognition(image):
    '''
    :param image: BW image of extraced face of size (224, 224)
    :return: flattened numpy array of features
    '''

    image = cv2.resize(image, (224, 224))
    image = np.stack([image] * 3)
    image = np.expand_dims(image, axis=0)
    inp = torch.tensor(image / 255, dtype=torch.float32)
    out_features = vgg.features(inp)
    out_features = out_features.detach().numpy()
    out_features = out_features.reshape((1, -1))

    return out_features

def expression_recognition(image):
    '''
    :param image: BW 48x48 pixel image of face
    :return: Int value of detected expression w.r.t classes
    '''

    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(np.expand_dims(image, 0), 0)
    image = torch.tensor(image / 255, dtype=torch.float32)
    out = fer(image)

    return out.argmax().item()

def replace_mask(image, c_index, no_assets):

    asset = cv2.imread('./assets/' + no_assets[c_index], cv2.IMREAD_UNCHANGED)
    asset = cv2.resize(asset, (image.shape[1], image.shape[0]))
    # asset = np.moveaxis(asset, 0, 1)
    mask = (asset[:,:,-1] == 255)
    asset = asset[:,:,:3] * mask[:,:,np.newaxis]
    temp = image * (1 - mask[:,:,np.newaxis])
    temp = temp + asset

    return temp

print('Reading baseline face')
baseline_face = cv2.imread('./user1/baseline.png')
bw_baseline_face = cv2.cvtColor(baseline_face, cv2.COLOR_BGR2GRAY)
face = face_cascade.detectMultiScale(bw_baseline_face)

for (x, y, w, h) in face:
    extracted_face = bw_baseline_face[x: x + w, y: y + h]

print('Generating features from baseline face')
baseline_feature = face_recognition(extracted_face)

fer = FERModel(1, 7)
fer.load_state_dict(torch.load('D:/Softwares/miniconda/torch_models/checkpoints/FER2013-Resnet9.pth',
                               map_location = torch.device('cpu')))
fer.eval()

cap = cv2.VideoCapture(0)

print('Starting live feed in...')
for i in reversed(range(10)):

    print(i, end=' ')
    time.sleep(1)
print('\n')


while(True):

    ret, frame = cap.read()
    frame = frame[60:410, :, :]
    bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(bw)

    try:
        for (x, y, w, h) in faces:
            out_features = face_recognition(bw[x:x+h, y:y+h])
            dist = distance.euclidean(baseline_feature, out_features)
            if dist < 60:
                c_index = expression_recognition(bw[x:x+h, y:y+h])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Detected expr:' + classes[c_index],
                            (5, 300), cv2.FONT_ITALIC, 0.8, (0, 0, 0))
                overlay = replace_mask(frame[y:y+h, x:x+w, :], c_index, no_assets)
                frame[y:y+h, x:x+w, :] = overlay

    except:
        pass

    cv2.imshow('depth map', frame)

    k = cv2.waitKey(1)

    if k % 256 == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
