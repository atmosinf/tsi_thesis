from django.http import HttpResponse
from django.shortcuts import render
from .models import *
from django.core.mail import EmailMessage
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading

import torch
from torchvision import transforms
from videoapp2.vgg import Vgg
import cv2
import numpy as np

def index(request):
    return render(request, 'index.html')

@gzip.gzip_page
def video_stream(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass
    return render(request, 'error.html')

#to capture video class
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def get_framearray(self):
        return self.frame

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        camarray = camera.get_framearray()
        pred, predlabel = get_prediction(camarray)
        # print(camtensor)
        # print(camtensor.shape)
        print(pred)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

vggmodel = Vgg()
vggmodel.eval()
device = torch.device('cpu')
vggmodel.load_state_dict(torch.load('saved_model/vggmodel_state_dict.pth', map_location=device))

def get_prediction(image):
    img = image
    transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((48,48)),
                                transforms.Grayscale(),
                                transforms.ToTensor()])
    imgtransformed = transform(img)
    imgtransformed.unsqueeze_(1)
    pred = vggmodel(imgtransformed)
    labels = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt", "unknown"]
    predlabel = labels[torch.argmax(pred, dim=1)]
    print(predlabel)

    return pred, predlabel
