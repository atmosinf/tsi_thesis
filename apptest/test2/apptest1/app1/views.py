from django.shortcuts import render
from django.http import HttpResponse

import torch
from torchvision import transforms
from app1.vgg import Vgg
import cv2
import numpy as np

# Create your views here.
import base64
from django.shortcuts import render
from .forms import ImageUploadForm

def index(request):
    image_uri = None
    prediction = None
    predlabel = None

    if request.method == 'POST':
        # in case of POST: get the uploaded image from the form and process it
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # retrieve the uploaded image and convert it to bytes (for PyTorch)
            image = form.cleaned_data['image']
            image_bytes = image.file.read()
            # convert and pass the image as base64 string to avoid storing it to DB or filesystem
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)

            # get predicted label with previously implemented PyTorch function
            try:
                nparr = np.fromstring(image_bytes, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                prediction, predlabel = get_prediction(img_np)
                prediction = list(prediction.detach().numpy()[0])
            except RuntimeError as re:
                print(re)

    else:
        # in case of GET: simply show the empty form for uploading images
        form = ImageUploadForm()

    # pass the form, image URI, and predicted label to the template to be rendered
    context = {
        'form': form,
        'image_uri': image_uri,
        'prediction': prediction,
        'predlabel': predlabel
    }
    return render(request, 'index.html', context)


vggmodel = Vgg()
vggmodel.eval()
device = torch.device('cpu')
vggmodel.load_state_dict(torch.load('saved_model/vggmodel_state_dict.pth', map_location=device))

def get_prediction(image_bytes):
    img = image_bytes
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