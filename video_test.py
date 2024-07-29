import torch
from torch.autograd import Variable

import cv2
from torchvision import transforms  # , utils
from PIL import Image
from model import U2NETP  # small version u2net 4.7 MB

import time


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")

model_dir = 'u2netp.pth'
net = U2NETP(3, 1)
net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))

if torch.backends.mps.is_available():
    net.to(device)
net.eval()

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])



cap = cv2.VideoCapture(0)
start_time = time.time()
# print(time.time() - start_time)
while True:
    print(time.time() - start_time)
    start_time = time.time()
    index, frame = cap.read()
    if not index:
        break
    image=Image.fromarray(frame)
    image = transform(image)
    inputs_test = image.type(torch.FloatTensor)

    if torch.backends.mps.is_available():
        inputs_test = Variable(inputs_test.to(device))
    else:
        inputs_test = Variable(inputs_test)

    inputs_test=torch.unsqueeze(inputs_test,dim=0)

    # start_time=time.time()
    d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
    # print(time.time()-start_time)

    # normalization
    pred = d1[:, 0, :, :]
    pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred))

    pred = pred[0].cpu().data.numpy()
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0
    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    mask = cv2.resize(pred, (frame.shape[1], frame.shape[0]))
    frame[mask!=255]=0
    result = frame

    cv2.imshow('Gray Frame', result)
    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()






