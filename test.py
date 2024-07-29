import os
from skimage import io
import torch
from torch.autograd import Variable

from torchvision import transforms  # , utils


import numpy as np
from PIL import Image
from model import U2NETP  # small version u2net 4.7 MB


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")


    # --------- 1. get image path and name ---------
    # model_name='u2netp'# fixed as u2netp

    # image_dir = os.path.join(os.getcwd(), 'images') # changed to 'images' directory which is populated while running the script
    prediction_dir = os.path.join(os.getcwd(), 'results/') # changed to 'results' directory which is populated after the predictions

    img_name = "/Users/xiaoxu/Desktop/IMG_0826.jpeg"
    print(img_name)
    model_dir = 'u2netp.pth'

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])

    from PIL import Image
    import pillow_heif
    if img_name.endswith(".HEIC") or img_name.endswith('.heic'):
        heif_file = pillow_heif.read_heif(img_name)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
    else:
        image = Image.open(img_name).convert('RGB')
    transformed_image = transform(image)
    print(transformed_image.shape)

    # --------- 3. model define ---------
    net = U2NETP(3, 1)
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))

    print("inferencing:", img_name.split(os.sep)[-1])
    inputs_test = transformed_image.type(torch.FloatTensor)

    if torch.backends.mps.is_available():
        net.to(device)
        inputs_test = Variable(inputs_test.to(device))
    else:
        inputs_test = Variable(inputs_test)

    inputs_test=torch.unsqueeze(inputs_test,dim=0)
    net.eval()
    import time
    start_time=time.time()
    d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
    print(time.time()-start_time)

    # normalization
    pred = d1[:, 0, :, :]
    pred = normPRED(pred)

    # save results to test_results folder
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)
    save_output(img_name, pred, prediction_dir)
    from cut import mask_cut_image
    mask_cut_image(img_name)




if __name__ == "__main__":
    main()
