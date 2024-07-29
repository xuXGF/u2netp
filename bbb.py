import cv2
import time
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from model import U2NETP

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


model_dir = 'u2netp.pth'
net = U2NETP(3, 1)
net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))

if torch.backends.mps.is_available():
    net.to(device)
net.eval()
# 定义你的 transform
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])

def process_frame(frame):
    # 直接使用 OpenCV 图像进行转换，不使用 PIL
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image)
    inputs_test = image.unsqueeze(0).to(device)

    with torch.no_grad():
        d1, _, _, _, _, _, _ = net(inputs_test)
        pred = d1[:, 0, :, :]
        pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred))
        pred = pred[0].cpu().numpy()
        pred = np.where(pred >= 0.5, 255, 0).astype(np.uint8)

    return pred

def display_frame(frame, pred):
    mask = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    frame[mask_resized != 255] = 0
    result = frame
    cv2.imshow('Processed Frame', result)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    start_time = time.time()
    frame_count = 0

    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 5 == 0:
                current_time = time.time()
                print(f"Frame time: {current_time - start_time:.4f}s")
                start_time = current_time

            # 使用多线程读取帧和多进程处理图像
            future = executor.submit(process_frame, frame)
            pred = future.result()
            executor.submit(display_frame, frame, pred)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()