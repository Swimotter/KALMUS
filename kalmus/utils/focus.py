import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import sys
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image)

# Function to convert tensor to PIL image
# def tensor_to_pil(tensor):
#     return transforms.ToPILImage() #(tensor.cpu())

# Function to perform K-means clustering on RGB pixel data
def run_kmeans_on_pixels(pixel_tensor, n_clusters=3):
    pixel_tensor = pixel_tensor.view(-1, 3)
    pixel_data = pixel_tensor.numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(pixel_data)
    return kmeans.labels_, kmeans.cluster_centers_


def find_focus(video_path):
    birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
    device = 'cuda'
    torch.set_float32_matmul_precision(['high', 'highest'][0])

    birefnet.to(device)
    birefnet.eval()

    tensor_to_pil = transforms.ToPILImage()

    cap = cv.VideoCapture(video_path)
    vw = cv.VideoWriter("./temp.mp4", cv.VideoWriter.fourcc(*'mp4v'), int(cap.get(cv.CAP_PROP_FPS)), (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        i += 1
        sys.stdout.write(str(i) + '\n')

        image_tensor = torch.from_numpy(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

        # Convert tensor to PIL image
        pil_image = tensor_to_pil(image_tensor.permute(2, 0, 1))  # Change to (channels, height, width)

        # Transform and send to CUDA as needed
        input_images = transform_image(pil_image).unsqueeze(0).to('cuda')

        # Prediction
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()

        # Resize `pred` to match `image_tensor`'s spatial dimensions
        pred_resized = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=image_tensor.shape[:2], mode='bilinear', align_corners=False).squeeze()

        pred_resized[pred_resized < 0.5] = 0
        pred_resized[pred_resized >= 0.5] = 1
        foreground = image_tensor.clone()
        foreground[~pred_resized.to(torch.bool)] = 0

        vw.write(cv.cvtColor(np.array(foreground), cv.COLOR_BGR2RGB))
    vw.release()