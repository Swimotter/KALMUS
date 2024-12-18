import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.io import read_video
from sklearn.cluster import KMeans
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

    # Read the video file
    # video_frames: Tensor[T, H, W, C] where T = number of frames, H = height, W = width, C = channels
    # audio_frames: Tensor[K, N] where K and N are audio dimensions
    # info: Metadata about the video and audio
    video_frames, audio_frames, info = read_video(video_path)

    tensor_to_pil = transforms.ToPILImage()
    vw = cv.VideoWriter("./temp.mp4", cv.VideoWriter.fourcc(*'H264'), 24, (video_frames[0].shape[1], video_frames[0].shape[0]))

    for i in range(video_frames.shape[0]):
        i = int(i)
        print(i)
        image_tensor = video_frames[i]

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