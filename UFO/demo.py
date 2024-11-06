import os
from PIL import Image
import torch
from torchvision import transforms
from model_video import build_model
import numpy as np
import cv2
import argparse
to_pil = transforms.ToPILImage()

def main(gpu_id, model_path, datapath, save_root_path, group_size, img_size, img_dir_name,crf):
    net = build_model(device).to(device)
    net=torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path, map_location=gpu_id, weights_only=True))
    net.eval()
    net = net.module.to(device)
    img_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_transform_gray = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.449], std=[0.226])])            
    
    with torch.no_grad():
        for p in range(len(datapath)):
            vc = cv2.VideoCapture(datapath[p])
            vc.set(cv2.CAP_PROP_POS_FRAMES, 800)
            rval = vc.isOpened()
            c=0
            frame_list=[]
            while rval:
                rval, frame = vc.read()
                if rval:
                    frame_list.append(frame)
                    c=c+1
                else:
                    break
            vc.release()

            idx=[]
            block_size=(len(frame_list)+group_size-1)//group_size
            for i in range(block_size):
                cur=i
                while cur<len(frame_list):
                    idx.append(cur)
                    cur+=block_size
            new_frame_list=[]
            for i in range(len(frame_list)):
                new_frame_list.append(frame_list[idx[i]])
            frame_list=new_frame_list
            
            frame_result=[]
            cur_class_rgb = torch.zeros(len(frame_list), 3, img_size, img_size)
            for i, frame in enumerate(frame_list):
                frame = Image.fromarray(frame)
                if frame.mode == 'RGB':
                    frame = img_transform(frame)
                else:
                    frame = img_transform_gray(frame)
                cur_class_rgb[i, :, :, :] = frame

            cur_class_mask = torch.zeros(len(frame_list), img_size, img_size)

            divided = len(frame_list) // group_size
            rested = len(frame_list) % group_size
            if divided != 0:
                for k in range(divided):
                    group_rgb = cur_class_rgb[(k * group_size): ((k + 1) * group_size)]
                    group_rgb = group_rgb.to(device)
                    _, pred_mask = net(group_rgb)
                    cur_class_mask[(k * group_size): ((k + 1) * group_size)] = pred_mask
            if rested != 0:
                group_rgb_tmp_l = cur_class_rgb[-rested:]
                group_rgb_tmp_r = cur_class_rgb[:group_size - rested]
                group_rgb = torch.cat((group_rgb_tmp_l, group_rgb_tmp_r), dim=0)
                group_rgb = group_rgb.to(device)
                _, pred_mask = net(group_rgb)
                cur_class_mask[(divided * group_size):] = pred_mask[:rested]

            for i, img in enumerate(frame_list):
                result = cur_class_mask[i, :, :]
                prediction = np.array(to_pil(result.data.squeeze().cpu()))

                img = Image.fromarray(img)
                w, h = img.size
                img=img.resize((prediction.shape[0],prediction.shape[1]),Image.BILINEAR)
                if crf==True:
                    prediction = crf_refine(np.array(img), prediction)
                result=torch.from_numpy(np.array(prediction)/255).view(prediction.shape[0],prediction.shape[1],1).repeat(1,1,3).numpy()
                img=np.array(img)
                result=(img/2+np.array([127,127,0]))*result+(1-result)*img
                result=Image.fromarray(result.astype(np.uint8))
                result = result.resize((w, h), Image.BILINEAR)
                frame_result.append(result)

            new_frame_result=[]
            for frame in frame_result:
              new_frame_result.append(frame)
            for i, frame in enumerate(frame_result):
              new_frame_result[idx[i]]=frame
            
            vw = cv2.VideoWriter(os.path.join(save_root_path[p], "test.mp4"), cv2.VideoWriter.fourcc(*'H264'), 24, (w, h))
            for img in new_frame_result:
                vw.write(np.array(img))
            vw.release()
            print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./models/video_best.pth', help="restore checkpoint")
    parser.add_argument('--data_path',default='./demo_mp4/video/kobe_1v1.mp4', help="dataset for evaluation")
    parser.add_argument('--output_dir', default='./demo_mp4/result', help='directory for result')
    parser.add_argument('--gpu_id', default='cuda:0', help='id of gpu')
    parser.add_argument('--crf', default=False, help='make outline clear')
    args = parser.parse_args()
    
    gpu_id = args.gpu_id
    device = torch.device(gpu_id)
    model_path = args.model

    val_datapath = [args.data_path]

    save_root_path = [args.output_dir]

    main(gpu_id, model_path, val_datapath, save_root_path, 5, 224, 'image',args.crf)
