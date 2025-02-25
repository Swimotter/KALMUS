from multiprocessing import freeze_support

import cv2

from kalmus.utils.focus import Focus

freeze_support()

frame = cv2.imread(r"C:\Users\mrjac\Downloads\Color Samples\Color4.png")
focus = Focus({"img_size": 640,
                "arch": 7,
                "multi_gpu": True,
                "RFB_aggregated_channel": [32, 64, 128],
                "frequency_radius": 16,
                "gamma": 0.1,
                "denoise": 0.93,
                "batch_size": 32,
                "num_workers": 0})
focus.set_loader(frame)
processed_frame = focus.test()
cv2.imshow("test", processed_frame)