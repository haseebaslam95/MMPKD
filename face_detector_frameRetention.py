from __future__ import annotations
import cv2
import sys
import os
import csv
import datetime

from src.detector import detect_faces
from utils.visualization_utils import show_bboxes
from PIL import Image
from facenet_pytorch import MTCNN
import pandas as pd
import numpy as np

annotations_path= '' #path to ratings_gold_standard
rootpath= '' #path to raw videos

all_videos_name=os.listdir(rootpath)

strt = 0
endt = 0

def get_msec(time_str):
	m, s, ms = time_str.split(':')
	print(time_str)
	return  int(m) * 60000 + int(s)*1000

def find_indices(ltc, item_to_find):
    indices = []
    for idx, value in enumerate(ltc):
        if value == item_to_find:
            indices.append(idx)
    return indices

def get_duration(cap):
		fps = cap.get(cv2.CAP_PROP_FPS)
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		duration = frame_count/fps
		return duration*1000

def main():
	video_name_list=[]



	for i in range(len(all_videos_name)):

		video_file_name = '--path to raw videos---'+'/'+ all_videos_name[i]
		df = pd.read_csv(annotations_path+'/'+all_videos_name[i].split('.')[0]+'.csv',header=None,sep=';')
		valencelist=list(df[1])[:-1]
		arousallist=list(df[0])[:-1]

		if not os.path.isdir('images/' + all_videos_name[i]):
			os.makedirs('images/' + all_videos_name[i][:-4])
		
		

		k = 0 
		j = 0
		frame_name_list=[]
		label_list = []
		id=[]
		ispositive=False
		cap = cv2.VideoCapture(video_file_name)
		video_ms=get_duration(cap)
		tmp_image=0

		while k < 300000:
			ispositive=False
			cap.set(cv2.CAP_PROP_POS_MSEC, k)
			success, image = cap.read()
			if cap.isOpened():

				if success:
					pil_im = Image.fromarray(image)
					bounding_boxes, landmarks = detect_faces(pil_im) 
					id.append(j)
					

					if len(bounding_boxes) == 0:
						print(j)
						# cv2.imwrite('images/' + all_videos_name[i][:-4] + '/image' + str(j) + '.jpg', np.zeros((224,224,3)))
						cv2.imwrite('images/' + all_videos_name[i][:-4] + '/image' + str(j) + '.jpg',tmp_image) 
						frame_name_list.append(all_videos_name[i][:-4]+'/image' + str(j) + '.jpg')
						k = k + 40
						j = j + 1
						continue

					x, y, w, h, _ = bounding_boxes[0]


					if image[int(y):int(h)].shape[0]==0:
						# cv2.imwrite('images/' + all_videos_name[i][:-4] + '/image' + str(j) + '.jpg', np.zeros((224,224,3)))
						cv2.imwrite('images/' + all_videos_name[i][:-4] + '/image' + str(j) + '.jpg',tmp_image)  
						frame_name_list.append(all_videos_name[i][:-4]+'/image' + str(j) + '.jpg')
						k = k + 40
						j = j + 1
						continue

					cv2.imwrite('images/' + all_videos_name[i][:-4] + '/image' + str(j) + '.jpg', image[int(y) : int(h), int(x):int(w)]) 
					frame_name_list.append(all_videos_name[i][:-4]+'/image' + str(j) + '.jpg')
					tmp_image= image[int(y) : int(h), int(x):int(w)]

				k = k + 40
				
				j = j + 1
		
		if(len(frame_name_list)==7499):
			csv_file = pd.DataFrame({'image_name': frame_name_list,'arousal': arousallist[:-1],'valence':valencelist[:-1] })
			csv_file.to_csv(all_videos_name[i]+'.csv',index=False)
			cap.release()
		elif(len(frame_name_list)==7500):
			csv_file = pd.DataFrame({'image_name': frame_name_list,'arousal': arousallist,'valence':valencelist })
			csv_file.to_csv(all_videos_name[i]+'.csv',index=False)
			cap.release()

	
if __name__ == "__main__":
	main()