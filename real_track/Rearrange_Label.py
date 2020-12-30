import pandas as pd
import csv
import os
import sys

try:
	inputPath = sys.argv[1]
	outputPath = sys.argv[2]
	segFile = sys.argv[3]
	if (not inputPath) or (not outputPath) or (not segFile):
		raise ''
except:
	print('usage: python Rearrange_Label.py <inputPath> <outputPath> <segFile>')
	exit(1)
	
# Open rally segment file
with open(segFile) as seg:
	rallys = csv.DictReader(seg, delimiter=',')
	for rally in rallys:
		total_frames = int(rally['End'])-int(rally['Start'])+2

		#Open previous label data
		with open(os.path.join(inputPath, rally['Score']+'.csv')) as csvfile:
			readCSV = csv.DictReader(csvfile, delimiter=',')
			frames = []
			x, y = [], []
			list1 = []
			for row in readCSV:
				frames.append(int(row['index']))
				x.append(int(row['x']))
				y.append(int(row['y']))
		visibility = [1 for _ in range(len(frames))]
		#Create DataFrame
		df_label = pd.DataFrame(columns=['Frame', 'Visibility', 'X', 'Y'])
		df_label['Frame'], df_label['Visibility'], df_label['X'], df_label['Y'] = frames, visibility, x, y
		#Compensate the non-labeled frames due to no visibility of badminton
		for i in range(0, total_frames):
			if i in list(df_label['Frame']):
				pass
			else:
				df_label = df_label.append(pd.DataFrame(data = {'Frame':[i], 'Visibility':[0], 'X':[0], 'Y':[0]}), ignore_index=True)

		#Sorting by 'Frame'
		df_label = df_label.sort_values(by=['Frame'])
		df_label.to_csv(os.path.join(outputPath, rally['Score']+'.csv'), encoding='utf-8', index=False)
