import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
#HOG SUBSAMPLE
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from sklearn.preprocessing import StandardScaler
from window import *
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.ndimage.measurements import label
from settings import *
from feature import *
from load_data import *
from train import *


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

# Draw bounding boxes based on labels
def draw_labeled_bboxes(img, labels):
	# Iterate through all detected cars
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()

		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])

		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		bbox_w = (bbox[1][0] - bbox[0][0])
		bbox_h = (bbox[1][1] - bbox[0][1])

		# Filter final detections for aspect ratios, e.g. thin vertical box is likely not a car
		aspect_ratio = bbox_w / bbox_h  # width / height
		#print('ar: %s' % (aspect_ratio,))

		# Also if small box "close" to the car (i.e. bounding box y location is high),
		# then probaby not a car
		bbox_area = bbox_w * bbox_h

		if bbox_area < small_bbox_area and bbox[0][1] > close_y_thresh:
			small_box_close = True
		else:
			small_box_close = False

		# Combine above filters with minimum bbox area filter
		if aspect_ratio > min_ar and aspect_ratio < max_ar and not small_box_close and bbox_area > min_bbox_area:
			# Draw the box on the image
			cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

	# Return the image
	return img


if __name__ == '__main__':
	svc = LinearSVC()
	X_scaler = StandardScaler()

	if use_pretrained:
		with open('model.p', 'rb') as f:
			save_dict = pickle.load(f)
		svc = save_dict['svc']
		X_scaler = save_dict['X_scaler']

		print('Loaded pre-trained model from model.p')
	else:
		print('Reading training data and training classifier from scratch')

		# with open('data.p', 'rb') as f:
		# 	data = pickle.load(f)
		# cars = data['vehicles']
		# notcars = data['non_vehicles']
		cars, notcars = load_data()
		train(cars, notcars, svc, X_scaler)

		print('Training complete, saving trained model to model.p')

		with open('model.p', 'wb') as f:
			pickle.dump({'svc': svc, 'X_scaler': X_scaler}, f)

	# Display predictions on all test_images
	imdir = 'test_images'
	for image_file in os.listdir(imdir):
		image = mpimg.imread(os.path.join(imdir, image_file))
		draw_image = np.copy(image)

		windows = slide_window(image, x_start_stop=(0, 1280), y_start_stop=(400, 656),
						xy_window=(96, 96), xy_overlap=(pct_overlap, pct_overlap))
		# windows = slide_window(image, x_start_stop=(0, 1280), y_start_stop=(400, 500),
		# 				xy_window=(96, 96), xy_overlap=(pct_overlap, pct_overlap))

		hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
								spatial_size=spatial_size, hist_bins=hist_bins,
								orient=orient, pix_per_cell=pix_per_cell,
								cell_per_block=cell_per_block,
								hog_channel=hog_channel, spatial_feat=spatial_feat,
								hist_feat=hist_feat, hog_feat=hog_feat)

		window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

		plt.imshow(window_img)
		plt.show()

		# Calculate and draw heat map
		heatmap = np.zeros((720, 1280))  # NOTE: Image dimensions hard-coded
		heatmap = add_heat(heatmap, hot_windows)
		heatmap = apply_threshold(heatmap, heatmap_thresh)
		labels = label(heatmap)
		print(labels[1], 'cars found')
		plt.imshow(labels[0], cmap='gray')
		plt.show()
		plt.savefig('cars_found')

		# Draw final bounding boxes
		draw_img = draw_labeled_bboxes(np.copy(image), labels)
		plt.imshow(draw_img)
		plt.show()
		plt.savefig('bounding_boxes')
