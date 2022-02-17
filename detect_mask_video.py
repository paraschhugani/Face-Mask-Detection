# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import datetime

import face_recognition

import os , psutil
pid = os.getpid()
CodePS = psutil.Process(pid)

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

def saveJpgImage(frame):
    #process image
	print("imwrite called")
	img_name = "/Users/paraschhugani/Desktop/Face_mask_Project/SampleData/opencv_frame_{}.jpg".format(datetime.datetime.now())
	cv2.imwrite(img_name, frame)
	# SaveFace(frame)
	return img_name.replace("/Users/paraschhugani/Desktop/Face_mask_Project/SampleData/","")

def SaveFace(frame):
	obama_image = face_recognition.load_image_file(frame)
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# Age Model And detector
age_proto = "age_deploy.prototxt"
age_model = "age_net.caffemodel"

# Gender Model and detector
gender_proto = "gender_deploy.prototxt"
gender_model = "gender_net.caffemodel"

print("[INFO] loading Age and Gender Detector model")
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)
age_list = ['(0-2 age)', '(4-6 age)', '(8-12 age)', '(15-20 age)', '(25-32 age)', '(38-43 age)', '(48-53 age)', '(60-100 age)']
gender_list = ['Female', 'Male']
Age_Gen_model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


pe = 0

# Initialize some variables for face recognistion
paras_image = face_recognition.load_image_file("paras.jpeg")
paras_face_encoding = face_recognition.face_encodings(paras_image)[0]
known_face_encodings = [paras_face_encoding]
known_face_names = ["Paras"]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# face recognition def
# video_capture = cv2.VideoCapture(0)
# ret, VC_frame = video_capture.read()
def dataFace(frame):

	# print("dataFace called")

	# this is used just to make new camera portal for better results but very slow
	video_capture = cv2.VideoCapture(0)
	ret, VC_frame = video_capture.read()

	small_frame = cv2.resize(VC_frame, (0, 0), fx=0.25, fy=0.25)

	# Convert the image from BGR color (which  OpenCV uses) to RGB color (which face_recognition uses)
	rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
	if process_this_frame:
		# print("Step 1")
		face_locations = face_recognition.face_locations(rgb_small_frame)
		# print("face location printing " + str(face_locations))
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

		face_names = []
		for face_encoding in face_encodings:
			# print("Step 2 , for loop")

            # See if the face is a match for the known face(s)
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
			name = "Unknown"
			# print(matches)

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
			face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
			# print(face_distances)
			try:

				best_match_index = np.argmin(face_distances)
				if matches[best_match_index]:
					# print("If face called")
					name = known_face_names[best_match_index]
				else:
					print("else called to save face")
					# Saving the image for face rgnition
					Temp_image_name = saveJpgImage(frame)

					Temp_Image_load = face_recognition.load_image_file("../SampleData/"+Temp_image_name)
					Temp_face_encoding = face_recognition.face_encodings(Temp_Image_load)[0]

					known_face_encodings.append(Temp_face_encoding)
					known_face_names.append(Temp_image_name.replace(".jpg",""))
					# print(known_face_encodings)

					# printing at thime of save face
					CodeMemoryUse = CodePS.memory_full_info()
					print("%.2f  MB" % float(CodeMemoryUse.rss*0.000001))
			except:
				print("except called")


# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)


	# printin the timezone
	# print(datetime.datetime.now(tz=datetime.timezone.utc))


	#printing process memory usage
	# CodeMemoryUse = CodePS.memory_full_info()
	# print("%.2f  MB" % float(CodeMemoryUse.rss*0.000001))

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		#calling face detectors to save people faces
		# print("printing box " + str(box))
		# dataFace(frame)

		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		tempblob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), Age_Gen_model_mean_values, swapRB = False)
		gender_net.setInput(tempblob)
		gender_pred = gender_net.forward()
		# print(gender_pred)
		gender = gender_list[gender_pred[0].argmax()]

		age_net.setInput(tempblob)
		age_pred = age_net.forward()
		age = age_list[age_pred[0].argmax()]

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		Gcolor = (255, 0, 0) if gender == "Male" else (255, 0, 242)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		Glabel = "{} {}".format(gender , age)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 2)

		cv2.putText(frame, Glabel, (startX, startY - 25),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, Gcolor, 2)

		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		# saveJpgImage(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break





# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
