from flask import Flask,  send_file, request
from werkzeug.utils import secure_filename
from YoloDetector import YoloDetector
import numpy as np
from sklearn.cluster import KMeans
import cv2
from PIL import Image
import uuid



def draw_on_frame(frame, results):

	try :

		for cls, objs in results.items():
			ocnt = 0
			all_xs = []
			all_ys =[]
			coords = []

			for x1, y1, x2, y2 in objs:
				x = int((x1 + x2) / 2)
				y = int((y1 + y2) / 2)
				all_ys.append(y)
				all_xs.append(x)
				coords.append((x,y))
				ocnt += 1

			kmeans = KMeans(n_clusters=12)

		X = np.array(all_ys).reshape((-1, 1))
		kmeans.fit(X)
		rows = kmeans.labels_

		lines = {i:[] for i in range(12)}


		for i, (x1, y1) in zip(rows, coords):
			lines[i].append((x1,y1))

		count = 1
		sorted_lines = [lines[w] for w in range(12)]
		sorted_lines = sorted(sorted_lines, key=lambda x:x[0][1])

		for line in sorted_lines:
			line = sorted(line, key=lambda x:x[0])
			for (x, y) in line:
				cv2.circle(frame, (x,y), 3, (0, 255, 0), 2)
				cv2.putText(frame, str(count), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), thickness=2)
				count += 1

		return frame, ocnt

	except Exception as e:

		print("An exception occurred: ", e)
		count = 0
		for cls, objs in results.items():
			for x1, y1, x2, y2 in objs:
				x = int((x1+x2)/2)
				y = int((y1+y2)/2)
				cv2.circle(frame, (x,y), 3, (0, 255, 0), 2)
				cv2.putText(frame, str(count), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), thickness=1)
				count += 1
		return frame, count




def circleDetection(fileName):
	detector = YoloDetector("Model/circle.cfg", "Model/circle.weights", ["circle"])

	frame = cv2.resize(cv2.imread(fileName), (800, 800))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurredImage = cv2.GaussianBlur(gray, (5, 5), 0)
	Canny = cv2.Canny(blurredImage, 100, 250, apertureSize=3)
	kernel = np.ones((3))
	dilate = cv2.dilate(Canny, kernel, iterations=1)
	ret, thresh = cv2.threshold(dilate, 120, 255, cv2.THRESH_BINARY)
	gray = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
	results = detector.detect(gray, conf=0.2)
	frame = draw_on_frame(frame, results)
	print('No of pipe is : ', frame[1])
	cv2.imwrite(fileName, frame[0])

	return frame[1]


app = Flask(__name__,static_url_path="/uploads",static_folder='uploads')


@app.route('/', methods = ['POST'])
def api_root():
	print("came")
	if request.method == 'POST' and request.files['image']:

		file = request.files['image']
		img = Image.open(file.stream)
		img_name ='uploads/'+str(uuid.uuid4())+".jpg"
		img.save(img_name)
		print(img_name)
		counts = circleDetection(img_name)

		return "http://127.0.0.1:5000/"+img_name+","+str(counts)

	else:
		return "Image is not uploaded"


if __name__ == '__main__':
    #app.run(host='149.56.17.28', debug=False)
    app.debug = True
    app.run()
