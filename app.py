import streamlit as st
# import subprocess




# Fungsi untuk instalasi requirements
# @st.cache(allow_output_mutation=True)
# def install_requirements():
#     subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

# # Install requirements hanya sekali di awal
# install_requirements()
# from streamlit_webrtc import webrtc_streamer
import cv2
from PIL import Image
import numpy as np

def perform_object_detection(image_path):
  config_file= 'models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
  frozen_model='models/frozen_inference_graph.pb'
  model = cv2.dnn_DetectionModel(frozen_model, config_file)

  classLabels = []
  file_names='models/labels.txt'
  with open(file_names, 'rt') as fpt:
      classLabels = fpt.read().rstrip('\n').split('\n')
  model.setInputSize(320,320)
  model.setInputScale(1.0/127.5)
  model.setInputMean((127.5,127,5,127.5))
  model.setInputSwapRB(True)
  # img=np.array(image_path)
  # img = cv2.imread(img)
  # Convert the PIL Image to NumPy array
  img_np = np.array(image_path)

  # Ensure that the image is in BGR format (OpenCV format)
  if img_np.shape[-1] == 4:
      img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
  else:
      img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

  ClassIndex, confidece, bbox= model.detect(img_np,confThreshold=0.5)
  font_scale=3
  font= cv2.FONT_HERSHEY_PLAIN
  for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
      cv2.rectangle(img_np, boxes, (255,0,0), 2)
      cv2.putText(img_np, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0,255,0), thickness=3)
  return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

# Streamlit web app code
st.title('Object Detection Coco 2020')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image file
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform object detection on button click
    if st.button('Run Object Detection'):
        result_image = perform_object_detection(image)
        # Display the result image
        st.subheader('Result')
        st.success('Success !',icon="❤️")
        st.image(result_image, caption='Result', use_column_width=True)
