import pandas as pd
import yt_dlp
from fer import FER

from differences import *
from collections import defaultdict
from ultralytics import YOLO



def yoloWork(model, frame):
    unique_objects = []


    results = model(frame, stream=False)
    for result in results:
        for obj in result.boxes:
            obj_class = obj.cls
            obj_name = model.names[int(obj_class)]

            if obj_name not in unique_objects:
                unique_objects.append(obj_name)
    return unique_objects



def process_video(video_url, model, model_ocr, emotion_model):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'quiet': True,
        'noplaylist': True,
    }

    info = yt_dlp.YoutubeDL(ydl_opts).extract_info(video_url, download=False)
    video_path = info["formats"][0]["manifest_url"]

    video_cv = cv2.VideoCapture(video_path)
    video_cv.set(3, 1280)
    video_cv.set(4, 700)

    length = int(video_cv.get(cv2.CAP_PROP_FRAME_COUNT))

    # max_possible_sum = 512 * 512 * 255
    k = 0
    sfps = 3

    ret, prev_frame = video_cv.read()

    prev_frame = calculate_histogram(transform_image(prev_frame)[0])
    separator = 0.7
    alpha = 1

    last_nonrepeating = 0

    current_frame_dict = defaultdict(int)
    current_frame_dict[last_nonrepeating] += 1
    unique_objects_dict = defaultdict(list)
    text_on_images_dict = defaultdict(str)
    emotions_on_images_dict = defaultdict(list)

    i = 1
    while i < length:
        i += 1
        ret, frame = video_cv.read()
        if not ret:
            break
        if i % sfps == 0:
            frame, frame_rgb = transform_image(frame)
            coef, separator, prev_frame = histogram_diff(frame, prev_frame, separator, alpha)

            if coef < separator:
                k += 1
                last_nonrepeating = i-1

                unique_objects = yoloWork(model, frame_rgb)
                text = model_ocr.readtext(frame)
                detected_emotions = emotion_model.detect_emotions(frame_rgb)

                if detected_emotions:
                    emotions_on_images_dict[str(last_nonrepeating)].append(detected_emotions)
                if text:
                    text_on_images_dict[str(last_nonrepeating)] = text
                if unique_objects:
                    unique_objects_dict[str(last_nonrepeating)].append(unique_objects)
            current_frame_dict[str(last_nonrepeating)] += 1

            # process.set_description(f"coef: {coef}, separator: {separator}")
    return current_frame_dict, unique_objects_dict, text_on_images_dict, emotions_on_images_dict


# model = YOLO("yolov10s.pt", verbose=False)
# import easyocr
# reader = easyocr.Reader(['ru', 'en'])
# print(process_video("https://rutube.ru/video/a76a751510b5f57a80f351d61207b186/?r=plwd", model, reader, FER))
