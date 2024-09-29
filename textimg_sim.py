import pylab as pl
import requests
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from fer import Video
from fer import FER

from video_processing import process_video
from ultralytics import YOLO
import easyocr


class S2T:
    def __init__(self, model_id="openai/whisper-small"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=False, use_safetensors=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=15,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        self.sentiment_analyzer = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = stopwords.words('russian')

    def transcribe_audio(self, audio_file):
        result = self.pipe(audio_file)
        return result['text']

    def process_text(self, text):
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        dense = tfidf_matrix.todense()

        top_keywords = []
        for i in range(len(dense)):
            sorted_items = dense[i].argsort().tolist()[0][-5:]
            top_keywords.append([feature_names[idx] for idx in sorted_items])

        return sentences, top_keywords, tfidf_matrix

    def rank_sentences(self, sentences, tfidf_matrix):
        similarity_matrix = cosine_similarity(tfidf_matrix)
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        key_sentences = [ranked_sentences[i][1] for i in range(min(10, len(ranked_sentences)))]

        return key_sentences

    def analyze_sentiment(self, sentences):
        sentiment_results = []
        for sentence in sentences:
            sentiment_result = self.sentiment_analyzer(sentence)
            sentiment_results.append({
                "sentence": sentence,
                "sentiment": sentiment_result
            })
        return sentiment_results

    def process_audio(self, audio_file):
        text = self.transcribe_audio(audio_file)
        sentences, top_keywords, tfidf_matrix = self.process_text(text)
        ranked_sentences = self.rank_sentences(sentences, tfidf_matrix)
        sentiment_results = self.analyze_sentiment(sentences)

        return {
            "transcribed_text": text,
            "ranked_sentences": ranked_sentences,
            "sentiment_analysis": sentiment_results
        }

# def get_otr():
#     reader = easyocr.Reader(['ru', 'en'])
#     model = YOLO("yolov10s.pt", verbose=False)
#     current_frame_dict, unique_objects_dict, text_on_images_dict = process_video("https://rutube.ru/video/a76a751510b5f57a80f351d61207b186/?r=plwd", model, reader)
# def text_img_sim(img, text):
#     pass

class BaseModel():
    def __init__(self):
        self.s2t_model = S2T()
        self.ocr_reader = easyocr.Reader(['ru', 'en'])
        self.model_video_detection = YOLO("yolov10s.pt", verbose=False)
        self.video_frames_skipping_model = process_video

    def process_text_and_skipping(self, audio, url):
        text = self.s2t_model.transcribe_audio(audio)
        audio_keys = self.s2t_model.process_audio(audio)
        current_frame_dict, unique_objects_dict, text_on_images_dict = self.get_video_features(url)
        all_json = {"text": text}
        all_json.update(**audio_keys, **current_frame_dict, **unique_objects_dict, **text_on_images_dict)
        return all_json


    def get_video_features(self, url):
        return process_video(url, self.model_video_detection, self.ocr_reader)


