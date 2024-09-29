from flask import Flask, request, render_template
from video_processor import VideoProcessor

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_file = request.files.get('video_file')
        video_link = request.form.get('video_link')
        if video_file:
            video_processor = VideoProcessor(video_file)
            print(1)
            return 'file'
        elif video_link:
            video_processor = VideoProcessor(video_link)
            return 'link'
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)