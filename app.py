from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
from nst import run_style_transfer  # You’ll create this function in nst.py

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    styled_image_path = None
    if request.method == 'POST':
        content = request.files['content']
        style = request.files['style']
        content_filename = secure_filename(content.filename)
        style_filename = secure_filename(style.filename)
        content.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
        style.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'output.jpg')
        run_style_transfer(
            os.path.join(app.config['UPLOAD_FOLDER'], content_filename),
            os.path.join(app.config['UPLOAD_FOLDER'], style_filename),
            result_path
        )
        styled_image_path = 'results/output.jpg'
    return render_template('index.html', styled_image_path=styled_image_path)

@app.route('/results/<filename>')
def result_file(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)