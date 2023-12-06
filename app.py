from flask import Flask, render_template, request, jsonify
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

model_name = 'Helsinki-NLP/tiny-marian-mt-en-de'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

def translate_text(text, target_lang='de'):
    inputs = tokenizer(text, return_tensors="pt")
    translation = model.generate(**inputs, target_lang=target_lang)
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    return translated_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    if request.method == 'POST':
        data = request.get_json()
        text_to_translate = data.get('text_to_translate')
        target_lang = data.get('target_lang', 'de')  # Default to German if not specified

        if not text_to_translate:
            return jsonify(error='Text to translate is required'), 400

        translated_text = translate_text(text_to_translate, target_lang)
        return jsonify(translation=translated_text)

if __name__ == '__main__':
    app.run(debug=True)
