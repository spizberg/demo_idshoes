import torch
import time
from flask import Flask, request, jsonify
from utilities import load_detector_torchscript, get_classes, convert_bytes_to_np, MODELS_FOLDER, predict_torchscript, load_classifier_torchscript


app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
detector_model = load_detector_torchscript(device)
classifier_model = None
list_classes = None
first_time = True

@app.route('/initialize/<marque>/', defaults={'model_type': None}, methods=['GET'])
@app.route('/initialize/<marque>/<model_type>/', methods=['GET'])
def initializeModel(marque, model_type):
    try:
        global classifier_model, list_classes, first_time
        class_filepath = f"classes/{marque}_classes.txt"
        list_classes = get_classes(class_filepath)        
        classifier_model = load_classifier_torchscript(marque, device, True if model_type else False)
        first_time = True
        return jsonify({"answer": "Good"})
    except:
        return jsonify({"answer": "Bad"})


@app.route('/predict/', methods=['POST'])
def makePrediction():
    global first_time
    r = request
    images = convert_bytes_to_np(r.data)
    start = time.time()
    if not first_time:
        predictions = predict_torchscript(images, detector_model, classifier_model, list_classes, device)
    else:
        for _ in range(10):
            predictions = predict_torchscript(images, detector_model, classifier_model, list_classes, device)
            first_time = False
    print(f"Temps écoulé: {time.time() - start}")
    torch.cuda.empty_cache()
    return jsonify(predictions)


if __name__ == '__main__':
    app.run()
