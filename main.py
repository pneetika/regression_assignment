from flask import Flask, request, render_template
from google.api_core.client_options import ClientOptions
from googleapiclient import discovery
import json

app = Flask(__name__)
CONFIG = dict()

@app.route('/')
def home():

    with open("./configs/config.json") as cfg:
        global CONFIG
        CONFIG = json.load(cfg)
    print(CONFIG)
    return render_template('index.html')

def get_gcp_response(features_vals, endpoint, project_id, model_name, model_version):
    client_options = ClientOptions(api_endpoint=endpoint)
    ml = discovery.build('ml', 'v1', client_options=client_options)

    request_body = {'instances': [features_vals]}
    gcp_request = ml.projects().predict(
        name='projects/{0}/models/{1}/versions/{2}'.format(project_id,
                                                           model_name,
                                                           model_version),
        body=request_body)

    response = gcp_request.execute()
    print(response)
    return response

@app.route('/predict', methods=['POST'])
def predict():

    features = request.form.to_dict()
    if features["all"]:
        all_attrs = features["all"]
        features_vals = all_attrs.split(', ')
        features_vals = list(map(float, features_vals))
    else:
        features.pop("all")
        features_txt = list(features.values())
        features_vals = [float(f) for f in features_txt]

    global CONFIG
    endpoint = CONFIG["gcp_endpoint"]
    project_id = CONFIG["project_id"]
    model_name = CONFIG["model_name"]
    model_version = CONFIG["model_version"]

    print("GCP Endpoint loaded ", endpoint)

    response = get_gcp_response(features_vals, endpoint, project_id, model_name, model_version)

    return render_template('index.html', prediction_text='House Price Predicted (in $1000\'s):        {0}'.format(response['predictions'][0]))


if __name__ == "__main__":
    app.run(debug=True)