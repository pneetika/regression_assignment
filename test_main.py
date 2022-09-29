import unittest
import flask
from main import predict, get_gcp_response
from werkzeug.datastructures import ImmutableMultiDict
import pytest

project_id = "boston-price-pred"
model_name = "boston_housing_price_pred"
model_version = "v1"
import unittest

class TestGCPHousePricePrediction(unittest.TestCase):
    def test_get_gcp_response(self):
        features = [0.09178, 0.0, 4.05, 0.0, 0.51, 6.416, 84.1, 2.6463, 5.0, 296.0, 16.6, 395.5, 9.04]
        gcp_endpoint = "https://us-central1-ml.googleapis.com"
        response = get_gcp_response(features, gcp_endpoint, project_id, model_name, model_version)

        assert isinstance(response['predictions'], list)


    def test_get_gcp_valid_response(self):
        features = [0.09178, 0.0, 4.05, 0.0, 0.51, 6.416, 84.1, 2.6463, 5.0, 296.0, 16.6, 395.5, 9.04]
        gcp_endpoint = "https://us-central1-ml.googleapis.com"
        response = get_gcp_response(features, gcp_endpoint, project_id, model_name, model_version)

        assert len(response['predictions']) > 0

    def test_get_gcp_valid_response_value(self):
        features = [0.09178, 0.0, 4.05, 0.0, 0.51, 6.416, 84.1, 2.6463, 5.0, 296.0, 16.6, 395.5, 9.04]
        gcp_endpoint = "https://us-central1-ml.googleapis.com"
        response = get_gcp_response(features, gcp_endpoint, project_id, model_name, model_version)

        assert round(response['predictions'][0], 2) == 23.09

if __name__ == '__main__':
    unittest.main()


