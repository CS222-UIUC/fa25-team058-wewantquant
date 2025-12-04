import pytest
from app.backendapp import flaskapp as app
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# ============================================================================
# INDEX PAGE TESTS
# ============================================================================

def test_index_page_loads_and_has_nav(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Welcome' in response.data or b'CS 222' in response.data
    assert b'Home' in response.data

# ============================================================================
# INFO PAGE TESTS
# ============================================================================

def test_index_page_loads_and_has_nav(client):
    response = client.get('/info')
    assert response.status_code == 200
    assert b'the team' in response.data
    assert b'Home' in response.data


# ============================================================================
# PREDICT PAGE TESTS (Combined)
# ============================================================================

def test_predict_page_content(client):
    response = client.get('/predict')
    assert response.status_code == 200

    # Form elements
    assert b'model-select' in response.data
    assert b'ticker-input' in response.data
    assert b'days-input' in response.data
    assert b'predict-btn' in response.data

    # Model options
    assert b'LSTM' in response.data
    assert b'Random Forest' in response.data
    assert b'Linear Regression' in response.data
    assert b'ARIMA' in response.data
    assert b'Prophet' in response.data


# ============================================================================
# PREDICT API ENDPOINT TESTS
# ============================================================================

def test_predict_post_success(client):
    response = client.post('/predict',
                            data=json.dumps({
                                'model': 'lstm',
                                'ticker': 'AAPL',
                                'days': 7
                            }),
                            content_type='application/json')

    assert response.status_code == 200
    data = json.loads(response.data)

    assert 'ticker' in data
    assert 'model_name' in data
    assert 'current_price' in data
    assert 'predictions' in data
    assert isinstance(data['predictions'], list)
    assert len(data['predictions']) == 7


def test_predict_post_different_days(client):
    for days in [0, 1, 7, 14, 30, 50, 100]:
        response = client.post('/predict',
                               data=json.dumps({
                                   'model': 'lstm',
                                   'ticker': 'AAPL',
                                   'days': days
                               }),
                               content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['predictions']) == days


def test_predict_post_all_models(client):
    models = ['lstm', 'random_forest', 'linear_regression', 'arima', 'prophet']
    for model in models:
        response = client.post('/predict',
                               data=json.dumps({
                                   'model': model,
                                   'ticker': 'AAPL',
                                   'days': 7
                               }),
                               content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'model_name' in data


def test_predict_post_different_tickers(client):
    tickers = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN']
    for ticker in tickers:
        response = client.post('/predict',
                               data=json.dumps({
                                   'model': 'lstm',
                                   'ticker': ticker,
                                   'days': 7
                               }),
                               content_type='application/json')
        assert response.status_code == 200
        assert json.loads(response.data)['ticker'] == ticker


# ============================================================================
# ERROR HANDLING TESTS (Combined)
# ============================================================================

def test_predict_invalid_inputs(client):
    invalid_payloads = [
        {},  # nothing
        {'ticker': 'AAPL', 'days': 7},  # missing model
        {'model': 'lstm', 'days': 7},  # missing ticker
        {'model': '', 'ticker': 'AAPL', 'days': 7},  # empty model
        {'model': 'lstm', 'ticker': '', 'days': 7},  # empty ticker
    ]

    for payload in invalid_payloads:
        response = client.post('/predict',
                               data=json.dumps(payload),
                               content_type='application/json')
        assert response.status_code == 400


def test_predict_post_invalid_json(client):
    response = client.post('/predict',
                           data='not valid json',
                           content_type='application/json')
    assert response.status_code in [400, 500]


def test_predict_post_missing_content_type(client):
    response = client.post('/predict',
                           data=json.dumps({
                               'model': 'lstm',
                               'ticker': 'AAPL',
                               'days': 7
                           }))
    assert response.status_code in [200, 400, 415]


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_predict_post_max_days(client):
    """Days > 100 should be capped at 100"""
    response = client.post('/predict',
                           data=json.dumps({
                               'model': 'lstm',
                               'ticker': 'AAPL',
                               'days': 101
                           }),
                           content_type='application/json')

    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data['predictions']) == 100  # capped


def test_predict_post_min_days(client):
    """Days <= 0 should default to 1"""
    for days in [0, -1, -5, -100]:
        response = client.post('/predict',
                               data=json.dumps({
                                   'model': 'lstm',
                                   'ticker': 'AAPL',
                                   'days': days
                               }),
                               content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['predictions']) == 1  # minimum forced



def test_predict_post_lowercase_ticker(client):
    response = client.post('/predict',
                           data=json.dumps({
                               'model': 'lstm',
                               'ticker': 'aapl',
                               'days': 7
                           }),
                           content_type='application/json')
    assert response.status_code == 200


def test_predict_post_special_characters_ticker(client):
    response = client.post('/predict',
                           data=json.dumps({
                               'model': 'lstm',
                               'ticker': 'BRK.B',
                               'days': 7
                           }),
                           content_type='application/json')
    assert response.status_code in [200, 400]


# ============================================================================
# HTTP METHOD TESTS
# ============================================================================

def test_predict_methods_not_allowed(client):
    assert client.put('/predict').status_code == 405
    assert client.delete('/predict').status_code == 405
    assert client.post('/').status_code == 405
    assert client.post('/info').status_code == 405


# ============================================================================
# INTEGRATION TEST
# ============================================================================

def test_full_user_flow(client):
    assert client.get('/').status_code == 200
    assert client.get('/predict').status_code == 200

    response = client.post('/predict',
                           data=json.dumps({
                               'model': 'lstm',
                               'ticker': 'AAPL',
                               'days': 7
                           }),
                           content_type='application/json')
    assert response.status_code == 200
    assert len(json.loads(response.data)['predictions']) == 7
