from fastapi.testclient import TestClient
from api.service import app
import jwt
from datetime import datetime, timedelta
import requests

client = TestClient(app)

### Variables
JWT_SECRET_KEY = "secret_key_for_bentoml_exam"
JWT_ALGORITHM = "HS256"
TEST_FEATURES = {
  "GRE_Score": 331,
  "TOEFL_Score": 120,
  "University_Rating": 3,
  "SOP": 4,
  "LOR_": 4,
  "CGPA": 8.96,
  "Research": 1
}

### Create token to test jwt
def create_token(user_id: str, expire_delta_hours: int):
    exp_time = datetime.utcnow() + timedelta(hours=expire_delta_hours)
    payload = {
        "sub": user_id,
        "exp": int(exp_time.timestamp())
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


### Authentication tests
# Missing token
def test_auth_missing_token():
    response = client.get("/auth-test")
    assert response.status_code == 401
    assert response.json() == {"detail": "Missing authentication token"}

# Expired token
def test_auth_expired_token():
    expired_token = create_token("user_test", expire_delta_hours=-1)
    response = client.get("/auth-test", headers={"Authorization": f"Bearer {expired_token}"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Token has expired"}

# Invalid token
def test_auth_invalid_token():
    response = client.get("/auth-test", headers={"Authorization": "Bearer FAUX_TOKEN"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}

# Valid token
def test_auth_valid_token():
    valid_token = create_token("user_test", expire_delta_hours=1)
    response = client.get("/auth-test", headers={"Authorization": f"Bearer {valid_token}"})
    assert response.status_code == 200
    assert response.json() == {"user": "user_test"}


### API connection tests

# Giving valid token for known users
def test_known_user():
    response = client.post("/login", json={"username": "user_test",
                                           "password": "DataScientest_mle"})
    assert response.status_code == 200
    assert "token" in response.json()

# Bad username test
def test_bad_username():
    response = client.post("/login", json={"username": "baduser",
                                            "password": "DataScientest_mle"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"

# Bad password test
def test_bad_username():
    response = client.post("/login", json={"username": "user_test",
                                            "password": "badpassword"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"


### API prediction tests

# Missing token on predict endpoint
def test_missing_token_predict():
    response = client.post("/predict", json=TEST_FEATURES)
    assert response.status_code == 401
    assert response.json() == {"detail": "Missing authentication token"}

# Correct prediction test
def test_correct_prediction():
    login_response = requests.post("http://localhost:3000/login", json={
        "username": "user_test",
        "password": "DataScientest_mle"
    })
    assert login_response.status_code == 200
    token = login_response.json()["token"]

    pred_response = requests.post(
        "http://localhost:3000/predict",
        json=TEST_FEATURES,
        headers={"Authorization": f"Bearer {token}"}
    )
    assert pred_response.status_code == 200
    assert pred_response.json()['admit_probability'] == 0.8600393935330517

# Incorrect inputs return error
def test_incorrect_prediction():
    login_response = requests.post("http://localhost:3000/login", json={
        "username": "user_test",
        "password": "DataScientest_mle"
    })
    assert login_response.status_code == 200
    token = login_response.json()["token"]
    pred_response = requests.post(
        "http://localhost:3000/predict",
        json={"GRE": 3, "TOEFL": "toto",},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert pred_response.status_code == 422
    details = pred_response.json()['detail']
    assert isinstance(details, list)
