### Import libraries
import bentoml
from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel
from fastapi.security import HTTPBearer
import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from datetime import datetime, timedelta
import pandas as pd

### FastAPI instance
app = FastAPI(title="Admission prediction", openapi_tags=[
    {
        'name': 'home',
        'description': 'Server informations'
    },
    {
        'name': 'test',
        'description': 'Returns Pytest results'
    },
    {
        'name': 'login',
        'description': 'Login page, must be registered user to predict.'
    },
    {
        'name': 'predict',
        'description': 'Prediction for admission'
    }
])

### Secret key and algorithm to authenticate
JWT_SECRET_KEY = "secret_key_for_bentoml_exam"
JWT_ALGORITHM = "HS256"

### Users
USERS = {
    "user123": "password123",
    "user456": "password456",
    "user_test": "DataScientest_mle"
}

### Authentication
class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path in ["/predict", "/auth-test"]:
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})
            try:
                token = token.split()[1]  # Remove 'Bearer ' prefix
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
            request.state.user = payload.get("sub")
        response = await call_next(request)
        return response


def create_jwt_token(user_id: str):
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {
        "sub": user_id,
        "exp": expiration
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token

### Add middleware to fastAPI
app.add_middleware(JWTAuthMiddleware)

### Getting model
model_runner = bentoml.sklearn.get("admission_lr:latest").to_runner()

### Service for API
svc = bentoml.Service("admission_api")
   
### Pydantic schemas
class LoginRequest(BaseModel):
    username: str
    password: str

class InputModel(BaseModel):
    GRE_Score: int
    TOEFL_Score: int
    University_Rating: int
    SOP: float
    LOR_: float
    CGPA: float
    Research: int

class PredictResponse(BaseModel):
    admit_probability: float

### FastAPI endpoints
@app.get("/home", tags=['home'])
def home():
    """ Bienvenue. """
    return {"message": "Examen Bentoml de Jérôme LE GAL",
            "session": "avr25_bootcamp_mle",
            "organisme": "DataScientest",
            "endpoints": ["/", "/login", "/predict", "/verify", "/docs"],
            "auth": "Utiliser /login afin de pouvoir réaliser des prédictions sur /predict"
            }

@app.post("/login", tags=['login'])
def login(credentials: LoginRequest):
    """ Interface d'autentification. """
    username = credentials.username
    password = credentials.password
    if username in USERS and USERS[username] == password:
        token = create_jwt_token(username)
        return {"token": token}
    else:
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})


@app.post("/predict", tags=['predict'], response_model=PredictResponse,
          dependencies=[Depends(HTTPBearer())])
async def prediction(features: InputModel, request: Request):
    """ Renvoi la probabilité d'admission. """
    user = request.state.user if hasattr(request.state, 'user') else None
    df = pd.DataFrame([{
        "GRE Score": features.GRE_Score,
        "TOEFL Score": features.TOEFL_Score,
        "University Rating": features.University_Rating,
        "SOP": features.SOP,
        "LOR ": features.LOR_,
        "CGPA": features.CGPA,
        "Research": features.Research
    }])
    proba = await model_runner.predict.async_run(df)
    return {"admit_probability": proba}


@app.get("/verify", tags=['test'])
def verify():
    """ Vérifie que l'API est fonctionnelle. """
    return {"message": "L'API est fonctionnelle."}

@app.get("/auth-test", tags=['test'])
async def auth_test(request: Request):
    """ Route factice protégée par le middleware JWT pour test """
    user = request.state.user if hasattr(request.state, 'user') else None
    return {"user": user}

svc.mount_asgi_app(app, path="/")