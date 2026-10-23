from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CORE Fitness AI Backend")

# Configurar CORS para permitir que tu Frontend (index.html) se comunique sin bloqueos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "online", "message": "El núcleo central de CORE está activo en la nube."}

@app.post("/generar-rutina")
def generar_rutina():
    return {"mensaje": "Ruta lista para conectar con Gemini y Firebase"}
