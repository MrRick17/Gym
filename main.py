import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai

app = FastAPI(title="CORE Fitness AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. CONEXIÓN SECRETA CON FIREBASE
try:
    firebase_creds = os.environ.get("FIREBASE_CREDENTIALS")
    if firebase_creds:
        cred_dict = json.loads(firebase_creds)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Conectado a Firebase exitosamente.")
except Exception as e:
    print(f"Error conectando a Firebase: {e}")

# 2. CONEXIÓN SECRETA CON GEMINI IA
model = None
try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
 
        # Intentamos inicializar el modelo estándar
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("Motor IA Gemini inicializado correctamente.")
except Exception as e:
    print(f"Error inicializando Gemini: {e}")

class RutinaRequest(BaseModel):
    uid: str

@app.get("/")
def home():
    return {"status": "online", "message": "El núcleo central de CORE está operativo."}

@app.post("/generar-rutina")
def generar_rutina(req: RutinaRequest):
    global model
    try:
        user_ref = db.collection('usuarios').document(req.uid)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="Usuario no encontrado.")
            
        user_data = user_doc.to_dict()
        objetivo = user_data.get('objetivo', 'Recomposición')
        peso = user_data.get('historial_pesos', [0])[-1]
        meta = user_data.get('peso_meta', 0)
        altura = user_data.get('altura', 0)
        edad = user_data.get('edad', 0)

        prompt = f"""
        Actúa como un entrenador personal experto y nutricionista deportivo de élite.
        Diseña una rutina semanal estructurada y una directriz nutricional para este cliente:
        - Edad: {edad} años
        - Altura: {altura} cm
        - Peso actual: {peso} kg
        - Peso meta: {meta} kg
        - Objetivo principal: {objetivo}
        
        Devuelve ÚNICAMENTE un objeto JSON válido con esta estructura exacta, sin texto adicional, sin bloque de código (markdown):
        {{
            "rutina": [
                {{"dia": "Día 1: Upper Power", "ejercicios": ["Bench Press: 4x5", "Rows: 4x6"]}},
                {{"dia": "Día 2: Lower Power", "ejercicios": ["Squats: 4x5", "Deadlifts: 3x5"]}},
                {{"dia": "Día 3: Descanso Activo", "ejercicios": ["Cardio ligero 30 min"]}}
            ],
            "nutricion": "Aquí tu directriz nutricional detallada con macros estimados, alimentos recomendados y calorías objetivo."
        }}
        """
        
        if model is None:
            raise HTTPException(status_code=500, detail="El modelo de IA no está disponible.")

        response = model.generate_content(prompt)
        texto_limpio = response.text.replace('```json', '').replace('```', '').strip()
        datos_ia = json.loads(texto_limpio)
        
        user_ref.update({
            'plan_entrenamiento': datos_ia['rutina'],
            'plan_nutricion': datos_ia['nutricion']
        })
        
        return {"status": "success", "message": "Plan analizado y actualizado exitosamente."}
        
    except Exception as e:
        print(f"Error procesando rutina: {e}")
        raise HTTPException(status_code=500, detail=str(e))
