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
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("Motor IA Gemini inicializado correctamente.")
except Exception as e:
    print(f"Error inicializando Gemini: {e}")

class RutinaRequest(BaseModel):
    uid: str

class ChatRequest(BaseModel):
    uid: str
    mensaje: str

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
        frecuencia = user_data.get('dias_entrenamiento', 'Lunes a Viernes')
        peso = user_data.get('historial_pesos', [0])[-1]
        meta = user_data.get('peso_meta', 0)
        altura = user_data.get('altura', 0)
        edad = user_data.get('edad', 0)

        prompt = f"""
        Actúa como un entrenador personal de élite y nutricionista deportivo avanzado.
        Diseña un plan de entrenamiento estructurado exactamente para la frecuencia de **{frecuencia}** y una directriz nutricional precisa para este atleta:
        - Edad: {edad} años
        - Altura: {altura} cm
        - Peso actual: {peso} kg
        - Peso meta: {meta} kg
        - Enfoque principal: {objetivo}
        - Disponibilidad semanal: {frecuencia}

        Requisitos estrictos:
        1. Si la frecuencia es 'Lunes a Viernes', genera exactamente 5 días. Si es 'Lunes a Sábado', genera exactamente 6 días.
        2. CADA DÍA DEBE INDICAR CLARAMENTE LOS GRUPOS MUSCULARES TRABAJADOS en el título (Ejemplo: 'Día 1: Pecho y Tríceps', 'Día 2: Espalda y Bíceps').
        3. Incluye ejercicios compuestos y de aislamiento con series y repeticiones profesionales (ej. 4x6-8).
        4. La nutrición debe incluir directrices claras, calorías, y mención de alimentos limpios (arepas, pechuga de pollo, huevos, pasta y papas).
        
        Devuelve ÚNICAMENTE un objeto JSON válido con esta estructura exacta, sin texto adicional, sin bloques de código markdown:
        {{
            "rutina": [
                {{"dia": "Día 1: Pecho y Tríceps (Lunes)", "ejercicios": ["Press de banca plano: 4x6-8", "Press inclinado con mancuernas: 3x8-10", "Aperturas en polea: 3x12", "Extensiones de tríceps en polea: 4x10-12"]}},
                {{"dia": "Día 2: Espalda y Bíceps (Martes)", "ejercicios": ["Jalón al pecho: 4x8-10", "Remo con barra: 4x6-8", "Remo en polea baja: 3x10", "Curl con barra EZ: 3x10"]}}
            ],
            "nutricion": "Directriz nutricional detallada enfocada en tu objetivo, especificando macros y alimentos base (arepas, pechuga de pollo, huevos, pasta y papas)."
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

@app.post("/chat")
def chat_asistente(req: ChatRequest):
    global model
    try:
        user_doc = db.collection('usuarios').document(req.uid).get()
        user_data = user_doc.to_dict() if user_doc.exists else {}
        nombre = user_data.get('nombre', 'Atleta')
        objetivo = user_data.get('objetivo', 'Hipertrofia')
        
        chat_prompt = f"""
        Eres Jarvis, el asistente de inteligencia artificial experto en fitness, nutrición y culturismo de la plataforma CORE Fitness.
        Estás hablando con {nombre}, cuyo objetivo principal es {objetivo}.
        Responde a su mensaje de forma motivadora, profesional, directa y concisa (máximo 3-4 frases o viñetas si es necesario):
        Mensaje del usuario: {req.mensaje}
        """
        if model is None:
            raise HTTPException(status_code=500, detail="Modelo no disponible.")
        response = model.generate_content(chat_prompt)
        return {"status": "success", "respuesta": response.text.strip()}
    except Exception as e:
        print(f"Error en chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
