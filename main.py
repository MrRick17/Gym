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

# 1. CONEXIÓN FIREBASE
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

# 2. CONEXIÓN GEMINI
model = None
try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-3.8-flash')
        print("Motor IA Gemini inicializado correctamente.")
except Exception as e:
    print(f"Error inicializando Gemini: {e}")

class RutinaRequest(BaseModel):
    uid: str

class ChatRequest(BaseModel):
    uid: str
    mensaje: str

class ProgresoRequest(BaseModel):
    uid: str
    peso: float
    fotos_b64: list[str] = []

@app.get("/")
def home():
    return {"status": "online", "message": "CORE Operativo."}

# ==========================================
# RUTINA Y NUTRICIÓN CENTRAL
# ==========================================
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
        Actúa como un entrenador personal de élite. Diseña un plan estructurado para la frecuencia **{frecuencia}**:
        - Peso actual: {peso} kg | Meta: {meta} kg | Enfoque: {objetivo}

        REGLAS ESTRICTAS:
        1. Genera los días exactos según la frecuencia (5 o 6 días). Específica los grupos musculares en el título.
        2. Ejercicios compuestos y de aislamiento con series/repeticiones.
        3. LA NUTRICIÓN NO DEBE CONTENER PLATILLOS NI RECETAS. Debe ser una lista directa y estricta indicando:
           - Fuentes de Proteínas, Carbohidratos y Grasas a consumir.
           - Alimentos PROHIBIDOS (ej. azúcar, fritos).
           - Resumen de Macros.
        
        Devuelve ÚNICAMENTE un JSON válido con esta estructura:
        {{
            "rutina": [
                {{"dia": "Día 1: Pecho y Tríceps", "ejercicios": ["Press de banca: 4x6-8", "Aperturas: 3x12"]}}
            ],
            "nutricion": "✅ PERMITIDOS: Pechuga, huevos, arroz, pasta.\n❌ PROHIBIDOS: Azúcar, ultraprocesados.\n📊 MACROS: 160g Proteína, 200g Carbs, 60g Grasas."
        }}
        """
        
        if model is None: raise HTTPException(status_code=500, detail="Modelo IA no disponible.")

        response = model.generate_content(prompt)
        texto_limpio = response.text.replace('```json', '').replace('```', '').strip()
        datos_ia = json.loads(texto_limpio)
        
        user_ref.update({
            'plan_entrenamiento': datos_ia['rutina'],
            'plan_nutricion': datos_ia['nutricion']
        })
        return {"status": "success", "message": "Plan optimizado."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# ESCÁNER BIOMÉTRICO (FOTOS + PESO)
# ==========================================
@app.post("/evaluar-progreso")
def evaluar_progreso(req: ProgresoRequest):
    global model
    try:
        user_ref = db.collection('usuarios').document(req.uid)
        user_doc = user_ref.get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="Usuario no encontrado.")
            
        # Preparar imágenes para la IA
        contents = [
            f"""
            Actúa como un juez de culturismo y nutricionista. El usuario reporta un nuevo peso de {req.peso} kg y adjunta 3 fotos.
            REGLA 1 (SEGURIDAD): Analiza las imágenes. Si NO son fotos de un cuerpo humano evaluable (ej. son memes, objetos, paisajes o fotos borrosas inservibles), detente y devuelve "error_fotos": true.
            REGLA 2: Si las fotos son reales y válidas, evalúa visualmente el estado físico actual del atleta y el nuevo peso.
            REGLA 3: Genera una nueva directriz de nutrición AJUSTADA (SIN PLATILLOS, SOLO QUÉ COMER, QUÉ NO COMER Y MACROS).
            
            Devuelve ÚNICAMENTE este JSON:
            {{
                "error_fotos": false,
                "feedback_visual": "El texto de tu análisis físico. Ejemplo: 'Notables ganancias en deltoides, pero...'",
                "nueva_nutricion": "✅ PERMITIDOS: ... \n❌ PROHIBIDOS: ... \n📊 MACROS: ..."
            }}
            """
        ]
        
        for b64 in req.fotos_b64:
            contents.append({"mime_type": "image/jpeg", "data": b64})

        response = model.generate_content(contents)
        texto_limpio = response.text.replace('```json', '').replace('```', '').strip()
        datos_ia = json.loads(texto_limpio)
        
        if datos_ia.get("error_fotos") == True:
            raise HTTPException(status_code=400, detail="Imágenes no válidas. Sube fotos reales de tu físico para el escáner.")

        # Si todo es correcto, actualizamos la base de datos
        user_data = user_doc.to_dict()
        historial = user_data.get('historial_pesos', [])
        historial.append(req.peso)
        
        user_ref.update({
            'historial_pesos': historial,
            'plan_nutricion': datos_ia['nueva_nutricion']
        })
        
        return {
            "status": "success", 
            "feedback": datos_ia.get("feedback_visual", "Análisis completado exitosamente.")
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error procesando fotos: {e}")
        raise HTTPException(status_code=500, detail="Error analizando la biometría visual.")

# ==========================================
# CHAT ASISTENTE (JARVIS)
# ==========================================
@app.post("/chat")
def chat_asistente(req: ChatRequest):
    global model
    try:
        user_doc = db.collection('usuarios').document(req.uid).get()
        user_data = user_doc.to_dict() if user_doc.exists else {}
        nombre = user_data.get('nombre', 'Atleta')
        objetivo = user_data.get('objetivo', 'Hipertrofia')
        
        chat_prompt = f"Eres Jarvis, el asistente de IA de CORE Fitness. Hablas con {nombre} (Objetivo: {objetivo}). Responde a: {req.mensaje}. Sé directo, profesional y conciso."
        response = model.generate_content(chat_prompt)
        return {"status": "success", "respuesta": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
