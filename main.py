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
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("Motor IA Gemini inicializado correctamente.")
except Exception as e:
    print(f"Error inicializando Gemini: {e}")

# --- MODELOS DE DATOS ---
class RutinaRequest(BaseModel):
    uid: str

class ProgresoRequest(BaseModel):
    uid: str
    peso: float
    fotos_b64: list[str] = []

class MensajeBio(BaseModel):
    role: str
    content: str

class BioChatRequest(BaseModel):
    uid: str
    historial: list[MensajeBio] = []
    nuevo_mensaje: str = ""

class AplicarPlanRequest(BaseModel):
    uid: str
    historial: list[MensajeBio] = []

@app.get("/")
def home():
    return {"status": "online"}

# ==========================================
# 1. RUTINA ESTÁNDAR (Botón Recalibrar de la pestaña Plan)
# ==========================================
@app.post("/generar-rutina")
def generar_rutina(req: RutinaRequest):
    global model
    try:
        user_ref = db.collection('usuarios').document(req.uid)
        user_doc = user_ref.get()
        if not user_doc.exists: raise HTTPException(status_code=404, detail="Usuario no encontrado.")
            
        user_data = user_doc.to_dict()
        objetivo = user_data.get('objetivo', 'Recomposición')
        frecuencia = user_data.get('dias_entrenamiento', 'Lunes a Viernes')
        peso = user_data.get('historial_pesos', [0])[-1]

        prompt = f"""
        Actúa como entrenador de élite. Plan para {frecuencia}. Peso: {peso}kg. Enfoque: {objetivo}.
        REGLAS:
        1. Genera los días exactos según la frecuencia. Grupos musculares en el título.
        2. Ejercicios compuestos y aislamiento con series/repeticiones.
        3. NUTRICIÓN DIRECTA Y ESTRICTA. CERO RECETAS O PLATILLOS. Solo lista de proteínas, carbohidratos, grasas permitidas, alimentos prohibidos y resumen de macros.
        
        Devuelve ÚNICAMENTE JSON:
        {{
            "rutina": [ {{"dia": "Día 1: Pecho y Tríceps", "ejercicios": ["Press: 4x8"]}} ],
            "nutricion": "✅ PERMITIDOS: ...\n❌ PROHIBIDOS: ...\n📊 MACROS: ..."
        }}
        """
        response = model.generate_content(prompt)
        datos_ia = json.loads(response.text.replace('```json', '').replace('```', '').strip())
        user_ref.update({'plan_entrenamiento': datos_ia['rutina'], 'plan_nutricion': datos_ia['nutricion']})
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 2. INICIO DE ESCÁNER BIOMÉTRICO (Analiza pero NO guarda el plan aún)
# ==========================================
@app.post("/evaluar-progreso")
def evaluar_progreso(req: ProgresoRequest):
    global model
    try:
        user_ref = db.collection('usuarios').document(req.uid)
        user_doc = user_ref.get()
        if not user_doc.exists: raise HTTPException(status_code=404)
            
        contents = [
            f"""
            Eres Jarvis, analista físico. El usuario reporta {req.peso} kg.
            REGLA 1: Si NO son fotos de un cuerpo humano evaluable, devuelve {{"error_fotos": true}}.
            REGLA 2: Si son válidas, evalúa visualmente.
            REGLA 3: Termina el análisis PREGUNTÁNDOLE al usuario qué grupo muscular quiere priorizar o qué prefiere ajustar, para adaptar su nueva rutina y dieta.
            Devuelve ÚNICAMENTE JSON:
            {{ "error_fotos": false, "feedback_visual": "Tu análisis y tu pregunta final aquí." }}
            """
        ]
        for b64 in req.fotos_b64: contents.append({"mime_type": "image/jpeg", "data": b64})

        response = model.generate_content(contents)
        datos_ia = json.loads(response.text.replace('```json', '').replace('```', '').strip())
        
        if datos_ia.get("error_fotos") == True:
            raise HTTPException(status_code=400, detail="Sube fotos reales de tu físico.")

        # Solo actualizamos el historial de pesos para la gráfica
        user_data = user_doc.to_dict()
        historial = user_data.get('historial_pesos', [])
        historial.append(req.peso)
        user_ref.update({'historial_pesos': historial})
        
        return {"status": "success", "feedback": datos_ia.get("feedback_visual")}
    except HTTPException as he: raise he
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 3. CHAT BIOMÉTRICO (Conversación intermedia)
# ==========================================
@app.post("/chat-biometrico")
def chat_biometrico(req: BioChatRequest):
    global model
    try:
        conversacion = "\n".join([f"{m.role}: {m.content}" for m in req.historial])
        prompt = f"""
        Eres Jarvis. Estás en medio de un análisis físico con el atleta.
        Historial de charla:
        {conversacion}
        Atleta: {req.nuevo_mensaje}
        
        Responde brevemente confirmando que tomarás en cuenta su petición para el nuevo plan.
        """
        response = model.generate_content(prompt)
        return {"status": "success", "respuesta": response.text.strip()}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 4. GUARDAR Y APLICAR AJUSTES FINALES
# ==========================================
@app.post("/aplicar-plan-biometrico")
def aplicar_plan(req: AplicarPlanRequest):
    global model
    try:
        user_ref = db.collection('usuarios').document(req.uid)
        user_data = user_ref.get().to_dict()
        
        frecuencia = user_data.get('dias_entrenamiento', 'Lunes a Viernes')
        peso = user_data.get('historial_pesos', [0])[-1]
        conversacion = "\n".join([f"{m.role}: {m.content}" for m in req.historial])

        prompt = f"""
        Actúa como entrenador de élite. Diseña la rutina de {frecuencia} FINAL basada en esta charla biométrica:
        {conversacion}
        Peso actual: {peso}kg.
        
        REGLAS:
        1. Adapta la rutina estrictamente a lo que el atleta pidió priorizar en la charla.
        2. NUTRICIÓN STRICTA (0 recetas). Solo proteínas, carbs, grasas permitidas, alimentos prohibidos y macros.
        
        Devuelve ÚNICAMENTE JSON:
        {{
            "rutina": [ {{"dia": "...", "ejercicios": ["..."]}} ],
            "nutricion": "✅ PERMITIDOS: ...\n❌ PROHIBIDOS: ...\n📊 MACROS: ..."
        }}
        """
        response = model.generate_content(prompt)
        datos_ia = json.loads(response.text.replace('```json', '').replace('```', '').strip())
        
        user_ref.update({'plan_entrenamiento': datos_ia['rutina'], 'plan_nutricion': datos_ia['nutricion']})
        return {"status": "success"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# El Chat Global Jarvis de la otra pestaña se mantiene igual
@app.post("/chat")
def chat_asistente(req: BioChatRequest):
    # ... (Mismo código que ya tenías para el chat global)
    return {"status": "success", "respuesta": "Módulo de chat global activo."}
