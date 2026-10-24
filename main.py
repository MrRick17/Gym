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
        # Usamos una temperatura más alta (0.7) para que no sea tan repetitivo y sea más creativo/personalizado
        model = genai.GenerativeModel('gemini-3.8-flash', generation_config={"temperature": 0.7})
        print("Motor IA Gemini inicializado correctamente.")
except Exception as e:
    print(f"Error inicializando Gemini: {e}")

# --- MODELOS DE DATOS ---
class RutinaRequest(BaseModel):
    uid: str

class RegistroBioRequest(BaseModel):
    uid: str
    peso: float
    frecuencia: str
    objetivo: str
    genero: str
    fotos_b64: list[str]

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
# 1. ANÁLISIS INICIAL (AL REGISTRARSE)
# ==========================================
@app.post("/analisis-registro")
def analisis_registro(req: RegistroBioRequest):
    global model
    try:
        user_ref = db.collection('usuarios').document(req.uid)
        
        contents = [
            f"""
            Eres el entrenador de élite y nutricionista de CORE Fitness. Genera un plan inicial TOTALMENTE PERSONALIZADO.
            PERFIL DEL ATLETA:
            - Género: {req.genero}
            - Peso inicial: {req.peso} kg
            - Frecuencia: {req.frecuencia}
            - Objetivo: {req.objetivo}
            
            REGLAS DE PERSONALIZACIÓN ESTRICTA:
            1. Analiza las fotos para evaluar el punto de partida físico (Si no son de un humano, devuelve "error_fotos": true).
            2. Si el objetivo es 'Volumen': Enfoca la rutina en cargas pesadas, repeticiones de 6-10, descansos largos y superávit calórico.
            3. Si el objetivo es 'Definición': Enfoca la rutina en intensidad, repeticiones de 10-15, agrega CARDIO HIIT al final y déficit calórico estricto.
            4. NUTRICIÓN: Prohibido dar recetas. Solo da las proteínas, carbohidratos, grasas limpias sugeridas para un {req.genero} de {req.peso}kg, qué evitar y los macros diarios exactos.
            
            Devuelve ÚNICAMENTE JSON VÁLIDO (sin markdown):
            {{
                "error_fotos": false,
                "rutina": [ {{"dia": "Día 1: Pecho y Tríceps", "ejercicios": ["Press de banca: 4x8"]}} ],
                "nutricion": "✅ PERMITIDOS: ...\\n❌ PROHIBIDOS: ...\\n📊 MACROS: ..."
            }}
            """
        ]
        
        for b64 in req.fotos_b64:
            contents.append({"mime_type": "image/jpeg", "data": b64})

        response = model.generate_content(contents)
        datos_ia = json.loads(response.text.replace('```json', '').replace('```', '').strip())
        
        if datos_ia.get("error_fotos") == True:
            return {"status": "warning", "message": "Atleta registrado, pero las fotos fueron rechazadas."}

        user_ref.update({'plan_entrenamiento': datos_ia['rutina'], 'plan_nutricion': datos_ia['nutricion']})
        return {"status": "success", "message": "Biometría analizada y plan generado."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error procesando la biometría inicial.")

# ==========================================
# 2. RUTINA ESTÁNDAR (Botón Recalibrar)
# ==========================================
@app.post("/generar-rutina")
def generar_rutina(req: RutinaRequest):
    global model
    try:
        user_ref = db.collection('usuarios').document(req.uid)
        user_doc = user_ref.get()
        if not user_doc.exists: raise HTTPException(status_code=404)
            
        user_data = user_doc.to_dict()
        objetivo = user_data.get('objetivo', 'Recomposición')
        frecuencia = user_data.get('dias_entrenamiento', 'Lunes a Viernes')
        peso = user_data.get('historial_pesos', [0])[-1]
        genero = user_data.get('genero', 'Desconocido')
        edad = user_data.get('edad', 20)

        prompt = f"""
        Actúa como entrenador de élite. Rediseña un plan TOTALMENTE ÚNICO para este atleta.
        ATLETA: {genero}, {edad} años, {peso} kg. Frecuencia: {frecuencia}. Objetivo principal: {objetivo}.

        REGLAS DE PERSONALIZACIÓN ESTRICTA:
        1. No repitas rutinas genéricas. Adapta los ejercicios a un {genero}.
        2. Si el objetivo es 'Volumen': Usa rangos de fuerza/hipertrofia (6-10 reps) y levantamientos pesados.
        3. Si el objetivo es 'Definición': Usa rangos de resistencia/hipertrofia (10-15 reps), agrega trabajo cardiovascular y pliométrico.
        4. NUTRICIÓN: Cero recetas. Solo fuentes de macros, restricciones y cantidad de calorías/macros exactos según su peso y objetivo.
        
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
# 3. ESCÁNER BIOMÉTRICO (PROGRESO)
# ==========================================
@app.post("/evaluar-progreso")
def evaluar_progreso(req: ProgresoRequest):
    global model
    try:
        user_ref = db.collection('usuarios').document(req.uid)
        user_doc = user_ref.get()
        if not user_doc.exists: raise HTTPException(status_code=404)
        
        user_data = user_doc.to_dict()
        genero = user_data.get('genero', 'Desconocido')
            
        contents = [
            f"""
            Eres Jarvis, analista visual. Atleta: {genero}, {req.peso} kg.
            REGLA 1: Si no hay cuerpo humano evaluable, devuelve {{"error_fotos": true}}.
            REGLA 2: Analiza el porcentaje de grasa y masa muscular basado en las imágenes y su género.
            REGLA 3: Termina preguntándole al usuario de forma natural (máx 2 líneas) qué aspecto muscular desea priorizar este mes para recalibrar.
            Devuelve JSON: {{ "error_fotos": false, "feedback_visual": "Tu análisis..." }}
            """
        ]
        for b64 in req.fotos_b64: contents.append({"mime_type": "image/jpeg", "data": b64})

        response = model.generate_content(contents)
        datos_ia = json.loads(response.text.replace('```json', '').replace('```', '').strip())
        
        if datos_ia.get("error_fotos") == True:
            raise HTTPException(status_code=400, detail="Sube fotos reales de tu físico.")

        historial = user_data.get('historial_pesos', [])
        historial.append(req.peso)
        user_ref.update({'historial_pesos': historial})
        
        return {"status": "success", "feedback": datos_ia.get("feedback_visual")}
    except HTTPException as he: raise he
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 4. CHAT BIOMÉTRICO
# ==========================================
@app.post("/chat-biometrico")
def chat_biometrico(req: BioChatRequest):
    global model
    try:
        conversacion = "\n".join([f"{m.role}: {m.content}" for m in req.historial])
        prompt = f"Eres Jarvis. Estás en un análisis físico. Historial:\n{conversacion}\nAtleta: {req.nuevo_mensaje}\nResponde en 1 línea confirmando que tomarás esto en cuenta para generar el plan. Sé carismático."
        response = model.generate_content(prompt)
        return {"status": "success", "respuesta": response.text.strip()}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 5. APLICAR PLAN BIOMÉTRICO
# ==========================================
@app.post("/aplicar-plan-biometrico")
def aplicar_plan(req: AplicarPlanRequest):
    global model
    try:
        user_ref = db.collection('usuarios').document(req.uid)
        user_data = user_ref.get().to_dict()
        
        frecuencia = user_data.get('dias_entrenamiento', 'Lunes a Viernes')
        peso = user_data.get('historial_pesos', [0])[-1]
        objetivo = user_data.get('objetivo', 'Volumen')
        genero = user_data.get('genero', 'Desconocido')
        conversacion = "\n".join([f"{m.role}: {m.content}" for m in req.historial])

        prompt = f"""
        Actúa como entrenador de élite. Diseña la rutina FINAL para un {genero} de {peso}kg ({frecuencia}) basada en sus peticiones:
        Peticiones del atleta:
        {conversacion}
        
        REGLAS:
        1. OBJETIVO ACTUAL: {objetivo}. Si es Volumen, menos reps y más peso. Si es Definición, más reps y cardio al final.
        2. ADAPTABILIDAD: Integra ESTRICTAMENTE lo que el atleta pidió priorizar en la charla.
        3. NUTRICIÓN EXACTA: Cero recetas. Macros ajustados a su género, peso y peticiones.
        
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

# ==========================================
# 6. CHAT JARVIS GLOBAL
# ==========================================
@app.post("/chat")
def chat_asistente(req: BioChatRequest):
    global model
    try:
        user_doc = db.collection('usuarios').document(req.uid).get()
        user_data = user_doc.to_dict() if user_doc.exists else {}
        nombre = user_data.get('nombre', 'Atleta')
        objetivo = user_data.get('objetivo', 'Volumen')
        genero = user_data.get('genero', 'Atleta')
        
        chat_prompt = f"Eres Jarvis, el asistente de IA de CORE Fitness. Hablas con {nombre} ({genero}, Objetivo: {objetivo}). Responde a: {req.nuevo_mensaje}. Sé directo, motivacional, sin redundancias y no uses formato JSON."
        response = model.generate_content(chat_prompt)
        return {"status": "success", "respuesta": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
