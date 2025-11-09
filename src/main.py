from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import cv2, numpy as np, os
from pathlib import Path
from src.utils import db

app = FastAPI()
app.mount("/app", StaticFiles(directory="src/web", html=True), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURACIÓN DE MODELOS Y RUTAS ---
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR_PATH)
MODEL_PATH = Path("data/modelos/lbph.yml")
LABELS_PATH = Path("data/modelos/label_map.npy")
FACES_DIR = Path("data/faces")
FACES_DIR.mkdir(parents=True, exist_ok=True)

recognizer, label_map = None, {}

if MODEL_PATH.exists():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
if LABELS_PATH.exists():
    label_map = np.load(str(LABELS_PATH), allow_pickle=True).item()

db.init_db()

# --- MODELOS DE ENTRADA ---

class EmpleadoIn(BaseModel):
    legajo: int
    nombre: str
    apellido: str
    dni: str
    puesto: str
    turno: str
    sector: str
    rol: str = "OPERARIO"

class RegistroIn(BaseModel):
    legajo: int
    evento: str
    ts_utc: str | None = None

class RolUpdate(BaseModel):
    rol: str

# --- ENDPOINTS ---

@app.post("/empleados")
def add_empleado(emp: EmpleadoIn):
    try:
        emp_id = db.add_empleado_con_rol(emp)
        # crear carpeta de fotos
        emp_dir = FACES_DIR / str(emp.legajo)
        emp_dir.mkdir(parents=True, exist_ok=True)
        return {"id": emp_id, "message": "Empleado agregado correctamente"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al insertar: {str(e)}")


@app.put("/empleados/{legajo}/rol")
def update_rol(legajo: int, body: RolUpdate):
    updated = db.update_empleado_rol(legajo, body.rol)
    if updated == 0:
        raise HTTPException(status_code=404, detail="Empleado no encontrado")
    return {"legajo": legajo, "rol": body.rol, "message": "Rol actualizado correctamente"}


# --- NUEVO: SUBIR FOTOS PARA RECONOCIMIENTO ---

@app.post("/empleados/{legajo}/fotos")
async def upload_fotos(legajo: int, files: list[UploadFile] = File(...)):
    emp_dir = FACES_DIR / str(legajo)
    emp_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            continue

        for i, (x, y, w, h) in enumerate(faces):
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (200, 200))
            fname = emp_dir / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}_{i}.jpg"
            cv2.imwrite(str(fname), face_resized)
            saved_files.append(str(fname))

    if not saved_files:
        raise HTTPException(status_code=400, detail="No se detectaron rostros en las imágenes")

    _retrain_model()
    return {"message": f"{len(saved_files)} fotos guardadas y modelo actualizado", "files": saved_files}


@app.get("/empleados/{legajo}/fotos")
def listar_fotos(legajo: int):
    emp_dir = FACES_DIR / str(legajo)
    if not emp_dir.exists():
        raise HTTPException(status_code=404, detail="No hay fotos para este legajo")
    fotos = [f"/{emp_dir}/{f.name}" for f in emp_dir.iterdir() if f.is_file()]
    return {"legajo": legajo, "fotos": fotos}


@app.post("/detect")
async def detect_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    response = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        msg, saved, legajo, nombre, evento = "❌ Persona no reconocida", False, None, None, None

        if recognizer and label_map:
            roi_resized = cv2.resize(roi_gray, (200, 200))
            label_id, conf = recognizer.predict(roi_resized)
            if label_id in label_map and conf < 80:
                label_txt = label_map[label_id]
                msg = f"✅ Validado: {label_txt} (conf={conf:.1f})"
                legajo, nombre = db.parse_label(label_txt)
                evento = db.infer_event_type(legajo)
                db.save_attendance(legajo, nombre, evento)
                saved = True

        response.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "message": msg, "saved": saved, "legajo": legajo,
            "nombre": nombre, "evento": evento
        })

    return {"faces": response}


@app.get("/attendance/today")
def attendance_today():
    return db.fetch_attendance_today()


@app.post("/registros")
def add_registro(reg: RegistroIn):
    if reg.evento not in ("INGRESO", "EGRESO"):
        raise HTTPException(status_code=400, detail="Evento inválido")
    ts = reg.ts_utc or datetime.utcnow().isoformat(timespec="seconds")
    conn = db.get_conn()
    try:
        id_emp = db._ensure_empleado_by_legajo(reg.legajo, "")
        cur = conn.execute(
            "INSERT INTO registros (id_empleado, ts_utc, evento) VALUES (?, ?, ?)",
            (id_emp, ts, reg.evento)
        )
        conn.commit()
        return {"id_registro": cur.lastrowid, "message": "Registro agregado correctamente"}
    finally:
        conn.close()


# --- FUNCIÓN DE ENTRENAMIENTO ---

def _retrain_model():
    global recognizer, label_map
    faces, labels, label_map = [], [], {}
    current_label = 0

    for emp_dir in FACES_DIR.iterdir():
        if not emp_dir.is_dir():
            continue
        for img_path in emp_dir.iterdir():
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(current_label)
        label_map[current_label] = str(emp_dir.name)
        current_label += 1

    if not faces:
        raise Exception("No hay imágenes para entrenar el modelo")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(str(MODEL_PATH))
    np.save(str(LABELS_PATH), label_map)