from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import cv2, numpy as np
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

class UsuarioIn(BaseModel):
    legajo: int
    nombre: str
    apellido: str
    email: str
    rol: str = "OPERARIO"

class RegistroIn(BaseModel):
    legajo: int
    evento: str
    ts_utc: str | None = None

class RolUpdate(BaseModel):
    rol: str


# --- ENDPOINTS DE USUARIOS ---

@app.get("/usuarios")
def listar_usuarios():
    """Listar todos los usuarios"""
    try:
        usuarios = db.fetch_usuarios()
        return {"usuarios": usuarios}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener usuarios: {e}")


@app.post("/usuarios")
def add_usuario(user: UsuarioIn):
    """Alta de nuevo usuario"""
    try:
        user_id = db.add_usuario_con_rol(user)
        user_dir = FACES_DIR / str(user.legajo)
        user_dir.mkdir(parents=True, exist_ok=True)
        return {"id": user_id, "message": "Usuario agregado correctamente"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al insertar usuario: {str(e)}")


@app.put("/usuarios/{legajo}/rol")
def update_rol(legajo: int, body: RolUpdate):
    """Modificar el rol de un usuario"""
    updated = db.update_usuario_rol(legajo, body.rol)
    if updated == 0:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return {"legajo": legajo, "rol": body.rol, "message": "Rol actualizado correctamente"}


# --- SUBIDA DE FOTOS ---

@app.post("/usuarios/{legajo}/fotos")
async def upload_fotos(legajo: int, files: list[UploadFile] = File(...)):
    """Subir fotos para el usuario (opcional para reconocimiento futuro)"""
    user_dir = FACES_DIR / str(legajo)
    user_dir.mkdir(parents=True, exist_ok=True)

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
            fname = user_dir / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}_{i}.jpg"
            cv2.imwrite(str(fname), face_resized)
            saved_files.append(str(fname))

    if not saved_files:
        raise HTTPException(status_code=400, detail="No se detectaron rostros en las imágenes")

    _retrain_model()
    return {"message": f"{len(saved_files)} fotos guardadas y modelo actualizado", "files": saved_files}


@app.get("/usuarios/{legajo}/fotos")
def listar_fotos(legajo: int):
    """Listar fotos del usuario"""
    user_dir = FACES_DIR / str(legajo)
    if not user_dir.exists():
        raise HTTPException(status_code=404, detail="No hay fotos para este usuario")
    fotos = [f"/{user_dir}/{f.name}" for f in user_dir.iterdir() if f.is_file()]
    return {"legajo": legajo, "fotos": fotos}


# --- REGISTROS (para asistencia u otro evento) ---

@app.post("/registros")
def add_registro(reg: RegistroIn):
    """Registrar evento de usuario (ingreso/egreso u otro tipo)"""
    if reg.evento not in ("INGRESO", "EGRESO"):
        raise HTTPException(status_code=400, detail="Evento inválido")
    ts = reg.ts_utc or datetime.utcnow().isoformat(timespec="seconds")
    conn = db.get_conn()
    try:
        id_usr = db._ensure_usuario_by_legajo(reg.legajo, "")
        cur = conn.execute(
            "INSERT INTO registros (id_usuario, ts_utc, evento) VALUES (?, ?, ?)",
            (id_usr, ts, reg.evento)
        )
        conn.commit()
        return {"id_registro": cur.lastrowid, "message": "Registro agregado correctamente"}
    finally:
        conn.close()


@app.get("/attendance/today")
def attendance_today():
    """Listar registros de hoy"""
    return db.fetch_attendance_today()


# --- FUNCIÓN DE ENTRENAMIENTO (opcional reconocimiento facial futuro) ---

def _retrain_model():
    global recognizer, label_map
    faces, labels, label_map = [], [], {}
    current_label = 0

    for user_dir in FACES_DIR.iterdir():
        if not user_dir.is_dir():
            continue
        for img_path in user_dir.iterdir():
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(current_label)
        label_map[current_label] = str(user_dir.name)
        current_label += 1

    if not faces:
        raise Exception("No hay imágenes para entrenar el modelo")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(str(MODEL_PATH))
    np.save(str(LABELS_PATH), label_map)
    