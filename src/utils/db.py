from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np
from pathlib import Path
from datetime import datetime
from src.utils import db
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/app", StaticFiles(directory="src/web", html=True), name="static")

# ----------------------------
# Configuración CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Modelos de reconocimiento
# ----------------------------
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR_PATH)
MODEL_PATH = Path("data/modelos/lbph.yml")
LABELS_PATH = Path("data/modelos/label_map.npy")

recognizer = None
label_map = {}

if MODEL_PATH.exists():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

if LABELS_PATH.exists():
    label_map = np.load(str(LABELS_PATH), allow_pickle=True).item()

db.init_db()

# ----------------------------
# Endpoints
# ----------------------------

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
        msg = "❌ Persona no reconocida"
        saved = False
        legajo = None
        nombre = None
        evento = None

        if recognizer and label_map:
            roi_resized = cv2.resize(roi_gray, (200, 200))
            label_id, conf = recognizer.predict(roi_resized)

            if label_id in label_map and conf < 80:
                label_txt = label_map[label_id]
                msg = f"✅ Validado: {label_txt} (conf={conf:.1f})"
                legajo, nombre = db.parse_label(label_txt)
                evento = db.infer_event_type(legajo)

                rowid = db.save_attendance(legajo, nombre, evento)
                saved = True
                print(f"[DB] attendance id={rowid} emp={legajo} {nombre} {evento}")

        response.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "message": msg, "saved": saved,
            "legajo": legajo, "nombre": nombre, "evento": evento
        })

    return {"faces": response}


@app.get("/attendance/today")
def attendance_today():
    return db.fetch_attendance_today()


@app.get("/metrics/attendance/today")
def attendance_metrics_today():
    conn = db.get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM empleados").fetchone()[0]
        presentes = conn.execute("""
            SELECT COUNT(DISTINCT id_empleado)
            FROM registros
            WHERE DATE(ts_utc) = DATE('now') AND evento = 'INGRESO'
        """).fetchone()[0]
        return {
            "total": total,
            "presentes": presentes,
            "ausentes": total - presentes
        }
    finally:
        conn.close()


@app.get("/metrics/attendance/last10")
def attendance_last10():
    conn = db.get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM empleados").fetchone()[0]
        rows = conn.execute("""
            WITH base AS (
              SELECT DATE(ts_utc) as fecha, id_empleado
              FROM registros
              WHERE evento = 'INGRESO'
              GROUP BY fecha, id_empleado
            )
            SELECT fecha, COUNT(id_empleado) as presentes
            FROM base
            WHERE fecha >= DATE('now', '-9 days')
            GROUP BY fecha
            ORDER BY fecha
        """).fetchall()

        out = []
        for fecha, presentes in rows:
            out.append({
                "fecha": fecha,
                "presentes": presentes,
                "ausentes": total - presentes
            })
        return out
    finally:
        conn.close()


# ----------------------------
# Modelos de datos
# ----------------------------
class EmpleadoIn(BaseModel):
    legajo: int
    nombre: str
    apellido: str
    dni: str
    puesto: str
    turno: str
    sector: str
    rol: str = "OPERARIO"  # Nuevo campo


@app.post("/empleados")
def add_empleado(emp: EmpleadoIn):
    if emp.rol.upper() not in ("GERENTE", "ADMINISTRADOR", "OPERARIO"):
        raise HTTPException(status_code=400, detail="Rol inválido")

    conn = db.get_conn()
    try:
        cur = conn.execute(
            """
            INSERT INTO empleados (legajo, nombre, apellido, dni, puesto, turno, sector, rol)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (emp.legajo, emp.nombre, emp.apellido, emp.dni, emp.puesto, emp.turno, emp.sector, emp.rol.upper())
        )
        return {"id": cur.lastrowid, "message": f"Empleado agregado correctamente con rol {emp.rol}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al insertar: {str(e)}")
    finally:
        conn.close()


# ----------------------------
# NUEVO ENDPOINT: /usuarios
# ----------------------------
@app.post("/usuarios")
def add_usuario(emp: EmpleadoIn):
    """
    Endpoint alternativo para registrar usuarios con rol,
    usado por pantallas de administración o alta de usuarios.
    """
    if emp.rol.upper() not in ("GERENTE", "ADMINISTRADOR", "OPERARIO"):
        raise HTTPException(status_code=400, detail="Rol inválido")

    conn = db.get_conn()
    try:
        cur = conn.execute(
            """
            INSERT INTO empleados (legajo, nombre, apellido, dni, puesto, turno, sector, rol)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (emp.legajo, emp.nombre, emp.apellido, emp.dni, emp.puesto, emp.turno, emp.sector, emp.rol.upper())
        )
        return {
            "id": cur.lastrowid,
            "rol": emp.rol.upper(),
            "message": f"Usuario '{emp.nombre} {emp.apellido}' agregado correctamente con rol {emp.rol}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al insertar usuario: {str(e)}")
    finally:
        conn.close()


# ----------------------------
# Registros manuales
# ----------------------------
class RegistroIn(BaseModel):
    legajo: int
    evento: str  # "INGRESO" o "EGRESO"
    ts_utc: str | None = None  # formato ISO opcional


@app.post("/registros")
def add_registro(reg: RegistroIn):
    if reg.evento not in ("INGRESO", "EGRESO"):
        raise HTTPException(status_code=400, detail="Evento debe ser 'INGRESO' o 'EGRESO'")

    ts = reg.ts_utc or datetime.utcnow().isoformat(timespec="seconds")

    try:
        conn = db.get_conn()
        id_empleado = db._ensure_empleado_by_legajo(reg.legajo, "")
        cur = conn.execute(
            "INSERT INTO registros (id_empleado, ts_utc, evento) VALUES (?, ?, ?)",
            (id_empleado, ts, reg.evento)
        )
        return {
            "id_registro": cur.lastrowid,
            "legajo": reg.legajo,
            "evento": reg.evento,
            "ts_utc": ts,
            "message": "Registro agregado correctamente"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al insertar: {str(e)}")
    finally:
        conn.close()