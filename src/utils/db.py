import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = Path("data/db/attendance.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DDL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS empleados (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  legajo INTEGER NOT NULL UNIQUE,
  nombre TEXT NOT NULL,
  apellido TEXT NOT NULL,
  dni TEXT NOT NULL,
  puesto TEXT NOT NULL,
  turno TEXT NOT NULL,
  sector TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS registros (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  id_empleado INTEGER NOT NULL,
  ts_utc TEXT NOT NULL,
  evento TEXT NOT NULL,
  FOREIGN KEY(id_empleado) REFERENCES empleados(id) ON DELETE CASCADE
);
"""

# -------------------------
# Conexión e inicialización
# -------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=5, isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db():
    conn = get_conn()
    try:
        for stmt in DDL.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(s)
    finally:
        conn.close()

# -------------------------
# Utilidades
# -------------------------
def parse_label(label: str):
    if not label:
        return None, None
    clean = label.replace("-", "_")
    parts = clean.split("_")
    if len(parts) >= 2:
        emp_id = parts[0]
        name = " ".join(parts[1:]).strip().replace("  ", " ")
        return emp_id, name
    return label, label

def _get_empleado_id_by_legajo(legajo: int):
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT id FROM empleados WHERE legajo = ?",
            (legajo,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()

def _ensure_empleado_by_legajo(legajo: int, nombre_completo: str) -> int:
    emp_id = _get_empleado_id_by_legajo(legajo)
    if emp_id is not None:
        return emp_id

    partes = (nombre_completo or "N/A").strip().split()
    nombre = partes[0] if partes else "N/A"
    apellido = " ".join(partes[1:]) if len(partes) > 1 else "N/A"

    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO empleados(legajo, nombre, apellido, dni, puesto, turno, sector) "
            "VALUES(?,?,?,?,?,?,?)",
            (legajo, nombre, apellido, "N/A", "N/A", "N/A", "N/A")
        )
        return cur.lastrowid
    finally:
        conn.close()

# -------------------------
# Consultas de negocio
# -------------------------
def get_last_event(emp_id: str, days: int = 1):
    if not emp_id:
        return None

    try:
        legajo = int(emp_id)
    except (TypeError, ValueError):
        return None

    id_empleado = _get_empleado_id_by_legajo(legajo)
    if id_empleado is None:
        return None

    dt_from = (datetime.utcnow() - timedelta(days=days)).isoformat(timespec="seconds")
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT evento, ts_utc
            FROM registros
            WHERE id_empleado = ? AND ts_utc >= ?
            ORDER BY ts_utc DESC
            LIMIT 1
            """,
            (id_empleado, dt_from),
        ).fetchone()
        if row:
            return {"evento": row[0], "ts_utc": row[1]}
        return None
    finally:
        conn.close()

def infer_event_type(emp_id: str) -> str:
    last = get_last_event(emp_id, days=1)
    if not last or last["evento"] in (None, "EGRESO"):
        return "INGRESO"
    return "EGRESO"

def fetch_attendance_today():
    today = datetime.now().date()
    start = datetime(today.year, today.month, today.day)
    end = start + timedelta(days=1)

    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT e.legajo,
                   (e.nombre || ' ' || e.apellido) AS nombre,
                   r.ts_utc,
                   r.evento
            FROM registros r
            JOIN empleados e ON e.id = r.id_empleado
            WHERE r.ts_utc >= ? AND r.ts_utc < ?
            ORDER BY r.ts_utc DESC
            """,
            (start.isoformat(timespec='seconds'), end.isoformat(timespec='seconds'))
        ).fetchall()
        cols = ["legajo", "nombre", "ts_utc", "evento"]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()

def save_attendance(legajo: str, name: str, event: str) -> int:
    if not legajo:
        raise ValueError("legajo requerido")

    try:
        legajo = int(legajo)
    except (TypeError, ValueError):
        raise ValueError("legajo debe ser numérico")

    # Asegura que exista empleados.id para ese legajo
    id_empleado = _ensure_empleado_by_legajo(legajo, name)

    ts = datetime.now().isoformat(timespec="seconds")
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO registros(id_empleado, ts_utc, evento) VALUES(?,?,?)",
            (id_empleado, ts, event)
        )
        return cur.lastrowid
    finally:
        conn.close()
