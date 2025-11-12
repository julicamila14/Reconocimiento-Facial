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
  sector TEXT NOT NULL,
  rol TEXT NOT NULL DEFAULT 'OPERARIO'
);

CREATE TABLE IF NOT EXISTS registros (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  id_empleado INTEGER NOT NULL,
  ts_utc TEXT NOT NULL,
  evento TEXT NOT NULL,
  FOREIGN KEY(id_empleado) REFERENCES empleados(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS usuarios (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  legajo INTEGER NOT NULL UNIQUE,
  nombre TEXT NOT NULL,
  apellido TEXT NOT NULL,
  email TEXT NOT NULL,
  rol TEXT NOT NULL DEFAULT 'OPERARIO'
);
"""

# -------------------------
# Conexión e inicialización
# -------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=5, isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row  
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
# Usuarios
# -------------------------
def fetch_usuarios():
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT legajo, nombre, apellido, email, rol FROM usuarios ORDER BY legajo"
        ).fetchall()
        cols = ["legajo", "nombre", "apellido", "email", "rol"]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()


def add_usuario_con_rol(user):
    conn = get_conn()
    try:
        if not isinstance(user, dict):
            user = user.dict()

        legajo = user.get("legajo")
        nombre = user.get("nombre")
        apellido = user.get("apellido")
        email = user.get("email")
        rol = user.get("rol", "OPERARIO")

        # Insertar usuario
        cur = conn.execute(
            """
            INSERT INTO usuarios (legajo, nombre, apellido, email, rol)
            VALUES (?, ?, ?, ?, ?)
            """,
            (legajo, nombre, apellido, email, rol),
        )

        # Crear empleado asociado si no existe
        existe = conn.execute(
            "SELECT id FROM empleados WHERE legajo = ?", (legajo,)
        ).fetchone()

        if not existe:
            conn.execute(
                """
                INSERT INTO empleados (legajo, nombre, apellido, dni, puesto, turno, sector, rol)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (legajo, nombre, apellido, "N/A", "N/A", "N/A", "N/A", rol),
            )

        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def update_usuario_rol(legajo, nuevo_rol):
    conn = get_conn()
    try:
        # Actualiza el rol tanto en usuarios como empleados
        conn.execute("UPDATE usuarios SET rol = ? WHERE legajo = ?", (nuevo_rol, legajo))
        conn.execute("UPDATE empleados SET rol = ? WHERE legajo = ?", (nuevo_rol, legajo))
        conn.commit()
        return True
    finally:
        conn.close()

# -------------------------
# Empleados (asistencia)
# -------------------------
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

def _ensure_empleado_by_legajo(legajo: int, nombre_completo: str, rol: str = "OPERARIO") -> int:
    emp_id = _get_empleado_id_by_legajo(legajo)
    if emp_id is not None:
        return emp_id

    partes = (nombre_completo or "N/A").strip().split()
    nombre = partes[0] if partes else "N/A"
    apellido = " ".join(partes[1:]) if len(partes) > 1 else "N/A"

    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO empleados(legajo, nombre, apellido, dni, puesto, turno, sector, rol) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (legajo, nombre, apellido, "N/A", "N/A", "N/A", "N/A", rol)
        )
        return cur.lastrowid
    finally:
        conn.close()

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
                   e.rol,
                   r.ts_utc,
                   r.evento
            FROM registros r
            JOIN empleados e ON e.id = r.id_empleado
            WHERE r.ts_utc >= ? AND r.ts_utc < ?
            ORDER BY r.ts_utc DESC
            """,
            (start.isoformat(timespec='seconds'), end.isoformat(timespec='seconds'))
        ).fetchall()
        cols = ["legajo", "nombre", "rol", "ts_utc", "evento"]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()

def save_attendance(legajo: str, name: str, event: str, rol: str = "OPERARIO") -> int:
    if not legajo:
        raise ValueError("legajo requerido")

    try:
        legajo = int(legajo)
    except (TypeError, ValueError):
        raise ValueError("legajo debe ser numérico")

    id_empleado = _ensure_empleado_by_legajo(legajo, name, rol)

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

def get_usuario_by_legajo(legajo: int):
    conn = get_conn()
    try:
        cur = conn.execute("SELECT * FROM usuarios WHERE legajo = ?", (legajo,))
        row = cur.fetchone()
        if row:
            return dict(row) 
        return None
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    print("Base de datos inicializada con tablas empleados, registros y usuarios.")

