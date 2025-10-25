import os, cv2, numpy as np
from pathlib import Path

DATA_DIR = Path("data/empleados")
MODEL_PATH = Path("data/modelos/lbph.yml")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_images_and_labels():
    images, labels, label_map = [], [], {}
    current_label = 0

    for person_dir in sorted(DATA_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        label_map[current_label] = person_dir.name
        for img_path in person_dir.glob("*.*"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Detección cara 
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(img, 1.2, 5)
            if len(faces) == 0:
                continue
            x,y,w,h = faces[0]
            face = cv2.resize(img[y:y+h, x:x+w], (200, 200))
            images.append(face)
            labels.append(current_label)
        current_label += 1
    return images, np.array(labels), label_map

def main():
    images, labels, label_map = load_images_and_labels()
    if len(images) == 0:
        raise RuntimeError("No se encontraron rostros en data/empleados/*")

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=10, grid_x=8, grid_y=8)
    recognizer.train(images, labels)
    recognizer.write(str(MODEL_PATH))

    # Guardar map de labels
    np.save(str(MODEL_PATH.parent / "label_map.npy"), label_map)
    print(f"[OK] Modelo entrenado: {MODEL_PATH}")
    print(f"[OK] Personas: {label_map}")

if __name__ == "__main__":
    main()
