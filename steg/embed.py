# steg/embed.py
from PIL import Image
import numpy as np
import os

SERVICE_BITS = 8  # 1 null byte as marker

# Вычисляем ёмоксть
def compute_capacity_bits_from_array(arr, lsb_count=1):
    h, w, c = arr.shape
    return int(h) * int(w) * int(c) * int(lsb_count)

def embed_message_with_checks(cover_path: str, stego_path: str, message_file: str, payload: float = 0.005):
    # загружаем картинку
    img = Image.open(cover_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    flat = arr.flatten() # преобразуем в массив байтов

    # загружаем сообщение
    with open(message_file, "rb") as f:
        msg_bytes_full = f.read()

    capacity_bits = compute_capacity_bits_from_array(arr, lsb_count=1)

    # Вычисляем величину допустимого сообщения в зависимости от payload
    allowed_bits = int(np.floor(payload * capacity_bits))
    if allowed_bits <= SERVICE_BITS:
        raise ValueError(f"Payload {payload} is too small: allowed_bits={allowed_bits}")

    message_bits_allowed = allowed_bits - SERVICE_BITS
    # округяем до 8
    message_bits_allowed = (message_bits_allowed // 8) * 8
    allowed_bytes = message_bits_allowed // 8

    # Если оригинальное сообщение больше допустимой длины - обрезаем, иначе - встраиваем целиком
    if len(msg_bytes_full) > allowed_bytes:
        msg_bytes = msg_bytes_full[:allowed_bytes]
        truncated = True
    else:
        msg_bytes = msg_bytes_full
        truncated = False


    if allowed_bytes == 0:
        raise ValueError("Нет места для сообщения.")

    # Добавляем нулевой байт-маркер
    final_bytes = msg_bytes + b'\x00'
    # Каждый байт преобразуется в 8-битную строку и все байты приводятся к ондной строке
    bits = ''.join(format(b, '08b') for b in final_bytes)
    n_bits = len(bits)
    if n_bits > flat.size:
        raise ValueError(f"Недостаточно месте для встраивания {n_bits} битов (capacity {flat.size}).")

    # Преобразовываем в массив
    bits_arr = np.array([0 if ch == '0' else 1 for ch in bits], dtype=np.uint8)
    # Встраиваем сообщение
    # Маска обнуляет младший бит
    mask = np.uint8(254)
    # Проводим побитовое И с маской, затем побитовое ИЛИ с битом сообщения
    flat[:n_bits] = (flat[:n_bits] & mask) | bits_arr

    # Преобразовываем массивы обратно и создаётся новое изображение
    new_arr = flat.reshape(arr.shape)
    new_img = Image.fromarray(new_arr, 'RGB')
    os.makedirs(os.path.dirname(stego_path) or ".", exist_ok=True)
    new_img.save(stego_path)

    meta = {
        "cover_path": os.path.abspath(cover_path),
        "stego_path": os.path.abspath(stego_path),
        "message_file": os.path.abspath(message_file),
        "payload": payload,
        "capacity_bits": int(capacity_bits),
        "allowed_bits": int(allowed_bits),
        "allowed_bytes": int(allowed_bytes),
        "message_bytes_used": int(len(msg_bytes)),
        "bits_written": int(n_bits),
        "truncated": bool(truncated)
    }
    return meta
