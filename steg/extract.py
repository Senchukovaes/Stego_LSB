# steg/extract.py
from PIL import Image
import numpy as np
import os
import json
from .metrics import analyze_pair_for_payloads, analyze_pair_experiment
from .embed import embed_message_with_checks, compute_capacity_bits_from_array, SERVICE_BITS
from .visualization import save_comparative_table

# Обновлённый список дефолтных payload'ов — теперь включает 50% и 70%
DEFAULT_PAYLOADS = [0.001, 0.005, 0.01, 0.05, 0.5, 0.7]

def safe_decode(data: bytes) -> str:
    # 1) Try UTF-8
    try:
        return data.decode("utf-8")
    except:
        pass

    # 2) Try CP1251 (Windows encoding)
    try:
        return data.decode("cp1251")
    except:
        pass

    # 3) Last resort: Latin-1
    return data.decode("latin-1", errors="replace")


def _read_meta_for_stego(stego_path: str):
    meta_path = stego_path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return None
    return None

# Поиск исходного изображения
def _infer_cover_from_stego(stego_path: str):
    base = os.path.splitext(os.path.basename(stego_path))[0]
    candidates = []
    candidates.append(os.path.join("imgs", "original", base + ".png"))
    for suf in ["_steg", "-steg", "_stego", "-stego", "_payload", "-payload"]:
        if base.endswith(suf):
            cand_name = base[:-len(suf)] + ".png"
            candidates.append(os.path.join("imgs", "original", cand_name))
            candidates.append(cand_name)
    candidates.append(base + ".png")
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None

def extract_message(stego_path: str, max_chars: int = 1000000):
    # Подгружаем изображение и преобразовываем в массив
    img = Image.open(stego_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    arr = arr[:, : , [2, 1, 0]]  # BGR

    flat = arr.flatten()

    max_bits = min(len(flat), max_chars * 8) # ограничиваем максимальное количество читаемых бит
    # Само извлечение (побитовое И с 00000001, получаем последние биты)
    bits = [str(int(flat[i] & 1)) for i in range(max_bits)]
    # Преобразовываем биты в байты и ищем маркер
    bytes_out = bytearray()
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:
            break
        b = int(''.join(byte_bits), 2)
        if b == 0:
            break
        bytes_out.append(b)
    # try:
    #     return bytes_out.decode('utf-8'), bytes(bytes_out)
    # except Exception:
    #     return bytes_out.decode('latin-1', errors='replace'), bytes(bytes_out)
    # декодируем текст
    return safe_decode(bytes_out), bytes(bytes_out)


def extract_message_and_analyze(stego_path: str, outdir: str = "results", payloads: list = None):
    if payloads is None:
        payloads = DEFAULT_PAYLOADS

    # Извлекаем сообщение
    msg_str, msg_bytes = extract_message(stego_path)
    base = os.path.splitext(os.path.basename(stego_path))[0]
    result_root = os.path.join(outdir, base)
    os.makedirs(result_root, exist_ok=True)

    # Сохранение необработанных байтов
    extracted_bin = os.path.join(result_root, f"{base}_extracted.bin")
    with open(extracted_bin, "wb") as f:
        f.write(msg_bytes)

    # Сохраняем восстановленный текст
    extracted_txt = os.path.join(result_root, f"{base}_extracted.txt")
    with open(extracted_txt, "w", encoding="utf-8", errors="replace") as f:
        f.write(msg_str)

    print("[extract] Extracted message saved to", extracted_txt)
    print("[extract] Message preview:", (msg_str[:200] + "...") if len(msg_str) > 200 else msg_str)

    # Находим оригинал картинки
    meta = _read_meta_for_stego(stego_path)
    if meta and meta.get("cover_path") and os.path.exists(meta["cover_path"]):
        cover_path = meta["cover_path"]
        orig_message_file = meta.get("message_file")
        print("[extract] Found cover in meta:", cover_path)
    else:
        cover_path = _infer_cover_from_stego(stego_path)
        orig_message_file = None
        print("[extract] Inferred cover:", cover_path)

    # Анализ
    payload_actual = float(meta["payload"]) if (meta and "payload" in meta) else 0.005
    print(f"[extract] Running full analysis for payload={payload_actual}")
    main_folder = os.path.join(result_root, f"payload_{int(payload_actual*10000)}")
    os.makedirs(main_folder, exist_ok=True)
    res_actual = None
    if cover_path:
        res_actual = analyze_pair_for_payloads(cover_path, stego_path, payload_actual, main_folder)
    else:
        print("[extract] Cover not found - skipping full analysis for actual stego.")

    # Эксперимент для разных payload
    # Use original message file if available; else use extracted bytes
    if orig_message_file and os.path.exists(orig_message_file):
        with open(orig_message_file, "rb") as f:
            master_bytes = f.read()
    else:
        master_bytes = msg_bytes

    experiments = []
    for p in payloads:
        payload_folder = os.path.join(result_root, f"payload_{int(p*10000)}")
        os.makedirs(payload_folder, exist_ok=True)
        if not cover_path:
            print(f"[experiment] Can't run payload {p}: cover not found.")
            continue
        # compute allowed bytes
        img = Image.open(cover_path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        capacity_bits = compute_capacity_bits_from_array(arr, lsb_count=1)
        allowed_bits = int(np.floor(p * capacity_bits))
        if allowed_bits <= SERVICE_BITS:
            print(f"[experiment] payload {p} gives no room (allowed_bits={allowed_bits}). Skipping.")
            continue
        message_bits = allowed_bits - SERVICE_BITS
        message_bits = (message_bits // 8) * 8
        allowed_bytes = message_bits // 8
        truncated_bytes = master_bytes[:allowed_bytes] if len(master_bytes) > allowed_bytes else master_bytes
        tmp_msg_path = os.path.join(payload_folder, "tmp_message.bin")
        with open(tmp_msg_path, "wb") as f:
            f.write(truncated_bytes)
        # create stego file (we produce stego images for experiments)
        stego_out = os.path.join(payload_folder, os.path.basename(stego_path).replace(".png", f"_payload{int(p*10000)}.png"))
        try:
            meta_info = embed_message_with_checks(cover_path, stego_out, tmp_msg_path, payload=p)
            with open(stego_out + ".meta.json", "w", encoding="utf-8") as f:
                json.dump(meta_info, f, indent=2, ensure_ascii=False)
            # lightweight analysis: only chi2 + AUC
            res = analyze_pair_experiment(cover_path, stego_out, payload=p, outdir=payload_folder)
            experiments.append(res)
        except Exception as e:
            print(f"[experiment] payload {p} failed: {e}")
            continue

    # Save experiments summary JSON and comparative PNG table in the top result folder
    summary_path = os.path.join(result_root, f"{base}_experiments_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "actual": res_actual,
            "experiments": experiments
        }, f, indent=2, ensure_ascii=False)
    print("[extract] Experiments finished, summary saved to", summary_path)

    # save comparative PNG table with chi2 and auc for all payloads (root result folder)
    comp_png = os.path.join(result_root, f"{base}_payload_comparison.png")
    save_comparative_table(experiments, comp_png)
    print("[extract] Comparative payload table saved to", comp_png)

    return {
        "extracted_text": extracted_txt,
        "actual_metrics": res_actual,
        "experiments_summary": summary_path
    }
