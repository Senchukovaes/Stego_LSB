# steg/chi_square.py
from PIL import Image
import numpy as np
import scipy.stats as stats


def _channel_histogram(rgb_bytes: bytes, channel: int) -> list:
    # Построение гистограммы для указанного канала
    if channel not in (0, 1, 2):
        raise ValueError("channel must be 0 (R), 1 (G) or 2 (B)")

    n_pixels = len(rgb_bytes) // 3
    hist = [0] * 256 # Создаём массив из 256 нулей
    base = channel
    # Считаем, сколько раз встречается каждое значение яркости и строим гистограмму частот
    for i in range(n_pixels):
        v = rgb_bytes[3 * i + base]  # Доступ к каналу R, G или B
        hist[v] += 1

    return hist

# хи-квадрат тест для одного канала изображения
def hi2_lsb_channel(rgb_bytes: bytes, channel: int):
    hist = _channel_histogram(rgb_bytes, channel)
    chi2 = 0.0
    used_pairs = 0

    # Анализ пар (2k, 2k+1)
    for k in range(0, 256, 2):
        o0 = hist[k]  # Наблюдаемая частота для 2k
        o1 = hist[k + 1]  # Наблюдаемая частота для 2k+1
        s = o0 + o1

        if s == 0:
            continue  # Пропуск пустых пар

        e = s / 2.0  # Ожидаемая частота при равномерном распределении
        # Если бы распределение было равномерным, значения встречались бы одинаково часто

        # Вычисление χи-квадрат статистики для пары
        chi2 += (o0 - e) * (o0 - e) / e + (o1 - e) * (o1 - e) / e
        used_pairs += 1

    # Степени свободы: количество пар - 1
    df = max(used_pairs - 1, 1) # max гарантирует как минимум одну степень свободы
    return chi2, df


def calculate_p_value(chi2_stat: float, df: int) -> float:
    # Расчет p-value по χ² статистике и степеням свободы
    return 1 - stats.chi2.cdf(chi2_stat, df)



# Считаем хи-квадрат для всех каналов
def hi2_lsb_all_channels(rgb_bytes: bytes) -> dict:
    # χ²-тест для всех цветовых каналов с p-value
    result = {}
    for ch, name in enumerate(("R", "G", "B")):
        chi2, df = hi2_lsb_channel(rgb_bytes, ch)
        p_value = calculate_p_value(chi2, df)
        result[name] = {
            "chi2_total": float(chi2),
            "df": int(df),
            "p_value": float(p_value)
        }
    return result


def chi_square_stat(image_path: str):
    # χ² по всем каналам.
    # Возвращаем dict с результатами для каждого канала и общими метриками.
    img = Image.open(image_path).convert("RGB")
    rgb_bytes = img.tobytes()

    channel_results = hi2_lsb_all_channels(rgb_bytes)

    # Вычисление средних значений по каналам (общий p-value)
    total_chi2 = np.mean([ch["chi2_total"] for ch in channel_results.values()])
    total_df = np.mean([ch["df"] for ch in channel_results.values()])
    total_p_value = calculate_p_value(total_chi2, int(total_df))

    return {
        "channels": channel_results,
        "overall": {
            "chi2_total": float(total_chi2),
            "df": int(total_df),
            "p_value": float(total_p_value)
        }
    }


# Подготовка изображения и разделение на блоки
def chi2_block_scores(image_path: str, block_size: int = 16):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    h, w, c = arr.shape
    scores = []

    # iterate blocks top-left to bottom-right
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = arr[y:min(y + block_size, h), x:min(x + block_size, w)]
            if block.size == 0:
                continue

            # Convert block to bytes for analysis
            block_img = Image.fromarray(block, 'RGB')
            rgb_bytes = block_img.tobytes()

            # Get average chi2 across channels
            channel_results = hi2_lsb_all_channels(rgb_bytes)
            avg_chi2 = np.mean([ch["chi2_total"] for ch in channel_results.values()])
            scores.append(float(avg_chi2))

    return scores