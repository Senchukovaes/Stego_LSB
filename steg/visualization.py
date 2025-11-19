# steg/visualization.py
import matplotlib.pyplot as plt
import os


def save_metrics_table(metrics_dict, save_path_png):
    """Таблица с полными метриками включая chi2_total, df и p-value"""
    os.makedirs(os.path.dirname(save_path_png) or ".", exist_ok=True)

    chi2_overall = metrics_dict.get('chi2', {}).get('overall', {})
    chi2_total = chi2_overall.get('chi2_total', 0)
    chi2_df = chi2_overall.get('df', 0)
    chi2_p = chi2_overall.get('p_value', 0)

    # Получаем результаты по каналам
    channels_data = metrics_dict.get('chi2', {}).get('channels', {})

    rows = [
        ["Cover", os.path.basename(metrics_dict.get("cover", ""))],
        ["Stego", os.path.basename(metrics_dict.get("stego", ""))],
        ["Payload", f"{metrics_dict.get('payload', 0):.4f}"],
        ["PSNR (dB)", f"{metrics_dict.get('psnr', 0):.4f}"],
        ["SSIM", f"{metrics_dict.get('ssim', 0):.6f}"],
        ["Chi2 Total", f"{chi2_total:.4f}"],
        ["Chi2 DF", f"{chi2_df}"],
        ["Chi2 p-value", f"{chi2_p:.6f}"],
        ["AUC", f"{metrics_dict.get('auc', 0):.4f}"],
        ["", ""],  # Пустая строка для разделения
        ["Channel Results", ""],
        ["R: chi2/df/p", f"{channels_data.get('R', {}).get('chi2_total', 0):.2f}/"
                         f"{channels_data.get('R', {}).get('df', 0)}/"
                         f"{channels_data.get('R', {}).get('p_value', 0):.4f}"],
        ["G: chi2/df/p", f"{channels_data.get('G', {}).get('chi2_total', 0):.2f}/"
                         f"{channels_data.get('G', {}).get('df', 0)}/"
                         f"{channels_data.get('G', {}).get('p_value', 0):.4f}"],
        ["B: chi2/df/p", f"{channels_data.get('B', {}).get('chi2_total', 0):.2f}/"
                         f"{channels_data.get('B', {}).get('df', 0)}/"
                         f"{channels_data.get('B', {}).get('p_value', 0):.4f}"]
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis('off')
    table = ax.table(
        cellText=rows,
        colLabels=["Метрика", "Значение"],
        loc='center',
        cellLoc='left'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    plt.tight_layout()
    plt.savefig(save_path_png, dpi=150)
    plt.close()


def plot_metrics(metrics_dict, save_path_png):
    """График с основными метриками"""
    os.makedirs(os.path.dirname(save_path_png) or ".", exist_ok=True)

    chi2_overall = metrics_dict.get('chi2', {}).get('overall', {})

    labels = ["PSNR", "SSIM", "Chi2 Total", "AUC"]
    psnr = metrics_dict.get("psnr", 0)
    ssim = metrics_dict.get("ssim", 0)
    chi2_total = chi2_overall.get('chi2_total', 0)
    auc = metrics_dict.get("auc", 0)

    # Нормализуем значения для графика
    values = [psnr / 100, ssim, chi2_total / 1000, auc]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values)
    plt.title("Normalized Metrics Comparison")
    plt.xticks(rotation=45)

    # Добавляем значения на столбцы
    for bar, value, label in zip(bars, [psnr, ssim, chi2_total, auc], labels):
        if label == "PSNR":
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        elif label == "SSIM":
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        elif label == "Chi2 Total":
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path_png, dpi=150)
    plt.close()


def save_comparative_table(results_list, save_path_png):
    """Сравнительная таблица для экспериментов с payload"""
    os.makedirs(os.path.dirname(save_path_png) or ".", exist_ok=True)
    headers = ["Payload", "Chi2 Total", "Chi2 DF", "p-value", "AUC"]
    table = []

    for r in results_list:
        p = r.get("payload", 0)
        chi2_overall = r.get("chi2", {}).get("overall", {})
        chi2_total = chi2_overall.get("chi2_total", 0)
        chi2_df = chi2_overall.get("df", 0)
        p_value = chi2_overall.get("p_value", 0)
        auc = r.get("auc", 0)

        table.append([
            f"{p * 100:.3f}%",
            f"{chi2_total:.2f}",
            f"{chi2_df}",
            f"{p_value:.6f}",
            f"{auc:.4f}"
        ])

    fig, ax = plt.subplots(figsize=(8, 0.8 + 0.4 * len(table)))
    ax.axis('off')
    tbl = ax.table(cellText=table, colLabels=headers, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    plt.tight_layout()
    plt.savefig(save_path_png, dpi=150)
    plt.close()