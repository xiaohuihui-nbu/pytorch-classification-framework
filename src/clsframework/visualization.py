"""Offline plots and HTML reports shared by CLI and Python workflows."""

import csv
import json
import re
from html import escape
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont, ImageOps


def _figure(rows=1, cols=1, size=(9, 6)):
    figure = Figure(figsize=size, layout="constrained")
    FigureCanvasAgg(figure)
    return figure, figure.subplots(rows, cols)


def _table(rows):
    return (
        "<table>"
        + "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>" for row in rows)
        + "</table>"
    )


def _html(directory, title, body):
    path = Path(directory) / "report.html"
    path.write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title><style>"
        "body{font:16px system-ui,sans-serif;background:#f3f6fb;color:#17243b;"
        "max-width:1100px;margin:32px auto;padding:0 24px}"
        "h1,h2{color:#173f70}section{background:white;padding:24px;margin:20px 0;"
        "border-radius:12px;overflow:auto}img{max-width:100%;height:auto}"
        "table{border-collapse:collapse;width:100%}td{padding:10px;border-bottom:1px solid #dde4ed}"
        "a{color:#1763b1}p{line-height:1.7}figure{margin:20px 0}figcaption{overflow-wrap:anywhere}"
        f"</style><body><h1>{escape(title)}</h1>{body}</body></html>",
        encoding="utf-8",
    )
    return path


def metric_sections(directory, report, classes):
    """Render real recorded metrics; binary/multilabel do not invent a multiclass matrix."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scalar = [(k, v) for k, v in report.items() if isinstance(v, (int, float, str)) or v is None]
    body = "<section><h2>指标</h2>" + _table(scalar) + "</section>"
    per_class = report.get("per_class", {})
    if per_class:
        labels = list(per_class)
        figure, ax = _figure(size=(9, max(4, min(20, len(labels) * 0.3))))
        ax.barh(range(len(labels)), [per_class[k]["f1"] for k in labels], color="#2674b8")
        ax.set(
            yticks=range(len(labels)),
            yticklabels=range(len(labels)),
            xlim=(0, 1),
            xlabel="F1",
            ylabel="Class index",
            title="Per-class F1",
        )
        figure.savefig(directory / "per_class.png", dpi=130)
        body += '<section><h2>逐类指标</h2><img src="per_class.png" alt="Per-class F1">'
        body += (
            _table(
                [("图中索引", "类别", "Precision", "Recall", "F1", "Support")]
                + [
                    (
                        i,
                        label,
                        *[per_class[label].get(k, "—") for k in ("precision", "recall", "f1", "support")],
                    )
                    for i, label in enumerate(labels)
                ]
            )
            + "</section>"
        )
    if "confusion_matrix" in report:
        matrix = np.asarray(report["confusion_matrix"])
        normalized = np.divide(
            matrix,
            matrix.sum(1, keepdims=True),
            out=np.zeros_like(matrix, dtype=float),
            where=matrix.sum(1, keepdims=True) != 0,
        )
        for name, values in (("confusion_matrix", matrix), ("confusion_matrix_normalized", normalized)):
            figure, ax = _figure()
            plot = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if "normalized" in name else None)
            figure.colorbar(plot, ax=ax)
            ax.set(xlabel="Predicted class index", ylabel="True class index", title=name.replace("_", " "))
            if len(classes) <= 30:
                ax.set_xticks(range(len(classes)))
                ax.set_yticks(range(len(classes)))
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        value = values[i, j]
                        ax.text(
                            j,
                            i,
                            f"{value:.2f}" if "normalized" in name else str(value),
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if value > values.max() * 0.55 else "black",
                        )
            figure.savefig(directory / f"{name}.png", dpi=130)
            body += f'<section><img src="{name}.png" alt="{name}"></section>'
        body += "<section><h2>混淆矩阵类别索引</h2>" + _table(list(enumerate(classes))) + "</section>"
    return body


def evaluation_report(directory, report, classes):
    return _html(
        directory,
        f"分类评估 · {report.get('split', 'validation')}",
        metric_sections(directory, report, classes),
    )


def training_report(run_dir, output=None):
    run = Path(run_dir)
    metrics_file = run / "metrics.jsonl"
    if not metrics_file.is_file():
        raise ValueError(f"No recorded validation metrics: {metrics_file}")
    metrics = {row["epoch"]: row for row in _jsonl(metrics_file)}
    if not metrics:
        raise ValueError("No recorded validation epochs")
    events = {
        row["epoch"]: row for row in _jsonl(run / "events.jsonl") if row.get("event") == "epoch_completed"
    }
    directory = Path(output) if output else run / "visuals"
    directory.mkdir(parents=True, exist_ok=output is None)
    keys = ["epoch", "train_loss", "accuracy_top1", "macro_f1", "map", "lr"]
    rows = [
        {"epoch": epoch + 1, **{k: metrics[epoch].get(k, events.get(epoch, {}).get(k)) for k in keys[1:]}}
        for epoch in sorted(metrics)
    ]
    with (directory / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = _figure(2, 2, (11, 7))
    score = "accuracy_top1" if any(r["accuracy_top1"] is not None for r in rows) else "map"
    for ax, key in zip(axes.flat, ("train_loss", score, "macro_f1", "lr"), strict=True):
        available = [r for r in rows if r[key] is not None]
        if available:
            ax.plot([r["epoch"] for r in available], [r[key] for r in available], "o-", color="#2674b8")
        else:
            ax.text(0.5, 0.5, "Not recorded", ha="center", transform=ax.transAxes)
        ax.set(title=key, xlabel="Epoch (1-based)")
        ax.grid(alpha=0.2)
    figure.savefig(directory / "results.png", dpi=130)
    latest = metrics[max(metrics)]
    # Prepared labels preserve the exact class order, including binary positive-class reports.
    class_file = run / "classes.json"
    if not class_file.is_file():
        class_file = run / "bundle/classes.json"  # Legacy runs.
    classes = (
        json.loads(class_file.read_text(encoding="utf-8"))
        if class_file.is_file()
        else list(latest["per_class"])
    )
    status_file = run / "status.json"
    status = json.loads(status_file.read_text(encoding="utf-8")) if status_file.is_file() else {}
    body = "<section><p>训练目录：" + escape(str(run.resolve())) + "</p>"
    body += "<p>曲线来自各轮实际记录。以下逐类指标和混淆矩阵来自最后记录轮的验证集，"
    body += "不是最佳权重的独立评估。未记录的 loss 留空；未计算验证 loss。</p>"
    body += (
        _table(status.items())
        + '</section><section><h2>训练曲线</h2><img src="results.png" alt="Training curves">'
    )
    body += '<p><a href="results.csv">下载逐轮 CSV</a></p></section>'
    body += metric_sections(directory, latest, classes)
    return _html(directory, "分类训练报告", body)


def _jsonl(path):
    return (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if path.is_file()
        else []
    )


def positive_int(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    return value


class PredictionResult(dict):
    """JSON-compatible prediction with an explicit PIL plot/save interface."""

    def plot(self, *, top_k=5):
        positive_int(top_k, "top_k", 100)
        with Image.open(self["path"]) as source:
            picture = ImageOps.exif_transpose(source).convert("RGB")
        picture.thumbnail((800, 700))
        ranked = sorted(self["probabilities"].items(), key=lambda row: row[1], reverse=True)[:top_k]
        width = max(640, picture.width)
        canvas = Image.new("RGB", (width, picture.height + 112 + 50 * len(ranked)), "#f3f6fb")
        canvas.paste(picture, ((width - picture.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        font = None
        for filename in ("C:/Windows/Fonts/msyh.ttc", "DejaVuSans.ttf"):
            try:
                font = ImageFont.truetype(filename, 18)
                break
            except OSError:
                pass
        font = font or ImageFont.load_default(size=18)
        labels = self.get("labels", [self.get("label", "")])
        decision = ", ".join(labels) or "(none)"
        draw.text((20, picture.height + 16), f"Prediction: {decision}"[:100], font=font, fill="#173f70")
        if "threshold" in self:
            draw.text((20, picture.height + 46), f"Threshold: {self['threshold']}", font=font, fill="#173f70")
        for i, (label, probability) in enumerate(ranked):
            y = picture.height + 86 + 50 * i
            draw.text((20, y), f"{label}: {probability:.2%}"[:100], font=font, fill="#17243b")
            draw.rectangle((20, y + 28, width - 20, y + 36), fill="#dce4ef")
            draw.rectangle((20, y + 28, 20 + int((width - 40) * probability), y + 36), fill="#2674b8")
        return canvas

    def save(self, filename, *, top_k=5):
        path = Path(filename)
        if path.exists():
            raise FileExistsError(path)
        rendered = self.plot(top_k=top_k)
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(path)
        return path


def prediction_report(results, directory, *, top_k=5, max_images=64):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    body = f"<section><p>共 {len(results)} 张预测，展示前 {min(len(results), max_images)} 张。完整概率见 JSONL。</p></section>"
    for i, result in enumerate(results[:max_images]):
        stem = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(result["path"]).stem)[:60]
        relative = f"images/{i + 1:06d}-{stem}.jpg"
        PredictionResult(result).save(directory / relative, top_k=top_k)
        decision = result.get("labels", [result.get("label", "")])
        body += f'<section><figure><img src="{relative}" alt="Prediction {i + 1}"><figcaption>'
        body += (
            escape(str(result["path"]))
            + " · "
            + escape(", ".join(decision) or "(none)")
            + "</figcaption></figure>"
        )
        body += (
            _table(sorted(result["probabilities"].items(), key=lambda row: row[1], reverse=True)[:top_k])
            + "</section>"
        )
    return _html(directory, "图片分类预测报告", body)
