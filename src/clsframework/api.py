"""High-level classification workflow shared by Python and the CLI."""

import json
import logging
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from .config import Config, Logging, load_config, resolve_config
from .logging_utils import operation_log

ALIASES = {
    "epochs": "trainer.max_epochs",
    "batch": "loader.batch_size_per_device",
    "workers": "loader.num_workers",
    "device": "trainer.accelerator",
    "imgsz": "preprocessing.image_size",
    "lr0": "optimizer.lr",
    "seed": "experiment.seed",
    "name": "experiment.name",
    "project": "experiment.output_root",
    "offline": "runtime.offline",
    "plots": "visualization.enabled",
}


def configure_threads():
    # Set before importing numerical libraries. Preserve explicit caller settings.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")


def normalize_overrides(overrides=None, options=None):
    """Accept typed Python values or existing --set strings, rejecting duplicate keys."""
    values = []
    if overrides is not None:
        if isinstance(overrides, Mapping):
            values.extend((key, json.dumps(value, default=str)) for key, value in overrides.items())
        elif isinstance(overrides, Sequence) and not isinstance(overrides, str):
            for value in overrides:
                if not isinstance(value, str) or "=" not in value:
                    raise ValueError("Overrides must contain key=value strings")
                values.append(value.split("=", 1))
        else:
            raise TypeError("overrides must be a mapping or a sequence of key=value strings")
    values.extend((key, json.dumps(value, default=str)) for key, value in (options or {}).items())
    result, seen = [], set()
    for key, value in values:
        key = ALIASES.get(key, key)
        if key.split(".", 1)[0] not in Config.model_fields:
            raise ValueError(f"Unsupported option: {key}")
        if any(key == old or key.startswith(old + ".") or old.startswith(key + ".") for old in seen):
            raise ValueError(f"Conflicting overrides: {key}")
        seen.add(key)
        result.append(f"{key}={value}")
    return result


class Classifier:
    """Create from a YAML/Config for training, or a .pt/run directory for inference."""

    def __init__(self, model: str | Path | Config, *, allow_plugins: bool = False, log_config=None):
        configure_threads()
        self.allow_plugins = allow_plugins
        self.last_log: Path | None = None
        self.last_output: Path | None = None
        self.last_report: Path | None = None
        self._log_settings = Logging.model_validate(log_config) if log_config is not None else None
        self.run_dir: Path | None = None
        self.checkpoint: Path | None = None
        self._model_source: Path | None = None
        # Retained for callers loading legacy bundle directories.
        self.bundle: Path | None = None
        self._config: Config | None = None
        self._trained_config: Config | None = None
        self._config_path: Path | None = None
        if isinstance(model, Config):
            self._config = resolve_config(model.model_dump(mode="json"), Path.cwd())
        else:
            path = Path(model).resolve()
            if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
                self._config_path = path
                self._config = load_config(path)
            elif path.is_dir() or (path.is_file() and path.suffix.lower() in {".pt", ".ckpt"}):
                from .inference import model_directory, model_source

                self._model_source = model_source(path)
                self.checkpoint = self._model_source if self._model_source.is_file() else None
                self.bundle = self._model_source if self._model_source.is_dir() else None
                directory = model_directory(self._model_source)
                self.run_dir = directory if (directory / "config.resolved.yaml").is_file() else None
                snapshot = directory / "config.resolved.yaml"
                if snapshot.is_file():
                    self._config_path = snapshot
                    self._config = load_config(snapshot)
            else:
                raise ValueError("model must be an existing YAML, framework .pt or saved run directory")

    def _configuration(self, config=None, overrides=None, options=None):
        updates = normalize_overrides(overrides, options)
        if config is not None:
            if isinstance(config, Config):
                return resolve_config(config.model_dump(mode="json"), Path.cwd(), updates)
            return load_config(Path(config), updates)
        if self._config is None:
            raise ValueError("This standalone model has no dataset config; pass config='path/to/model.yaml'")
        base = self._config_path.parent if self._config_path else Path.cwd()
        return resolve_config(self._config.model_dump(mode="json"), base, updates)

    def _require_model(self):
        if self._model_source is None:
            raise ValueError("Train first, or create Classifier from a saved .pt/run directory")
        return self._model_source

    @contextmanager
    def _operation(self, mode, cfg=None):
        cfg = cfg or self._trained_config or self._config
        settings = self._log_settings or (cfg.logging if cfg else Logging())
        root = cfg.experiment.output_root if cfg else Path.cwd() / "runs"
        name = cfg.experiment.name if cfg else "inference"
        with operation_log(mode, settings, root.parent / "logs" / name) as path:
            self.last_log = path
            yield

    def prepare(self, *, overrides=None, **kwargs):
        """Validate data and cache the selected pretrained initialization without training."""
        from .data import prepare_data, save_prepared
        from .engine import resolve_runtime
        from .models import build_model
        from .utils import write_json

        cfg = self._configuration(overrides=overrides, options=kwargs)
        with self._operation("prepare", cfg):
            resolve_runtime(cfg)
            data = prepare_data(cfg)
            _, architecture, preprocessing, provenance = build_model(cfg, data.classes)
            directory = cfg.runtime.cache_dir / "prepared" / data.fingerprint
            save_prepared(data, directory)
            write_json(
                directory / "model_prepare.json",
                {"architecture": architecture, "preprocessing": preprocessing, "provenance": provenance},
            )
            logging.getLogger(__name__).info("数据/权重准备完成：%s", directory)
            return {"prepared": directory, "fingerprint": data.fingerprint, **data.report}

    def train(
        self, *, resume=None, finetune_from=None, fresh=False, stop_after_epoch=None, overrides=None, **kwargs
    ) -> Path:
        """Fine-tune from config; resume accepts a trusted framework checkpoint path."""
        cfg = self._configuration(overrides=overrides, options=kwargs)
        if not isinstance(fresh, bool):
            raise TypeError("fresh must be a boolean")
        if sum((resume is not None, finetune_from is not None, fresh)) > 1:
            raise ValueError("Choose one of resume, finetune_from or fresh")
        if fresh or resume is not None or finetune_from is not None:
            cfg.checkpoint.auto_resume = False
        if fresh:
            cfg.checkpoint.resume_from = None
            cfg.checkpoint.finetune_from = None
        if finetune_from is not None:
            if isinstance(finetune_from, bool):
                raise ValueError("finetune_from requires 'last' or a framework checkpoint/run path")
            cfg.checkpoint.finetune_from = (
                "last" if finetune_from == "last" else Path(finetune_from).resolve()
            )
            if cfg.checkpoint.resume_from is not None:
                raise ValueError("Choose strict resume or weight-only finetuning, not both")
        if resume is not None:
            if isinstance(resume, bool):
                raise ValueError("resume requires a checkpoint path, not a boolean")
            cfg.checkpoint.resume_from = Path(resume).resolve()
            cfg.checkpoint.finetune_from = None
        if stop_after_epoch is not None and (
            isinstance(stop_after_epoch, bool)
            or not isinstance(stop_after_epoch, int)
            or stop_after_epoch < 1
        ):
            raise ValueError("stop_after_epoch must be a positive integer")
        return self._train(cfg, stop_after_epoch=stop_after_epoch)

    def smoke(self, *, overrides=None, **kwargs) -> Path:
        """Run two train/validation batches in a separate one-epoch experiment."""
        cfg = self._configuration(overrides=overrides, options=kwargs)
        cfg.experiment.name += "-smoke"
        cfg.trainer.max_epochs = 1
        cfg.scheduler.warmup_epochs = 0
        cfg.checkpoint.resume_from = None
        cfg.checkpoint.finetune_from = None
        cfg.checkpoint.auto_resume = False
        return self._train(cfg, smoke=True)

    def _train(self, cfg, **kwargs):
        from .engine import train

        mode = "smoke" if kwargs.get("smoke") else "train"
        with self._operation(mode, cfg):
            directory = train(cfg, self._config_path, **kwargs)
        if self.last_log is not None:
            from .utils import append_json

            append_json(directory / "operation_logs.jsonl", {"operation": mode, "log": str(self.last_log)})
        self.run_dir = directory
        self.checkpoint = directory / "checkpoints/best.pt"
        self._model_source = self.checkpoint if self.checkpoint.is_file() else None
        self.bundle = None
        self._trained_config = cfg.model_copy(deep=True)
        self.last_output = directory
        report_path = directory / "visuals/report.html"
        self.last_report = report_path if cfg.visualization.enabled and report_path.is_file() else None
        return directory

    def report(self, *, output=None):
        """Render recorded training metrics without rerunning training or evaluation."""
        from .visualization import training_report

        if self.run_dir is None:
            raise ValueError("Train first, or load a saved run directory")
        self.last_report = training_report(self.run_dir, output)
        return self.last_report

    def val(self, *, config=None, split="val", output=None, overrides=None, **kwargs):
        """Evaluate a saved model on a locked dataset split (val by default)."""
        source = self._require_model()
        cfg = self._configuration(config if config is not None else self._trained_config, overrides, kwargs)
        from .engine import resolve_runtime
        from .inference import evaluate_model

        with self._operation("test" if split == "test" else "val", cfg):
            resolve_runtime(cfg)
            report = evaluate_model(source, cfg, split, output, self.allow_plugins)
            self.last_output = Path(report["output_dir"])
            self.last_report = self.last_output / "report.html" if cfg.visualization.enabled else None
            metrics = {key: value for key, value in report.items() if isinstance(value, (int, float))}
            logging.getLogger(__name__).info("评估完成：split=%s, metrics=%s", split, metrics)
            return report

    def test(self, *, config=None, output=None, overrides=None, **kwargs):
        """Evaluate the held-out test split."""
        return self.val(config=config, split="test", output=output, overrides=overrides, **kwargs)

    def predict(
        self,
        source: str | Path,
        *,
        output=None,
        batch=16,
        save=False,
        project=None,
        name=None,
        top_k=None,
        max_images=None,
    ) -> list[dict]:
        """Return predictions for a local image/directory; optionally save JSONL."""
        model_path = self._require_model()
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ValueError("batch must be a positive integer")
        source = Path(source)
        import torch

        from .config import Visualization
        from .inference import predict
        from .visualization import PredictionResult, positive_int, prediction_report

        settings = self._trained_config or self._config
        settings = settings.visualization if settings else Visualization()
        top_k = positive_int(settings.top_k if top_k is None else top_k, "top_k", 100)
        max_images = positive_int(
            settings.max_images if max_images is None else max_images, "max_images", 1000
        )
        if not isinstance(save, bool):
            raise TypeError("save must be a boolean")
        if (project is not None or name is not None) and (not save or output is not None):
            raise ValueError("project/name require save=True without an explicit output")
        if name is not None and (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or any(c in name for c in '/\\<>:"|?*')
        ):
            raise ValueError("name must be a simple directory name")
        if save and output is None:
            from datetime import datetime
            from uuid import uuid4

            directory = Path(project or "runs/predict") / (
                name or f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
            )
            if directory.exists():
                raise FileExistsError(directory)
            output = directory / "predictions.jsonl"
        gallery = Path(output).parent / f"{Path(output).stem}_visuals" if save else None
        if gallery is not None and gallery.exists():
            raise FileExistsError(gallery)
        self.last_output = None
        self.last_report = None

        torch.set_num_threads(2)
        with self._operation("predict"), TemporaryDirectory(prefix="cls-predict-") as temporary:
            if not source.exists():
                raise FileNotFoundError(source)
            target = Path(output) if output is not None else Path(temporary) / "predictions.jsonl"
            predict(model_path, source, target, self.allow_plugins, batch)
            results = [
                PredictionResult(json.loads(line)) for line in target.read_text(encoding="utf-8").splitlines()
            ]
            self.last_output = target.resolve() if output is not None else None
            if save:
                self.last_report = prediction_report(
                    results, gallery, top_k=top_k, max_images=max_images
                ).resolve()
            logging.getLogger(__name__).info(
                "推理完成：source=%s, samples=%s, output=%s", source, len(results), output
            )
            return results

    def __call__(self, source: str | Path, **kwargs):
        return self.predict(source, **kwargs)

    def export(self, *, format="onnx", output=None):
        """Export ONNX with numerical verification; other formats are rejected."""
        if format != "onnx":
            raise ValueError("Only format='onnx' is supported")
        source = self._require_model()
        import torch

        from .inference import export_onnx, model_directory

        torch.set_num_threads(2)
        target = Path(output) if output is not None else model_directory(source) / "exports/model.onnx"
        with self._operation("export"):
            report = export_onnx(source, target, self.allow_plugins)
            logging.getLogger(__name__).info("ONNX 导出及数值检查完成：%s", target)
            return report
