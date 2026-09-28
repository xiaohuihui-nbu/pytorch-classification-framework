import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from .api import ALIASES, Classifier, configure_threads
from .config import Config, load_config

app = typer.Typer(no_args_is_help=True, help="PyTorch configuration-driven image classification")
ConfigOption = Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)]
Overrides = Annotated[list[str] | None, typer.Option("--set", help="Override key=value; repeatable")]


def show(value):
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


@app.command()
def init(recipe: Annotated[str, typer.Option()], output: Annotated[Path, typer.Option()]):
    """Generate an editable, complete YAML config without downloading resources."""
    import yaml

    if output.exists():
        raise typer.BadParameter(f"Refusing to overwrite {output}")
    cfg = Config()
    cfg.experiment.name = recipe
    if recipe in {"cifar10_resnet18", "mnist_resnet18"}:
        cfg.dataset.provider = "torchvision"
        cfg.dataset.name = recipe.split("_")[0]
        cfg.dataset.download = True
        cfg.dataset.root = Path("data") / cfg.dataset.name
    elif recipe == "custom_resnet18":
        cfg.dataset.root = Path("data/my_dataset")
    elif recipe == "multilabel":
        cfg.task.type = "multilabel"
        cfg.dataset.provider = "manifest"
        cfg.dataset.root = Path("data/my_dataset")
        cfg.dataset.manifest = Path("data/my_dataset/samples.jsonl")
        cfg.dataset.classes_file = Path("data/my_dataset/classes.json")
        cfg.loss.name = "bce_with_logits"
        cfg.evaluation.monitor = "val/map"
    else:
        raise typer.BadParameter("Choose cifar10_resnet18, mnist_resnet18, custom_resnet18 or multilabel")
    cfg.model.weights.source = "provider_default"
    cfg.preprocessing.source = "model_weights"
    cfg.preprocessing.image_size = "auto"
    cfg.optimizer.lr = 0.0003
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True), encoding="utf-8")
    show({"config": output.resolve(), "next": f"cls doctor -c {output}"})


@app.command("list")
def list_components(kind: str):
    """List integrated catalogs; validation status is in docs/VALIDATION.md."""
    from .registry import DATASET_CATALOG, LOSS_CATALOG, MODEL_CATALOG

    catalogs = {
        "models": {
            "timm": MODEL_CATALOG,
            "builtin": ["tiny_cnn"],
            "torchvision": "model names via get_model",
        },
        "datasets": {"torchvision": DATASET_CATALOG, "custom": ["imagefolder", "manifest"]},
        "losses": LOSS_CATALOG,
    }
    if kind not in catalogs:
        raise typer.BadParameter("Choose models, datasets or losses")
    show(catalogs[kind])


@app.command()
def schema(output: Annotated[Path, typer.Option()]):
    """Write the exact supported JSON Schema for configuration editors."""
    from .utils import write_json

    write_json(output, Config.model_json_schema())
    show({"schema": output})


@app.command()
def doctor(config: ConfigOption, overrides: Overrides = None):
    """Read-only config/environment check. Does not download or train."""
    from .engine import environment

    cfg = load_config(config, overrides)
    show(
        {
            "environment": environment(),
            "config": cfg.model_dump(mode="json"),
            "dataset_root_exists": cfg.dataset.root.exists(),
            "validation": "schema passed; run prepare for data integrity",
        }
    )


@app.command()
def prepare(config: ConfigOption, overrides: Overrides = None):
    """Prepare selected data/weights and write an immutable manifest snapshot."""
    show(Classifier(config).prepare(overrides=overrides))


@app.command()
def train(
    config: ConfigOption,
    overrides: Overrides = None,
    resume: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    finetune_from: Annotated[
        str | None, typer.Option(help="last, or trusted framework run/checkpoint path")
    ] = None,
    fresh: bool = False,
    stop_after_epoch: Annotated[int | None, typer.Option(min=1)] = None,
):
    """Train; --resume accepts only trusted checkpoints created by this framework."""
    model = Classifier(config)
    directory = model.train(
        overrides=overrides,
        resume=resume,
        finetune_from=finetune_from,
        fresh=fresh,
        stop_after_epoch=stop_after_epoch,
    )
    show({"run_dir": directory, "report": model.last_report, "log": model.last_log})


@app.command()
def smoke(config: ConfigOption, overrides: Overrides = None):
    """Train and validate two real batches using a separate smoke run."""
    show({"run_dir": Classifier(config).smoke(overrides=overrides)})


@app.command()
def test(
    bundle: Annotated[Path, typer.Option(exists=True)],
    config: Annotated[Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)] = None,
    split: str = "test",
    output: Path | None = None,
    allow_plugins: bool = False,
    overrides: Overrides = None,
):
    """Evaluate the locked split independently using a saved inference bundle."""
    show(
        Classifier(bundle, allow_plugins=allow_plugins).val(
            config=config, split=split, output=output, overrides=overrides
        )
    )


@app.command()
def val(
    bundle: Annotated[Path, typer.Option(exists=True)],
    config: Annotated[Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)] = None,
    split: str = "val",
    output: Path | None = None,
    allow_plugins: bool = False,
    overrides: Overrides = None,
):
    """Evaluate the validation split; config defaults to the run's saved snapshot."""
    test(bundle, config, split, output, allow_plugins, overrides)


@app.command()
def predict(
    bundle: Annotated[Path, typer.Option(exists=True)],
    inputs: Annotated[Path, typer.Option("--input", exists=True)],
    output: Annotated[Path | None, typer.Option()] = None,
    batch_size: int = 16,
    allow_plugins: bool = False,
    save: bool = True,
    project: Path | None = None,
    name: str | None = None,
    topk: Annotated[int | None, typer.Option(min=1, max=100)] = None,
    max_images: Annotated[int | None, typer.Option(min=1, max=1000)] = None,
):
    """Predict images on CPU using bundle preprocessing and labels."""
    model = Classifier(bundle, allow_plugins=allow_plugins)
    results = model.predict(
        inputs,
        output=output,
        batch=batch_size,
        save=save,
        project=project,
        name=name,
        top_k=topk,
        max_images=max_images,
    )
    show(
        {
            "samples": len(results),
            "output": model.last_output,
            "report": model.last_report,
            **({"results": results} if model.last_output is None else {}),
        }
    )


@app.command()
def report(
    run_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Path | None = None,
):
    """Generate offline plots/HTML from recorded epochs without loading weights."""
    from .visualization import training_report

    show({"report": training_report(run_dir, output)})


@app.command("export")
def export_model(
    bundle: Annotated[Path, typer.Option(exists=True)],
    output: Annotated[Path, typer.Option()],
    allow_plugins: bool = False,
    format: str = "onnx",
):
    """Export ONNX and check numerical parity; requires the export extra."""
    show(Classifier(bundle, allow_plugins=allow_plugins).export(output=output, format=format))


def normalize_arguments(arguments: list[str]) -> list[str]:
    """Translate mode key=value syntax while preserving the existing Typer options."""
    if arguments and arguments[0] == "classify":
        arguments = arguments[1:]
    if not arguments or not any("=" in arg and not arg.startswith("--") for arg in arguments[1:]):
        return arguments
    mode = arguments[0]
    if mode not in {"train", "smoke", "prepare", "doctor", "val", "test", "predict", "export", "report"}:
        return arguments
    config_modes = {"train", "smoke", "prepare", "doctor", "val", "test"}
    option_names = {"output": "--output", "allow_plugins": "--allow-plugins"}
    if mode in config_modes:
        option_names.update(config="--config", cfg="--config")
    option_names["model"] = "--config" if mode in {"train", "smoke", "prepare", "doctor"} else "--bundle"
    if mode == "report":
        option_names["model"] = "--run-dir"
    if mode == "train":
        option_names.update(
            resume="--resume",
            stop_after_epoch="--stop-after-epoch",
            finetune_from="--finetune-from",
            fresh="--fresh",
        )
    if mode in {"val", "test"}:
        option_names["split"] = "--split"
    if mode == "predict":
        option_names.update(
            source="--input",
            batch="--batch-size",
            save="--save",
            project="--project",
            name="--name",
            topk="--topk",
            max_images="--max-images",
        )
    if mode == "export":
        option_names["format"] = "--format"
    result, seen = [mode], set()
    option_value = False
    for arg in arguments[1:]:
        # Values of --set and other options may themselves contain '='.
        if option_value:
            result.append(arg)
            option_value = False
            continue
        if arg.startswith("-"):
            flag = arg.split("=", 1)[0]
            canonical_flag = {
                "-c": "--config",
                "--no-allow-plugins": "--allow-plugins",
                "--no-fresh": "--fresh",
                "--no-save": "--save",
            }.get(flag, flag)
            if canonical_flag != "--set":
                if canonical_flag in seen:
                    raise ValueError(f"Duplicate argument: {flag}")
                seen.add(canonical_flag)
            result.append(arg)
            option_value = "=" not in arg and arg not in {
                "--allow-plugins",
                "--no-allow-plugins",
                "--fresh",
                "--no-fresh",
                "--save",
                "--no-save",
                "--help",
            }
            continue
        if "=" not in arg:
            raise ValueError(f"Expected key=value, got: {arg}")
        key, value = arg.split("=", 1)
        canonical = option_names.get(key, ALIASES.get(key, key))
        if canonical in seen:
            raise ValueError(f"Duplicate argument: {key}")
        seen.add(canonical)
        if key in option_names:
            if key in {"allow_plugins", "fresh", "save"}:
                if value.lower() not in {"true", "false"}:
                    raise ValueError(f"{key} must be true or false")
                result.append(canonical if value.lower() == "true" else canonical.replace("--", "--no-", 1))
            else:
                result.extend([canonical, value])
        elif mode in config_modes and canonical.split(".", 1)[0] in Config.model_fields:
            result.extend(["--set", f"{canonical}={value}"])
        else:
            raise ValueError(f"Unsupported {mode} argument: {key}")
    return result


def main():
    configure_threads()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        arguments = normalize_arguments(sys.argv[1:])
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise SystemExit(2) from exc
    app(args=arguments)


if __name__ == "__main__":
    main()
