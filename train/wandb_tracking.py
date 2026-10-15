"""Optional, process-local W&B logging for independent strategy/seed runs."""
from __future__ import annotations

import contextlib
import contextvars
import importlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


DEFAULTS = {
    "enabled": False,
    "project": "DNA-5mc-classifier",
    "entity": None,
    "name": None,
    "group": None,
    "mode": "online",
    "tags": [],
}
_active = contextvars.ContextVar("modification_wandb_run", default=None)


def validate_config(settings):
    if type(settings["enabled"]) is not bool:
        raise ValueError("wandb.enabled must be boolean")
    for key in ("project", "entity", "name", "group"):
        value = settings[key]
        if value is None and key != "project":
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"wandb.{key} must be a nonempty string" + (" or null" if key != "project" else ""))
    if settings["mode"] not in ("online", "offline", "disabled"):
        raise ValueError("wandb.mode must be online, offline, or disabled")
    if not isinstance(settings["tags"], list) or any(not isinstance(t, str) or not t.strip() for t in settings["tags"]):
        raise ValueError("wandb.tags must be a list of nonempty strings")


def check_available(cfg):
    settings = cfg.get("wandb", DEFAULTS)
    mode = os.environ.get("WANDB_MODE", settings["mode"])
    if mode not in ("online", "offline", "disabled"):
        raise ValueError("WANDB_MODE must be online, offline, or disabled")
    if settings["enabled"] and mode != "disabled":
        if "wandb" not in sys.modules and importlib.util.find_spec("wandb") is None:
            raise RuntimeError("wandb.enabled=true but wandb is not installed. Install it with "
                               "the training Python: python -m pip install wandb, or set wandb.enabled=false.")


@contextlib.contextmanager
def _preserve_rng():
    # SDK calls receive Python scalars only; they never operate on CUDA tensors.
    # Preserve the training process's CPU RNG streams, including Dropout's RNG.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


class Session:
    def __init__(self, cfg, run_config, run_dir):
        self.settings = cfg.get("wandb", DEFAULTS)
        self.mode = os.environ.get("WANDB_MODE", self.settings["mode"])
        self.run_dir = Path(run_dir)
        self.config = run_config
        self.run = None
        self.failed = False
        self.info = {}

    def start(self):
        if not self.settings["enabled"] or self.mode == "disabled":
            return
        check_available({"wandb": self.settings})
        suffix = f"{self.config['strategy']}_seed{self.config['seed']}"
        prefix = self.settings["name"] or self.config["experiment"]
        with _preserve_rng():
            sdk = importlib.import_module("wandb")
            try:
                self.run = sdk.init(
                    project=self.settings["project"], entity=self.settings["entity"],
                    name=f"{prefix}/{suffix}", group=self.settings["group"] or self.config["experiment"],
                    tags=self.settings["tags"], config=self.config, mode=self.mode,
                    dir=str(self.run_dir.resolve()), save_code=False,
                    settings=sdk.Settings(console="off", disable_git=True),
                )
                self.run.define_metric("global_step")
                self.run.define_metric("epoch")
                self.run.define_metric("train/*", step_metric="global_step")
                self.run.define_metric("perf/*", step_metric="global_step")
                self.run.define_metric("valid/*", step_metric="epoch")
                self.run.define_metric("epoch_train/*", step_metric="epoch")
            except Exception as exc:
                raise RuntimeError("W&B initialization failed. Check project/entity and wandb login, "
                                   "or use WANDB_MODE=offline / wandb.enabled=false.") from exc
        self.info = dict(id=self.run.id, name=self.run.name, project=self.run.project,
                         entity=self.run.entity, mode=self.mode,
                         group=self.settings["group"] or self.config["experiment"],
                         url=self.run.url if self.mode == "online" else None)
        self._write_info()
        print(f"[WANDB] mode={self.mode} name={self.run.name} id={self.run.id} "
              f"url={self.info['url'] or '(offline)'}", flush=True)

    def _write_info(self):
        path = self.run_dir / "wandb_run.json"
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(self.info, indent=2) + "\n")
        temporary.replace(path)

    def call(self, operation):
        if self.run is None or self.failed:
            return
        try:
            with _preserve_rng():
                operation(self.run)
        except Exception as exc:
            self.failed = True
            print(f"[WANDB] logging disabled after {type(exc).__name__}: {exc}; "
                  "local training and result files continue", flush=True)

    def finish(self, exit_code):
        if self.run is None:
            return
        try:
            with _preserve_rng():
                self.run.finish(exit_code=exit_code)
        except Exception as exc:
            print(f"[WANDB] finish failed: {type(exc).__name__}: {exc}", flush=True)
        if self.info:
            self.info.update(exit_code=exit_code, logging_failed=self.failed)
            try:
                self._write_info()
            except OSError as exc:
                print(f"[WANDB] cannot update wandb_run.json: {exc}", flush=True)


@contextlib.contextmanager
def track_run(cfg, run_config, run_dir):
    """Start inside the training worker and always finish before its next job."""
    session = Session(cfg, run_config, run_dir)
    token = _active.set(session)
    try:
        session.start()
        yield
    except BaseException as exc:
        session.finish(130 if isinstance(exc, KeyboardInterrupt) else 1)
        raise
    else:
        session.finish(0)
    finally:
        _active.reset(token)


def update_config(run_config):
    session = _active.get()
    if session is not None:
        session.call(lambda run: run.config.update(run_config, allow_val_change=True))


def log_train(epoch, global_step, batch_loss, interval_loss, running_loss, lr,
              samples_per_s, data_wait_s, elapsed_s, eta_s):
    session = _active.get()
    if session is not None:
        # Custom axes avoid W&B dropping validation records at the same step as
        # the last training batch. Do not pass W&B's internal `step=` argument.
        session.call(lambda run: run.log({
            "epoch": epoch, "global_step": global_step,
            "train/batch_loss": batch_loss, "train/interval_loss": interval_loss,
            "train/running_loss": running_loss, "train/lr": lr,
            "perf/samples_per_s": samples_per_s, "perf/data_wait_s": data_wait_s,
            "perf/elapsed_s": elapsed_s, "perf/eta_s": eta_s,
        }))


def log_epoch(row, global_step):
    session = _active.get()
    if session is not None:
        data = {"epoch": row["epoch"], "global_step": global_step,
                "epoch_train/loss": row["train_loss"]}
        data.update({f"valid/{key[6:]}": value for key, value in row.items()
                     if key.startswith("valid_") and value is not None})
        session.call(lambda run: run.log(data))


def log_results(metrics, run_config):
    session = _active.get()
    if session is not None:
        summary = {"best_epoch": metrics["best_epoch"], "best_valid_metric": metrics["best_valid_metric"],
                   "monitor_metric": metrics["monitor_metric"], "effective_monitor": metrics["effective_monitor"],
                   "epochs_completed": len(metrics["history"]),
                   "early_stopped": len(metrics["history"]) < run_config["training"]["epochs"],
                   "trainable_parameters": run_config["trainable_parameters"]}
        for split in ("train", "valid", "test"):
            for level in ("read", "site"):
                for key, value in metrics.get(f"{split}_{level}_level", {}).items():
                    if value is not None:
                        summary[f"final/{split}_{level}/{key}"] = value
        session.call(lambda run: run.summary.update(summary))
