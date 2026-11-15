"""Register trained models in the MLflow Model Registry and promote the best one.

Each model trained by the pipeline is registered as a new version of
`registry.model_name` with descriptive tags. The version with the best test
`registry.selection_metric` gets the `production` alias, but only if it beats
the version currently holding that alias.
"""

from __future__ import annotations

import json

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from retinaops.common import ROOT, dvc_data_hash, git_commit, load_params, setup_mlflow, write_json


def main() -> None:
    params = load_params()
    reg = params["registry"]
    metric, name, alias = reg["selection_metric"], reg["model_name"], reg["production_alias"]
    mlflow = setup_mlflow(params)
    client = MlflowClient()

    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(
            name, description="5-class diabetic retinopathy grading on APTOS 2019 fundus images."
        )

    candidates = []
    for arch in params["models"]:
        run_id = json.loads((ROOT / "models" / arch / "run.json").read_text())["run_id"]
        metrics = json.loads((ROOT / "reports" / arch / "metrics.json").read_text())

        existing = client.search_model_versions(f"name='{name}' and run_id='{run_id}'")
        if existing:
            mv = existing[0]
            print(f"{arch}: run {run_id[:8]} already registered as v{mv.version}")
        else:
            mv = mlflow.register_model(f"runs:/{run_id}/model", name)
            tags = {
                "model_arch": arch,
                "role": "baseline" if arch == "resnet50" else "main",
                f"test_{metric}": metrics[metric],
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "data_version_md5": dvc_data_hash(),
                "git_commit": git_commit(),
            }
            for k, v in tags.items():
                client.set_model_version_tag(name, mv.version, k, str(v))
            print(f"{arch}: registered as v{mv.version} (test {metric}={metrics[metric]:.4f})")
        candidates.append({"arch": arch, "version": mv.version, "run_id": run_id, metric: metrics[metric]})

    best = max(candidates, key=lambda c: c[metric])

    current = None
    try:
        cur = client.get_model_version_by_alias(name, alias)
        cur_score = float(client.get_run(cur.run_id).data.metrics.get(f"test_{metric}", float("-inf")))
        current = {"version": cur.version, metric: cur_score}
    except MlflowException:
        pass

    promoted = current is None or best[metric] > current[metric] or current["version"] == best["version"]
    if promoted:
        client.set_registered_model_alias(name, alias, best["version"])
        print(f"Promoted v{best['version']} ({best['arch']}, {metric}={best[metric]:.4f}) to '{alias}'")
    else:
        print(f"Kept v{current['version']} as '{alias}' ({metric}={current[metric]:.4f} "
              f">= candidate {best[metric]:.4f})")

    write_json(
        {"registered_model": name, "selection_metric": metric, "candidates": candidates,
         "best_candidate": best, "previous_production": current, "promoted": promoted,
         "production_version": best["version"] if promoted else current["version"]},
        ROOT / "reports" / "registry.json",
    )


if __name__ == "__main__":
    main()
