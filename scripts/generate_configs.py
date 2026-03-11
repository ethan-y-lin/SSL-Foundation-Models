import yaml
from pathlib import Path
import argparse
import copy


def generate_config(
    template_config,
    output_root,
    ssl_method,
    dataset,
    split,
    seed,
    model
):

    # Load template
    with open(template_config, "r") as f:
        cfg = yaml.safe_load(f)

    # Update keys
    cfg["algorithm"] = ssl_method
    cfg["dataset"] = dataset

    save_path = f"./saved_models/lora/{ssl_method}/{dataset}/{split}/{seed}/{model}"

    cfg["save_dir"] = save_path
    cfg["load_path"] = save_path + "/log/latest_model.pth"
    cfg["save_name"] = f"{ssl_method}_{dataset}_{split}_{seed}_{model}"

    # Create directory
    config_dir = Path(output_root) / "lora" / ssl_method / dataset / split / str(seed) / model
    config_dir.mkdir(parents=True, exist_ok=True)

    # Write config
    config_path = config_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print(f"Created: {config_path}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--template", required=True)
    parser.add_argument("--output_root", default="config")

    parser.add_argument("--ssl_method", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--model", required=True)

    args = parser.parse_args()

    generate_config(
        args.template,
        args.output_root,
        args.ssl_method,
        args.dataset,
        args.split,
        args.seed,
        args.model,
    )


if __name__ == "__main__":
    main()

# python scripts/generate_configs.py --template "config/lora/fixmatch/resisc45/k1/seed0/clip/config.yaml" --ssl_method "fixmatch" --dataset "resisc45" --split "k1" --model "clip" --seed "seed42"