#!/usr/bin/env python3
"""
Automatic Config Template Generator

This script generates experiment config templates based on model configs.
It can be used as a pre-commit hook or run manually.

Usage:
    python scripts/generate_config_templates.py
    python scripts/generate_config_templates.py --config path/to/config.yaml
    python scripts/generate_config_templates.py --model llama3_expr24
    python scripts/generate_config_templates.py --dry-run
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


class ConfigTemplateGenerator:
    """Generate experiment config templates from model configs."""

    def __init__(self, config_path: str = ".config-template-gen.yaml"):
        self.root_dir = Path(__file__).parent.parent
        self.config_path = self.root_dir / config_path
        self.config = self._load_config()
        self.model_dir = self.root_dir / "configs" / "model"
        self.experiment_dir = self.root_dir / "configs" / "experiment"

    def _load_config(self) -> Dict[str, Any]:
        """Load the generation configuration."""
        if not self.config_path.exists():
            print(f"❌ Config file not found: {self.config_path}")
            print("Creating default config...")
            self._create_default_config()

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _create_default_config(self):
        """Create a default config file if it doesn't exist."""
        default_config = {
            "generate": {},
            "ignore": ["_template.yaml", "test.yaml"],
            "settings": {
                "create_template_dir": True,
                "backup_existing": True,
                "add_timestamp": True,
                "customize_fields": ["exp_name", "seed", "tags"],
            },
        }
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

    def _should_ignore(self, model_name: str) -> bool:
        """Check if a model should be ignored."""
        ignore_list = self.config.get("ignore", [])
        return any(pattern in model_name for pattern in ignore_list)

    def _is_enabled(self, model_name: str) -> bool:
        """Check if template generation is enabled for this model."""
        model_config = self.config.get("generate", {}).get(model_name, {})
        return model_config.get("enabled", False)

    def _load_model_config(self, model_path: Path) -> Dict[str, Any]:
        """Load a model config file."""
        with open(model_path) as f:
            return yaml.safe_load(f)

    def _create_template_header(self, source_config: str) -> str:
        """Create a header comment for the template."""
        settings = self.config.get("settings", {})
        if not settings.get("add_timestamp", True):
            return ""

        header_template = settings.get(
            "comment_header", "# AUTO-GENERATED TEMPLATE\n# Source: {source_config}\n"
        )

        return header_template.format(
            source_config=source_config,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _create_template_content(
        self, model_name: str, model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create template content from model config."""
        gen_config = self.config["generate"][model_name]

        # Base template structure
        template = {
            "# @package _global_": None,
            "defaults": [
                {"override /data": "CUSTOMIZE_DATA"},
                {"override /model": model_name},
                {"override /callbacks": "default"},
                {"override /trainer": "default"},
            ],
            "tags": gen_config.get("default_tags", ["template"]),
            "exp_name": "CUSTOMIZE_EXP_NAME",
            "seed": 42,
        }

        # Add trainer configuration
        template["trainer"] = {
            "max_epochs": "CUSTOMIZE_EPOCHS",
            "gradient_clip_val": 0.5,
            "accumulate_grad_batches": 4,
            "default_root_dir": "${paths.output_dir}",
            "precision": "bf16-true",
            "devices": 1,
        }

        # Add model overrides section (keep empty for template)
        template["model"] = {
            "optimizer": {
                "lr": "CUSTOMIZE_LR",
            },
            "# Add model-specific overrides here": None,
        }

        # Add data configuration
        template["data"] = {
            "num_workers": 8,
        }

        # Add logger configuration
        template["logger"] = {
            "wandb": {
                "project": "CUSTOMIZE_PROJECT",
                "offline": False,
            }
        }

        return template

    def _backup_file(self, file_path: Path):
        """Create a backup of existing file."""
        if file_path.exists():
            backup_path = file_path.with_suffix(
                f".yaml.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy2(file_path, backup_path)
            print(f"   📦 Backed up to: {backup_path.name}")

    def _write_template(
        self, output_path: Path, header: str, content: Dict[str, Any], description: str = ""
    ):
        """Write template to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Backup existing file
        if self.config.get("settings", {}).get("backup_existing", True):
            self._backup_file(output_path)

        with open(output_path, "w") as f:
            # Write header
            if header:
                f.write(header)
                f.write("\n")

            # Write description
            if description:
                f.write(f"# {description}\n")
                f.write(f"# Run with: python chemgfn/train.py experiment={output_path.stem}\n\n")

            # Write content
            yaml.dump(
                content,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=100,
            )

    def _validate_yaml(self, file_path: Path) -> bool:
        """Validate generated YAML file."""
        try:
            with open(file_path) as f:
                yaml.safe_load(f)
            return True
        except yaml.YAMLError as e:
            print(f"   ❌ YAML validation failed: {e}")
            return False

    def generate_template(self, model_name: str, dry_run: bool = False) -> bool:
        """Generate a single template for a model config."""
        # Check if enabled
        if not self._is_enabled(model_name):
            print(f"⏭️  Skipping {model_name} (disabled in config)")
            return False

        # Get generation config
        gen_config = self.config["generate"][model_name]
        model_path = self.model_dir / f"{model_name}.yaml"

        if not model_path.exists():
            print(f"❌ Model config not found: {model_path}")
            return False

        print(f"🔄 Generating template for: {model_name}")

        # Load model config
        try:
            model_config = self._load_model_config(model_path)
        except Exception as e:
            print(f"   ❌ Failed to load model config: {e}")
            return False

        # Create template content
        header = self._create_template_header(f"configs/model/{model_name}.yaml")
        content = self._create_template_content(model_name, model_config)
        description = gen_config.get("description", "")

        # Get output path
        output_template = gen_config.get(
            "output_template", f"configs/experiment/templates/{model_name}_template.yaml"
        )
        output_path = self.root_dir / output_template

        if dry_run:
            print(f"   📄 Would generate: {output_path}")
            print(f"   📝 Description: {description}")
            return True

        # Write template
        try:
            self._write_template(output_path, header, content, description)
            print(f"   ✅ Generated: {output_path}")
        except Exception as e:
            print(f"   ❌ Failed to write template: {e}")
            return False

        # Validate
        if self.config.get("settings", {}).get("validate_yaml", True):
            if self._validate_yaml(output_path):
                print(f"   ✅ YAML validation passed")
            else:
                return False

        return True

    def generate_all(self, dry_run: bool = False) -> Dict[str, bool]:
        """Generate all enabled templates."""
        results = {}

        print("=" * 60)
        print("🚀 Config Template Generator")
        print("=" * 60)

        # Get all model configs
        model_files = sorted(self.model_dir.glob("*.yaml"))
        enabled_models = [m for m in self.config.get("generate", {}).keys() if self._is_enabled(m)]

        print(f"\nFound {len(model_files)} model configs")
        print(f"Enabled: {len(enabled_models)} models\n")

        for model_name in enabled_models:
            results[model_name] = self.generate_template(model_name, dry_run)
            print()

        # Print summary
        if self.config.get("post_generation", {}).get("print_summary", True):
            self._print_summary(results, dry_run)

        return results

    def _print_summary(self, results: Dict[str, bool], dry_run: bool):
        """Print generation summary."""
        success = sum(1 for v in results.values() if v)
        failed = len(results) - success

        print("=" * 60)
        print("📊 Generation Summary")
        print("=" * 60)
        print(f"✅ Successful: {success}")
        print(f"❌ Failed: {failed}")

        if dry_run:
            print("\n⚠️  DRY RUN - No files were modified")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate experiment config templates")
    parser.add_argument(
        "--config", default=".config-template-gen.yaml", help="Path to generation config file"
    )
    parser.add_argument("--model", help="Generate template for specific model only")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without writing files"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available models and their status"
    )

    args = parser.parse_args()

    # Create generator
    generator = ConfigTemplateGenerator(args.config)

    # List mode
    if args.list:
        print("📋 Model Configs Status:")
        print("-" * 60)
        for model_name, config in generator.config.get("generate", {}).items():
            status = "✅ ENABLED" if config.get("enabled") else "❌ DISABLED"
            output = config.get("output_template", "N/A")
            print(f"{status} | {model_name:20s} → {output}")
        return 0

    # Generate specific model
    if args.model:
        success = generator.generate_template(args.model, args.dry_run)
        return 0 if success else 1

    # Generate all
    results = generator.generate_all(args.dry_run)
    all_success = all(results.values())
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
