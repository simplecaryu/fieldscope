"""Tests for fieldscope CLI."""

import json
from pathlib import Path

from click.testing import CliRunner

from fieldscope.cli import main


class TestCliRun:
    def test_run_requires_query(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run"])
        assert result.exit_code != 0

    def test_run_with_query_creates_output(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "spintronics", "--output-dir", str(tmp_path), "--auto-accept", "--dry-run"],
        )
        assert result.exit_code == 0
        # Should have created a run directory
        subdirs = list(tmp_path.iterdir())
        assert len(subdirs) == 1
        run_dir = subdirs[0]
        assert (run_dir / "state.json").exists()

    def test_run_dry_run_shows_stages(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "spintronics", "--output-dir", str(tmp_path), "--dry-run"],
        )
        assert result.exit_code == 0
        assert "keyword_expansion" in result.output

    def test_run_with_config(self, tmp_path):
        config_file = tmp_path / "fieldscope.toml"
        config_file.write_text('[seeds]\ntop_k = 20\n')
        out_dir = tmp_path / "output"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "test", "--config", str(config_file), "--output-dir", str(out_dir), "--dry-run"],
        )
        assert result.exit_code == 0

    def test_run_from_stage(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run", "test",
                "--output-dir", str(tmp_path),
                "--from-stage", "adaptive_clustering",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "adaptive_clustering" in result.output
        assert "keyword_expansion" not in result.output

    def test_run_invalid_from_stage(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "test", "--output-dir", str(tmp_path), "--from-stage", "bogus", "--dry-run"],
        )
        assert result.exit_code != 0


class TestCliInit:
    def test_init_creates_toml(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["init"])
            assert result.exit_code == 0
            assert Path("fieldscope.toml").exists()

    def test_init_does_not_overwrite(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("fieldscope.toml").write_text("existing")
            result = runner.invoke(main, ["init"])
            assert result.exit_code != 0
            assert "already exists" in result.output


class TestCliStatus:
    def test_status_with_valid_run(self, tmp_path):
        # Set up a fake run directory
        run_dir = tmp_path / "20260312_143022_test"
        run_dir.mkdir()
        state = {
            "run_id": "20260312_143022_test",
            "query": "test",
            "config_snapshot": {},
            "completed_stages": ["keyword_expansion", "initial_retrieval"],
            "current_stage": None,
            "stage_outputs": {},
        }
        (run_dir / "state.json").write_text(json.dumps(state))

        runner = CliRunner()
        result = runner.invoke(
            main, ["status", "20260312_143022_test", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "keyword_expansion" in result.output
        assert "initial_retrieval" in result.output

    def test_status_nonexistent_run(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["status", "nonexistent", "--output-dir", str(tmp_path)])
        assert result.exit_code != 0
