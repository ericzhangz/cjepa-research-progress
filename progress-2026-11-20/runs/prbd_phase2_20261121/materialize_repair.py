"""Restore the packaged Phase 2 repair evidence for local replay, using only the standard library."""

import argparse
import filecmp
import os
from pathlib import Path
import re
import shutil
import tempfile
from zipfile import ZipFile


ARCHIVE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ARCHIVE.parents[1] / "reproduce-output/prbd_phase2_repair_20261121"

# Each group lists the archived filenames, including the compressed prediction.
GROUPS = {
    "": """
        PROTOCOL.md REPORT.md admission.json response_geometry.json static_matched_roots.json
        static_physical_nonzero_roots.json static_results.json switch_adajepa_events.json
        switch_frozen_events.json switch_full_events.json switch_gru_events.json
        switch_optta_events.json switch_results.json switch_uniform_events.json
    """,
    "audit": """
        ORIGINAL_CHECKPOINT_AUDIT.md REPAIR_REGULARIZER_AUDIT.md original_checkpoint_audit.json
        original_final_fixed_tensors.pt original_final_static_density.pt
        original_initial_fixed_tensors.pt original_initial_static_density.pt
        repair_regularizer_final.json repair_regularizer_initial.json
    """,
    "oracle": """
        REPORT.md final.pt final_checks.json initial.pt initial_checks.json protocol.json result.json
        serialization_checks.json static_per_root.json static_predictions.pt static_results.json
        switch_adajepa_events.json switch_frozen_events.json switch_gru_events.json
        switch_nominal_events.json switch_optta_events.json switch_oracle_events.json
        switch_per_root.json switch_predictions.pt.zip switch_results.json switch_uniform_events.json
        training_full_measure.json training_result.json updates.jsonl
    """,
    "unlabeled": """
        batched_solve_regression.json blindness_and_causality_checks.json budget_verification.json
        checkpoint_reload_checks.json final.pt final_checkpoint_reload_checks.json final_checks.json
        initial.pt initial_checks.json initial_objective.json initial_static_writes.pt
        objective_history.json pred_static_prbd.pt pred_switch_prbd.pt preflight_structural_checks.json
        preflight_throughput.json probability_checks.json protocol.json result.json updates.jsonl
    """,
}


def materialize(output):
    files = {
        "repair_" + (group + "__" if group else "") + name: Path(group) / name.removesuffix(".zip")
        for group, names in GROUPS.items() for name in names.split()
    }
    output = output.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "run"
        for published, relative in files.items():
            source = ARCHIVE / published
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if published.endswith(".pt.zip"):
                with ZipFile(source) as archive:
                    if archive.namelist() != [relative.name]:
                        raise ValueError(f"{source} must contain exactly {relative.name}")
                    with archive.open(relative.name) as member, target.open("xb") as writer:
                        shutil.copyfileobj(member, writer)
            elif target.suffix == ".md":
                # Links in the flat release point to neighboring repair_* files.
                def restore_link(match):
                    link = os.path.relpath(files[match[1]], relative.parent)
                    return "](" + Path(link).as_posix() + ")"

                content = re.sub(r"\]\((repair_[\w.]+)\)", restore_link, source.read_text(encoding="utf-8"))
                target.write_text(content, encoding="utf-8", newline="\n")
            else:
                shutil.copyfile(source, target)

        if not output.exists() and not output.is_symlink():
            staged.rename(output)
        else:
            # Check the entire tree before filling any missing members. Replay outputs
            # may already have changed, so compare file contents before reusing them.
            for directory in [output, *(output / group for group in GROUPS if group)]:
                if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
                    raise FileExistsError(f"Conflicting directory: {directory}")
            missing = []
            for relative in files.values():
                dest = output / relative
                if dest.is_symlink() or (dest.exists() and not dest.is_file()):
                    raise FileExistsError(f"Conflicting file: {dest}")
                if not dest.exists():
                    missing.append(relative)
                elif not filecmp.cmp(staged / relative, dest, shallow=False):
                    raise FileExistsError(f"Different contents at {dest}")
            for relative in missing:
                dest = output / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                # An atomic, exclusive link installs only the private staged copy.
                os.link(staged / relative, dest)
    print(f"Materialized {len(files)} repair files at {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="destination run directory")
    materialize(parser.parse_args().output)
