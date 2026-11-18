"""Compute §49.8 lifecycle effects from fixed replay and physical candidate scores."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from research.p1a.stage_a_fourroom import CENTER, write_json

HERE = Path(__file__).resolve().parent
QUAL = HERE.parent / 'stage_a_fourroom_clean_20261117'
PHASES = ('initial_probe_retained_fork', 'reversal_1_retained_fork', 'reversal_2_retained_fork')
EPSILON = 1e-5


def records(rows, label):
    return [(row, mode) for row in rows for tag, mode in row['phase_labels'] if tag == label]


def key(row):
    return row['episode'].rsplit('_', 1)[0]


def regret(row, mode, choice='written'):
    return row['choices'][choice]['first_block_regret_by_mode'][mode]


def frozen_fit_alignment(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    mapping = data['physical_to_explanation']
    if data.get('frozen_before_heldout') is not True or not data.get('source'):
        raise ValueError('An explicit mapping must identify its fit-only source and be frozen before held-out evaluation')
    if mapping is not None and sorted(mapping) != [0, 1]:
        raise ValueError('The physical-to-explanation mapping must be a permutation of [0, 1], or null when unresolved')
    return mapping, {'path': str(path.resolve()), **data}


def probe_confirmation(rows, probe, arrival, alignment):
    events = records(rows, f'{probe}_probe_arrival_{arrival}')
    useful = [(row, mode) for row, mode in events if row['first_mode_difference_raw_step'] is not None]
    signed = [(2 * alignment[mode] - 1) * row['evidence'] for row, mode in useful] if alignment else None
    return {'probe': probe, 'arrival': arrival, 'legal_detectable_count': len(useful),
            'correct_evidence_direction': sum(value > 0 for value in signed) if signed is not None else None,
            'median_oriented_evidence': float(np.median(signed)) if signed else None,
            'postwrite_support_on_actual_majority': sum(row['probabilities'][alignment[mode]] > .5
                                                       for row, mode in useful) if alignment else None,
            'events': [{'episode': row['episode'], 'root': row['root'], 'physical_mode_evaluation_only': mode,
                        'evidence': row['evidence'],
                        'oriented_evidence': (2 * alignment[mode] - 1) * row['evidence'] if alignment else None,
                        'support_on_actual_explanation': row['probabilities'][alignment[mode]] if alignment else None}
                       for row, mode in useful]}



def region(fork):
    x, y = fork['start']
    return ('N' if y < CENTER else 'S') + ('W' if x < CENTER else 'E')


def choice_details(row, mode, branch, fork):
    choice = row['choices'][f'branch_{branch}']
    correct, opposite = fork['true_indices'][mode], fork['true_indices'][1-mode]
    return {'explanation': branch, 'candidate': choice['selected'],
            'first_block_group': choice['first_block_group'],
            'first_block_regret': choice['first_block_regret_by_mode'][mode],
            'full_plan_regret': choice['full_plan_regret_by_mode'][mode],
            'correct_route_margin': choice['scores'][opposite] - choice['scores'][correct]}


def local_diagnostics(rows, tasks):
    retained = [entry for phase in PHASES for entry in records(rows, phase)]
    result = {}
    for area in sorted({region(fork) for fork in tasks.values()}):
        local = [(row, mode) for row, mode in retained if region(tasks[key(row)]) == area]
        # Keep both predeclared assignments. These indices never enter forecasting or deployment.
        permutations = []
        for mapping in itertools.permutations((0, 1)):
            selected = []
            for row, mode in local:
                fork = tasks[key(row)]
                both = [choice_details(row, mode, branch, fork) for branch in (0, 1)]
                selected.append({'episode': row['episode'], 'root': row['root'],
                                 'physical_mode_evaluation_only': mode,
                                 'mapped_explanation': mapping[mode], 'branch_choices': both,
                                 'support_on_mapped_explanation': row['probabilities'][mapping[mode]],
                                 'mapped_choice': both[mapping[mode]]})
            confirmations = []
            for probe in ('initial_probe', 'reversal_1', 'reversal_2'):
                for arrival in (1, 2):
                    events = [(row, mode) for row, mode in records(rows, f'{probe}_probe_arrival_{arrival}')
                              if region(tasks[key(row)]) == area
                              and row['first_mode_difference_raw_step'] is not None]
                    signed = [(2 * mapping[mode]-1)*row['evidence'] for row, mode in events]
                    confirmations.append({'probe': probe, 'arrival': arrival, 'visible_count': len(events),
                                          'correct_evidence_direction': sum(value > 0 for value in signed),
                                          'median_oriented_evidence': float(np.median(signed)) if signed else None,
                                          'postwrite_support_on_mapped_majority': sum(
                                              row['probabilities'][mapping[mode]] > .5 for row, mode in events)})
            permutations.append({
                'analysis_only_physical_to_explanation': list(mapping), 'retained_queries': len(local),
                'mean_first_block_regret': float(np.mean([item['mapped_choice']['first_block_regret']
                                                         for item in selected])),
                'correct_route_direction': sum(item['mapped_choice']['correct_route_margin'] > EPSILON
                                               for item in selected),
                'probe_arrivals': confirmations,
                'retained_choices': selected,
            })
        result[area] = {'diagnostic_only': True, 'fixed_permutations': permutations}
    return result


def compare_runs(baseline_directory, baseline, directory, result):
    before = json.loads((baseline_directory / 'paired_unlabeled_replay.json').read_text(encoding='utf-8'))['rows']
    after = json.loads((directory / 'paired_unlabeled_replay.json').read_text(encoding='utf-8'))['rows']
    def retained_index(rows):
        return {(row['episode'], row['root'], phase, mode): row for phase in PHASES
                for row, mode in records(rows, phase)}
    old_index, new_index = retained_index(before), retained_index(after)
    if old_index.keys() != new_index.keys():
        raise ValueError('The two runs must use the same held-out retained query roots')
    paired = []
    for identity, old in old_index.items():
        new = new_index[identity]
        episode, root, phase, mode = identity
        choices = {}
        names = ('written', 'before', 'uniform', 'keep_old', 'branch_0', 'branch_1')
        if 'omit_probe_writes' in old['choices'] and 'omit_probe_writes' in new['choices']:
            names += ('omit_probe_writes',)
        for name in names:
            a, b = old['choices'][name], new['choices'][name]
            if len(a['scores']) != 384 or len(b['scores']) != 384:
                raise ValueError('The coverage comparison requires all 384 fixed candidates')
            choices[name] = {'baseline_candidate': a['selected'], 'expanded_candidate': b['selected'],
                             'baseline_first_block_regret': a['first_block_regret_by_mode'][mode],
                             'expanded_first_block_regret': b['first_block_regret_by_mode'][mode],
                             'expanded_minus_baseline_regret': b['first_block_regret_by_mode'][mode]
                                - a['first_block_regret_by_mode'][mode]}
        paired.append({'episode': episode, 'root': root, 'phase': phase,
                       'physical_mode_evaluation_only': mode, 'choices': choices})
    with np.load(baseline_directory / 'lifecycle.npz', allow_pickle=False) as archive:
        baseline_count = int(archive['fit_menu_a'].shape[2])
    with np.load(directory / 'lifecycle.npz', allow_pickle=False) as archive:
        expanded_count = int(archive['fit_menu_a'].shape[2])
    comparison = {'baseline_directory': str(baseline_directory.resolve()),
                  'expanded_directory': str(directory.resolve()), 'fixed_candidates_per_query': 384,
                  'baseline_fitting_candidates_per_menu': baseline_count,
                  'expanded_fitting_candidates_per_menu': expanded_count,
                  'paired_retained_queries': len(paired),
                  'baseline': baseline, 'expanded': result,
                  'retained_first_regret_difference': {name: result['retained_first_regret'][name]
                                                        - baseline['retained_first_regret'][name]
                                                        if name in baseline['retained_first_regret'] else None
                                                       for name in result['retained_first_regret']},
                  'retained_query_comparison': paired}
    write_json(directory / 'coverage_comparison.json', comparison)


def summarize(directory, qualification, alignment_path=None):
    raw = json.loads((directory / 'paired_unlabeled_replay.json').read_text(encoding='utf-8'))
    rows = raw['rows']
    physical = json.loads((qualification / 'physical.json').read_text(encoding='utf-8'))
    tasks = {f['task']: f for f in physical['forks']}
    quiet = [(row, mode) for row, mode in records(rows, 'neutral_control') if row['evidence'] is not None]
    assert len(quiet) == 48 and all(row['neutral'] for row, _ in quiet)
    delta = [regret(row, mode) - regret(row, mode, 'before') for row, mode in quiet]
    neutral = {'count': len(quiet), 'first_block_changes': sum(
        r['choices']['written']['first_block_group'] != r['choices']['before']['first_block_group']
        for r, _ in quiet), 'physical_worsening': sum(x > EPSILON for x in delta),
        'mean_after_minus_before_regret': float(np.mean(delta)),
        'median_absolute_evidence': float(np.median([abs(row['evidence'] or 0.) for row, _ in quiet]))}
    retained = [record for phase in PHASES for record in records(rows, phase)]
    assert len(retained) == 24
    maps = []
    for mapping in itertools.permutations((0, 1)):
        outcome = [regret(row, mode, f'branch_{mapping[mode]}') for row, mode in retained]
        maps.append({'analysis_only_physical_to_explanation': mapping,
                     'mean_first_block_regret': float(np.mean(outcome))})
    if alignment_path is None:
        alignment = min(maps, key=lambda a: a['mean_first_block_regret'])['analysis_only_physical_to_explanation']
        alignment_info = {'source': 'diagnostic held-out global permutation with lower fixed-menu regret',
                          'physical_to_explanation': list(alignment), 'frozen_before_heldout': False}
    else:
        alignment, alignment_info = frozen_fit_alignment(alignment_path)
    phases, n = {}, 0
    choices = ('written', 'before', 'uniform', 'keep_old')
    if all('omit_probe_writes' in row['choices'] for row, _ in retained):
        choices += ('omit_probe_writes',)
    for phase in PHASES:
        matched = records(rows, phase)
        pairs = []
        for task, fork in tasks.items():
            (a, mode_a), (b, mode_b) = [(row, mode) for row, mode in matched if key(row) == task]
            i, j = fork['true_indices']
            assert a['root'] == b['root'] and mode_a != mode_b
            gaps = []
            for row, mode in ((a, mode_a), (b, mode_b)):
                scores = row['choices']['written']['scores']
                gaps.append((1-2*mode) * (scores[j] - scores[i]))
            both = all(gap > EPSILON for gap in gaps)
            n += both
            own_gain = [regret(b, mode_a) - regret(a, mode_a), regret(a, mode_b) - regret(b, mode_b)]
            pairs.append({'task': task, 'correct_direction_both': both, 'correct_route_margins': gaps,
                          'correct_history_gain': own_gain, 'support_on_actual_explanation': [
                              a['probabilities'][alignment[mode_a]] if alignment else None,
                              b['probabilities'][alignment[mode_b]] if alignment else None]})
        phases[phase] = {
            'pairs': pairs,
            'mean_regret': {name: float(np.mean([regret(row, mode, name) for row, mode in matched]))
                            for name in choices},
            'correct_history_mean_gain': float(np.mean([g for pair in pairs for g in pair['correct_history_gain']])),
        }
    confirmations = [probe_confirmation(rows, reversal, arrival, alignment)
                     for reversal in ('initial_probe', 'reversal_1', 'reversal_2') for arrival in (1, 2)]
    cost_summary = {name: float(np.mean([regret(row, mode, name) for row, mode in retained])) for name in choices}
    old = {(r['episode'], r['mode_eval_only']): r['old_anchor_mse_by_branch'] for r, _ in retained}
    old_checks = raw.get('old_anchor_checks')
    result = {
        'arm': raw['arm'], 'fit_weights_fixed_in_replay': True, 'neutral': neutral,
        'paired_direction_correct': n, 'paired_direction_total': len(retained) // 2,
        'retained_first_regret': cost_summary, 'global_explanation_alignments': maps,
        'evidence_alignment': alignment_info,
        'retained_phases': phases, 'probe_arrivals': confirmations,
        'local_explanation_alignments': local_diagnostics(rows, tasks),
        'old_anchor_mse_recomputed_exact_after_episode': True,
        'old_anchor_checks': old_checks,
        'old_anchor_max_abs_difference': max(check['max_abs_difference'] for check in old_checks)
                                         if old_checks else None,
        'omit_probe_write_arrivals': raw.get('omit_probe_write_arrivals'),
        'omit_probe_write_control': raw.get('omit_probe_write_control'),
        'retained_query_count': len(retained),
        'retained_queries': [{'episode': row['episode'], 'root': row['root'], 'phase': phase,
                             'physical_mode_evaluation_only': mode,
                             'first_block_regret': {name: regret(row, mode, name) for name in choices},
                             'selected_candidates': {name: row['choices'][name]['selected'] for name in choices},
                             'support_on_actual_explanation': row['probabilities'][alignment[mode]] if alignment else None}
                            for phase in PHASES for row, mode in records(rows, phase)],
        'old_anchor_mse_by_episode_mode_for_audit': {f'{k[0]}_{k[1]}': value for k, value in old.items()},
        'mean_issue_seconds_measured': float(np.mean([r['issue_seconds'] for r in rows if r['issue_seconds'] is not None])),
        'mean_write_seconds_measured': float(np.mean([r['write_seconds'] for r in rows if r['write_seconds'] is not None])),
        'mean_conditional_forecast_seconds_measured': float(np.mean([r['conditional_forecast_seconds'] for r in rows])),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=HERE)
    parser.add_argument('--qualification', type=Path, default=QUAL)
    parser.add_argument('--baseline-directory', type=Path)
    parser.add_argument('--fit-alignment', type=Path)
    args = parser.parse_args()
    result = summarize(args.directory, args.qualification, args.fit_alignment)
    write_json(args.directory / 'paired_unlabeled_summary.json', result)
    if args.baseline_directory is not None:
        baseline = summarize(args.baseline_directory, args.qualification)
        compare_runs(args.baseline_directory, baseline, args.directory, result)
    print({k: result[k] for k in ('neutral', 'paired_direction_correct', 'retained_first_regret',
                                  'global_explanation_alignments')}, flush=True)


if __name__ == '__main__':
    main()
