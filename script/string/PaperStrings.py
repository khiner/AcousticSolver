#!/usr/bin/env python3
"""Reproduce the string convergence sweeps of Russo et al. (2026), Figs. 3–7.

Runs native CPU solvers without retaining trajectories. JSON results retain failed
points and distinguish the inferred published array indexing from physical times.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
from itertools import product
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EPS = sys.float_info.epsilon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/string-paper/strings')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-exponent', type=int, default=9)
    parser.add_argument('--reference-exponent', type=int, default=10)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 1 <= args.max_exponent <= 9 or not args.max_exponent < args.reference_exponent <= 10:
        parser.error('Require workers 1..4 and 1 <= max-exponent < reference-exponent <= 10')
    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = ['src/String/String.h', 'src/String/String.cpp', 'src/String/StringPaper.cpp', 'script/string/PaperStrings.py']
    sources = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths}
    if args.plot_only:
        summary = json.loads((args.output / 'results.json').read_text())
        for path in source_paths[:-1]:
            if summary['source_sha256'][path] != sources[path]:
                raise RuntimeError(f'Numerical source changed since recorded execution: {path}')
        summary['source_sha256']['script/string/PaperStrings.py'] = sources['script/string/PaperStrings.py']
        summary['analysis_only'] = True
        write_artifacts(summary, args.output, summary['rows'][0]['source_fingerprint'])
        return
    binary = ROOT / 'build/StringPaper'
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    numeric_sources = {key: value for key, value in sources.items() if not key.endswith('.py')}
    fingerprint = hashlib.sha256(json.dumps(dict(numeric_sources, binary=binary_hash), sort_keys=True).encode()).hexdigest()
    jobs_dir = args.output / 'jobs'
    jobs_dir.mkdir(exist_ok=True)

    def run(job, describe=False):
        key = hashlib.sha256(json.dumps({'source': fingerprint, 'job': job}, sort_keys=True).encode()).hexdigest()[:20]
        output = jobs_dir / f'{key}.json'
        if output.exists() and not describe:
            cached = json.loads(output.read_text())
            if cached.get('source_fingerprint') != fingerprint or cached.get('job') != job:
                raise RuntimeError(f'Cached result provenance mismatch: {output}')
            return cached
        request = jobs_dir / f'{key}{".describe" if describe else ""}.input.json'
        request.write_text(json.dumps(dict(job, **({'describe': True} if describe else {})), indent=2))
        completed = subprocess.run([str(binary), str(request)], check=True, text=True, capture_output=True)
        result = json.loads(completed.stdout)
        if not describe:
            result['source_fingerprint'] = fingerprint
            temporary = output.with_suffix('.tmp')
            temporary.write_text(json.dumps(result, indent=2) + '\n')
            temporary.replace(output)
        return result

    jobs = []
    initials = ('mode', 'raised-cosine')
    exponents = range(1, args.max_exponent + 1)
    for variant, initial, shift, grid, exponent in product(
            ('ge-a-unsplit', 'ge-a-split', 'ge-b-split'), initials, (EPS, 1000.),
            ('longitudinal', 'transverse'), exponents):
        jobs.append(dict(variant=variant, initial=initial, shift=shift, grid=grid, exponent=exponent, method='sav'))
    for family, initial in product(('cubic', 'kc'), initials):
        for split, grid in (('split', 'transverse'), ('unsplit', 'transverse'), ('unsplit', 'below')):
            for shift, exponent in product((EPS,) if grid == 'below' else (EPS, 1000.), exponents):
                jobs.append(dict(variant=f'{family}-{split}', initial=initial, shift=shift / 2, paper_shift=shift,
                                 grid=grid, exponent=exponent, method='sav'))

    # The reference samples each coarse half-step at its own actual clock time.
    descriptions = [run(job, describe=True) for job in jobs]
    references = []
    for variant in ('ge-b-split', 'cubic-split', 'kc-split'):
        family = variant.split('-')[0]
        for initial in ('mode', 'raised-cosine'):
            times = sorted({(math.floor(item['sample_rate'] * .06) - .5) * item['dt'] for item in descriptions
                            if item['job']['variant'].split('-')[0] == family and item['job']['initial'] == initial})
            targets = [dict(key=format(t, '.17g'), time=t) for t in times]
            references.append(dict(variant=variant, initial=initial, shift=EPS, grid='max', exponent=args.reference_exponent,
                                   method='reference', targets=targets))
    table_jobs = [dict(variant=variant, initial=initial, shift=EPS, grid='max', exponent=exponent, method='reference', table2=True)
                  for variant in ('ge-a-split', 'ge-b-split') for initial in ('mode', 'raised-cosine')
                  for exponent in range(3, args.reference_exponent + 1)
                  if not (variant == 'ge-b-split' and exponent == args.reference_exponent)]
    all_jobs = references + table_jobs + jobs
    rows = []
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run, job): job for job in all_jobs}
        for number, future in enumerate(concurrent.futures.as_completed(pending), 1):
            result = future.result()
            rows.append(result)
            job = pending[future]
            print(f"{number}/{len(all_jobs)} {job['variant']} {job['initial']} a={job['exponent']} {job['grid']} "
                  f"{job['method']} {result['status']} {result['wall_seconds']:.2f}s", flush=True)
    ref = {(r['job']['variant'].split('-')[0], r['job']['initial']): r for r in rows if r['job']['method'] == 'reference' and not r['job'].get('table2')}

    for baseline in ref.values():
        if baseline['status'] != 'complete' or not baseline['paper_observation']:
            raise RuntimeError('High-resolution reference did not complete; reproduction is incomplete')

    def psi(potential, variant, shift):
        return math.sqrt(max(0., 2 * potential + (0 if variant == 'kc-split' else shift if variant.startswith('ge-') else 2 * shift)))

    for row in rows:
        job = row['job']
        if job['method'] != 'sav' or row['status'] != 'complete':
            continue
        reference = ref[job['variant'].split('-')[0], job['initial']]
        obs, baseline = row['paper_observation'], reference['paper_observation']
        row['errors'] = {key: abs(obs[key] - baseline[key]) for key in ('u', 'v')}
        row['errors']['psi'] = abs(obs['psi'] - psi(baseline['potentials'][job['variant']], job['variant'], job['shift']))
        captured = reference['reference_captures'][format(obs['psi_time'], '.17g')]
        row['aligned_psi_error'] = abs(obs['psi'] - psi(captured['potentials'][job['variant']], job['variant'], job['shift']))
    # Printed values are independent paper evidence; retain deltas without relaxing them to a pass/fail gate.
    table2 = {
        'ge-a-split': [
            [-.001542536764, .000000455045, -.000067010490, -.000004989832],
            [-.001537884027, .000000534644, -.000108258834, -.000005085885],
            [-.001536638849, .000000554477, -.000143945383, -.000005081376],
            [-.001536158952, .000000563468, -.000149147281, -.000005060437],
            [-.001535953014, .000000567741, -.000151706166, -.000005044582],
            [-.001535913124, .000000568515, -.000152133566, -.000005032951],
            [-.001535830208, .000000570587, -.000152218477, -.000005037279],
            [-.001535822314, .000000571029, -.000152228140, -.000005035934]],
        'ge-b-split': [
            [-.001542536880, .000000454636, -.000067010580, -.000004987832],
            [-.001537884054, .000000534567, -.000108258848, -.000005085587],
            [-.001536638855, .000000554461, -.000143945391, -.000005081310],
            [-.001536158954, .000000563464, -.000149147283, -.000005060421],
            [-.001535953015, .000000567740, -.000151706168, -.000005044578],
            [-.001535913125, .000000568515, -.000152133566, -.000005032950],
            [-.001535830208, .000000570587, -.000152218477, -.000005037279],
            [-.001535822314, .000000571029, -.000152228140, -.000005035934]]}
    table2_comparison = []
    for row in rows:
        job = row['job']
        if job['method'] != 'reference' or job['variant'] not in table2 or job['exponent'] < 3 or not row['paper_observation']:
            continue
        offset = 0 if job['initial'] == 'mode' else 2
        expected = table2[job['variant']][job['exponent'] - 3][offset:offset+2]
        actual = [row['paper_observation'][key] for key in ('u', 'v')]
        table2_comparison.append(dict(variant=job['variant'], initial=job['initial'], exponent=job['exponent'],
                                      published=expected, actual=actual, absolute_delta=[abs(a-b) for a,b in zip(actual,expected)]))
    table3 = {
        ('ge-a-unsplit', EPS): [2.18e-9,4.37e-10,1.11e-7,9.39e-9,5.17e-10,2.74e-6],
        ('ge-a-unsplit', 1000.): [2.42e-9,4.35e-10,9.84e-11,1.30e-8,1.34e-9,3.76e-9],
        ('ge-a-split', EPS): [3.63e-8,1.35e-7,1.17e-4,9.49e-9,4.46e-9,2.62e-5],
        ('ge-a-split', 1000.): [7.89e-9,4.42e-10,1.65e-11,9.66e-9,1.35e-9,4.50e-10],
        ('ge-b-split', EPS): [7.89e-9,4.42e-10,5.81e-11,9.66e-9,1.35e-9,4.30e-11],
        ('ge-b-split', 1000.): [7.89e-9,4.42e-10,1.85e-11,9.66e-9,1.35e-9,1.08e-10]}
    table3_comparison = []
    for row in rows:
        job = row['job']
        key = (job['variant'], job['shift'])
        if job['method'] != 'sav' or job['exponent'] != 9 or job['grid'] != 'transverse' or key not in table3 or 'errors' not in row:
            continue
        offset = 0 if job['initial'] == 'mode' else 3
        expected = table3[key][offset:offset+3]
        actual = [row['errors']['u'], row['errors']['v'], row['aligned_psi_error']]
        table3_comparison.append(dict(variant=job['variant'], initial=job['initial'], shift=job['shift'],
                                      published=expected, actual=actual, unaligned_psi_error=row['errors']['psi'],
                                      absolute_delta=[abs(a-b) for a,b in zip(actual,expected)]))
    summary = dict(table2_comparison=table2_comparison, table3_comparison=table3_comparison, paper='https://doi.org/10.1007/s11071-026-12708-0', pdf_sha256=hashlib.sha256((ROOT / 'build/string-paper/paper.pdf').read_bytes()).hexdigest() if (ROOT / 'build/string-paper/paper.pdf').exists() else None,
                   native_binary_sha256=binary_hash,
                   source_sha256=sources, upstream_commit=subprocess.check_output(['git', '-C', str(ROOT / 'external/nLinStringsConv-SAV'), 'rev-parse', 'HEAD'], text=True).strip(),
                   max_exponent=args.max_exponent, reference_exponent=args.reference_exponent, seconds=.06,
                   complete_published_resolution=args.max_exponent == 9 and args.reference_exponent == 10,
                   attempted_jobs=len(all_jobs), completed_jobs=sum(r['status'] == 'complete' for r in rows),
                   nonfinite_jobs=sum(r['status'] == 'nonfinite' for r in rows), computational_failures=0,
                   invocation_wall_seconds=time.monotonic() - start,
                   recorded_case_wall_seconds_sum=sum(row['wall_seconds'] for row in rows), analysis_only=False,
                   conventions=['Literal upstream initialization and adjusted dt; fs=floor(1/dt).',
                                'Published Table 2 matches Out(timeSamples), corresponding to displacement time (timeSamples-1)*dt.',
                                'Psi(timeSamples) corresponds to (timeSamples-0.5)*dt; plots use actual-half-time-aligned reference errors, while errors.psi retains the unaligned array-index comparison.',
                                'RC uses upstream half-width floor(0.25*(N-1)); paper width terminology is ambiguous.',
                                'MAX selects the larger requested analytical hL/hT bound; actual grid uses upstream rounding and margins.',
                                'Below-bound NS grid uses an even interval count rounded upward from L/(0.99*hT), retaining the requested timestep.',
                                'Cubic split and unsplit plus KC unsplit map paper epsilon to upstream shiftV=epsilon/2; KC split uses effective epsilon=0 as derived in the paper and upstream.',
                                'Nonfinite points are retained as failed points, never converted into zero error.'], rows=rows)
    write_artifacts(summary, args.output, fingerprint)


def write_artifacts(summary, output, fingerprint):
    (output / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (output / 'results.csv').open('w') as stream:
        fields = ['variant', 'initial', 'method', 'exponent', 'grid', 'shift', 'paper_shift', 'status', 'intervals', 'sample_rate', 'dt', 'wall_seconds', 'u', 'v', 'psi', 'u_error', 'v_error', 'psi_error', 'aligned_psi_error']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader()
        for row in summary['rows']:
            values = dict(row['job'], **{k:v for k,v in row.items() if k != 'job'}, **row.get('paper_observation', {}))
            values.update({key+'_error':value for key,value in row.get('errors', {}).items()})
            writer.writerow(values)
    plot(summary, output)
    artifacts = [output / 'results.json', output / 'results.csv'] + [output / f'figure{number}.{extension}' for number in range(3, 8) for extension in ('pdf', 'png', 'svg')]
    manifest = {path.name: {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in artifacts}
    (output / 'manifest.json').write_text(json.dumps({'source_fingerprint': fingerprint, 'artifacts': manifest}, indent=2) + '\n')


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    def plot_curve(axes, rows, fields, initial, **style):
        rows = sorted(rows, key=lambda r: r['job']['exponent'])
        for ax, key in zip(axes, fields):
            errors = [max(1e-18, r['aligned_psi_error'] if key == 'psi' else r['errors'][key])
                      if 'errors' in r else float('nan') for r in rows]
            ax.loglog([r['sample_rate'] for r in rows], errors, marker='o' if initial == 'mode' else 's', **style)
            ax.set_ylabel(f'Absolute {key} error')
            ax.grid(True, which='both', alpha=.2)
        axes[-1].set_xlabel('Adjusted sample rate (Hz)')

    rows = summary['rows']
    figures = []
    fig, ax = plt.subplots(figsize=(6, 4))
    curve = sorted([r for r in rows if r['job']['exponent'] >= 2 and r['job']['variant'] == 'ge-b-split' and r['job']['initial'] == 'mode'
                    and r['job']['grid'] in ('longitudinal', 'max') and r['job']['shift'] == EPS], key=lambda r:r['job']['exponent'])
    curve = list({r['job']['exponent']: r for r in curve}.values())
    k = np.array([1 / math.ldexp(44100., r['job']['exponent']) for r in curve])
    hL, hT = np.array([r['requested_hL'] for r in curve]), np.array([r['requested_hT'] for r in curve])
    ax.loglog(k, hL, label='$h_L$'); ax.loglog(k, hT, label='$h_T$')
    ax.fill_between(k, np.maximum(hL, hT), .1, color='gold', alpha=.25)
    ax.set(xlabel='Requested timestep (s)', ylabel='Grid spacing (m)', title='Figure 3: geometric stability bounds')
    ax.legend(); figures.append(('figure3', fig))
    colors = {'longitudinal': 'tab:blue', 'transverse': 'tab:orange', 'max': 'tab:green'}
    for number, variant in enumerate(('ge-a-unsplit', 'ge-a-split', 'ge-b-split'), 4):
        fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
        for column, shift in enumerate((EPS, 1000.)):
            for initial in ('mode', 'raised-cosine'):
                for grid in ('longitudinal', 'transverse', 'max'):
                    candidates = [r for r in rows if r['job']['variant'] == variant and r['job']['method'] == 'sav'
                                  and r['job']['initial'] == initial and r['job']['shift'] == shift]
                    candidates = [r for r in candidates if r['job']['grid'] == (('longitudinal' if r['requested_hL'] >= r['requested_hT'] else 'transverse') if grid == 'max' else grid)]
                    plot_curve(axes[:, column], candidates, ('u', 'v', 'psi'), initial,
                          linestyle='-' if initial == 'mode' else '--', color=colors[grid], label=f'{grid}, {initial}')
            axes[0, column].set_title(f'{variant}, shift={shift:.3g}')
        axes[0,0].legend(fontsize=7)
        figures.append((f'figure{number}',fig))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for column, family in enumerate(('cubic', 'kc')):
        for initial in ('mode', 'raised-cosine'):
            for variant, grid in ((family+'-split','transverse'),(family+'-unsplit','transverse'),(family+'-unsplit','below')):
                candidates = [r for r in rows if r['job']['variant'] == variant and r['job']['initial'] == initial
                              and r['job']['grid'] == grid and r['job'].get('paper_shift') == EPS]
                plot_curve(axes[:, column], candidates, ('u', 'psi'), initial,
                      label=f"{variant.removeprefix(family+'-')} {grid}, {initial}")
        axes[0,column].set_title(f'{family}, shift=machine epsilon')
    axes[0,0].legend(fontsize=7)
    figures.append(('figure7',fig))
    for name,figure in figures:
        if name != 'figure3':
            figure.suptitle('Reconstruction: published displacement indexing; reference psi aligned to each coarse half-step', fontsize=9)
        figure.tight_layout()
        for extension in ('pdf','png','svg'):figure.savefig(output/f'{name}.{extension}',dpi=160)
        plt.close(figure)


if __name__ == '__main__':
    main()
