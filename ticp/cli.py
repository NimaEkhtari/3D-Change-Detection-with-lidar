# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:56:16 2024

@author: nekhtari

Command line tool, installed as the `ticp` command:

    ticp before.laz after.laz --window 150 --step 25 --classes 2

main.py in the repository calls main() with its own defaults, so the values
set at the top of main.py can still be overridden from the command line.
"""

import os
import sys
import logging
import argparse
from dataclasses import asdict

from .icp import ICPConfig
from .workflow import compare, output_name


def build_parser(defaults=None):
    d = asdict(ICPConfig())
    d.update(point_clouds=[], out_dir='results', prefix='ticp_', plot=False)
    d.update(defaults or {})

    p = argparse.ArgumentParser(prog='ticp', description='Moving-window translation-only ICP for 3D change detection.')
    p.add_argument('point_clouds', nargs='*', default=d['point_clouds'],
                   help='two or more point clouds in time order, every consecutive pair is processed')
    p.add_argument('--out-dir', default=d['out_dir'])
    p.add_argument('--prefix', default=d['prefix'])
    p.add_argument('--window', type=float, default=d['window_size'], help='window size')
    p.add_argument('--step', type=float, default=d['step_size'], help='step between windows')
    p.add_argument('--margin', type=float, default=d['margin'])
    p.add_argument('--classes', type=int, nargs='*', default=d['classes'],
                   help='LAS classes to use, e.g. --classes 2 6 (just --classes = all points)')
    p.add_argument('--min-points', type=int, default=d['min_points'])
    p.add_argument('--convergence', type=float, default=d['convergence'])
    p.add_argument('--max-iter', type=int, default=d['max_iter'])
    p.add_argument('--outlier-threshold', type=float, default=d['outlier_threshold'])
    p.add_argument('--normal-knn', type=int, default=d['normal_knn'])
    p.add_argument('--bounds', type=float, nargs=4, default=d['bounds'],
                   metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'))
    p.add_argument('--workers', type=int, default=d['n_workers'])
    p.add_argument('--crs', default=d['crs'])
    p.add_argument('--plot', action=argparse.BooleanOptionalAction, default=d['plot'],
                   help='show a quick plot of each result')
    return p



def print_progress():
    """ Progress callback that prints every 5% """
    state = {'last': -5, 'message': None}

    def progress(fraction, message):
        if fraction is None:
            if message != state['message']:
                print(f'{message} ...')
            state['message'] = message
            return
        pct = int(100 * fraction)
        if pct >= state['last'] + 5 or (pct == 100 and state['last'] < 100):
            print(f'{pct}% done')
            state['last'] = pct
    return progress



def main(argv=None, defaults=None):
    args = build_parser(defaults).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    if len(args.point_clouds) < 2:
        raise SystemExit('Need at least two point clouds (before and after)')

    config = ICPConfig(
        window_size=args.window,
        step_size=args.step,
        margin=args.margin,
        classes=args.classes or None,
        min_points=args.min_points,
        convergence=args.convergence,
        max_iter=args.max_iter,
        outlier_threshold=args.outlier_threshold,
        normal_knn=args.normal_knn,
        bounds=args.bounds,
        n_workers=args.workers,
        crs=args.crs,
    )

    events = args.point_clouds
    for pre_event, pos_event in zip(events[:-1], events[1:]):
        print(f'\n{pre_event}\n  -> {pos_event}')
        name = output_name(pre_event, pos_event, config, args.prefix)
        result = compare(pre_event, pos_event, config, output=os.path.join(args.out_dir, name),
                         progress=print_progress())
        if args.plot:
            from .plotting import plot_results
            plot_results(result)



if __name__ == '__main__':
    main()
