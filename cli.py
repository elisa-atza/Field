#!/usr/bin/env python3
"""Command-line entry point for the field pipeline.

Each pipeline stage is a subcommand here; stages communicate through files
on disk, so they are run one at a time, in order. Which year, fields, dates
and folders they work on is set in pipeline_config.toml, not here.

    python cli.py <command> --help      # per-command options
    python cli.py --list                # list every command and its step

The interactive steps open a matplotlib window and wait for you; the command
returns once you finish (middle-click) or close the window.

`python cli.py run` goes through every stage in order, skipping whatever is
already on disk, so it can be stopped and started again at any point.
"""

import argparse
import os
import sys

from field_paths import (
    FIELDS, SEGMENTATION, color_bounds_path, corrected_grid_path, dates,
    field_config, grid_path, image_path, latest_canopy_table, markers_path,
    plattegrond_paths, polygons_path, quadrant_path, raw_canopy_path,
    reference_date, sorted_markers_path,
)


def cmd_crop(a):
    import Step1_Crop_big_orthophotos as step
    step.main(a.path, a.field, a.date)


def cmd_markers(a):
    import Step2_Find_red_markers as step
    step.main(a.field, a.date)


def cmd_boundaries(a):
    import Step3_Select_field_boundaries as step
    step.main(a.field)


def cmd_build_grid(a):
    import Step4_Build_grid as step
    step.main(a.field)


def cmd_correct_grid(a):
    import Step5_Manual_correction_grid as step
    step.main(a.field)


def cmd_sort_markers(a):
    import Step6_Sort_and_assign_markers as step
    step.main(a.field, a.date)


def cmd_plattegronds(a):
    import Step7_0_Plattegronds as step
    step.main()


def cmd_make_polygons(a):
    import Step7_1_Create_polygons_from_grids as step
    step.main(a.field)


def cmd_transform_polygons(a):
    import Step7_2_Transform_polygons as step
    step.main(a.field, a.date, transformation_method=a.transformation_method)


def cmd_timelapse(a):
    import Step7_3_Produce_timelapse as step
    step.main(a.segmentation)


def cmd_color_bounds(a):
    import Step8_Select_day_spec_unbiased_color_bounds as step
    step.main(a.field)


def cmd_canopy(a):
    import Step9_Save_canopy_to_file as step
    step.main(a.field, a.date)


def cmd_check_grids(a):
    import Step5_Manual_correction_grid as step
    step.show_all_grids(a.field)


def cmd_check_canopy(a):
    import Step9_Save_canopy_to_file as step
    for d in [a.date] if a.date else dates(a.field):
        step.check_canopy(a.field, d)


def pipeline_stages(field):
    '''Every stage of a full run for one field, in order, as
    (label, done, run, interactive): done() says whether its output already
    exists, run() does it. Built lazily -- the grid stages depend on which
    grids build-grid produced.'''
    ref = reference_date(field)
    others = [d for d in dates(field) if d != ref]
    split = field_config(field).get('split_quadrants', False)

    def step(name):
        return __import__(name)

    def grid_indices():
        return [f'{q}_{h}' if split else str(q) for q in range(1, 5) for h in ((1, 2) if split else (None,))]

    for d in dates(field):
        yield (f'markers {d}', lambda d=d: os.path.exists(markers_path(field, d)),
               lambda d=d: step('Step2_Find_red_markers').main(field, d), True)
    for q in range(1, 5):
        yield (f'boundaries quadrant {q}', lambda q=q: os.path.exists(quadrant_path(field, q, 'tops')),
               lambda q=q: step('Step3_Select_field_boundaries').main(field, q), True)
        if split:
            def mids(q=q):
                print(f'Quadrant {q}: click the line between its two blocks this time '
                      '(left/right pair per block of ridges, fewer than 12 points).')
                step('Step3_Select_field_boundaries').main(field, q)
            yield (f'boundaries quadrant {q} middle line',
                   lambda q=q: os.path.exists(quadrant_path(field, q, 'mids')), mids, True)
    yield ('build-grid', lambda: all(os.path.exists(grid_path(field, i)) for i in grid_indices()),
           lambda: step('Step4_Build_grid').main(field), False)
    # overview before correcting; shown while any grid is still uncorrected
    yield ('check grids', lambda: all(os.path.exists(corrected_grid_path(field, i)) for i in grid_indices()),
           lambda: step('Step5_Manual_correction_grid').show_all_grids(field), False)
    for i in grid_indices():
        yield (f'correct-grid {i}', lambda i=i: os.path.exists(corrected_grid_path(field, i)),
               lambda i=i: step('Step5_Manual_correction_grid').main(field, i), True)
    for d in others:
        yield (f'sort-markers {d}', lambda d=d: os.path.exists(sorted_markers_path(field, d)),
               lambda d=d: step('Step6_Sort_and_assign_markers').main(field, d), True)
    yield ('plattegronds', lambda: all(os.path.exists(p) for p in plattegrond_paths(field)),
           lambda: step('Step7_0_Plattegronds').main(), False)
    yield ('make-polygons', lambda: os.path.exists(polygons_path(field, ref)),
           lambda: step('Step7_1_Create_polygons_from_grids').main(field), False)
    for d in others:
        yield (f'transform-polygons {d}', lambda d=d: os.path.exists(polygons_path(field, d)),
               lambda d=d: step('Step7_2_Transform_polygons').main(field, d), False)
    if SEGMENTATION == 'per_date':
        def bounds_done():
            if not os.path.exists(color_bounds_path(field)):
                return False
            import pandas as pd
            return set(dates(field)) <= set(pd.read_csv(color_bounds_path(field))['Date'])
        yield ('color-bounds', bounds_done,
               lambda: step('Step8_Select_day_spec_unbiased_color_bounds').main(field), True)
    for d in dates(field):
        yield (f'canopy {d}', lambda d=d: os.path.exists(raw_canopy_path(field, d)),
               lambda d=d: step('Step9_Save_canopy_to_file').produce_raw_file_canopy(field, d), False)

    # The table is rebuilt when a date's raw canopy is newer than the latest
    # table; the canopy-as-field checks are shown after a rebuild.
    rebuilt = []

    def tables_done():
        table = latest_canopy_table(field)
        return table is not None and all(
            os.path.getmtime(raw_canopy_path(field, d)) <= os.path.getmtime(table)
            for d in dates(field) if os.path.exists(raw_canopy_path(field, d)))

    def build_tables():
        step('Step9_Save_canopy_to_file').fill_df_field(field, dates(field))
        rebuilt.append(True)
    yield ('canopy table', tables_done, build_tables, False)
    for d in dates(field):
        yield (f'check canopy {d}', lambda: not rebuilt,
               lambda d=d: step('Step9_Save_canopy_to_file').check_canopy(field, d), False)


def cmd_run(a):
    missing = []
    for d in dates(a.field):
        try:
            image_path(a.field, d)
        except (FileNotFoundError, FileExistsError) as e:
            missing.append(str(e))
    if missing:
        sys.exit('cannot start, fix the images first:\n  ' + '\n  '.join(missing))

    for label, done, run, interactive in pipeline_stages(a.field):
        if done():
            if a.status:
                print(f'  done     {label}')
            continue
        if a.status:
            print(f'  to do    {label}')
            continue
        print(f'\n=== {label} ===')
        run()
        if interactive and not done():
            sys.exit(f'{label}: nothing was saved (window closed without finishing?). '
                     'Run `python cli.py run` again to retry from here.')
    if not a.status:
        print('\nAll stages done.')


def build_parser():
    p = argparse.ArgumentParser(
        prog='cli.py',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest='command', metavar='command')

    def add(name, step, help, interactive=False):
        extra = ' [interactive: opens a matplotlib window]' if interactive else ''
        text = f'({step}) {help}{extra}'
        return sub.add_parser(name, help=text, description=text)

    def field_arg(sp):
        # with a single configured field there is nothing to choose
        if len(FIELDS) == 1:
            sp.add_argument('--field', choices=FIELDS, default=FIELDS[0],
                            help=f'default: {FIELDS[0]}, the only configured field')
        else:
            sp.add_argument('--field', choices=FIELDS, required=True)

    def date_arg(sp, required=True, help=''):
        sp.add_argument('--date', required=required,
                        help=help or 'one of the dates configured for the field')

    sp = add('crop', 'Step1', "crop a full orthophoto into the field's images folder", interactive=True)
    sp.add_argument('--path', required=True, help='full orthophoto (.tif or .npy)')
    field_arg(sp)
    date_arg(sp, help='date string to name the crop with, as listed in the config')
    sp.set_defaults(func=cmd_crop)

    sp = add('markers', 'Step2', 'find red ground markers and click the ones to keep', interactive=True)
    field_arg(sp)
    date_arg(sp)
    sp.set_defaults(func=cmd_markers)

    sp = add('boundaries', 'Step3', 'select ridge-block corners of one quadrant on the reference date', interactive=True)
    field_arg(sp)
    sp.set_defaults(func=cmd_boundaries)

    sp = add('build-grid', 'Step4', 'detect plant gaps on the reference date and snap them to a grid')
    field_arg(sp)
    sp.set_defaults(func=cmd_build_grid)

    sp = add('correct-grid', 'Step5', 'click grid points to move them (one grid per run)', interactive=True)
    field_arg(sp)
    sp.set_defaults(func=cmd_correct_grid)

    sp = add('sort-markers', 'Step6', "put a date's markers in the reference date's order", interactive=True)
    field_arg(sp)
    date_arg(sp, help='target date (any configured date except the reference)')
    sp.set_defaults(func=cmd_sort_markers)

    sp = add('plattegronds', 'Step7_0', 'read the trial-layout spreadsheet into lot-number matrices')
    sp.set_defaults(func=cmd_plattegronds)

    sp = add('make-polygons', 'Step7_1', 'build one polygon per lot from the corrected grids (reference date)')
    field_arg(sp)
    sp.set_defaults(func=cmd_make_polygons)

    sp = add('transform-polygons', 'Step7_2', 'warp the reference-date polygons onto another date')
    field_arg(sp)
    date_arg(sp, help='target date (any configured date except the reference)')
    sp.add_argument('--transformation-method', choices=['affine', 'interpolation'], default='affine',
                    help='affine: least-squares similarity fit; interpolation: thin-plate spline')
    sp.set_defaults(func=cmd_transform_polygons)

    sp = add('timelapse', 'Step7_3', 'produce per-variety GIFs across all dates (documentation only)')
    sp.add_argument('--segmentation', action='store_true', help='render segmented (green/black) frames')
    sp.set_defaults(func=cmd_timelapse)

    sp = add('color-bounds', 'Step8', 'choose per-date hue bounds for canopy segmentation', interactive=True)
    field_arg(sp)
    sp.set_defaults(func=cmd_color_bounds)

    sp = add('canopy', 'Step9', 'count canopy per plot/ridge; write one table per date and one joined table')
    field_arg(sp)
    date_arg(sp, required=False, help='a single date; default: every configured date')
    sp.set_defaults(func=cmd_canopy)

    sp = add('check-grids', 'Step5', 'show all detected grids over the reference image', interactive=True)
    field_arg(sp)
    sp.set_defaults(func=cmd_check_grids)

    sp = add('check-canopy', 'Step9', 'show canopy per ridge next to the image', interactive=True)
    field_arg(sp)
    date_arg(sp, required=False, help='a single date; default: every configured date')
    sp.set_defaults(func=cmd_check_canopy)

    sp = add('run', 'all', 'go through every stage in order, skipping what is already done')
    field_arg(sp)
    sp.add_argument('--status', action='store_true', help='only list which stages are done')
    sp.set_defaults(func=cmd_run)

    return p, sub


def main():
    parser, sub = build_parser()
    if len(sys.argv) > 1 and sys.argv[1] in ('--list', '-l'):
        for name, p in sub.choices.items():
            print(f'{name:20s} {p.description or ""}')
        return
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    date = getattr(args, 'date', None)
    if date and date not in dates(args.field):
        parser.error(f'--date {date!r} is not one of the dates configured for {args.field}: '
                     f'{", ".join(dates(args.field))} (reference: {reference_date(args.field)})')
    args.func(args)


if __name__ == '__main__':
    main()
