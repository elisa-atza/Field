import numpy as np
import pandas as pd
from field_paths import FIELDS, field_config, plattegrond_paths, plattegronden_xlsx


def lotnum_plattegrond(field):
    '''Lot numbers 1..720 laid out the way they run across each field.'''
    if field in ('M', 'V'):
        platt=np.arange(1,721)
        platt=platt.reshape((12,60))
        return np.rot90(platt)
    if field == 'S':
        platt=np.arange(1,721)
        platt=np.flipud(platt)
        platt=platt.reshape((6,120))
        return np.rot90(platt,3)
    raise ValueError(f'no lot-number layout defined for field {field!r}')


def main():
    for field in FIELDS:
        exel=pd.read_excel(plattegronden_xlsx(), sheet_name=field_config(field)['plattegrond_sheet'])
        exel=exel.dropna(axis=1, how='all')
        exel=exel.dropna(axis=0, how='all')
        exel.drop(exel.columns[[0]],axis=1,inplace=True)

        named_path, lotnum_path = plattegrond_paths(field)
        np.save(named_path, exel.values)
        np.save(lotnum_path, lotnum_plattegrond(field))

# This module is a pipeline stage: its main() is called from cli.py, not
# run directly.
