#Test python file
import os
import pathlib

print(os.getcwd())
if os.path.lexists('/mhome/damtp/q/dfs28/Project/Project_data/files/flowsheet_nearly_interpolated.csv'):
    print('Os: The path exists')
else:
    print('Os: The path does not exist')

if pathlib.Path('~/Project/Project_data/files/flowsheet_nearly_interpolated.csv').is_file():
    print('Pathlib: The path exists')
else:
    print('Pathlib: The path does not exist')

print('Flowsheet_all_times exists: {}'.format(os.path.exists('/local/data/public/2020/dfs28/PICU_data/flowsheet_all_times.csv')))
