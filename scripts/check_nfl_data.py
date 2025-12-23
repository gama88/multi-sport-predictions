"""Check what NFL data we have available."""
import pandas as pd

# Use existing spreadspoke data
df = pd.read_csv('data/nfl/spreadspoke_scores.csv')
print('Spreadspoke NFL Data:')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Seasons: {sorted(df["schedule_season"].unique())[-5:]}')
print()
print('Sample:')
print(df.tail(3))
