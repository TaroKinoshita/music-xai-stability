import pandas as pd

df = pd.read_csv('selected_50.csv')

print(f"Total: {len(df)}")
print(f"\nType distribution:")
print(df['type'].value_counts())

print(f"\nPrototypical by genre:")
proto = df[df['type'] == 'prototypical']
print(proto['true_genre'].value_counts())

print(f"\nBoundary songs:")
boundary = df[df['type'] == 'boundary']
print(f"Count: {len(boundary)}")
print(f"Avg confidence: {boundary['confidence'].mean():.2%}")

print(df)