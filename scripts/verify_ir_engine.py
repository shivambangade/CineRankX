"""Verification script for IR engine on real dataset."""

import pandas as pd

from src.ir_engine import IREngine

print("Loading merged dataset...")
df = pd.read_csv("data/processed/movies_merged.csv")
print(f"Loaded {len(df)} movies\n")

print("Fitting IR engine...")
engine = IREngine()
engine.fit(df)
engine.save()
print("IR engine fitted and saved\n")

print("=" * 70)
print("VERIFICATION 1: Search for 'batman'")
print("=" * 70)
results = engine.search("batman", limit=10)
for i, result in enumerate(results, 1):
    print(
        f"{i}. {result['title']:50s} (ID: {result['movieId']:5d}) "
        f"Similarity: {result['similarity']:.4f}"
    )

print("\n" + "=" * 70)
print("VERIFICATION 2: Autocomplete 'bat'")
print("=" * 70)
results = engine.autocomplete("bat", limit=10)
for i, result in enumerate(results, 1):
    print(f"{i}. {result['title']:50s} (ID: {result['movieId']:5d})")

print("\n" + "=" * 70)
print("VERIFICATION 3: Movies similar to 'Toy Story' (ID: 1)")
print("=" * 70)
toy_story = df[df["movieId"] == 1]
if len(toy_story) > 0:
    print(f"Reference: {toy_story.iloc[0]['title']}\n")
    results = engine.similar_to(1, limit=10)
    for i, result in enumerate(results, 1):
        print(
            f"{i}. {result['title']:50s} (ID: {result['movieId']:5d}) "
            f"Similarity: {result['similarity']:.4f}"
        )
else:
    print("Toy Story (ID: 1) not found in dataset")

print("\n" + "=" * 70)
print("All verifications complete!")
print("=" * 70)
