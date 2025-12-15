import json
from rdkit import Chem
from mordred import Calculator, descriptors

# Load your model features
with open("model_features.json") as f:
    model_features = json.load(f)

# Check if MINaaN is in your features
print("Checking for MINaaN in model_features.json:")
minaan_found = False
for feat in model_features:
    if isinstance(feat, str) and 'mina' in feat.lower():
        print(f"Found similar: {feat}")
        if feat.lower() == 'minaann' or feat == 'MINaaN':
            minaan_found = True

if not minaan_found:
    print("MINaaN not found in model_features.json")
    print("Actual features containing 'mina':")
    for feat in model_features:
        if isinstance(feat, str) and 'mina' in feat.lower():
            print(f"  {feat}")

# Test with a molecule
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
mol = Chem.MolFromSmiles(smiles)
calc = Calculator(descriptors, ignore_3D=True)

# Calculate descriptors
result = calc(mol)

# Find all descriptors containing 'mina'
print("\nMordred descriptors containing 'mina':")
for desc_name, value in result.items():
    desc_str = str(desc_name)
    if 'mina' in desc_str.lower():
        print(f"{desc_str}: {value}")

# Check exact name
print("\nChecking exact names:")
for desc_name in result.keys():
    desc_str = str(desc_name)
    if desc_str in ['MINaaN', 'minaaN', 'MinAaN']:
        print(f"Found exact: {desc_str} = {result[desc_name]}")
