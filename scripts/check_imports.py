
try:
    import transformers
    print("transformers: INSTALLED")
except ImportError:
    print("transformers: NOT INSTALLED")

try:
    import molclr
    print("molclr: INSTALLED")
except ImportError:
    print("molclr: NOT INSTALLED")

try:
    import mole
    print("mole: INSTALLED")
except ImportError:
    print("mole: NOT INSTALLED")

try:
    import deepchem
    print("deepchem: INSTALLED")
except ImportError:
    print("deepchem: NOT INSTALLED")
