try:
    from tdc.single_pred import ADME
    from rdkit import Chem
    print("Сите библиотеки се инсталирани правилно")
except ImportError as e:
    print(f"Проблем: {e}")