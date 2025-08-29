#!/usr/bin/env python3
"""
Script to test if required modules can be imported successfully.
"""

def test_imports():
    """Test all module imports and report status."""
    
    modules_to_test = [
        ('sklearn', 'import sklearn'),
        ('matplotlib.pyplot', 'import matplotlib.pyplot as plt'),
        ('numpy', 'import numpy as np'),
        ('pandas', 'import pandas as pd'),
        ('pynwb', 'import pynwb'),
        ('seaborn', 'import seaborn as sns'),
        ('jax.numpy', 'import jax.numpy as jnp'),
        ('umap', 'import umap'),
        ('dynamax.hidden_markov_model.GaussianHMM', 'from dynamax.hidden_markov_model import GaussianHMM')
    ]
    
    results = []
    
    print("Testing module imports...")
    print("=" * 50)
    
    for module_name, import_statement in modules_to_test:
        try:
            exec(import_statement)
            status = "✓ SUCCESS"
            results.append((module_name, True, None))
            print(f"{module_name:<35} {status}")
        except ImportError as e:
            status = "✗ FAILED"
            results.append((module_name, False, str(e)))
            print(f"{module_name:<35} {status}")
            print(f"    Error: {e}")
        except Exception as e:
            status = "✗ ERROR"
            results.append((module_name, False, str(e)))
            print(f"{module_name:<35} {status}")
            print(f"    Unexpected error: {e}")
    
    print("\n" + "=" * 50)
    
    # Summary
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"Summary: {successful}/{total} modules imported successfully")
    
    if successful == total:
        print("🎉 All modules are working correctly!")
        return True
    else:
        print("\n❌ Some modules failed to import. You may need to install:")
        for module_name, success, error in results:
            if not success:
                print(f"  - {module_name}")
        return False

def test_basic_functionality():
    """Test basic functionality of successfully imported modules."""
    
    print("\nTesting basic functionality...")
    print("=" * 50)
    
    try:
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        print(f"✓ NumPy: Created array {arr}")
    except Exception as e:
        print(f"✗ NumPy test failed: {e}")
    
    try:
        import pandas as pd
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        print(f"✓ Pandas: Created DataFrame with shape {df.shape}")
    except Exception as e:
        print(f"✗ Pandas test failed: {e}")
    
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.close(fig)  # Close to avoid display issues
        print("✓ Matplotlib: Successfully created and closed a figure")
    except Exception as e:
        print(f"✗ Matplotlib test failed: {e}")
    
    try:
        import seaborn as sns
        print(f"✓ Seaborn: Version {sns.__version__}")
    except Exception as e:
        print(f"✗ Seaborn test failed: {e}")
    
    try:
        import sklearn
        print(f"✓ Scikit-learn: Version {sklearn.__version__}")
    except Exception as e:
        print(f"✗ Scikit-learn test failed: {e}")
    
    try:
        import jax.numpy as jnp
        arr = jnp.array([1, 2, 3])
        print(f"✓ JAX: Created array {arr}")
    except Exception as e:
        print(f"✗ JAX test failed: {e}")
    
    try:
        import pynwb
        print(f"✓ PyNWB: Version {pynwb.__version__}")
    except Exception as e:
        print(f"✗ PyNWB test failed: {e}")
    
    try:
        from dynamax.hidden_markov_model import GaussianHMM
        print("✓ Dynamax: Successfully imported GaussianHMM")
    except Exception as e:
        print(f"✗ Dynamax test failed: {e}")
    
    try:
        import umap
        print(f"✓ UMAP: Version {umap.__version__}")
        # Test basic UMAP functionality
        reducer = umap.UMAP(n_neighbors=5, n_components=2, random_state=42)
        print("✓ UMAP: Successfully created UMAP reducer instance")
    except Exception as e:
        print(f"✗ UMAP test failed: {e}")

if __name__ == "__main__":
    print("Module Import Test Script")
    print("========================")
    
    # Test imports
    all_imports_successful = test_imports()
    
    # Test basic functionality if imports were successful
    if all_imports_successful:
        test_basic_functionality()
    
    print("\nTest completed!")