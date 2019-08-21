from pathlib        import Path
from libs.index     import IndexManager
import libs.dirs    as dirs

csvPath = Path(dirs.test) / "test_assets/test_dataset_indexed.csv"

ind = IndexManager(csvPath)
ind.check_files()