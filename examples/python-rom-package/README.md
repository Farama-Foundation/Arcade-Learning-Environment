# ALE Python ROM Plugin

This example demonstrates distributing ROMs via a Python package. A package which would like to register ROMs with the ALE must hook into the `ale-py.roms` entry point with a function which returns a list of ROM filenames. The signature should look like:

```py
def roms() -> List[pathlib.Path]:
  pass
```

The ALE will only register ROMs which are supported, i.e., that pass `ALEInterface.isSupportedROM`. To check if your ROMs are supported you can run `ale-import-roms roms/ --dry-run`. For a full list of supported ROMs see `md5.txt`.


## Example Package

To use the example package simply place all you ROMs (with .bin extension) in the `roms/` directory. You can now install the package locally, build a wheel, etc. and the supported ROMs in `roms/` will be visible to the ALE. You'll now be able to import ROMs as

```py
# e.g., if you imported the supported version of Freeway
from ale_py.roms import Freeway

# Print all registered ROMs
import ale_py.roms as roms
print(roms.__all__)
```
