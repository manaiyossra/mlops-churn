import sys
import numpy, pandas, sklearn, matplotlib, mlflow, fastapi, uvicorn, joblib

print("Python :", sys.version.split()[0])
print("Venv actif :", sys.prefix != sys.base_prefix)
for m in (numpy, pandas, sklearn, matplotlib, mlflow, fastapi, uvicorn, joblib):
    print(f"{m.__name__:12} {m.__version__}")
print("Environment is set up correctly!")
