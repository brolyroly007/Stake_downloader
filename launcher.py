import uvicorn
import os
import sys
import multiprocessing

# Necesario para PyInstaller con multiprocessing (uvicorn usa workers)
multiprocessing.freeze_support()

if __name__ == "__main__":
    # Asegurar que estamos en el directorio correcto
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))
    
    print("Iniciando servidor web en http://localhost:8000...")
    
    # Importar app aquí para evitar efectos secundarios al importar
    from app import app
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
