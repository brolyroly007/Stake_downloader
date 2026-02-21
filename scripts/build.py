import PyInstaller.__main__
import shutil
from pathlib import Path
import os

def build():
    print("=== Building Viral Monitor Executable ===")
    
    # Limpiar dist/build anteriores
    if Path("dist").exists():
        shutil.rmtree("dist")
    if Path("build").exists():
        shutil.rmtree("build")

    # Definir argumentos de PyInstaller
    args = [
        'main.py',                      # Script principal
        '--name=ViralMonitor',          # Nombre del exe
        '--onefile',                    # Un solo archivo
        '--clean',                      # Limpiar cache
        
        # Incluir directorios de datos
        '--add-data=web;web',           # Carpeta web
        
        # Imports ocultos necesarios
        '--hidden-import=app',
        '--hidden-import=uvicorn.logging',
        '--hidden-import=uvicorn.loops',
        '--hidden-import=uvicorn.loops.auto',
        '--hidden-import=uvicorn.protocols',
        '--hidden-import=uvicorn.protocols.http',
        '--hidden-import=uvicorn.protocols.http.auto',
        '--hidden-import=uvicorn.lifespan',
        '--hidden-import=uvicorn.lifespan.on',
        '--hidden-import=engineio.async_drivers.aiohttp',
        '--hidden-import=playwright',
        
        # Excluir módulos innecesarios para reducir tamaño
        '--exclude-module=tkinter',
        '--exclude-module=matplotlib',
        '--exclude-module=notebook',
        '--exclude-module=scipy',
    ]
    
    # Ejecutar PyInstaller
    PyInstaller.__main__.run(args)
    
    print("\n=== Build Complete ===")
    print(f"Executable created at: {os.path.abspath('dist/ViralMonitor.exe')}")
    print("\nIMPORTANT: Ensure 'ffmpeg' is installed/available in the system PATH or in the same folder as the exe.")

if __name__ == "__main__":
    build()
