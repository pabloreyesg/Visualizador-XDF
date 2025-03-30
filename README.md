# Visualizador y Procesador de Señales XDF

Este proyecto es una interfaz gráfica desarrollada en Python para visualizar, recortar y procesar señales almacenadas en archivos `.xdf`, un formato común en adquisiciones de datos fisiológicos (como OpenBCI o LabRecorder). La aplicación permite cargar señales, visualizar canales con superposición de triggers, recortar señales entre marcadores específicos y aplicar procesamiento con [`NeuroKit2`](https://neurokit2.readthedocs.io/en/latest/).

## 🧰 Características principales

- Carga y visualización de archivos `.xdf`.
- Extracción automática de canales y triggers.
- Visualización de múltiples canales superpuestos con sus eventos.
- Recorte de señales según eventos (triggers) seleccionados.
- Exportación de segmentos recortados a archivos `.csv`.
- Procesamiento automático de señales EDA, ECG y pupilometría con NeuroKit2.
- Interfaz gráfica sencilla usando `tkinter`.

## 📦 Requisitos

Este proyecto requiere Python 3.7 o superior y las siguientes librerías:

```bash
pip install numpy matplotlib pyxdf neurokit2