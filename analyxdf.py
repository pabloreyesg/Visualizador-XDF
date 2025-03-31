import sys
import numpy as np
import pyxdf
import csv
import neurokit2 as nk
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore

# Variables globales
canales_dict = {}       # Diccionario: clave = etiqueta; valor = dict con 'stream', 'time' y 'data'
triggers = []           # Lista de tuplas: (tiempo, marker)
recortes_guardados = {} # Diccionario para recortes guardados
TRIGGER_TOLERANCE = 0.01

def obtener_nombres_de_canales(info):
    nombres = []
    if info is None:
        return nombres
    if "desc" in info and info["desc"] and info["desc"][0] is not None:
        desc0 = info["desc"][0]
        if "channels" in desc0 and desc0["channels"] and desc0["channels"][0] is not None:
            channels_container = desc0["channels"][0]
            channels_data = channels_container.get("channel", None)
            if channels_data:
                if isinstance(channels_data, list):
                    for ch in channels_data:
                        if ch is not None and "label" in ch and ch["label"]:
                            label = ch["label"][0] if isinstance(ch["label"], list) else ch["label"]
                            nombres.append(label)
                        else:
                            nombres.append(None)
                elif isinstance(channels_data, dict):
                    if "label" in channels_data and channels_data["label"]:
                        label = channels_data["label"][0] if isinstance(channels_data["label"], list) else channels_data["label"]
                        nombres.append(label)
                if nombres and all(n is None for n in nombres):
                    nombres = []
                if nombres:
                    return nombres
    if "name" in info and info["name"]:
        base_name = info["name"][0] if isinstance(info["name"], list) else info["name"]
    else:
        base_name = None
    try:
        if "channel_count" in info and info["channel_count"]:
            count = int(info["channel_count"][0])
        else:
            count = 1
    except Exception:
        count = 1
    if base_name:
        if count == 1:
            nombres = [base_name]
        else:
            nombres = [f"{base_name} {i+1}" for i in range(count)]
    else:
        nombres = [f"Canal {i+1}" for i in range(count)]
    return nombres

def get_expected_sampling_rate(info):
    tipo = info.get("type", None)
    if tipo:
        if isinstance(tipo, list):
            tipo = tipo[0]
        tipo = tipo.upper()
        if any(x in tipo for x in ["ACC", "GYRO", "MAG"]):
            return 25
        elif "PPG" in tipo:
            return 25
        elif "TEMP" in tipo:
            return 7
        elif "EDA" in tipo:
            return 15
    return None

def recortar_senal(time_stamps, data_arr, t_start, t_end):
    """
    Recorta la señal entre t_start y t_end e incluye obligatoriamente ambos marcadores.
    Si t_start o t_end no están presentes, se insertan mediante interpolación.
    """
    try:
        t_start = float(t_start)
        t_end = float(t_end)
    except Exception:
        return time_stamps, data_arr

    # Convertir a arrays de NumPy
    time_arr = np.array(time_stamps)
    data_arr = np.array(data_arr)
    
    # Seleccionar muestras dentro del intervalo [t_start, t_end]
    mask = (time_arr >= t_start) & (time_arr <= t_end)
    time_recortado = time_arr[mask]
    data_recortado = data_arr[mask]

    # Incluir t_start si no está presente
    if time_recortado.size == 0 or not np.isclose(time_recortado[0], t_start):
        idx = np.searchsorted(time_arr, t_start)
        if idx == 0:
            value_start = data_arr[0]
        elif idx == len(time_arr):
            value_start = data_arr[-1]
        else:
            t0, t1 = time_arr[idx-1], time_arr[idx]
            v0, v1 = data_arr[idx-1], data_arr[idx]
            value_start = v0 + (v1 - v0) * (t_start - t0) / (t1 - t0)
        time_recortado = np.insert(time_recortado, 0, t_start)
        data_recortado = np.insert(data_recortado, 0, value_start)

    # Incluir t_end si no está presente
    if not np.isclose(time_recortado[-1], t_end):
        idx = np.searchsorted(time_arr, t_end)
        if idx == 0:
            value_end = data_arr[0]
        elif idx == len(time_arr):
            value_end = data_arr[-1]
        else:
            t0, t1 = time_arr[idx-1], time_arr[idx]
            v0, v1 = data_arr[idx-1], data_arr[idx]
            value_end = v0 + (v1 - v0) * (t_end - t0) / (t1 - t0)
        time_recortado = np.append(time_recortado, t_end)
        data_recortado = np.append(data_recortado, value_end)
    
    return time_recortado.tolist(), data_recortado.tolist()

def pupil_process(data, sampling_rate):
    processed_signal = np.array(data) * 0.95  # Ejemplo: atenuar la señal
    metrics = {"Media": np.mean(processed_signal), "Desviación": np.std(processed_signal)}
    return {"processed_signal": processed_signal, "metrics": metrics}

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Visualizador y Procesador de Señales XDF")
        self.resize(600, 400)
        self.setupUI()

    def setupUI(self):
        centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(centralWidget)
        layout = QtWidgets.QVBoxLayout(centralWidget)

        instrucciones = QtWidgets.QLabel("Seleccione una opción desde el menú.")
        layout.addWidget(instrucciones)

        menubar = self.menuBar()
        archivoMenu = menubar.addMenu("Archivo")
        cargarAction = QtWidgets.QAction("Cargar archivo XDF", self)
        cargarAction.triggered.connect(self.cargar_archivo)
        archivoMenu.addAction(cargarAction)
        salirAction = QtWidgets.QAction("Salir", self)
        salirAction.triggered.connect(self.close)
        archivoMenu.addAction(salirAction)

        procesarMenu = menubar.addMenu("Procesamiento")
        graficarAction = QtWidgets.QAction("Graficar canales", self)
        graficarAction.triggered.connect(self.abrir_menu_graficar)
        procesarMenu.addAction(graficarAction)
        cortarAction = QtWidgets.QAction("Cortar señal según triggers", self)
        cortarAction.triggered.connect(self.abrir_menu_cortar_triggers)
        procesarMenu.addAction(cortarAction)
        neurokitAction = QtWidgets.QAction("Procesar con NeuroKit", self)
        neurokitAction.triggered.connect(self.procesar_neurokit)
        procesarMenu.addAction(neurokitAction)

    def cargar_archivo(self):
        global canales_dict, triggers, recortes_guardados
        canales_dict = {}
        triggers = []
        recortes_guardados = {}

        ruta_archivo, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Seleccionar archivo XDF", "", "Archivos XDF (*.xdf)")
        if not ruta_archivo:
            return

        try:
            data, header = pyxdf.load_xdf(ruta_archivo)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudo cargar el archivo:\n{e}")
            return

        numeric_streams = []
        marker_streams = []
        for stream in data:
            y = stream.get("time_series", None)
            if y is None:
                continue
            if isinstance(y, list):
                marker_streams.append(stream)
            elif isinstance(y, np.ndarray):
                numeric_streams.append(stream)
            else:
                QtWidgets.QMessageBox.critical(self, "Error", "Formato de stream desconocido")
                return

        for m in marker_streams:
            for t, marker in zip(m.get("time_stamps", []), m.get("time_series", [])):
                if isinstance(marker, list) and marker:
                    triggers.append((t, marker[0]))
                elif isinstance(marker, str):
                    triggers.append((t, marker))
        triggers.sort(key=lambda x: x[0])

        for s_idx, stream in enumerate(numeric_streams):
            y = stream.get("time_series", None)
            time_stamps = stream.get("time_stamps", [])
            info = stream.get("info", {})

            channel_names = obtener_nombres_de_canales(info)
            if isinstance(y, np.ndarray):
                n_canales = 1 if y.ndim == 1 else y.shape[1]
            else:
                n_canales = 1

            if not channel_names or len(channel_names) < n_canales:
                if "name" in info and info["name"]:
                    base_name = info["name"][0] if isinstance(info["name"], list) else info["name"]
                else:
                    base_name = f"Stream {s_idx+1}"
                if n_canales == 1:
                    channel_names = [base_name]
                else:
                    channel_names = [f"{base_name} {i+1}" for i in range(n_canales)]

            for i in range(n_canales):
                etiqueta = f"Stream {s_idx+1} - {channel_names[i]}"
                canal_data = y if y.ndim == 1 else y[:, i]
                canales_dict[etiqueta] = {
                    'stream': stream,
                    'canal_idx': i,
                    'time': time_stamps,
                    'data': canal_data
                }
        QtWidgets.QMessageBox.information(self, "Carga completada", "Archivo cargado y canales extraídos correctamente.")

    def abrir_menu_graficar(self):
        if not canales_dict:
            QtWidgets.QMessageBox.warning(self, "Advertencia", "Primero debes cargar un archivo XDF.")
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Graficar canales")
        dialog.resize(500, 400)
        layout = QtWidgets.QVBoxLayout(dialog)

        label = QtWidgets.QLabel("Seleccione uno o varios canales para graficar:")
        layout.addWidget(label)

        listWidget = QtWidgets.QListWidget()
        listWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for clave in sorted(canales_dict.keys()):
            item = QtWidgets.QListWidgetItem(clave)
            listWidget.addItem(item)
        layout.addWidget(listWidget)

        btnGraficar = QtWidgets.QPushButton("Graficar")
        layout.addWidget(btnGraficar)

        def graficar():
            selectedItems = listWidget.selectedItems()
            if not selectedItems:
                QtWidgets.QMessageBox.warning(dialog, "Advertencia", "Debes seleccionar al menos un canal.")
                return
            for item in selectedItems:
                clave = item.text()
                canal_info = canales_dict[clave]
                time_stamps = canal_info['time']
                data_arr = canal_info['data']
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(time_stamps, data_arr, label=clave)
                ax.set_title(f"Señal - {clave}")
                ax.set_xlabel("Tiempo (s)")
                ax.set_ylabel("Valor")
                ax.grid(True)
                if np.size(time_stamps) > 0:
                    tmin, tmax = min(time_stamps), max(time_stamps)
                    for t, marker in triggers:
                        if tmin <= t <= tmax:
                            ax.axvline(x=t, linestyle='--', color='red', lw=0.5, alpha=0.7)
                            ylim = ax.get_ylim()
                            ax.text(t, ylim[1], f" {marker}", rotation=90,
                                    verticalalignment='top', color='red', fontsize=8)
                ax.legend()
                plt.tight_layout()
                plt.show()
            dialog.accept()

        btnGraficar.clicked.connect(graficar)
        dialog.exec_()

    def abrir_menu_cortar_triggers(self):
        if not canales_dict or not triggers:
            QtWidgets.QMessageBox.warning(self, "Advertencia", "Asegúrate de haber cargado el archivo y que existan triggers.")
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Cortar señal según triggers")
        dialog.resize(500, 450)
        layout = QtWidgets.QVBoxLayout(dialog)

        labelInicio = QtWidgets.QLabel("Seleccione el marcador de INICIO:")
        layout.addWidget(labelInicio)
        comboInicio = QtWidgets.QComboBox()
        lista_triggers = [f"{t:.3f} s - {marker}" for t, marker in triggers]
        comboInicio.addItems(lista_triggers)
        layout.addWidget(comboInicio)

        labelFin = QtWidgets.QLabel("Seleccione el marcador de FIN:")
        layout.addWidget(labelFin)
        comboFin = QtWidgets.QComboBox()
        comboFin.addItems(lista_triggers)
        layout.addWidget(comboFin)

        labelCanales = QtWidgets.QLabel("Seleccione uno o varios canales a recortar:")
        layout.addWidget(labelCanales)
        listWidget = QtWidgets.QListWidget()
        listWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for clave in sorted(canales_dict.keys()):
            item = QtWidgets.QListWidgetItem(clave)
            listWidget.addItem(item)
        layout.addWidget(listWidget)

        btnCortar = QtWidgets.QPushButton("Cortar y mostrar señal")
        btnGuardarArchivo = QtWidgets.QPushButton("Guardar recorte en archivo")
        btnGuardarApp = QtWidgets.QPushButton("Guardar recorte en aplicación")
        btnLayout = QtWidgets.QHBoxLayout()
        btnLayout.addWidget(btnCortar)
        btnLayout.addWidget(btnGuardarArchivo)
        btnLayout.addWidget(btnGuardarApp)
        layout.addLayout(btnLayout)

        def cortar_y_mostrar():
            inicio_sel = comboInicio.currentText()
            fin_sel = comboFin.currentText()
            if not inicio_sel or not fin_sel:
                QtWidgets.QMessageBox.warning(dialog, "Advertencia", "Debes seleccionar ambos marcadores (inicio y fin).")
                return
            try:
                t_start = float(inicio_sel.split(" s -")[0])
                t_end = float(fin_sel.split(" s -")[0])
            except Exception as e:
                QtWidgets.QMessageBox.critical(dialog, "Error", f"No se pudo interpretar los triggers seleccionados:\n{e}")
                return
            if t_end <= t_start:
                QtWidgets.QMessageBox.warning(dialog, "Advertencia", "El marcador de fin debe ser mayor que el de inicio.")
                return

            selectedItems = listWidget.selectedItems()
            if not selectedItems:
                QtWidgets.QMessageBox.warning(dialog, "Advertencia", "Debes seleccionar al menos un canal.")
                return

            for item in selectedItems:
                clave = item.text()
                canal_info = canales_dict[clave]
                time_stamps = canal_info['time']
                data_arr = canal_info['data']
                t_recort, data_recort = recortar_senal(time_stamps, data_arr, t_start, t_end)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(t_recort, data_recort, label=f"{clave} (recortada)")
                ax.set_title(f"Señal recortada - {clave}")
                ax.set_xlabel("Tiempo (s)")
                ax.set_ylabel("Valor")
                ax.grid(True)
                ax.legend()
                plt.tight_layout()
                plt.show()
            dialog.accept()

        # Función para guardar la señal recortada completa, incluyendo t_start y t_end,
        # y asignando TODOS los marcadores (concatenando si es necesario)
        def guardar_en_archivo():
            inicio_sel = comboInicio.currentText()
            fin_sel = comboFin.currentText()
            if not inicio_sel or not fin_sel:
                QtWidgets.QMessageBox.warning(dialog, "Advertencia", "Debes seleccionar ambos marcadores (inicio y fin).")
                return
            try:
                t_start = float(inicio_sel.split(" s -")[0])
                t_end = float(fin_sel.split(" s -")[0])
            except Exception as e:
                QtWidgets.QMessageBox.critical(dialog, "Error", f"No se pudo interpretar los triggers seleccionados:\n{e}")
                return
            if t_end <= t_start:
                QtWidgets.QMessageBox.warning(dialog, "Advertencia", "El marcador de fin debe ser mayor que el de inicio.")
                return

            # Obtener todos los triggers que caen en el intervalo
            markers_in_range = [(tt, marker) for (tt, marker) in triggers if t_start <= tt <= t_end]

            selectedItems = listWidget.selectedItems()
            if not selectedItems:
                QtWidgets.QMessageBox.warning(dialog, "Advertencia", "Debes seleccionar al menos un canal para exportar.")
                return

            for item in selectedItems:
                clave = item.text()
                canal_info = canales_dict[clave]
                time_stamps = canal_info['time']
                data_arr = canal_info['data']
                t_recort, data_recort = recortar_senal(time_stamps, data_arr, t_start, t_end)

                # Construir filas para toda la señal recortada
                filas = [(t, val, "") for t, val in zip(t_recort, data_recort)]
                # Asignar TODOS los marcadores a la muestra más cercana (concatenando si ya existe)
                for tt, marker in markers_in_range:
                    diffs = np.abs(np.array(t_recort) - tt)
                    idx = int(np.argmin(diffs))
                    if filas[idx][2] != "":
                        filas[idx] = (filas[idx][0], filas[idx][1], f"{filas[idx][2]}; {marker}")
                    else:
                        filas[idx] = (filas[idx][0], filas[idx][1], marker)
                filas.sort(key=lambda x: x[0])

                archivo_export, _ = QtWidgets.QFileDialog.getSaveFileName(dialog, f"Exportar {clave} a CSV", "", "CSV (*.csv)")
                if archivo_export:
                    try:
                        with open(archivo_export, 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(["Tiempo (s)", "Valor", "Trigger"])
                            for row in filas:
                                writer.writerow(row)
                        QtWidgets.QMessageBox.information(dialog, "Exportación", f"Se exportó {clave} exitosamente.")
                    except Exception as e:
                        QtWidgets.QMessageBox.critical(dialog, "Error", f"No se pudo exportar {clave}:\n{e}")
            dialog.accept()

        def guardar_en_aplicacion():
            inicio_sel = comboInicio.currentText()
            fin_sel = comboFin.currentText()
            if not inicio_sel or not fin_sel:
                QtWidgets.QMessageBox.warning(dialog, "Advertencia", "Debes seleccionar ambos marcadores (inicio y fin).")
                return
            try:
                t_start = float(inicio_sel.split(" s -")[0])
                t_end = float(fin_sel.split(" s -")[0])
            except Exception as e:
                QtWidgets.QMessageBox.critical(dialog, "Error", f"No se pudo interpretar los triggers seleccionados:\n{e}")
                return
            if t_end <= t_start:
                QtWidgets.QMessageBox.warning(dialog, "Advertencia", "El marcador de fin debe ser mayor que el de inicio.")
                return

            selectedItems = listWidget.selectedItems()
            if not selectedItems:
                QtWidgets.QMessageBox.warning(dialog, "Advertencia", "Debes seleccionar al menos un canal para guardar.")
                return

            for item in selectedItems:
                clave = item.text()
                canal_info = canales_dict[clave]
                time_stamps = canal_info['time']
                data_arr = canal_info['data']
                t_recort, data_recort = recortar_senal(time_stamps, data_arr, t_start, t_end)
                recortes_guardados[clave] = (t_recort, data_recort)
            QtWidgets.QMessageBox.information(dialog, "Guardado", "Señal recortada guardada en la aplicación para procesamiento futuro.")
            dialog.accept()

        btnCortar.clicked.connect(cortar_y_mostrar)
        btnGuardarArchivo.clicked.connect(guardar_en_archivo)
        btnGuardarApp.clicked.connect(guardar_en_aplicacion)
        dialog.exec_()

    def procesar_neurokit(self):
        """
        Se muestra un diálogo con una lista que contiene:
          - Los canales originales (prefijados con "Original: ")
          - Los recortes guardados (prefijados con "Recorte: ")
        Al seleccionar un ítem, se procesa la señal:
          * Si es un recorte, se utiliza la señal guardada en recortes_guardados.
          * Si es original, se usa la señal completa de canales_dict.
        Para el procesamiento se utiliza la información del canal (sampling rate, tipo, etc.).
        """
        if not canales_dict:
            QtWidgets.QMessageBox.warning(self, "Advertencia", "Primero debes cargar un archivo XDF.")
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Procesar canales con NeuroKit")
        dialog.resize(500, 400)
        layout = QtWidgets.QVBoxLayout(dialog)

        label = QtWidgets.QLabel("Seleccione uno o varios canales para procesar:")
        layout.addWidget(label)

        listWidget = QtWidgets.QListWidget()
        listWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        # Agregar canales originales
        for clave in sorted(canales_dict.keys()):
            item = QtWidgets.QListWidgetItem("Original: " + clave)
            listWidget.addItem(item)
        # Agregar recortes guardados, si existen
        if recortes_guardados:
            for clave in sorted(recortes_guardados.keys()):
                item = QtWidgets.QListWidgetItem("Recorte: " + clave)
                listWidget.addItem(item)
        layout.addWidget(listWidget)

        btnProcesar = QtWidgets.QPushButton("Procesar canales seleccionados")
        layout.addWidget(btnProcesar)

        def procesar_seleccion():
            selectedItems = listWidget.selectedItems()
            if not selectedItems:
                QtWidgets.QMessageBox.warning(dialog, "Advertencia", "Debes seleccionar al menos un canal para procesar.")
                return
            for item in selectedItems:
                label_text = item.text()
                if label_text.startswith("Recorte: "):
                    clave = label_text.replace("Recorte: ", "")
                    # Usar el recorte guardado
                    time_stamps, data_arr = recortes_guardados[clave]
                    # Se obtiene la información original para el canal
                    info = canales_dict[clave]["stream"].get("info", {})
                else:
                    clave = label_text.replace("Original: ", "")
                    canal_info = canales_dict[clave]
                    time_stamps = canal_info["time"]
                    data_arr = canal_info["data"]
                    info = canal_info["stream"].get("info", {})

                effective_srate = info.get("effective_srate", None)
                try:
                    sampling_rate = float(effective_srate) if effective_srate is not None else None
                except Exception:
                    sampling_rate = None
                expected_sr = get_expected_sampling_rate(info)
                if expected_sr is not None and sampling_rate is not None and sampling_rate != expected_sr:
                    QtWidgets.QMessageBox.information(dialog, "Información",
                        f"Para {clave} se encontró una tasa de muestreo {sampling_rate} Hz, pero se espera {expected_sr} Hz.\nSe usará {expected_sr} Hz para el procesamiento.")
                    sampling_rate = expected_sr
                if sampling_rate is None or sampling_rate == 0:
                    QtWidgets.QMessageBox.warning(dialog, "Advertencia", f"No se encontró una tasa de muestreo válida para {clave}.")
                    continue

                tipo = info.get("type", None)
                if tipo and isinstance(tipo, list):
                    tipo = tipo[0]
                tipo = tipo.upper() if tipo else ""
                if "EDA" in tipo:
                    try:
                        eda_signal = np.array(data_arr).flatten()
                        eda_cleaned = nk.eda_clean(eda_signal, sampling_rate=sampling_rate)
                        signals, info_processed = nk.eda_process(eda_cleaned, sampling_rate=sampling_rate)
                        nk.eda_plot(signals)
                        plt.show()
                    except Exception as e:
                        QtWidgets.QMessageBox.critical(dialog, "Error", f"Error al procesar EDA en {clave}:\n{e}")
                elif "ECG" in tipo:
                    signals, info_processed = nk.ecg_process(data_arr, sampling_rate=sampling_rate)
                    nk.ecg_plot(signals, sampling_rate=sampling_rate, show=True)
                elif "PUPIL" in tipo:
                    processed = pupil_process(data_arr, sampling_rate=sampling_rate)
                    processed_signal = processed["processed_signal"]
                    metrics = processed["metrics"]
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(time_stamps, processed_signal, label="Pupilometry Processed")
                    ax.set_title("Señal procesada de pupilometría")
                    ax.set_xlabel("Tiempo (s)")
                    ax.set_ylabel("Valor")
                    ax.grid(True)
                    ax.legend()
                    plt.tight_layout()
                    plt.show()
                    print("Métricas de pupilometría:")
                    for key, value in metrics.items():
                        print(f"{key}: {value}")
                else:
                    QtWidgets.QMessageBox.information(dialog, "Información", f"No hay procesamiento NeuroKit implementado para el tipo '{tipo}' en el canal {clave}.")
            dialog.accept()

        btnProcesar.clicked.connect(procesar_seleccion)
        dialog.exec_()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
