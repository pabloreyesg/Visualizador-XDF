import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
import numpy as np
import pyxdf
import csv
import neurokit2 as nk  # Requiere: pip install neurokit2

# Variables globales para almacenar información de canales, triggers y recortes guardados
canales_dict = {}       # Llave: etiqueta descriptiva; Valor: dict con 'stream', 'time' y 'data'
triggers = []           # Lista de tuplas: (tiempo, marker)
recortes_guardados = {} # Diccionario: llave = etiqueta del canal, valor = (tiempos, datos) recortados

# Tolerancia para asociar un trigger (en segundos)
TRIGGER_TOLERANCE = 0.01

def obtener_nombres_de_canales(info):
    """
    Intenta extraer los nombres de los canales desde el header.
    Primero se intenta con "desc". Si no se encuentra, se utiliza "name" y "channel_count".
    Retorna una lista de nombres.
    """
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
    """
    Devuelve la tasa de muestreo esperada basada en el campo "type" del header.
      - Movimiento (ACC, GYRO, MAG): 25 Hz
      - PPG: 25 Hz
      - Temperatura (TEMP): 7 Hz
      - EDA: 15 Hz
    """
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

def cargar_archivo():
    """Carga el archivo XDF y extrae la información de canales, triggers y limpia recortes previos."""
    global canales_dict, triggers, recortes_guardados
    canales_dict = {}
    triggers = []
    recortes_guardados = {}

    ruta_archivo = filedialog.askopenfilename(
        title="Seleccionar archivo XDF",
        filetypes=[("Archivos XDF", "*.xdf")]
    )
    if not ruta_archivo:
        return

    try:
        data, header = pyxdf.load_xdf(ruta_archivo)
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{e}")
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
            messagebox.showerror("Error", "Formato de stream desconocido")
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
    messagebox.showinfo("Carga completada", "Archivo cargado y canales extraídos correctamente.")

def recortar_senal(time_stamps, data_arr, t_start, t_end):
    """Recorta la señal para quedarse con datos entre t_start y t_end."""
    try:
        t_start = float(t_start)
        t_end = float(t_end)
    except Exception:
        return time_stamps, data_arr

    mask = np.array(time_stamps) >= t_start
    mask &= np.array(time_stamps) <= t_end
    time_recortado = np.array(time_stamps)[mask]
    data_recortado = np.array(data_arr)[mask]
    return time_recortado.tolist(), data_recortado.tolist()

def abrir_menu_graficar():
    """Abre la ventana para graficar canales con triggers superpuestos."""
    if not canales_dict:
        messagebox.showwarning("Advertencia", "Primero debes cargar un archivo XDF.")
        return

    win = tk.Toplevel(root)
    win.title("Graficar canales")
    win.geometry("500x400")

    lbl = tk.Label(win, text="Seleccione uno o varios canales para graficar:")
    lbl.pack(pady=5)

    listbox = tk.Listbox(win, selectmode=tk.MULTIPLE, width=60)
    listbox.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
    for clave in sorted(canales_dict.keys()):
        listbox.insert(tk.END, clave)

    btn = tk.Button(win, text="Graficar", command=lambda: graficar_canales(listbox))
    btn.pack(pady=10)

def graficar_canales(listbox):
    seleccionados = [listbox.get(i) for i in listbox.curselection()]
    if not seleccionados:
        messagebox.showwarning("Advertencia", "Debes seleccionar al menos un canal.")
        return
    for clave in seleccionados:
        canal_info = canales_dict[clave]
        time_stamps = canal_info['time']
        data_arr = canal_info['data']
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_stamps, data_arr, label=clave)
        ax.set_title(f"Señal - {clave}")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Valor")
        ax.grid(True)
        if time_stamps is not None and np.size(time_stamps) > 0:
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

def abrir_menu_cortar_triggers():
    """Abre una ventana para recortar la señal según triggers y para guardar el recorte."""
    if not canales_dict or not triggers:
        messagebox.showwarning("Advertencia", "Asegúrate de haber cargado el archivo y que existan triggers.")
        return

    win = tk.Toplevel(root)
    win.title("Cortar señal según triggers")
    win.geometry("500x450")

    tk.Label(win, text="Seleccione el marcador de INICIO:").pack(pady=5)
    combo_inicio = ttk.Combobox(win, state="readonly", width=40)
    lista_triggers = [f"{t:.3f} s - {marker}" for t, marker in triggers]
    combo_inicio['values'] = lista_triggers
    combo_inicio.pack(pady=5)

    tk.Label(win, text="Seleccione el marcador de FIN:").pack(pady=5)
    combo_fin = ttk.Combobox(win, state="readonly", width=40)
    combo_fin['values'] = lista_triggers
    combo_fin.pack(pady=5)

    tk.Label(win, text="Seleccione uno o varios canales a recortar:").pack(pady=5)
    listbox = tk.Listbox(win, selectmode=tk.MULTIPLE, width=60)
    listbox.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
    for clave in sorted(canales_dict.keys()):
        listbox.insert(tk.END, clave)

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=10)

    def cortar_y_mostrar():
        inicio_sel = combo_inicio.get()
        fin_sel = combo_fin.get()
        if not inicio_sel or not fin_sel:
            messagebox.showwarning("Advertencia", "Debes seleccionar ambos marcadores (inicio y fin).")
            return
        try:
            t_start = float(inicio_sel.split(" s -")[0])
            t_end = float(fin_sel.split(" s -")[0])
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo interpretar los triggers seleccionados:\n{e}")
            return
        if t_end <= t_start:
            messagebox.showwarning("Advertencia", "El marcador de fin debe ser mayor que el de inicio.")
            return

        seleccionados = [listbox.get(i) for i in listbox.curselection()]
        if not seleccionados:
            messagebox.showwarning("Advertencia", "Debes seleccionar al menos un canal.")
            return

        for clave in seleccionados:
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
        win.destroy()

    def guardar_en_archivo():
        """Recorta y guarda la señal en un archivo CSV, incluyendo todos los triggers en el intervalo.
        Si un trigger no se asocia a una muestra exacta, se busca la muestra más cercana y se usa su valor."""
        inicio_sel = combo_inicio.get()
        fin_sel = combo_fin.get()
        if not inicio_sel or not fin_sel:
            messagebox.showwarning("Advertencia", "Debes seleccionar ambos marcadores (inicio y fin).")
            return
        try:
            t_start = float(inicio_sel.split(" s -")[0])
            t_end = float(fin_sel.split(" s -")[0])
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo interpretar los triggers seleccionados:\n{e}")
            return
        if t_end <= t_start:
            messagebox.showwarning("Advertencia", "El marcador de fin debe ser mayor que el de inicio.")
            return

        # Obtener todos los triggers que caen en el intervalo (sin tolerancia para contarlos)
        markers_in_range = [(tt, marker) for (tt, marker) in triggers if t_start <= tt <= t_end]

        seleccionados = [listbox.get(i) for i in listbox.curselection()]
        if not seleccionados:
            messagebox.showwarning("Advertencia", "Debes seleccionar al menos un canal para exportar.")
            return

        for clave in seleccionados:
            canal_info = canales_dict[clave]
            time_stamps = canal_info['time']
            data_arr = canal_info['data']
            t_recort, data_recort = recortar_senal(time_stamps, data_arr, t_start, t_end)
            filas = []
            # Agregar cada muestra con su trigger (si existe) a una tolerancia
            for t, val in zip(t_recort, data_recort):
                trigger_marker = ""
                for tt, marker in markers_in_range:
                    if abs(t - tt) <= TRIGGER_TOLERANCE:
                        trigger_marker = marker
                        break
                filas.append((t, val, trigger_marker))
            # Para cada trigger en markers_in_range que no se asoció a ninguna muestra,
            # buscar la muestra más cercana y usar su valor en lugar de "NA"
            for tt, marker in markers_in_range:
                if not any(abs(f[0] - tt) <= TRIGGER_TOLERANCE and f[2] == marker for f in filas):
                    # Buscar el índice de la muestra más cercana a tt
                    diffs = np.abs(np.array(t_recort) - tt)
                    idx = int(np.argmin(diffs))
                    valor_cercano = data_recort[idx]
                    filas.append((tt, valor_cercano, marker))
            filas.sort(key=lambda x: x[0])

            archivo_export = filedialog.asksaveasfilename(
                title=f"Exportar {clave} a CSV",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")]
            )
            if archivo_export:
                try:
                    with open(archivo_export, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["Tiempo (s)", "Valor", "Trigger"])
                        for row in filas:
                            writer.writerow(row)
                    triggers_exportados = [row for row in filas if row[2] != ""]
                    msg = (f"Se exportó {clave} exitosamente.\n"
                           f"Triggers en intervalo: {len(markers_in_range)}\n"
                           f"Triggers exportados: {len(triggers_exportados)}")
                    messagebox.showinfo("Exportación", msg)
                except Exception as e:
                    messagebox.showerror("Error", f"No se pudo exportar {clave}:\n{e}")

    def guardar_en_aplicacion():
        """Recorta y guarda la señal en la aplicación para procesamiento futuro."""
        inicio_sel = combo_inicio.get()
        fin_sel = combo_fin.get()
        if not inicio_sel or not fin_sel:
            messagebox.showwarning("Advertencia", "Debes seleccionar ambos marcadores (inicio y fin).")
            return
        try:
            t_start = float(inicio_sel.split(" s -")[0])
            t_end = float(fin_sel.split(" s -")[0])
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo interpretar los triggers seleccionados:\n{e}")
            return
        if t_end <= t_start:
            messagebox.showwarning("Advertencia", "El marcador de fin debe ser mayor que el de inicio.")
            return

        seleccionados = [listbox.get(i) for i in listbox.curselection()]
        if not seleccionados:
            messagebox.showwarning("Advertencia", "Debes seleccionar al menos un canal para guardar.")
            return

        for clave in seleccionados:
            canal_info = canales_dict[clave]
            time_stamps = canal_info['time']
            data_arr = canal_info['data']
            t_recort, data_recort = recortar_senal(time_stamps, data_arr, t_start, t_end)
            recortes_guardados[clave] = (t_recort, data_recort)
        messagebox.showinfo("Guardado", "Señal recortada guardada en la aplicación para procesamiento futuro.")

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=10)
    btn_cortar = tk.Button(btn_frame, text="Cortar y mostrar señal", command=cortar_y_mostrar)
    btn_cortar.grid(row=0, column=0, padx=5)
    btn_guardar_archivo = tk.Button(btn_frame, text="Guardar recorte en archivo", command=guardar_en_archivo)
    btn_guardar_archivo.grid(row=0, column=1, padx=5)
    btn_guardar_app = tk.Button(btn_frame, text="Guardar recorte en aplicación", command=guardar_en_aplicacion)
    btn_guardar_app.grid(row=0, column=2, padx=5)

def procesar_neurokit():
    """Procesa canales usando NeuroKit (EDA, ECG, Pupilometría)."""
    if not canales_dict:
        messagebox.showwarning("Advertencia", "Primero debes cargar un archivo XDF.")
        return

    win = tk.Toplevel(root)
    win.title("Procesar canales con NeuroKit")
    win.geometry("500x400")

    lbl = tk.Label(win, text="Seleccione uno o varios canales para procesar:")
    lbl.pack(pady=5)

    listbox = tk.Listbox(win, selectmode=tk.MULTIPLE, width=60)
    listbox.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
    for clave in sorted(canales_dict.keys()):
        listbox.insert(tk.END, clave)

    def procesar_seleccion():
        seleccionados = [listbox.get(i) for i in listbox.curselection()]
        if not seleccionados:
            messagebox.showwarning("Advertencia", "Debes seleccionar al menos un canal para procesar.")
            return
        for clave in seleccionados:
            canal_info = canales_dict[clave]
            stream = canal_info['stream']
            data_arr = canal_info['data']
            time_stamps = canal_info['time']
            info = stream.get("info", {})
            effective_srate = info.get("effective_srate", None)
            try:
                sampling_rate = float(effective_srate) if effective_srate is not None else None
            except Exception:
                sampling_rate = None
            expected_sr = get_expected_sampling_rate(info)
            if expected_sr is not None and sampling_rate is not None and sampling_rate != expected_sr:
                messagebox.showinfo("Información",
                    f"Para {clave} se encontró una tasa de muestreo {sampling_rate} Hz, pero se espera {expected_sr} Hz.\nSe usará {expected_sr} Hz para el procesamiento.")
                sampling_rate = expected_sr
            if sampling_rate is None or sampling_rate == 0:
                messagebox.showwarning("Advertencia", f"No se encontró una tasa de muestreo válida para {clave}.")
                continue

            tipo = info.get("type", None)
            if tipo and isinstance(tipo, list):
                tipo = tipo[0]
            tipo = tipo.upper() if tipo else ""
            if "EDA" in tipo:
                signals, info_processed = nk.eda_process(data_arr, sampling_rate=sampling_rate)
                nk.eda_plot(signals, show=True)
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
                messagebox.showinfo("Información", f"No hay procesamiento NeuroKit implementado para el tipo '{tipo}' en el canal {clave}.")
        win.destroy()

    btn = tk.Button(win, text="Procesar canales seleccionados", command=procesar_seleccion)
    btn.pack(pady=10)

# Función ficticia para procesamiento de pupilometría (reemplazar según la librería elegida)
def pupil_process(data, sampling_rate):
    processed_signal = np.array(data) * 0.95  # Ejemplo: atenuar la señal
    metrics = {"Media": np.mean(processed_signal), "Desviación": np.std(processed_signal)}
    return {"processed_signal": processed_signal, "metrics": metrics}

# Menú principal
root = tk.Tk()
root.title("Visualizador y Procesador de Señales XDF")
root.geometry("400x200")
root.resizable(False, False)

menubar = tk.Menu(root)
root.config(menu=menubar)

menu_archivo = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Archivo", menu=menu_archivo)
menu_archivo.add_command(label="Cargar archivo XDF", command=cargar_archivo)
menu_archivo.add_separator()
menu_archivo.add_command(label="Salir", command=root.quit)

menu_proc = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Procesamiento", menu=menu_proc)
menu_proc.add_command(label="Graficar canales", command=abrir_menu_graficar)
menu_proc.add_command(label="Cortar señal según triggers", command=abrir_menu_cortar_triggers)
menu_proc.add_command(label="Procesar con NeuroKit", command=procesar_neurokit)

root.mainloop()
