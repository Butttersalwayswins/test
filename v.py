# audio_visualizer.py
import sys
import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QMessageBox, QCheckBox, QSlider
import logging
import time

# Setup logging
logging.basicConfig(filename='visualizer.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters
CHUNK = 1024  # Chunk size for audio processing
UPDATE_INTERVAL = 100  # Update interval in ms
NUM_BARS = 30  # Number of equalizer bars
SMOOTHING_ALPHA = 0.9  # Smoothing factor for equalizer
VISUAL_MODE = "Equalizer"  # Default visualization mode
STEREO_MODE = "Average"  # Default stereo mode
BEAT_MIN_INTERVAL = 0.5  # Minimum time (seconds) between beat detections
BEAT_DETECTION_ENABLED = True  # Default beat detection state
BEAT_COLOR = (255, 0, 0, 255)  # Default beat detection color (Red)
WAVEFORM_COLOR = (255, 255, 255, 255)  # Default waveform color (White)
WATERFALL_BASE_COLOR = (0, 128, 255, 255)  # Default waterfall base color (Blue-ish)
FREQUENCY_SPECTRUM_COLOR = (0, 255, 0, 255)  # Default frequency spectrum color (Green)
SILENT_THRESHOLD = 1e-7  # Threshold for detecting silent audio
SPECTROGRAM_WIDTH = 80  # Number of time steps in spectrogram
SPECTROGRAM_NFFT = 512  # FFT size for spectrogram
SPECTROGRAM_INTENSITY_SCALE = 2.0  # Default intensity scale for spectrogram
WATERFALL_DEPTH = 20  # Number of curves in waterfall plot

# Color options for beat detection, waveform, waterfall, and frequency spectrum
COLOR_OPTIONS = {
    "Red": (255, 0, 0, 255),
    "Blue": (0, 0, 255, 255),
    "Green": (0, 255, 0, 255),
    "Yellow": (255, 255, 0, 255),
    "White": (255, 255, 255, 255),
    "Pink": (255, 105, 180, 255)
}

# Colormap options for spectrogram
COLORMAP_OPTIONS = ["plasma", "viridis", "inferno", "magma"]
SPECTROGRAM_COLORMAP = "plasma"  # Default colormap

# Audio state
device_index = None
samplerate = 44100
audio_buffer_left = np.zeros(CHUNK, dtype=np.float32)
audio_buffer_right = np.zeros(CHUNK, dtype=np.float32)
prev_heights = np.zeros(NUM_BARS)
prev_energy = 0
last_beat_time = 0
stream = None
timer = None
bars = []
waveform = None
spectrogram = None
spectrogram_data = None
waterfall_curves = []
waterfall_data = []
frequency_spectrum = None

# Find audio device
try:
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if "CABLE Output" in dev['name'] and dev['max_input_channels'] > 0:
            device_index = i
            break
    if device_index is None:
        device_index = sd.default.device[0]
        logging.info(f"Fallback to default input device: {devices[device_index]['name']}")
    samplerate = int(sd.query_devices(device_index)['default_samplerate'])
    logging.info(f"Using device {device_index}: {devices[device_index]['name']}, samplerate: {samplerate}")
except Exception as e:
    logging.error(f"Device initialization failed: {e}")
    print(f"Error: Audio device initialization failed. Check visualizer.log.")
    sys.exit(1)

# GUI setup
try:
    app = QtWidgets.QApplication(sys.argv)
    main_widget = QtWidgets.QWidget()
    main_widget.setWindowTitle("Desktop Audio Visualizer")
    main_layout = QHBoxLayout()
    main_widget.setLayout(main_layout)

    # Plot widget
    plot_widget = pg.GraphicsLayoutWidget()
    plot = plot_widget.addPlot(title="Visualizer")
    plot.setYRange(0, 1)
    plot.setLabel('bottom', 'Frequency (Hz)')
    main_layout.addWidget(plot_widget)

    # Control panel
    control_widget = QtWidgets.QWidget()
    control_layout = QVBoxLayout()
    device_combo = QComboBox()
    device_combo.addItems([dev['name'] for dev in devices if dev['max_input_channels'] > 0])
    device_combo.setCurrentIndex(device_index)
    control_layout.addWidget(QLabel("Input Device:"))
    control_layout.addWidget(device_combo)
    vis_combo = QComboBox()
    vis_combo.addItems(["Equalizer", "Waveform", "Spectrogram", "Waterfall", "Frequency Spectrum"])
    control_layout.addWidget(QLabel("Visualization Mode:"))
    control_layout.addWidget(vis_combo)
    stereo_combo = QComboBox()
    stereo_combo.addItems(["Average", "Left", "Right"])
    control_layout.addWidget(QLabel("Stereo Mode:"))
    control_layout.addWidget(stereo_combo)
    beat_checkbox = QCheckBox("Enable Beat Detection")
    beat_checkbox.setChecked(BEAT_DETECTION_ENABLED)
    control_layout.addWidget(beat_checkbox)
    beat_color_combo = QComboBox()
    beat_color_combo.addItems(COLOR_OPTIONS.keys())
    beat_color_combo.setCurrentText("Red")
    control_layout.addWidget(QLabel("Beat Detection Color:"))
    control_layout.addWidget(beat_color_combo)
    waveform_color_combo = QComboBox()
    waveform_color_combo.addItems(COLOR_OPTIONS.keys())
    waveform_color_combo.setCurrentText("White")
    control_layout.addWidget(QLabel("Waveform Color:"))
    control_layout.addWidget(waveform_color_combo)
    waterfall_color_combo = QComboBox()
    waterfall_color_combo.addItems(COLOR_OPTIONS.keys())
    waterfall_color_combo.setCurrentText("Blue")
    control_layout.addWidget(QLabel("Waterfall Color:"))
    control_layout.addWidget(waterfall_color_combo)
    frequency_spectrum_color_combo = QComboBox()
    frequency_spectrum_color_combo.addItems(COLOR_OPTIONS.keys())
    frequency_spectrum_color_combo.setCurrentText("Green")
    control_layout.addWidget(QLabel("Frequency Spectrum Color:"))
    control_layout.addWidget(frequency_spectrum_color_combo)
    spectrogram_colormap_combo = QComboBox()
    spectrogram_colormap_combo.addItems(COLORMAP_OPTIONS)
    spectrogram_colormap_combo.setCurrentText(SPECTROGRAM_COLORMAP)
    control_layout.addWidget(QLabel("Spectrogram Colormap:"))
    control_layout.addWidget(spectrogram_colormap_combo)
    intensity_slider = QSlider(QtCore.Qt.Horizontal)
    intensity_slider.setMinimum(1)
    intensity_slider.setMaximum(100)
    intensity_slider.setValue(int(SPECTROGRAM_INTENSITY_SCALE * 10))
    control_layout.addWidget(QLabel("Spectrogram Intensity Scale:"))
    control_layout.addWidget(intensity_slider)
    control_widget.setLayout(control_layout)
    main_layout.addWidget(control_widget)

    main_widget.resize(900, 500)
    main_widget.show()
except Exception as e:
    logging.error(f"GUI initialization failed: {e}")
    print(f"Error: GUI failed to open. Check visualizer.log.")
    sys.exit(1)

def show_error(message):
    """Display an error message to the user."""
    QMessageBox.critical(main_widget, "Error", message)

def get_color_gradient(value):
    """Generate a color gradient based on amplitude for equalizer bars."""
    try:
        value = np.clip(value, 0, 1)
        if value < 0.33:
            ratio = value / 0.33
            r, g, b = 0, int(255 * ratio), int(255 * (1 - ratio))
        elif value < 0.66:
            ratio = (value - 0.33) / 0.33
            r, g, b = int(255 * ratio), 255, 0
        else:
            ratio = (value - 0.66) / 0.34
            r, g, b = 255, int(255 * (1 - ratio)), 0
        return (r, g, b)
    except Exception as e:
        logging.error(f"Color gradient error: {e}")
        return (0, 0, 255)

def get_waterfall_color(index):
    """Generate fading color for waterfall plot curves based on WATERFALL_BASE_COLOR."""
    alpha = int(255 * (index + 1) / WATERFALL_DEPTH)  # Fade from transparent to opaque
    r, g, b, _ = WATERFALL_BASE_COLOR
    return (r, g, b, alpha)

def audio_callback(indata, frames, time, status):
    """Callback for real-time audio input."""
    try:
        global audio_buffer_left, audio_buffer_right
        if status:
            logging.warning(f"Audio callback status: {status}")
        if indata.shape[1] == 2:
            audio_buffer_left = indata[:, 0].copy()
            audio_buffer_right = indata[:, 1].copy()
        else:
            audio_buffer_left = audio_buffer_right = indata[:, 0].copy()
        logging.debug(f"Audio buffer stats: left_mean={np.mean(audio_buffer_left):.4f}, "
                      f"right_mean={np.mean(audio_buffer_right):.4f}, "
                      f"left_std={np.std(audio_buffer_left):.4f}")
    except Exception as e:
        logging.error(f"Audio callback error: {e}")

def init_equalizer_bars():
    """Initialize equalizer bars."""
    global bars
    plot.clear()
    bars = []
    bar_x = np.arange(NUM_BARS)
    bar_width = 0.8
    for x in bar_x:
        bar = pg.BarGraphItem(x=[x], height=[0], width=bar_width, brush='b')
        plot.addItem(bar)
        bars.append(bar)
        logging.debug(f"Initialized bar {x} for Equalizer")
    plot.setYRange(0, 1)
    plot.setXRange(0, NUM_BARS)
    plot.setLabel('bottom', 'Frequency (Hz)')
    plot.setLabel('left', 'Amplitude')
    logging.debug("Equalizer initialized")

def init_waveform():
    """Initialize waveform plot with selected color."""
    global waveform
    plot.clear()
    waveform = plot.plot(pen=pg.mkPen(WAVEFORM_COLOR))
    plot.setYRange(-1, 1)
    plot.setXRange(0, CHUNK / samplerate)
    plot.setLabel('bottom', 'Time (s)')
    plot.setLabel('left', 'Amplitude')
    logging.debug(f"Waveform initialized with color {WAVEFORM_COLOR}")

def init_spectrogram():
    """Initialize spectrogram."""
    global spectrogram, spectrogram_data
    plot.clear()
    spectrogram_data = np.zeros((SPECTROGRAM_NFFT // 2, SPECTROGRAM_WIDTH))
    spectrogram = pg.ImageItem()
    plot.addItem(spectrogram)
    spectrogram.setImage(spectrogram_data, autoLevels=False, levels=(-4, 1))
    plot.setYRange(0, samplerate / 2)
    plot.setXRange(0, SPECTROGRAM_WIDTH * UPDATE_INTERVAL / 1000.0)
    plot.setLabel('left', 'Frequency (Hz)')
    plot.setLabel('bottom', 'Time (s)')
    colormap = pg.colormap.get(SPECTROGRAM_COLORMAP)
    spectrogram.setLookupTable(colormap.getLookupTable())
    logging.debug(f"Spectrogram initialized with colormap {SPECTROGRAM_COLORMAP}, shape {spectrogram_data.shape}")

def init_waterfall():
    """Initialize waterfall plot with selected base color."""
    global waterfall_curves, waterfall_data
    plot.clear()
    waterfall_curves = []
    waterfall_data = [np.zeros(SPECTROGRAM_NFFT // 2) for _ in range(WATERFALL_DEPTH)]
    freqs = np.linspace(0, samplerate / 2, SPECTROGRAM_NFFT // 2)
    for i in range(WATERFALL_DEPTH):
        curve = plot.plot(x=freqs + i * (samplerate / 2 / WATERFALL_DEPTH), y=waterfall_data[i], pen=pg.mkPen(get_waterfall_color(i)))
        waterfall_curves.append(curve)
    plot.setYRange(0, 1)
    plot.setXRange(0, samplerate / 2 + (WATERFALL_DEPTH - 1) * (samplerate / 2 / WATERFALL_DEPTH))
    plot.setLabel('left', 'Amplitude')
    plot.setLabel('bottom', 'Frequency (Hz)')
    logging.debug(f"Waterfall initialized with {WATERFALL_DEPTH} curves, base color {WATERFALL_BASE_COLOR}")

def init_frequency_spectrum():
    """Initialize frequency spectrum plot with selected color."""
    global frequency_spectrum
    plot.clear()
    frequency_spectrum = plot.plot(pen=pg.mkPen(FREQUENCY_SPECTRUM_COLOR))
    plot.setYRange(0, 1)
    plot.setXRange(20, samplerate / 2)
    plot.setLabel('left', 'Amplitude')
    plot.setLabel('bottom', 'Frequency (Hz)')
    plot.getAxis('bottom').setScale(1.0)
    logging.debug(f"Frequency Spectrum initialized with color {FREQUENCY_SPECTRUM_COLOR}")

def update_visualization():
    """Update the visualization (Equalizer, Waveform, Spectrogram, Waterfall, Frequency Spectrum)."""
    try:
        global prev_heights, prev_energy, bars, waveform, spectrogram, spectrogram_data, waterfall_curves, waterfall_data, frequency_spectrum, last_beat_time
        audio = (audio_buffer_left + audio_buffer_right) / 2 if STEREO_MODE == "Average" else \
                audio_buffer_left if STEREO_MODE == "Left" else audio_buffer_right

        # Check for silent or invalid audio data
        is_silent = np.all(audio == 0) or np.any(np.isnan(audio)) or np.std(audio) < SILENT_THRESHOLD
        if is_silent:
            logging.debug("Silent audio detected")
            if VISUAL_MODE == "Equalizer" and len(bars) == NUM_BARS:
                prev_heights = np.zeros(NUM_BARS)
                for i, bar in enumerate(bars):
                    bar.setOpts(height=0, brush=pg.mkBrush(0, 0, 255, 255))
            elif VISUAL_MODE == "Waveform" and waveform is not None:
                waveform.setData(np.linspace(0, CHUNK / samplerate, CHUNK), np.zeros(CHUNK))
            elif VISUAL_MODE == "Spectrogram" and spectrogram is not None:
                spectrogram_data = np.zeros((SPECTROGRAM_NFFT // 2, SPECTROGRAM_WIDTH))
                spectrogram.setImage(spectrogram_data, autoLevels=False, levels=(-4, 1))
            elif VISUAL_MODE == "Waterfall" and len(waterfall_curves) == WATERFALL_DEPTH:
                waterfall_data = [np.zeros(SPECTROGRAM_NFFT // 2) for _ in range(WATERFALL_DEPTH)]
                freqs = np.linspace(0, samplerate / 2, SPECTROGRAM_NFFT // 2)
                for i, curve in enumerate(waterfall_curves):
                    curve.setData(x=freqs + i * (samplerate / 2 / WATERFALL_DEPTH), y=waterfall_data[i])
            elif VISUAL_MODE == "Frequency Spectrum" and frequency_spectrum is not None:
                freqs = np.linspace(0, samplerate / 2, SPECTROGRAM_NFFT // 2)
                frequency_spectrum.setData(freqs, np.zeros(SPECTROGRAM_NFFT // 2))
            return

        # Log audio amplitude for debugging
        audio_amplitude = np.std(audio)
        logging.debug(f"Audio amplitude (std): {audio_amplitude:.6f}")

        # Beat detection
        if BEAT_DETECTION_ENABLED:
            current_time = time.time()
            energy = np.sum(audio ** 2)
            beat_detected = (energy > 2 * prev_energy and energy > 0.2 and 
                             current_time - last_beat_time > BEAT_MIN_INTERVAL)
            prev_energy = energy
            logging.debug(f"Energy: {energy:.4f}, Beat detected: {beat_detected}")
            if beat_detected:
                plot_widget.setBackground(BEAT_COLOR)
                QtCore.QTimer.singleShot(100, lambda: plot_widget.setBackground((0, 0, 0, 255)))
                last_beat_time = current_time

        if VISUAL_MODE == "Equalizer":
            if len(bars) != NUM_BARS:
                init_equalizer_bars()
            window = np.hanning(len(audio))
            fft_vals = np.abs(np.fft.rfft(audio * window))
            fft_vals *= audio_amplitude
            fft_vals = np.clip(fft_vals, 0, 1)
            freqs = np.logspace(np.log10(20), np.log10(samplerate / 2), NUM_BARS + 1)
            indices = np.clip((freqs / (samplerate / 2) * len(fft_vals)).astype(int), 0, len(fft_vals) - 1)
            heights = []
            for i in range(NUM_BARS):
                slice_vals = fft_vals[indices[i]:indices[i + 1]]
                height = np.mean(slice_vals) if len(slice_vals) > 0 else 0
                heights.append(height)
            heights = np.nan_to_num(heights, nan=0.0)
            heights = SMOOTHING_ALPHA * np.array(heights) + (1 - SMOOTHING_ALPHA) * prev_heights
            prev_heights = heights
            logging.debug(f"Equalizer heights: {heights[:5]}...")
            for i, height in enumerate(heights):
                r, g, b = get_color_gradient(height)
                bars[i].setOpts(height=[height], brush=pg.mkBrush(r, g, b, 255))
            plot.getAxis('bottom').setTicks([[(i, f"{freqs[i]:.0f}") for i in range(0, NUM_BARS, 5)]])

        elif VISUAL_MODE == "Waveform":
            if waveform is None:
                init_waveform()
            waveform.setData(np.linspace(0, CHUNK / samplerate, CHUNK), audio)

        elif VISUAL_MODE == "Spectrogram":
            if spectrogram is None:
                init_spectrogram()
            window = np.hanning(len(audio))
            fft_vals = np.abs(np.fft.rfft(audio * window, n=SPECTROGRAM_NFFT))
            fft_vals *= audio_amplitude * SPECTROGRAM_INTENSITY_SCALE
            fft_vals = np.log10(np.clip(fft_vals + 1e-10, 1e-10, np.inf))
            fft_vals = np.clip(fft_vals, -4, 1)
            spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
            spectrogram_data[:, -1] = fft_vals[:SPECTROGRAM_NFFT // 2]
            spectrogram.setImage(spectrogram_data, autoLevels=False, levels=(-4, 1))
            logging.debug(f"Spectrogram data range: min={np.min(spectrogram_data):.4f}, max={np.max(spectrogram_data):.4f}")

        elif VISUAL_MODE == "Waterfall":
            if len(waterfall_curves) != WATERFALL_DEPTH:
                init_waterfall()
            window = np.hanning(len(audio))
            fft_vals = np.abs(np.fft.rfft(audio * window, n=SPECTROGRAM_NFFT))
            fft_vals *= audio_amplitude
            fft_vals = np.clip(fft_vals, 0, 1)
            waterfall_data.pop(0)  # Remove oldest spectrum
            waterfall_data.append(fft_vals[:SPECTROGRAM_NFFT // 2])
            freqs = np.linspace(0, samplerate / 2, SPECTROGRAM_NFFT // 2)
            for i, curve in enumerate(waterfall_curves):
                curve.setData(x=freqs + i * (samplerate / 2 / WATERFALL_DEPTH), y=waterfall_data[i], pen=pg.mkPen(get_waterfall_color(i)))
            logging.debug(f"Waterfall updated, newest spectrum max: {np.max(waterfall_data[-1]):.4f}")

        elif VISUAL_MODE == "Frequency Spectrum":
            if frequency_spectrum is None:
                init_frequency_spectrum()
            window = np.hanning(len(audio))
            fft_vals = np.abs(np.fft.rfft(audio * window, n=SPECTROGRAM_NFFT))
            fft_vals *= audio_amplitude
            fft_vals = np.clip(fft_vals, 0, 1)
            freqs = np.linspace(0, samplerate / 2, SPECTROGRAM_NFFT // 2)
            frequency_spectrum.setData(freqs, fft_vals[:SPECTROGRAM_NFFT // 2])
            logging.debug(f"Frequency Spectrum updated, max amplitude: {np.max(fft_vals):.4f}")

    except Exception as e:
        logging.error(f"Visualization update error: {e}")
        show_error(f"Visualization update failed: {str(e)}")

def change_device(index):
    """Switch to a new audio input device."""
    try:
        global stream, device_index, samplerate
        if stream is not None:
            stream.stop()
            stream.close()
        device_index = index
        samplerate = int(sd.query_devices(device_index)['default_samplerate'])
        dev_info = sd.query_devices(device_index)
        channels = min(dev_info['max_input_channels'], 2)
        stream = sd.InputStream(device=device_index, channels=channels, samplerate=samplerate, 
                               callback=audio_callback, blocksize=CHUNK)
        stream.start()
        logging.info(f"Switched to device {index}: {devices[index]['name']}")
    except Exception as e:
        logging.error(f"Device change error: {e}")
        show_error(f"Failed to switch device: {str(e)}")

def change_visual_mode(mode):
    """Change visualization mode (Equalizer/Waveform/Spectrogram/Waterfall/Frequency Spectrum)."""
    try:
        global VISUAL_MODE, waveform, bars, spectrogram, spectrogram_data, waterfall_curves, waterfall_data, frequency_spectrum, prev_heights
        VISUAL_MODE = mode
        waveform = None
        bars = []
        spectrogram = None
        spectrogram_data = None
        waterfall_curves = []
        waterfall_data = []
        frequency_spectrum = None
        prev_heights = np.zeros(NUM_BARS)
        plot.clear()
        plot.getAxis('bottom').setTicks([])
        plot.getAxis('left').setLabel('')
        plot.getAxis('bottom').setLabel('')
        logging.debug(f"Plot cleared for mode switch to {mode}, items remaining: {len(plot.items)}")
        if VISUAL_MODE == "Equalizer":
            init_equalizer_bars()
        elif VISUAL_MODE == "Waveform":
            init_waveform()
        elif VISUAL_MODE == "Spectrogram":
            init_spectrogram()
        elif VISUAL_MODE == "Waterfall":
            init_waterfall()
        elif VISUAL_MODE == "Frequency Spectrum":
            init_frequency_spectrum()
        logging.info(f"Visualization mode changed to: {mode}")
    except Exception as e:
        logging.error(f"Visual mode change error: {e}")
        show_error(f"Failed to change visualization mode: {str(e)}")

def change_stereo_mode(mode):
    """Change stereo mode (Average/Left/Right)."""
    try:
        global STEREO_MODE
        STEREO_MODE = mode
        logging.info(f"Stereo mode changed to: {mode}")
    except Exception as e:
        logging.error(f"Stereo mode change error: {e}")
        show_error(f"Failed to change stereo mode: {str(e)}")

def toggle_beat_detection(state):
    """Toggle beat detection on/off."""
    try:
        global BEAT_DETECTION_ENABLED
        BEAT_DETECTION_ENABLED = state == QtCore.Qt.Checked
        logging.info(f"Beat detection enabled: {BEAT_DETECTION_ENABLED}")
        if not BEAT_DETECTION_ENABLED:
            plot_widget.setBackground((0, 0, 0, 255))
    except Exception as e:
        logging.error(f"Beat detection toggle error: {e}")
        show_error(f"Failed to toggle beat detection: {str(e)}")

def change_beat_color(color_name):
    """Change beat detection color."""
    try:
        global BEAT_COLOR
        BEAT_COLOR = COLOR_OPTIONS[color_name]
        logging.info(f"Beat detection color changed to: {color_name}")
    except Exception as e:
        logging.error(f"Beat color change error: {e}")
        show_error(f"Failed to change beat color: {str(e)}")

def change_waveform_color(color_name):
    """Change waveform line color."""
    try:
        global WAVEFORM_COLOR, waveform
        WAVEFORM_COLOR = COLOR_OPTIONS[color_name]
        if waveform is not None and VISUAL_MODE == "Waveform":
            waveform.setPen(pg.mkPen(WAVEFORM_COLOR))
        logging.info(f"Waveform color changed to: {color_name}")
    except Exception as e:
        logging.error(f"Waveform color change error: {e}")
        show_error(f"Failed to change waveform color: {str(e)}")

def change_waterfall_color(color_name):
    """Change waterfall base color."""
    try:
        global WATERFALL_BASE_COLOR, waterfall_curves
        WATERFALL_BASE_COLOR = COLOR_OPTIONS[color_name]
        if len(waterfall_curves) == WATERFALL_DEPTH and VISUAL_MODE == "Waterfall":
            for i, curve in enumerate(waterfall_curves):
                curve.setPen(pg.mkPen(get_waterfall_color(i)))
        logging.info(f"Waterfall color changed to: {color_name}")
    except Exception as e:
        logging.error(f"Waterfall color change error: {e}")
        show_error(f"Failed to change waterfall color: {str(e)}")

def change_frequency_spectrum_color(color_name):
    """Change frequency spectrum line color."""
    try:
        global FREQUENCY_SPECTRUM_COLOR, frequency_spectrum
        FREQUENCY_SPECTRUM_COLOR = COLOR_OPTIONS[color_name]
        if frequency_spectrum is not None and VISUAL_MODE == "Frequency Spectrum":
            frequency_spectrum.setPen(pg.mkPen(FREQUENCY_SPECTRUM_COLOR))
        logging.info(f"Frequency Spectrum color changed to: {color_name}")
    except Exception as e:
        logging.error(f"Frequency Spectrum color change error: {e}")
        show_error(f"Failed to change frequency spectrum color: {str(e)}")

def change_spectrogram_colormap(colormap_name):
    """Change spectrogram colormap."""
    try:
        global SPECTROGRAM_COLORMAP, spectrogram
        SPECTROGRAM_COLORMAP = colormap_name
        if spectrogram is not None:
            colormap = pg.colormap.get(SPECTROGRAM_COLORMAP)
            spectrogram.setLookupTable(colormap.getLookupTable())
            logging.debug(f"Colormap updated to {SPECTROGRAM_COLORMAP}")
        logging.info(f"Spectrogram colormap changed to: {colormap_name}")
    except Exception as e:
        logging.error(f"Spectrogram colormap change error: {e}")
        show_error(f"Failed to change spectrogram colormap: {str(e)}")

def change_intensity_scale(value):
    """Change spectrogram intensity scale."""
    try:
        global SPECTROGRAM_INTENSITY_SCALE
        SPECTROGRAM_INTENSITY_SCALE = value / 10.0
        logging.info(f"Spectrogram intensity scale changed to: {SPECTROGRAM_INTENSITY_SCALE}")
    except Exception as e:
        logging.error(f"Intensity scale change error: {e}")
        show_error(f"Failed to change intensity scale: {str(e)}")

# Connect signals
try:
    device_combo.currentIndexChanged.connect(change_device)
    vis_combo.currentTextChanged.connect(change_visual_mode)
    stereo_combo.currentTextChanged.connect(change_stereo_mode)
    beat_checkbox.stateChanged.connect(toggle_beat_detection)
    beat_color_combo.currentTextChanged.connect(change_beat_color)
    waveform_color_combo.currentTextChanged.connect(change_waveform_color)
    waterfall_color_combo.currentTextChanged.connect(change_waterfall_color)
    frequency_spectrum_color_combo.currentTextChanged.connect(change_frequency_spectrum_color)
    spectrogram_colormap_combo.currentTextChanged.connect(change_spectrogram_colormap)
    intensity_slider.valueChanged.connect(change_intensity_scale)
except Exception as e:
    logging.error(f"Signal connection error: {e}")
    show_error(f"Failed to connect signals: {str(e)}")
    sys.exit(1)

# Audio stream
try:
    dev_info = sd.query_devices(device_index)
    channels = min(dev_info['max_input_channels'], 2)
    stream = sd.InputStream(device=device_index, channels=channels, samplerate=samplerate, 
                           callback=audio_callback, blocksize=CHUNK)
    stream.start()
    logging.info("Audio stream started")
except Exception as e:
    logging.error(f"Stream initialization error: {e}")
    show_error(f"Failed to start audio stream: {str(e)}")
    sys.exit(1)

# Timer for updates
try:
    timer = QtCore.QTimer()
    timer.timeout.connect(update_visualization)
    timer.start(UPDATE_INTERVAL)
    logging.info("Timer started")
except Exception as e:
    logging.error(f"Timer initialization error: {e}")
    show_error(f"Failed to start timer: {str(e)}")
    sys.exit(1)

if __name__ == '__main__':
    try:
        logging.info("Visualizer application started")
        # Initialize the equalizer as the default visualization
        init_equalizer_bars()
        sys.exit(app.exec_())
    except Exception as e:
        logging.error(f"Visualizer application error: {e}")
        show_error(f"Visualizer application failed: {str(e)}")
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        logging.info("Visualizer application closed")
