# Installing the MRT2 Audio Unit (AUv3)

Please follow these steps carefully to register the plugin extension with macOS and load it in your DAW.

---

## 1. Installation & Registration

macOS registers the location of the plugin extension (`.appex`) the first time the host app (`MRT2 (AU).app`) is opened. If you move the app *after* opening it, macOS will lose track of it or register duplicate paths, causing the plugin to fail to load in your DAW.

### Step-by-Step Installation:
1. **Move `MRT2 (AU).app` to a permanent location** (we recommend `/Applications` or your user's `~/Applications` folder). **Do this first** before opening the app.
2. **Open `MRT2 (AU).app` once** to register the embedded Audio Unit extension with the macOS system.
3. **Important for updates:** Whenever you update the plugin, replace the old app at the exact same path, then open the new app once to refresh registration.

### Troubleshooting Registration (`pluginkit`)

If the plugin does not appear in your DAW, the macOS registration database might be stale or pointing to a previous location. You can inspect and clean it using the terminal:

1. **Verify registered plugin paths:**
   ```bash
   pluginkit -mAv | grep -i mrt
   ```
2. **Remove stale/incorrect paths:**
   If you see duplicate or old paths, unregister them using their specific path:
   ```bash
   pluginkit -r "/path/to/old/stale/MRT2_AU.appex"
   ```
3. **Manually register the new path:**
   ```bash
   pluginkit -a "/Applications/MRT2 (AU).app/Contents/PlugIns/MRT2_AU.appex"
   ```

---

## 2. Host / DAW Settings

The inference engine runs in real-time and expects specific audio configuration from the host:

* **AUv3 Search/Support:** Make sure your DAW is configured to scan for **Audio Unit v3 (AUv3)** plugins. The instrument should appear under **AUv3 Instruments** (or **Audio Units** -> **Google** -> **MRT2**).
  * *Ableton Live Note:* You may need to go to Preferences -> Plug-ins and explicitly turn on "Use Audio Units v3" for it to scan.
* **Sample Rate:** You **MUST** set your DAW / audio interface sample rate to **48,000 Hz (48 kHz)** before opening the plugin. Other sample rates (like 44.1 kHz) will cause playback pitch distortion or silence because the model weights are strictly trained on 48 kHz audio.
* **Buffer Size:** For low-latency performance, set your audio buffer size as low as your system can handle (e.g., **64, 128, or 256 samples**).

---

## 3. Reporting Issues

If you encounter bugs, performance issues, or crash logs, please report them at:
👉 https://github.com/magenta/magenta-realtime/issues
