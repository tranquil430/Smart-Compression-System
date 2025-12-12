import os
import sys
import json
import time
import datetime
import threading
import subprocess
import shutil
import tempfile
import pickle
import zipfile
import schedule
import webview
import pystray
import pyzipper
import joblib
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

# Try to import the adaptive logging module (formerly filter_model2)
try:
    # Assuming the file was renamed to adaptive_log.py and sits in the models/ or root dir
    # If it is in the same directory, import directly.
    import adaptive_log
    RETRAINING_AVAILABLE = True
except ImportError:
    try:
        # Fallback if user kept the original name
        import filter_model2 as adaptive_log
        RETRAINING_AVAILABLE = True
    except ImportError:
        adaptive_log = None
        RETRAINING_AVAILABLE = False
        print("[WARNING] Adaptive logging module not found. Retraining disabled.")


def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) 
        # Adjusting base_path to go up one level if this script is in src/
        if not os.path.exists(os.path.join(base_path, relative_path)):
            # Fallback to current dir if not found in parent
            base_path = os.path.abspath(os.path.dirname(__file__))
            
    return os.path.join(base_path, relative_path)


class SmartCompressionSystem:
    def __init__(self):
        self.window = None
        self.selected_folders = []
        self.output_path = ""
        self.schedule_time = ""
        self.file_types = []
        self.use_password = False
        self.password = ""
        self.status = ""
        self.virus_scan_enabled = False
        self.time_threshold_days = 0
        self.output_format = "compressed_{filename}"
        self.smart_compression_enabled = False
        self.group_by_file_type = False
        self.compression_method = "zip"
        
        self.total_original_size = 0
        self.total_compressed_size = 0
        self.all_compressed_files = []
        self.file_type_stats = {}
        
        self.archive_log_file = "_scs_archive.log"
        self.scheduler_thread = None
        self.scheduler_stop_event = threading.Event()
        self.stop_event = threading.Event()
        self.current_subprocess = None

        self._load_models()
        self.html = self._load_ui_resources()

    def _load_ui_resources(self):
        """Loads HTML, CSS, and JS assets."""
        # Adjust paths based on new structure (assets/ folder)
        html_path = resource_path(os.path.join("assets", "index.html"))
        js_path = resource_path(os.path.join("assets", "app.js"))
        css_path = resource_path(os.path.join("assets", "styles.css"))
        
        if not os.path.exists(html_path):
            return "<html><body><h2>UI assets not found</h2></body></html>"
            
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        # Inject CSS
        if os.path.exists(css_path):
            with open(css_path, "r", encoding="utf-8") as f:
                css_content = f.read()
            html_content = html_content.replace('</head>', f'<style>{css_content}</style></head>')
            
        # Inject JS
        if os.path.exists(js_path):
            with open(js_path, "r", encoding="utf-8") as f:
                js_content = f.read()
            html_content = html_content.replace('<script src="app.js"></script>', f'<script>{js_content}</script>')
            
        return html_content

    def _load_models(self):
        """Initializes machine learning models."""
        # XGBoost Optimizer
        xgb_path = resource_path(os.path.join("models", "xgb.pkl"))
        if not os.path.exists(xgb_path):
            xgb_path = resource_path("xgb.pkl") # Fallback

        self.model = None
        try:
            if os.path.exists(xgb_path):
                with open(xgb_path, 'rb') as f:
                    (self.model, self.model1_encoder, self.model1_features, 
                     self.model1_scaler, self.model1_bins, self.model1_bin_labels, 
                     self.model1_numerical_cols) = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load optimizer model: {e}")

        # Decision Filter (Model 1)
        filter1_path = resource_path(os.path.join("models", "1st_filter.pkl"))
        if not os.path.exists(filter1_path): filter1_path = resource_path("1st_filter.pkl")
        
        self.decision_model = None
        try:
            if os.path.exists(filter1_path):
                with open(filter1_path, 'rb') as f:
                    self.decision_model = pickle.load(f)
        except Exception:
            pass

        # Adaptive Model (Model 2/3)
        self.model3 = None
        self.le_action = None
        self.le_filetype = None
        self.le_time = None
        
        try:
            def load_priority(filename):
                # Check local models/ dir first, then root, then bundle
                paths = [
                    os.path.join("models", filename),
                    filename,
                    resource_path(filename)
                ]
                for p in paths:
                    if os.path.exists(p):
                        return joblib.load(p)
                raise FileNotFoundError(f"Could not find {filename}")

            self.model3 = load_priority("2nd_filter.pkl")
            self.le_action = load_priority("le_action.pkl")
            self.le_filetype = load_priority("le_filetype.pkl")
            self.le_time = load_priority("le_time.pkl")
        except Exception as e:
            print(f"[WARNING] Adaptive model failed to load: {e}")

    # -------------------------
    # UI Interop API
    # -------------------------
    
    def open_file_explorer(self, path):
        try:
            if os.path.exists(path):
                os.startfile(path)
                return {"status": "ok"}
            return {"status": "error", "message": "Path not found"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_dashboard_stats(self):
        return {
            "original_size": self.total_original_size,
            "compressed_size": self.total_compressed_size,
            "file_list": self.all_compressed_files,
            "file_types": self.file_type_stats
        }

    def update_compression_method(self, method):
        self.compression_method = method
        return {"status": "ok"}

    def update_output_format(self, format_string):
        self.output_format = format_string
        return {"status": "ok"}

    def update_time_threshold(self, days):
        self.time_threshold_days = int(days)
        return {"status": "ok"}

    def add_folders(self):
        if not self.window: return []
        result = self.window.create_file_dialog(webview.FOLDER_DIALOG, allow_multiple=True)
        if result:
            for folder in result:
                if folder not in self.selected_folders:
                    self.selected_folders.append(folder)
        return self.selected_folders

    def remove_folder(self, folder_path):
        if folder_path in self.selected_folders:
            self.selected_folders.remove(folder_path)
        return self.selected_folders

    def select_output_path(self):
        if not self.window: return ""
        result = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        if result: self.output_path = result[0]
        return self.output_path
    
    def set_schedule(self, time_str):
        self.schedule_time = time_str
        if self.scheduler_thread:
            self.scheduler_stop_event.set()
            self.scheduler_thread.join(timeout=2.0) 
        schedule.clear()
        self.scheduler_stop_event.clear()
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        return {"status": "ok", "message": f"Schedule active for {time_str}."}

    def update_excluded_types(self, types):
        self.file_types = types or []
        return {"status": "ok"}

    def start_compression_thread(self):
        if not self.selected_folders: return {"message": "Error: No folders selected!"}
        if not self.output_path: return {"message": "Error: No output path selected!"}
        
        self.stop_event.clear()
        threading.Thread(target=self._process_compression, daemon=True).start()
        return {"message": "Compression started..."}

    def stop_compression_process(self):
        self.stop_event.set()
        if self.current_subprocess:
            try:
                self.current_subprocess.terminate()
            except Exception: pass
        return {"status": "stopping"}

    def toggle_virus_scan(self, enabled):
        self.virus_scan_enabled = bool(enabled)
        return {"status": "ok"}

    def toggle_password(self, enabled):
        self.use_password = bool(enabled)
        return {"status": "ok"}

    def update_password(self, password):
        self.password = str(password or "")
        return {"status": "ok"}

    def toggle_smart_compression(self, enabled):
        self.smart_compression_enabled = bool(enabled)
        if self.smart_compression_enabled:
            self.time_threshold_days = 0
            self.file_types = []
        return {"status": "ok"}
    
    def toggle_group_by_type(self, enabled):
        self.group_by_file_type = bool(enabled)
        return {"status": "ok"}
        
    def delete_current_input_folders(self):
        if not self.selected_folders:
            return {"success": False, "message": "No input folders selected."}
        
        deleted = 0
        failed = []
        for folder in list(self.selected_folders):
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                    self.selected_folders.remove(folder)
                    deleted += 1
                except Exception as e:
                    failed.append(f"{os.path.basename(folder)}")
            else:
                self.selected_folders.remove(folder)
        
        msg = f"Deleted {deleted} folders."
        if failed: msg += f" Failed: {', '.join(failed)}"
        return {"success": True, "message": msg, "remaining": self.selected_folders}

    def delete_all_archives(self):
        if not self.output_path: 
            return {"success": False, "message": "Output path not set."}
        
        log_path = os.path.join(self.output_path, self.archive_log_file)
        if not os.path.exists(log_path): 
            return {"success": True, "message": "No archive history found."}

        deleted, failed, surviving = 0, 0, []
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                paths = [l.strip() for l in f if l.strip()]
            
            for path in paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        deleted += 1
                    except Exception:
                        failed += 1
                        surviving.append(path)
            
            with open(log_path, 'w', encoding='utf-8') as f:
                for path in surviving: f.write(f"{path}\n")
            
            return {"success": True, "message": f"Deleted {deleted} archives. {failed} failed."}
        except Exception as e:
            return {"success": False, "message": f"Error: {e}"}

    # -------------------------
    # Internal Logic
    # -------------------------

    def _log_archive_path(self, archive_path):
        if not self.output_path: return
        log_path = os.path.join(self.output_path, self.archive_log_file)
        try:
            with open(log_path, 'a', encoding='utf-8') as f: 
                f.write(f"{archive_path}\n")
        except Exception: pass

    def _get_time_of_day(self):
        h = datetime.datetime.now().hour
        if 5 <= h < 12: return 'morning'
        if 12 <= h < 17: return 'afternoon'
        if 17 <= h < 21: return 'evening'
        return 'night'

    def _log_retraining_data(self, file_path, file_size):
        if not RETRAINING_AVAILABLE or not adaptive_log: return
        try:
            ext = os.path.splitext(file_path)[1].lower().lstrip('.') or "unknown"
            adaptive_log.log_user_action(
                file_type=ext,
                file_size=file_size,
                similar_history=1,
                time_of_day=self._get_time_of_day(),
                action="compress"
            )
        except Exception: pass

    def _filter_files(self):
        """Orchestrates file selection based on Smart vs Manual modes."""
        if self.smart_compression_enabled and self.decision_model:
            return self._smart_filter()
        return self._manual_filter()

    def _smart_filter(self):
        now = time.time()
        candidates = []
        seen = set()
        
        for folder in self.selected_folders:
            for root, _, files in os.walk(folder):
                for file in files:
                    fp = os.path.join(root, file)
                    if fp in seen: continue
                    seen.add(fp)
                    
                    try:
                        # Baseline: Ignore files < 24 hours old
                        if (now - os.path.getctime(fp)) < 86400: continue
                        
                        size = os.path.getsize(fp)
                        ext = os.path.splitext(file)[1].lower()
                        
                        candidates.append({
                            "folder": folder, "file_path": fp,
                            "days_since_access": (now - os.path.getatime(fp)) / 86400,
                            "days_since_mod": (now - os.path.getmtime(fp)) / 86400,
                            "size_mb": size / (1024*1024), "ext": ext,
                            "file_size": size, "file_type_str": ext.lstrip('.')
                        })
                    except Exception: pass
        
        if not candidates: return []
        df = pd.DataFrame(candidates)
        
        # 1. Apply Decision Model (Model 1)
        try:
            cols = ["days_since_access", "days_since_mod", "size_mb", "ext"]
            # Ensure columns exist
            for c in cols:
                if c not in df.columns: df[c] = ".unknown" if c == 'ext' else 0
            
            preds = self.decision_model.predict(df[cols])
            df = df[preds == 'compress']
        except Exception: pass # Fallback to all if model fails

        # 2. Apply Adaptive Model (Model 3)
        if self.model3 and not df.empty:
            try:
                df['time_of_day'] = self._get_time_of_day()
                
                # Encode features manually using the loaded encoders
                # Note: This requires strict error handling for unseen labels
                def encode_safe(val, encoder):
                    return encoder.transform([val])[0] if val in encoder.classes_ else -1
                
                df['ft_enc'] = df['file_type_str'].apply(lambda x: encode_safe(x, self.le_filetype))
                df['tod_enc'] = df['time_of_day'].apply(lambda x: encode_safe(x, self.le_time))
                
                # Filter valid encodings
                df_valid = df[(df['ft_enc'] != -1) & (df['tod_enc'] != -1)].copy()
                
                if not df_valid.empty:
                    X_adaptive = df_valid[['ft_enc', 'file_size', 'tod_enc']].copy()
                    X_adaptive.columns = ["file_type", "file_size", "time_of_day"]
                    X_adaptive['similar_history'] = 0 # Placeholder feature
                    
                    # Reorder cols to match training
                    X_adaptive = X_adaptive[["file_type", "file_size", "similar_history", "time_of_day"]]
                    
                    adaptive_preds = self.model3.predict(X_adaptive)
                    decoded_preds = self.le_action.inverse_transform(adaptive_preds)
                    
                    df = df_valid[decoded_preds == 'compress']
            except Exception as e:
                print(f"[WARNING] Adaptive filter error: {e}")

        if df.empty: return []
        return list(zip(df['folder'], df['file_path']))

    def _manual_filter(self):
        files_out = []
        seen = set()
        now = time.time()
        threshold_sec = self.time_threshold_days * 86400

        for folder in self.selected_folders:
            for root, _, files in os.walk(folder):
                for file in files:
                    fp = os.path.join(root, file)
                    if fp in seen: continue
                    seen.add(fp)

                    if self.file_types and any(file.lower().endswith(ext) for ext in self.file_types):
                        continue
                        
                    if threshold_sec > 0:
                        try:
                            if (now - os.path.getatime(fp)) < threshold_sec: continue
                        except Exception: pass
                        
                    files_out.append((folder, fp))
        return files_out

    def _scan_files(self, files):
        clean_files = []
        infected = []
        stats = {
            "original_size": 0,
            "file_list": [],
            "file_types": {}
        }
        
        total = len(files)

        for i, (folder, path) in enumerate(files):
            if self.stop_event.is_set(): return [], {}, "Stopped."

            filename = os.path.basename(path)
            
            # Update UI
            if self.window and i % 5 == 0:
                safe_name = json.dumps(filename)
                self.window.evaluate_js(f"updateScanProgress({{ 'total_files': {total}, 'current_file': {i+1}, 'filename': {safe_name} }})")

            # Virus Check
            if self.virus_scan_enabled:
                if self._check_virus(path):
                    infected.append(filename)
                    self._notify_ui(f"Virus detected: {filename}", False)
                    continue

            try:
                # Basic accessibility check
                with open(path, 'rb') as f: f.read(1024)
                
                size = os.path.getsize(path)
                ext = os.path.splitext(filename)[1].lower() or ".file"
                
                stats["original_size"] += size
                stats["file_list"].append((filename, size))
                stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
                clean_files.append((folder, path))
                
            except Exception:
                # Could be permission error or AV blocking access
                pass

        msg = ""
        if infected:
            msg = f" (Skipped {len(infected)} infected files)"
            
        return clean_files, stats, msg

    def _check_virus(self, path):
        """Uses Windows Defender CLI for scanning."""
        import glob
        
        # Locate MpCmdRun.exe
        paths = [
            os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Windows Defender", "MpCmdRun.exe"),
            os.path.join(os.environ.get("ProgramData", "C:\\ProgramData"), "Microsoft", "Windows Defender", "Platform", "*", "MpCmdRun.exe")
        ]
        
        exe = None
        for p in paths:
            if "*" in p:
                candidates = sorted(glob.glob(p))
                if candidates: p = candidates[-1]
            if os.path.exists(p) and not os.path.isdir(p):
                exe = p
                break
        
        if not exe: return False # Cannot scan
        
        try:
            # -ScanType 3 is custom file scan
            res = subprocess.run(
                [exe, "-Scan", "-ScanType", "3", "-File", path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            return res.returncode == 2 # 2 = Infected
        except Exception:
            return False

    def _create_archive(self, archive_path, files, method):
        # Ensure unique filename
        if os.path.exists(archive_path):
            base, ext = os.path.splitext(archive_path)
            c = 1
            while os.path.exists(f"{base} ({c}){ext}"): c += 1
            archive_path = f"{base} ({c}){ext}"

        try:
            if method == 'zip':
                # Python native zip (supports AES via pyzipper)
                mode = pyzipper.AESZipFile if self.use_password else zipfile.ZipFile
                kwargs = {'compression': pyzipper.ZIP_DEFLATED}
                if self.use_password:
                    kwargs['encryption'] = pyzipper.WZ_AES
                
                with mode(archive_path, 'w', **kwargs) as zf:
                    if self.use_password: zf.setpassword(self.password.encode())
                    
                    total = len(files)
                    for i, (folder, path) in enumerate(files):
                        if self.stop_event.is_set(): return None
                        
                        zf.write(path, os.path.relpath(path, folder))
                        self._log_retraining_data(path, os.path.getsize(path))
                        
                        if self.window and i % 5 == 0:
                            self.window.evaluate_js(f"updateCompressionProgress({{'total_files': {total}, 'current_file': {i+1}}})")
                            
            else:
                # External CLI (7z, RAR)
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as tf:
                    for _, path in files: tf.write(f"{path}\n")
                    list_path = tf.name

                cmd = [method, 'a']
                if method == 'rar': cmd.append('-ep1')
                if method == '7z': cmd.append('-spf')
                
                cmd.extend([archive_path, f'@{list_path}'])
                if self.use_password: cmd.append(f'-p{self.password}')

                if self.window: 
                    self.window.evaluate_js(f"showIndeterminateProgress('Compressing via {method.upper()}...')")

                self.current_subprocess = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                self.current_subprocess.communicate()
                
                os.remove(list_path)
                if self.current_subprocess.returncode != 0:
                    raise Exception("External process failed")
                
                # Batch log
                for _, path in files:
                    try: self._log_retraining_data(path, os.path.getsize(path))
                    except: pass

            return archive_path
        except Exception as e:
            print(f"[ERROR] Archiving failed: {e}")
            return None

    def _process_compression(self):
        """Main worker method."""
        msg = "Unknown error"
        success = False
        start_time = time.time()
        
        try:
            # 1. Filter
            files = self._filter_files()
            if not files:
                msg = "No files matched criteria."
                return

            # 2. Scan
            clean_files, stats, inf_msg = self._scan_files(files)
            if not clean_files:
                msg = f"No clean files available. {inf_msg}"
                return

            if self.stop_event.is_set():
                msg = "Stopped by user."
                return

            # 3. Compress
            self.total_original_size = 0
            self.total_compressed_size = 0
            
            # Generate base filename
            now = datetime.datetime.now()
            name_tmpl = self.output_format.replace("{date}", now.strftime("%Y%m%d")) \
                                          .replace("{time}", now.strftime("%H%M%S")) \
                                          .replace("{filename}", f"backup_{now.strftime('%H%M%S')}")

            if not self.group_by_file_type:
                # Single Archive
                dest = os.path.join(self.output_path, f"{name_tmpl}.{self.compression_method}")
                final_path = self._create_archive(dest, clean_files, self.compression_method)
                
                if final_path:
                    self.total_original_size = stats['original_size']
                    self.total_compressed_size = os.path.getsize(final_path)
                    self.all_compressed_files = stats['file_list']
                    self.file_type_stats = stats['file_types']
                    self._log_archive_path(final_path)
                    success = True
                    msg = "Compression complete." + inf_msg
            else:
                # Grouped Archives
                groups = {}
                for f, p in clean_files:
                    ext = os.path.splitext(p)[1].lower() or "unk"
                    groups.setdefault(ext, []).append((f, p))
                
                results = []
                for ext, g_files in groups.items():
                    if self.stop_event.is_set(): break
                    
                    sub_dest = os.path.join(self.output_path, f"{name_tmpl}_{ext.lstrip('.')}.{self.compression_method}")
                    res = self._create_archive(sub_dest, g_files, self.compression_method)
                    
                    if res:
                        self.total_original_size += sum(os.path.getsize(p) for _, p in g_files)
                        self.total_compressed_size += os.path.getsize(res)
                        self._log_archive_path(res)
                        results.append(ext)
                
                self.all_compressed_files = stats['file_list']
                self.file_type_stats = stats['file_types']
                
                if results:
                    success = True
                    msg = f"Grouped compression done ({len(results)} groups). {inf_msg}"
                else:
                    msg = "Grouped compression failed or stopped."

        except Exception as e:
            msg = f"Critical error: {str(e)}"
            print(f"[EXCEPTION] {e}")
        finally:
            # UI Teardown
            duration = f"{time.time() - start_time:.2f}s"
            if self.window:
                self.window.evaluate_js(f"handleCompressionComplete({json.dumps({ 'message': msg, 'success': success, 'stats': self.get_dashboard_stats(), 'folders_deleted': False, 'compression_time': duration })})")

    def _notify_ui(self, message, is_success=True):
        if self.window:
            safe = json.dumps(message)
            self.window.evaluate_js(f"showNotification({safe}, {json.dumps(is_success)})")

    def _run_scheduler(self):
        schedule.every().day.at(self.schedule_time).do(self.start_compression_thread)
        while not self.scheduler_stop_event.is_set():
            schedule.run_pending()
            self.scheduler_stop_event.wait(60)

    def shutdown(self):
        self.scheduler_stop_event.set()
        if RETRAINING_AVAILABLE and adaptive_log:
            print("[INFO] Triggering model retraining...")
            adaptive_log.retrain_model_if_needed()

# -------------------------
# System Tray & Entry Point
# -------------------------

class TrayApp:
    def __init__(self, api, window):
        self.api = api
        self.window = window
        self.icon = pystray.Icon("SCS", self._create_icon(), "Smart Compression", self._create_menu())

    def _create_icon(self):
        # Create a simple generated icon if asset missing
        path = resource_path(os.path.join("assets", "tray_icon.png"))
        if os.path.exists(path):
            return Image.open(path)
        
        img = Image.new('RGBA', (64, 64), (0,0,0,0))
        draw = ImageDraw.Draw(img)
        draw.ellipse((10, 10, 54, 54), fill="#4F46E5")
        return img

    def _create_menu(self):
        return pystray.Menu(
            pystray.MenuItem("Show Dashboard", self.show_ui),
            pystray.MenuItem("Exit", self.quit_app)
        )

    def show_ui(self, icon, item):
        self.window.show()

    def quit_app(self, icon, item):
        self.api.shutdown()
        self.icon.stop()
        self.window.destroy()
        sys.exit()

    def run(self):
        self.icon.run_detached()


if __name__ == "__main__":
    if os.name == 'nt':
        # Fix for high-DPI displays on Windows
        import ctypes
        try: ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except: pass

    api = SmartCompressionSystem()
    
    window = webview.create_window(
        'Smart Compression System', 
        html=api.html, 
        js_api=api, 
        width=1000, 
        height=720,
        min_size=(800, 600)
    )
    api.window = window
    
    tray = TrayApp(api, window)
    threading.Thread(target=tray.run, daemon=True).start()
    
    # Hide window on close instead of destroying (min to tray)
    window.events.closing += lambda: (window.hide(), False)[1]
    
    webview.start(gui='edgechromium', debug=False, http_server=True)