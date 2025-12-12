// --- Global variables ---
let fileChart = null;
let excludedSet = new Set();
let currentMethod = 'zip'; 

function updateFolderList(folders) {
    const folderList = document.getElementById("folder-list");
    if (!folderList) return;
    folderList.innerHTML = ""; 
    if (!folders || folders.length === 0) {
        folderList.innerHTML = '<p style="font-size: 13px; color: #717182; text-align:center;">No folders selected.</p>';
        return;
    }
    folders.forEach(f => {
        const div = document.createElement("div");
        div.className = "file-item";
        
        const safeFolderPath = f.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '&quot;');
        
        // --- CHANGED: Click event and Title ---
        div.style.cursor = "pointer";
        div.title = "Click to open in Explorer"; // Updated tooltip
        div.setAttribute("onclick", `openInExplorer('${safeFolderPath}')`); // Changed to onclick
        // --------------------------------------

        div.innerHTML = `
            <svg class="file-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>
            <div class="file-info"><p class="file-name">${f}</p></div>
            <button onclick="event.stopPropagation(); removeFolder('${safeFolderPath}')" style="background:transparent; border:none; cursor:pointer; color:#f44336; font-size:20px; font-weight:bold; padding:0 5px;">&times;</button>
            `;
        folderList.appendChild(div);
    });
}

function addFolders() {
    pywebview.api.add_folders().then(folders => {
        updateFolderList(folders);
        if (!folders || folders.length === 0) showNotification("No folders selected.", false);
    });
}

function removeFolder(folderPath) {
    pywebview.api.remove_folder(folderPath).then(folders => {
        updateFolderList(folders);
        showNotification("Folder removed.", true);
    });
}

function addOutputFolder() {
    pywebview.api.select_output_path().then(path => {
        const outputPathDiv = document.getElementById('outputPath');
        if (outputPathDiv) {
            outputPathDiv.innerHTML = '';
            if (path) {
                const pathItem = document.createElement('div');
                pathItem.className = 'list-item';
                pathItem.innerText = path;
                
                // --- CHANGED: Click event and Title ---
                const safePath = path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
                pathItem.style.cursor = "pointer";
                pathItem.title = "Click to open in Explorer"; // Updated tooltip
                pathItem.setAttribute("onclick", `openInExplorer('${safePath}')`); // Changed to onclick
                // --------------------------------------

                outputPathDiv.appendChild(pathItem);
            } else {
                outputPathDiv.innerText = "Not selected";
            }
        }
    });
}

function startCompression() {
    showNotification('Starting compression...', true);
    
    // 1. Toggle Buttons (Show Stop, Hide Start)
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    if (startBtn) startBtn.style.display = 'none';
    if (stopBtn) {
        stopBtn.style.display = 'block';
        stopBtn.innerText = "Stop Compression";
    }

    // 2. Reset UI elements
    const spaceSavedBar = document.getElementById('spaceSaved');
    if (spaceSavedBar) {
        spaceSavedBar.style.width = '0%';
        const title = document.querySelector('.space-saved-title');
        if (title) title.innerText = 'SPACE SAVED';
    }
    
    const exactDisplay = document.getElementById('exactSpaceSavedDisplay');
    if (exactDisplay) exactDisplay.innerText = '0 B';

    updateCompressionProgress({ total_files: 0, current_file: 0 });
    document.getElementById('progressText').innerText = 'Initializing...';
    
    // 3. Call Python
    pywebview.api.start_compression_thread().then(response => { 
        document.getElementById('progressText').innerText = response.message;
        
        // --- FIX START: Check for immediate validation errors ---
        if (response.message && response.message.startsWith("Error")) {
            // If Python says "Error: No folders selected", revert buttons immediately
            if (startBtn) startBtn.style.display = 'block';
            if (stopBtn) stopBtn.style.display = 'none';
            showNotification(response.message, false);
        }
        // --- FIX END ---
    });
}
function setCompressionMethod(method) {
    currentMethod = method; 
    const buttonGroup = document.getElementById('compressionMethodGroup');
    buttonGroup.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`method-${method}`).classList.add('active');
    pywebview.api.update_compression_method(method);
    
    // REMOVED: The logic that forced Smart Filtering to turn on
}

function togglePasswordField() {
    var checkbox = document.getElementById('togglePassword');
    var passwordInput = document.getElementById('passwordInput');
    pywebview.api.toggle_password(checkbox.checked);
    passwordInput.style.display = checkbox.checked ? 'block' : 'none';
}

function updatePassword() { pywebview.api.update_password(document.getElementById('passwordInput').value); }
function toggleVirusScan() { pywebview.api.toggle_virus_scan(document.getElementById('virusScan').checked); }
function updateTimeThreshold() { pywebview.api.update_time_threshold(parseInt(document.getElementById('timeThreshold').value) || 0); }
function updateFilenameFormat() { pywebview.api.update_output_format(document.getElementById('filenameFormat').value.trim()); }

function showIndeterminateProgress(message) {
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    if (!progressBar || !progressText) return;
    progressBar.innerText = ''; 
    progressBar.style.width = '100%'; 
    progressBar.classList.add('indeterminate'); 
    progressText.innerText = message;
}

function updateScanProgress(data) {
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    if (!progressBar || !progressText) return;
    progressBar.classList.remove('indeterminate');
    if (data.total_files > 0) {
        let percent = Math.round((data.current_file / data.total_files) * 100);
        progressBar.style.width = percent + '%';
        progressBar.innerText = percent + '%';
        progressText.innerText = `Scanning: ${data.current_file} of ${data.total_files} - ${data.filename}`;
    }
}

function updateCompressionProgress(data) {
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    if (!progressBar || !progressText) return;
    progressBar.classList.remove('indeterminate');
    let totalFiles = data.total_files;
    let currentFile = data.current_file;
    if (totalFiles > 0) {
        let percent = Math.round((currentFile / totalFiles) * 100);
        progressBar.style.width = percent + '%';
        progressBar.innerText = percent + '%';
        progressText.innerText = `Compressing: ${currentFile} of ${totalFiles} files`;
    } else {
        progressBar.style.width = '0%';
        progressBar.innerText = '0%';
        progressText.innerText = 'No files to compress.';
    }
}

function addExtension() {
    const select = document.getElementById('excludeSelect');
    const value = select.value;
    if (value && !excludedSet.has(value)) {
        excludedSet.add(value);
        updateExcludedList();
        pywebview.api.update_excluded_types(Array.from(excludedSet));
    }
}

function updateExcludedList() {
    const listDiv = document.getElementById('excludedExtensions');
    listDiv.innerHTML = '';
    excludedSet.forEach(ext => {
        const item = document.createElement('div');
        item.className = 'list-item';
        item.innerHTML = `${ext} <button onclick="removeExtension('${ext}')" style="color: red; cursor: pointer; margin-left: 10px; background: transparent; border: none; font-weight: bold;">&times;</button>`;
        listDiv.appendChild(item);
    });
}

function removeExtension(ext) {
    excludedSet.delete(ext);
    updateExcludedList();
    pywebview.api.update_excluded_types(Array.from(excludedSet));
}

function toggleSmartCompression() {
    const enabled = document.getElementById('smartCompression').checked;
    pywebview.api.toggle_smart_compression(enabled);
    document.getElementById('timeThreshold').disabled = enabled;
    const excludeSection = document.getElementById('excludeSection');
    if (excludeSection) {
        excludeSection.style.opacity = enabled ? '0.5' : '1';
        excludeSection.style.pointerEvents = enabled ? 'none' : 'auto';
    }
    
    if (enabled) {
        showNotification('Smart Filtering enabled. Manual rules disabled.', true);
    } else {
        showNotification('Smart Filtering disabled. Manual rules active.', true);
    }
    // REMOVED: The logic that reverted to ZIP if Auto was selected
}
function showNotification(message, success = true) {
    // --- DEBUG LOGGING ---
    console.log(`[UI NOTIFICATION] Type: ${success ? 'Success' : 'Error'}, Message: ${message}`);
    // ---------------------

    const notification = document.getElementById('notification');
    if (!notification) {
        console.error("Notification element not found in DOM!");
        return;
    }

    // Force a redraw by removing the class first if it exists
    notification.classList.remove('show');
    
    // Set content and style
    notification.innerText = message;
    notification.style.background = success ? 'linear-gradient(135deg, #4CAF50, #45a049)' : 'linear-gradient(135deg, #f44336, #e53935)';
    
    // Small timeout to allow the class removal to register before re-adding
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);

    // Hide after 4 seconds
    setTimeout(() => {
        notification.classList.remove('show');
    }, 4000);
}

function handleCompressionComplete(data) {
    // --- NEW: Reset Buttons ---
    document.getElementById('startBtn').style.display = 'block';
    document.getElementById('stopBtn').style.display = 'none';
    // --------------------------

    showNotification(data.message, data.success);
    if (data.success && data.stats) updateDashboard(data.stats);
    
    // ... (rest of the existing function) ...
    // Keep your existing logic for scanTime, compTime, and folders_deleted
    const scanDisplay = document.getElementById('scanTimeDisplay');
    const compDisplay = document.getElementById('compTimeDisplay');
    
    if (scanDisplay && data.scan_time) scanDisplay.innerText = data.scan_time;
    if (compDisplay && data.compression_time) compDisplay.innerText = data.compression_time;

    if (data.folders_deleted) {
        updateFolderList([]); 
    }

    const progressText = document.getElementById('progressText');
    if (progressText) progressText.innerText = data.message;
    
    const progressBar = document.getElementById('progressBar');
    if (progressBar) {
        progressBar.classList.remove('indeterminate');
        if (data.success) {
            progressBar.style.width = '100%';
            progressBar.innerText = 'Done';
        } else if (data.message.includes("Stopped")) {
            // Visual feedback for stop
            progressBar.style.width = '0%'; 
            progressBar.innerText = 'Stopped';
            progressBar.style.background = '#EF4444'; // Red
        } else {
            progressBar.style.width = '0%';
            progressBar.innerText = '0%';
        }
        
        // Reset bar color after a delay if it was red
        setTimeout(() => { progressBar.style.background = ''; }, 3000);
    }
}

function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

function updateDashboard(stats) {
    if (!stats) return;
    updateSpaceSaved(stats.original_size, stats.compressed_size);
    updateFileTypes(stats.file_types);
    updateCompressedList(stats.file_list);
    updateChart(stats.file_types);
}

function updateSpaceSaved(original_size, compressed_size) {
    const bar = document.getElementById('spaceSaved');
    
    // Update the text numbers for Original and Compressed
    const origDisplay = document.getElementById('originalSizeDisplay');
    const compDisplay = document.getElementById('compressedSizeDisplay');
    
    if (origDisplay) origDisplay.innerText = formatBytes(original_size);
    if (compDisplay) compDisplay.innerText = formatBytes(compressed_size);

    // --- NEW: Calculate and display the exact saved amount ---
    const exactDisplay = document.getElementById('exactSpaceSavedDisplay');
    let savedAmount = original_size - compressed_size;
    if (savedAmount < 0) savedAmount = 0;
    
    if (exactDisplay) {
        // Update the green badge with the exact formatted number
        exactDisplay.innerText = formatBytes(savedAmount);
    }
    // ---------------------------------------------------------

    if (!bar) return;
    
    // Calculate percentage for the bar
    let saved_percent = original_size > 0 ? Math.round(((original_size - compressed_size) / original_size) * 100) : 0;
    if (saved_percent < 0) saved_percent = 0; 
    
    bar.style.width = saved_percent + '%';
    
    // Update the title to show percentage (Optional: Keeps the percentage in the title text)
    const title = document.querySelector('.space-saved-title');
    if (title) title.innerText = `SPACE SAVED (${saved_percent}%)`;
}

function updateFileTypes(file_types) {
    const list = document.getElementById('fileTypes');
    if (!list) return;
    list.innerHTML = '';
    const sortedTypes = Object.entries(file_types).sort(([,a],[,b]) => b-a);
    if (sortedTypes.length === 0) { list.innerText = 'None'; return; }
    sortedTypes.slice(0, 5).forEach(([ext, count]) => {
        const div = document.createElement('div');
        div.style.cssText = 'flex: 1; text-align: center;';
        div.innerHTML = `<span style="font-weight: 600; color: #6B70D9;">${ext}</span><br>${count}`;
        list.appendChild(div);
    });
}

function updateCompressedList(file_list) {
    const ol = document.getElementById('compressedList');
    if (!ol) return;
    ol.innerHTML = '';
    if (!file_list || file_list.length === 0) { ol.innerHTML = '<li>No data yet</li>'; return; }
    file_list.slice(0, 10).forEach(([name, size]) => {
        const li = document.createElement('li');
        li.innerText = `${name} (${formatBytes(size)})`;
        ol.appendChild(li);
    });
}

function updateChart(file_types) {
    const canvas = document.getElementById('chart');
    if (!canvas) return; 
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const labels = Object.keys(file_types);
    const data = Object.values(file_types);
    if (fileChart) fileChart.destroy(); 
    if (labels.length === 0) {
        ctx.font = "14px -apple-system"; ctx.fillStyle = "#717182"; ctx.textAlign = "center";
        ctx.fillText("No file data to display", canvas.width / 2, canvas.height / 2);
        return;
    }
    fileChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: ['rgba(136, 141, 242, 0.8)', 'rgba(189, 221, 252, 0.8)', 'rgba(107, 112, 217, 0.8)', 'rgba(150, 150, 150, 0.8)', 'rgba(255, 206, 86, 0.8)'],
                borderColor: 'rgba(255, 255, 255, 0.7)', borderWidth: 2
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { 
                legend: { position: 'bottom', labels: { color: '#2a2a3e', padding: 10 } },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.label || '';
                            if (label) {
                                label += ': ';
                            }
                            let value = context.parsed;
                            let total = context.dataset.data.reduce((a, b) => a + b, 0);
                            let percentage = Math.round((value / total) * 100) + '%';
                            return label + value + ' (' + percentage + ')';
                        }
                    }
                }
            }
        }
    });
}

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("select-folders-btn").addEventListener("click", addFolders);
    if (window.pywebview) {
        pywebview.api.get_dashboard_stats().then(stats => updateDashboard(stats));
    } else {
        updateDashboard({ original_size: 0, compressed_size: 0, file_list: [], file_types: {} });
    }
});

function confirmDeleteArchives() {
    if (confirm("Are you sure you want to delete ALL archives created by this app? This cannot be undone.")) {
        showNotification("Deleting all tracked archives...", true);
        const deleteStatus = document.getElementById('deleteStatus');
        if (deleteStatus) deleteStatus.innerText = "Deleting...";
        pywebview.api.delete_all_archives().then(response => {
            showNotification(response.message, response.success);
            if (deleteStatus) deleteStatus.innerText = response.message;
        });
    } else { showNotification("Deletion cancelled.", false); }
}

function setSchedule() {
    const timeValue = document.getElementById('scheduleTime').value;
    if (!timeValue) { showNotification('Please select a time for the schedule.', false); return; }
    pywebview.api.set_schedule(timeValue).then(response => {
        const scheduleStatus = document.getElementById('scheduleStatus');
        if (scheduleStatus) scheduleStatus.innerText = response.message;
        showNotification(response.message, true);
    });
}

function toggleGroupFileTypes() {
    const enabled = document.getElementById('groupFileTypes').checked;
    pywebview.api.toggle_group_by_type(enabled);
    if (enabled) showNotification('Grouping by file type enabled.', true);
}

function confirmDeleteInputFolders() {
    if (confirm("Are you sure you want to delete the currently selected input folders? This cannot be undone.")) {
        pywebview.api.delete_current_input_folders().then(response => {
            showNotification(response.message, response.success);
            if (response.remaining) {
                updateFolderList(response.remaining);
            } else if (response.success) {
                updateFolderList([]); 
            }
        });
    }
}

function stopCompression() {
    // Disable button to prevent double clicks
    const stopBtn = document.getElementById('stopBtn');
    if(stopBtn) stopBtn.innerText = "Stopping...";
    
    pywebview.api.stop_compression_process().then(response => {
        showNotification("Stopping compression process...", false);
    });
}

function openInExplorer(path) {
    if (!path) return;
    pywebview.api.open_file_explorer(path);
}