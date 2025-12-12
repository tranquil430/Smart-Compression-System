# Smart Compression System (SCS)

SCS is an intelligent file management tool that automates file archiving using Machine Learning. It utilizes a hybrid model approach to predict which files should be compressed, selects the optimal compression algorithm (7z, RAR, ZIP) based on file entropy, and learns from user interactions over time.

## Key Features

* **Intelligent Filtering:** Uses a Random Forest classifier to identify archival candidates based on access time, modification dates, and file types.
* **Adaptive Optimization:** An XGBoost model analyzes file entropy and size to select the most efficient compression algorithm (e.g., choosing 7z for high-compression targets vs. ZIP for speed).
* **User-in-the-Loop Learning:** The system adapts to manual overrides, retraining its decision engine based on your specific usage patterns.
* **Automated Security:** Integrated virus scanning using Windows Defender API before processing.
* **Dashboard UI:** A clean, local web-based interface built with `pywebview` for managing tasks, scheduling, and visualizing storage savings.

## Architecture

The system relies on three distinct modeling stages:
1.  **Candidate Selection:** A baseline Random Forest model filters files based on metadata policies.
2.  **Algorithm Selector:** An XGBoost classifier predicts the optimal compression format.
3.  **Adaptive Feedback:** A secondary classifier retrains on runtime logs to refine future suggestions.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/smart-compression-system.git](https://github.com/yourusername/smart-compression-system.git)
    cd smart-compression-system
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **External Requirements:**
    * **7-Zip** (added to system PATH)
    * **WinRAR** (optional, for RAR support)

## Usage

Run the main application entry point:

```bash
python src/app.py