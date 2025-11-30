# ECG Stress Detection - Setup Instructions

## Prerequisites
- Python 3.8 or higher
- Node.js 18 or higher
- pip and npm

## Backend Setup

### 1. Navigate to Backend Directory
```bash
cd backend
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Place Model Files
Ensure your trained models are in the `models/` directory:
```
backend/
└── models/
    ├── random_forest.pkl
    └── svc.pkl
```

### 6. Run FastAPI Server
```bash
uvicorn main:app --reload
```

The backend will be available at `http://localhost:8000`

---

## Frontend Setup

### 1. Navigate to Frontend Directory
```bash
cd frontend
```

### 2. Install Dependencies
```bash
npm i
```

### 3. Run Development Server
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

---

## Usage

1. Open `http://localhost:3000` in your browser
2. Select a model (Random Forest or SVC)
3. Upload a CSV file containing ECG data
4. Click "Analyze ECG Data"
5. View the prediction results and extracted features

---

## CSV File Format

Your CSV file should have an "ECG" column with signal data:

```csv
ECG,Label
-0.1781616211,0
-0.1802215576,0
-0.1833343506,0
```

---

## Stopping the Servers

**Backend:**
Press `Ctrl+C` in the terminal running the FastAPI server

**Frontend:**
Press `Ctrl+C` in the terminal running the Next.js server