import os
import uuid
import shutil
from datetime import datetime

import mysql.connector
from ultralytics import YOLO
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from fpdf import FPDF

# ── Load YOLO Model (once at startup) ────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
yolo_model = YOLO(MODEL_PATH)

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="AI-Based Brain Tumor Detection System")

# Secret key for session middleware (demo only)
app.add_middleware(SessionMiddleware, secret_key="brain-tumor-demo-secret-key-2026")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 template directory
templates = Jinja2Templates(directory="templates")

# Paths
UPLOAD_DIR = os.path.join("static", "uploads")
DETECTION_DIR = os.path.join("static", "detections")
REPORTS_DIR = "reports"

# Ensure directories exist
for d in [UPLOAD_DIR, DETECTION_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── MySQL Database Configuration ─────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "akc2sysit25$",
    "database": "answer_evaluation",
}


def get_db():
    """Get a new MySQL connection."""
    return mysql.connector.connect(**DB_CONFIG)


def init_db():
    """Create the patients table if it doesn't exist."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INT AUTO_INCREMENT PRIMARY KEY,
            patient_name VARCHAR(255) NOT NULL,
            age INT NOT NULL,
            gender VARCHAR(50),
            date VARCHAR(50),
            result VARCHAR(50),
            confidence FLOAT,
            image VARCHAR(500),
            detection_image VARCHAR(500)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()


# Initialize DB table on startup
init_db()

# Demo credentials
DEMO_USERNAME = "admin"
DEMO_PASSWORD = "admin123"


# ── Helper: read all patient records from DB ─────────────────────────────────
def read_patients():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM patients ORDER BY id ASC")
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    return records


# ── Helper: compute statistics from DB ───────────────────────────────────────
def get_stats():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT COUNT(*) AS total FROM patients")
    total = cursor.fetchone()["total"]
    cursor.execute("SELECT COUNT(*) AS cnt FROM patients WHERE result = %s", ("Positive",))
    positive = cursor.fetchone()["cnt"]
    negative = total - positive
    cursor.close()
    conn.close()
    return {"total": total, "positive": positive, "negative": negative}


# ── Helper: insert a patient record into DB ──────────────────────────────────
def save_patient(patient_name, age, gender, result, confidence, image_filename, detection_filename):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO patients (patient_name, age, gender, date, result, confidence, image, detection_image)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        (
            patient_name,
            age,
            gender,
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            result,
            confidence,
            image_filename,
            detection_filename,
        ),
    )
    conn.commit()
    last_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return last_id


# ── Helper: get a single patient by id ───────────────────────────────────────
def get_patient_by_id(patient_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM patients WHERE id = %s", (patient_id,))
    patient = cursor.fetchone()
    cursor.close()
    conn.close()
    return patient


# ── Helper: delete a patient record and its files ────────────────────────────
def delete_patient(patient_id):
    patient = get_patient_by_id(patient_id)
    if not patient:
        return False
    # Remove uploaded image
    img = patient.get("image", "")
    if img:
        path = os.path.join(UPLOAD_DIR, img)
        if os.path.exists(path):
            os.remove(path)
    # Remove detection image
    det = patient.get("detection_image", "")
    if det:
        path = os.path.join(DETECTION_DIR, det)
        if os.path.exists(path):
            os.remove(path)
    # Delete DB row
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM patients WHERE id = %s", (patient_id,))
    conn.commit()
    cursor.close()
    conn.close()
    return True


# ── Helper: run YOLO detection and save annotated image ──────────────────────
# Model classes:  0 = negative,  1 = positive
CLASS_NEGATIVE = 0
CLASS_POSITIVE = 1


def run_yolo_detection(source_path, detection_path):
    """
    Run the YOLO model on the MRI image.
    Returns (result, confidence) where result is 'Positive'/'Negative'.
    Saves the annotated image with bounding boxes to detection_path.
    """
    results = yolo_model(source_path)
    r = results[0]
    boxes = r.boxes

    if len(boxes) > 0:
        # Separate positive and negative detections
        positive_indices = [
            i for i in range(len(boxes)) if int(boxes.cls[i]) == CLASS_POSITIVE
        ]

        if positive_indices:
            # Tumor detected — pick the highest-confidence positive box
            best_idx = max(positive_indices, key=lambda i: float(boxes.conf[i]))
            confidence = float(boxes.conf[best_idx])
            result = "Positive"
        else:
            # Only negative detections — pick the highest-confidence one
            best_idx = boxes.conf.argmax()
            confidence = float(boxes.conf[best_idx])
            result = "Negative"
    else:
        # No detections at all
        confidence = 0.0
        result = "Negative"

    # Save annotated image with bounding boxes drawn by YOLO
    r.save(detection_path)

    return result, round(confidence, 4)


# ── Helper: draw a horizontal separator line ─────────────────────────────────
def _pdf_separator(pdf, y=None, color=(0, 180, 216), width=0.6):
    pdf.set_draw_color(*color)
    pdf.set_line_width(width)
    y = y or pdf.get_y()
    pdf.line(15, y, 195, y)


# ── Helper: draw a section heading ───────────────────────────────────────────
def _pdf_section_heading(pdf, title, icon_char=""):
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(15, 76, 129)
    text = f"{icon_char}  {title}" if icon_char else title
    pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
    _pdf_separator(pdf, color=(220, 230, 240), width=0.3)
    pdf.ln(4)


# ── Helper: generate professional PDF report ─────────────────────────────────
def generate_pdf_report(patient, patient_id):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=25)

    # ── HEADER BAND ──────────────────────────────────────────────────────
    # Dark blue header background
    pdf.set_fill_color(15, 76, 129)
    pdf.rect(0, 0, 210, 38, "F")

    # Accent stripe
    pdf.set_fill_color(0, 180, 216)
    pdf.rect(0, 38, 210, 2, "F")

    # System name
    pdf.set_y(8)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "Brain-Tumor-AI", new_x="LMARGIN", new_y="NEXT", align="C")

    # Subtitle
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(200, 220, 240)
    pdf.cell(0, 7, "AI-Based Brain Tumor Detection System", new_x="LMARGIN", new_y="NEXT", align="C")

    # Report ID & date right-aligned in header area
    pdf.set_y(8)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(180, 200, 220)
    report_date = datetime.now().strftime("%B %d, %Y  %H:%M")
    pdf.cell(0, 5, f"Report #{patient_id + 1}  |  {report_date}", new_x="LMARGIN", new_y="NEXT", align="R")

    pdf.set_y(46)

    # ── REPORT TITLE ─────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, "Diagnostic Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(2)
    _pdf_separator(pdf)
    pdf.ln(4)

    # ── PATIENT INFORMATION ──────────────────────────────────────────────
    _pdf_section_heading(pdf, "Patient Information")

    # Table-style patient info with alternating row shading
    info_fields = [
        ("Patient Name", patient.get("patient_name", "N/A")),
        ("Age", patient.get("age", "N/A")),
        ("Gender", patient.get("gender", "N/A")),
        ("Date of Diagnosis", patient.get("date", "N/A")),
    ]

    col_label_w = 60
    col_value_w = 120
    row_h = 9

    for i, (label, value) in enumerate(info_fields):
        if i % 2 == 0:
            pdf.set_fill_color(240, 245, 250)
        else:
            pdf.set_fill_color(255, 255, 255)

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(80, 90, 100)
        pdf.cell(col_label_w, row_h, f"  {label}", fill=True, new_x="END")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(col_value_w, row_h, f"  {value}", fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)

    # ── DIAGNOSIS RESULT ─────────────────────────────────────────────────
    _pdf_section_heading(pdf, "Diagnosis Result")

    is_positive = patient.get("result") == "Positive"
    diagnosis_text = "Tumor Detected" if is_positive else "No Tumor Detected"
    confidence_val = patient.get("confidence", "N/A")

    # Result box with colored background
    box_y = pdf.get_y()
    if is_positive:
        pdf.set_fill_color(253, 237, 236)
        border_color = (231, 76, 60)
        text_color = (192, 57, 43)
        status_icon = "!"
    else:
        pdf.set_fill_color(234, 250, 241)
        border_color = (46, 204, 113)
        text_color = (39, 174, 96)
        status_icon = "+"

    # Draw result box
    pdf.set_draw_color(*border_color)
    pdf.set_line_width(0.8)
    pdf.rect(15, box_y, 180, 28, "DF")

    # Diagnosis text
    pdf.set_xy(20, box_y + 3)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(*text_color)
    pdf.cell(170, 10, f"[{status_icon}]  {diagnosis_text}", new_x="LMARGIN", new_y="NEXT")

    # Confidence score inside the box
    pdf.set_x(20)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 90, 100)
    try:
        conf_pct = f"{float(confidence_val) * 100:.1f}%"
    except (ValueError, TypeError):
        conf_pct = str(confidence_val)
    pdf.cell(170, 9, f"Confidence Score:  {conf_pct}", new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(box_y + 32)

    # ── Confidence bar graphic ───────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = 15, pdf.get_y(), 180, 6
    # Background
    pdf.set_fill_color(230, 230, 230)
    pdf.rect(bar_x, bar_y, bar_w, bar_h, "F")
    # Filled portion
    try:
        fill_w = bar_w * float(confidence_val)
    except (ValueError, TypeError):
        fill_w = 0
    if is_positive:
        pdf.set_fill_color(231, 76, 60)
    else:
        pdf.set_fill_color(46, 204, 113)
    if fill_w > 0:
        pdf.rect(bar_x, bar_y, fill_w, bar_h, "F")

    pdf.set_y(bar_y + bar_h + 6)

    # ── MRI SCAN IMAGE ───────────────────────────────────────────────────
    detection_img = patient.get("detection_image", "")
    det_path = os.path.join(DETECTION_DIR, detection_img) if detection_img else ""
    if det_path and os.path.exists(det_path):
        _pdf_section_heading(pdf, "MRI Scan - AI Detection Output")

        # Center the image
        img_w = 120
        img_x = (210 - img_w) / 2
        pdf.image(det_path, x=img_x, w=img_w)
        pdf.ln(3)

        # Caption
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(130, 140, 150)
        pdf.cell(0, 6, "AI Detection Result from MRI Scan", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)

    # ── SUMMARY ──────────────────────────────────────────────────────────
    _pdf_section_heading(pdf, "Summary")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 70, 80)
    summary_text = (
        "This report was automatically generated by the Brain-Tumor-AI system using a "
        "deep learning model (YOLOv8) trained to analyze MRI brain scan images for the "
        "presence of tumors. The model processes uploaded MRI scans and identifies regions "
        "of interest, providing a confidence-scored diagnosis to assist medical professionals."
    )
    pdf.multi_cell(0, 6, summary_text)
    pdf.ln(6)

    # ── FOOTER BAND ──────────────────────────────────────────────────────
    # Ensure we're at least near the bottom
    footer_y = max(pdf.get_y() + 10, 260)
    if footer_y > 275:
        pdf.add_page()
        footer_y = 260

    _pdf_separator(pdf, y=footer_y, color=(200, 210, 220))

    pdf.set_y(footer_y + 3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(15, 76, 129)
    pdf.cell(0, 5, "Generated by Brain-Tumor-AI  |  AI-Assisted Diagnostic Tool", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(150, 155, 165)
    disclaimer = (
        "Disclaimer: This system is intended for research and educational purposes only "
        "and should not replace professional medical diagnosis."
    )
    pdf.multi_cell(0, 4, disclaimer, align="C")

    # ── Save PDF ─────────────────────────────────────────────────────────
    pdf_filename = f"report_{patient_id}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
    pdf.output(pdf_path)
    return pdf_path


# ── Routes ───────────────────────────────────────────────────────────────────

# 1. Home Page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


# 2. Login Page (GET)
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


# 2. Login Page (POST) – simple demo authentication
@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == DEMO_USERNAME and password == DEMO_PASSWORD:
        request.session["user"] = username
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse(
        "login.html", {"request": request, "error": "Invalid username or password"}
    )


# Logout
@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


# 3. Dashboard — now passes live stats
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=303)
    stats = get_stats()
    return templates.TemplateResponse("dashboard.html", {"request": request, "stats": stats})


# 4. MRI Upload Page
@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("upload.html", {"request": request})


# 5. Prediction Route — now saves detection image
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    patient_name: str = Form(...),
    age: int = Form(...),
    gender: str = Form("Not specified"),
    mri_image: UploadFile = File(...),
):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=303)

    # Save uploaded image with a unique filename
    ext = os.path.splitext(mri_image.filename)[1] or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(save_path, "wb") as buf:
        shutil.copyfileobj(mri_image.file, buf)

    # ── Run real YOLO detection ──────────────────────────────────────────
    detection_name = f"det_{unique_name}"
    detection_path = os.path.join(DETECTION_DIR, detection_name)
    result, confidence = run_yolo_detection(save_path, detection_path)

    # Save record to DB
    patient_id = save_patient(patient_name, age, gender, result, confidence, unique_name, detection_name)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "patient_name": patient_name,
        "age": age,
        "gender": gender,
        "result": result,
        "confidence": confidence,
        "image_path": f"/static/uploads/{unique_name}",
        "detection_path": f"/static/detections/{detection_name}",
        "patient_id": patient_id,
    })


# 6. Reports Page
@app.get("/reports", response_class=HTMLResponse)
async def reports(request: Request):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=303)
    patients = read_patients()
    return templates.TemplateResponse("reports.html", {"request": request, "patients": patients})


# ── Download PDF report for a patient ────────────────────────────────────────
@app.get("/download-report/{patient_id}")
async def download_pdf_report(request: Request, patient_id: int):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=303)

    patient = get_patient_by_id(patient_id)
    if not patient:
        return RedirectResponse(url="/reports", status_code=303)

    pdf_path = generate_pdf_report(patient, patient_id)

    safe_name = patient.get("patient_name", "patient").replace(" ", "_")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"report_{safe_name}.pdf",
    )


# ── Delete a patient record ──────────────────────────────────────────────────
@app.get("/delete-patient/{patient_id}")
async def delete_patient_route(request: Request, patient_id: int):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=303)
    delete_patient(patient_id)
    return RedirectResponse(url="/reports", status_code=303)
