import os
import uuid
from flask import Flask, request, render_template, redirect, url_for, Response
from openai import OpenAI
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import time
from datetime import date
import openai  # 예외 클래스들 사용용




app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB

# --- 요청 제한 설정 ---
RATE_LIMIT_PER_DAY = 20  # IP 당 하루 최대 20회 (원하면 숫자 조절)
_rate_limit_store = {}   # { ip: {"day": "2025-11-18", "count": n} }

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def detect_lang():
    """
    1순위: ?lang=ko / ?lang=en 쿼리 파라미터
    2순위: 브라우저 Accept-Language 헤더
    기본값: ko
    """
    lang = request.args.get("lang")
    if lang in ("ko", "en"):
        return lang

    header = (request.headers.get("Accept-Language") or "").lower()

    # 가장 단순한 판단: en으로 시작하거나 en-* 이 포함되면 영어
    if header.startswith("en") or "en-" in header:
        return "en"

    return "ko"


def check_rate_limit(ip: str):
    """
    IP 기준 하루 요청 횟수 제한.
    return (is_limited: bool, remaining: int)
    """
    today = date.today().isoformat()
    data = _rate_limit_store.get(ip)

    if not data or data.get("day") != today:
        # 첫 요청이거나 날짜가 바뀐 경우 초기화
        _rate_limit_store[ip] = {"day": today, "count": 1}
        return False, RATE_LIMIT_PER_DAY - 1

    if data["count"] >= RATE_LIMIT_PER_DAY:
        return True, 0

    data["count"] += 1
    remaining = RATE_LIMIT_PER_DAY - data["count"]
    return False, remaining


def extract_text_from_pdf(path: str) -> str:
    # 1차: PyMuPDF로 텍스트 추출
    doc = fitz.open(path)
    texts = []
    for page in doc:
        text = page.get_text("text") or ""
        texts.append(text)

    text_all = "\n".join(texts).strip()

    # 텍스트가 거의 없으면 OCR로 재시도
    # (길이 기준은 필요에 따라 조절)
    if len(text_all) < 50:
        ocr_text = ocr_pdf_with_tesseract(path)
        ocr_text = (ocr_text or "").strip()

        # OCR로 얻은 텍스트가 있으면 그걸 사용
        if len(ocr_text) >= 10:
            return ocr_text

    return text_all

def ocr_pdf_with_tesseract(path: str, max_pages: int = 5) -> str:
    """
    PyMuPDF가 텍스트를 못 뽑을 때, 이미지로 렌더링 후 Tesseract로 OCR.
    max_pages: 너무 무거워지지 않게 앞쪽 몇 페이지만 OCR.
    """
    doc = fitz.open(path)
    texts = []

    # 페이지 수 제한
    num_pages = min(len(doc), max_pages)

    for page_index in range(num_pages):
        page = doc[page_index]

        # 페이지를 이미지로 렌더링 (dpi는 200~300 정도가 적당)
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 한국어 + 영어 OCR
        ocr_text = pytesseract.image_to_string(img, lang="kor+eng")
        texts.append(ocr_text)

    return "\n".join(texts)


def split_text_into_chunks(text: str, max_chars: int = 5000, overlap: int = 500):
    """
    긴 텍스트를 여러 청크로 나누는 함수.
    max_chars: 청크 최대 길이
    overlap: 청크 사이 겹치는 길이(맥락 보존용)
    """
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)

        # 다음 청크 시작 위치 (겹치게 해서 맥락 유지)
        if end >= length:
            break
        start = end - overlap

    return chunks


def summarize_text(text: str, level: str = "medium", lang: str = "ko") -> str:
    """
    긴 PDF도 처리할 수 있게:
    - 짧은 텍스트: 한 번에 요약
    - 긴 텍스트: 여러 청크로 나눠 부분 요약 → 최종 취합 요약
    """

    # 언어별 안내문 생성
    if lang == "en":
        if level == "short":
            instruction = "Summarize the following PDF in 3–4 concise sentences in English."
        elif level == "long":
            instruction = "Summarize the following PDF in about one A4 page in English, with headings and bullet points."
        else:
            instruction = "Summarize the following PDF into 3–5 key bullet points in English."
    else:
        if level == "short":
            instruction = "다음 PDF를 3~4개의 간결한 문장으로 요약해 주세요."
        elif level == "long":
            instruction = "다음 PDF를 한 페이지 분량으로 제목과 불릿 포인트를 포함해서 자세히 요약해 주세요."
        else:
            instruction = "다음 PDF를 3~5개의 핵심 불릿 포인트로 요약해 주세요."


    # 1) 짧은 텍스트는 기존 방식 그대로 (단, 10,000자 자르기 제거)
    if len(text) <= 9000:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a professional document summarizer."},
                {"role": "user", "content": instruction + "\n\n" + text}
            ],
            timeout=15
        )
        return completion.choices[0].message.content.strip()

    # 2) 긴 텍스트는 청크로 나눠서 부분 요약 → 최종 요약
    chunks = split_text_into_chunks(text, max_chars=6000, overlap=500)

    # 안전 장치: 너무 많은 청크면 상한 걸기 (비정상 초대형 문서 방지)
    max_chunks = 6
    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]

    partial_summaries = []

    # 2-1) 각 청크별 부분 요약
    for idx, chunk in enumerate(chunks, start=1):
        language_label = "English" if lang == "en" else "Korean"

        chunk_instruction = (
            f"You are summarizing part {idx} of a long PDF document. "
            f"Focus on the key ideas and ignore minor details. "
            f"Write the summary in {language_label}.\n\n"
            f"{instruction}"
        )


        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a professional document summarizer."},
                {"role": "user", "content": chunk_instruction + "\n\n" + chunk}
            ],
            timeout=15
        )
        partial = completion.choices[0].message.content.strip()
        partial_summaries.append(f"[Part {idx}]\n{partial}")

    # 2-2) 부분 요약들을 한 번 더 요약해서 최종 버전 생성
    combined = "\n\n".join(partial_summaries)

    final_instruction = (
        "Below are summaries of each part of a long PDF document.\n"
        f"Combine them into ONE final summary in {language_label}.\n"
    )


    if level == "short":
        final_instruction += "Make it 3–4 sentences."
    elif level == "long":
        final_instruction += "Make it about one A4 page with clear structure (headings and bullet points)."
    else:
        final_instruction += "Make it 3–7 key bullet points."

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a professional document summarizer."},
            {"role": "user", "content": final_instruction + "\n\n" + combined}
        ],
        timeout=15
    )

    return completion.choices[0].message.content.strip()



@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    error = None

    # 언어 감지
    lang = detect_lang()
    template_name = "index_en.html" if lang == "en" else "index.html"

    if request.method == "POST":
        # --- IP 기준 요청 제한 체크 ---
        forwarded_for = request.headers.get("X-Forwarded-For", "")
        if forwarded_for:
            ip = forwarded_for.split(",")[0].strip()
        else:
            ip = request.remote_addr or "unknown"

        limited, remaining = check_rate_limit(ip)
        if limited:
            error = f"오늘 무료 사용 횟수({RATE_LIMIT_PER_DAY}회)를 모두 사용했습니다. 내일 다시 이용해 주세요."
            return render_template(template_name, summary=None, error=error)

        file = request.files.get("pdf")
        level = request.form.get("level", "medium")

        if not file or file.filename == "":
            # 언어별 에러 메시지 나누고 싶으면 여기서 분기
            if lang == "en":
                error = "No PDF file was uploaded."
            else:
                error = "PDF 파일이 업로드되지 않았습니다."
            return render_template(template_name, summary=None, error=error)

        original_name = file.filename or "upload.pdf"
        _, ext = os.path.splitext(original_name)
        ext = ext.lower() or ".pdf"
        safe_name = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

        try:
            file.save(filepath)
            text = extract_text_from_pdf(filepath)

            if not text.strip():
                if lang == "en":
                    error = "Could not extract any text from the PDF."
                else:
                    error = "PDF에서 텍스트를 추출할 수 없습니다."
                return render_template(template_name, summary=None, error=error)

            try:
                summary = summarize_text(text, level=level, lang=lang)
            except openai.APITimeoutError:
                error = "AI 요약 요청 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."
                summary = None
            except openai.RateLimitError:
                error = "AI 요약 요청이 너무 많습니다. 잠시 후 다시 시도해 주세요."
                summary = None
            except openai.APIError:
                error = "AI 요약 처리 중 서버 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
                summary = None
            except Exception:
                error = "AI 요약 처리 중 알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
                summary = None

            return render_template(template_name, summary=summary, error=error)

        except Exception:
            if lang == "en":
                error = "An error occurred while processing the document. Please try again."
            else:
                error = "문서 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
            return render_template(template_name, summary=None, error=error)

        finally:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                pass

    # GET: 처음 접속 화면
    return render_template(template_name, summary=None, error=None)


@app.route("/terms")
def terms_ko():
    return render_template("terms.html")

@app.route("/privacy")
def privacy_ko():
    return render_template("privacy.html")

@app.route("/terms-en")
def terms_en():
    return render_template("terms_en.html")

@app.route("/privacy-en")
def privacy_en():
    return render_template("privacy_en.html")

@app.route("/pdf-summary")
def pdf_summary_page():
    return render_template("pdf_summary.html")

@app.route("/ocr")
def ocr_page():
    return render_template("ocr.html")

@app.route("/extract-text")
def extract_text_page():
    return render_template("extract_text.html")

@app.route("/ai-pdf-reader")
def ai_pdf_reader_page():
    return render_template("ai_pdf_reader.html")

@app.route("/pdf-summary-kr")
def pdf_summary_kr():
    return render_template("pdf_summary_kr.html")

@app.route("/ocr-kr")
def ocr_kr():
    return render_template("ocr_kr.html")

@app.route("/extract-text-kr")
def extract_text_kr():
    return render_template("extract_text_kr.html")

@app.route("/ai-pdf-reader-kr")
def ai_pdf_reader_kr():
    return render_template("ai_pdf_reader_kr.html")

@app.route("/sitemap.xml")
def sitemap():
    pages = []

    # 기본 페이지들
    static_urls = [
        ("/", "2025-01-01"),
        ("/privacy", "2025-01-01"),
        ("/terms", "2025-01-01"),
        ("/privacy-en", "2025-01-01"),
        ("/terms-en", "2025-01-01"),
        ("/pdf-summary", "2025-01-01"),
        ("/ocr", "2025-01-01"),
        ("/extract-text", "2025-01-01"),
        ("/ai-pdf-reader", "2025-01-01"),
        ("/pdf-summary-kr", "2025-01-01"),
        ("/ocr-kr", "2025-01-01"),
        ("/extract-text-kr", "2025-01-01"),
        ("/ai-pdf-reader-kr", "2025-01-01"),
    ]

    domain = "https://quickpdfsum.com"

    for url, lastmod in static_urls:
        pages.append(f"""
        <url>
            <loc>{domain}{url}</loc>
            <lastmod>{lastmod}</lastmod>
            <changefreq>monthly</changefreq>
            <priority>0.8</priority>
        </url>
        """)

    sitemap_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        {''.join(pages)}
    </urlset>
    """

    return Response(sitemap_xml, mimetype="application/xml")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
