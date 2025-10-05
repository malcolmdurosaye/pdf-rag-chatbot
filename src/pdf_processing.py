# NOTE: Image-based PDFs rely on PDFium via pypdfium2 for OCR-ready rendering.

from io import BytesIO

from PyPDF2 import PdfReader
import pypdfium2 as pdfium
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import streamlit as st
from typing import List

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text from a PDF (text-based or image-based with enhanced OCR).
    """
    text = ""
    try:
        pdf_bytes = None
        reader_source = pdf_file

        if hasattr(pdf_file, "read"):
            pdf_bytes = pdf_file.read()
            if hasattr(pdf_file, "seek"):
                pdf_file.seek(0)
            reader_source = BytesIO(pdf_bytes)
        elif isinstance(pdf_file, (bytes, bytearray)):
            pdf_bytes = bytes(pdf_file)
            reader_source = BytesIO(pdf_bytes)

        pdf_reader = PdfReader(reader_source)
        image_cache = None
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text and len(page_text.strip()) > 10:
                text += page_text + "\n"
            else:
                if image_cache is None:
                    try:
                        if pdf_bytes is not None:
                            pdfium_source = BytesIO(pdf_bytes)
                        else:
                            pdfium_source = pdf_file
                            fspath = getattr(pdf_file, "__fspath__", None)
                            if callable(fspath):
                                pdfium_source = fspath()

                        pdf_document = pdfium.PdfDocument(pdfium_source)
                        try:
                            image_cache = []
                            for render_index in range(len(pdf_document)):
                                page = pdf_document[render_index]
                                page_image = page.render(scale=1.5).to_pil()
                                image_cache.append(page_image)
                        finally:
                            close_method = getattr(pdf_document, "close", None)
                            if callable(close_method):
                                close_method()
                    except pdfium.PdfiumError as ocr_error:
                        st.warning(
                            "Image extraction for OCR failed due to a rendering error. "
                            f"Details: {ocr_error}. Skipping this file."
                        )
                        return ""
                    except Exception as ocr_error:
                        st.warning(
                            "Image extraction for OCR failed. This PDF may be image-based and your system may not support it yet. "
                            f"Details: {ocr_error}. Skipping this file."
                        )
                        return ""

                if page_num >= len(image_cache):
                    st.warning(f"No rendered image available for page {page_num + 1}. Skipping OCR for this page.")
                    continue

                image = image_cache[page_num].copy()
                image = image.convert('L')
                image = ImageEnhance.Contrast(image).enhance(2.0)
                image = image.filter(ImageFilter.MedianFilter())
                try:
                    ocr_text = pytesseract.image_to_string(image, lang='eng+spa+fra')
                except pytesseract.TesseractNotFoundError:
                    st.warning(
                        "Tesseract OCR executable is not available. Install Tesseract, add it to PATH, and restart the app."
                    )
                    return ""

                if len(ocr_text.strip()) > 10:
                    text += ocr_text + "\n"
                else:
                    st.warning(f"Low-quality OCR output on page {page_num + 1}. Consider improving PDF quality.")
    except Exception as e:
        st.warning(f"PDF extraction failed: {str(e)}. Skipping this file.")
        return ""
    return text

def split_text(text: str, chunk_size: int = 500) -> List[str]:
    """
    Split text into chunks with quality validation.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1
        if current_length > chunk_size:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text.strip()) > 50:  # Ensure chunk has meaningful content
                chunks.append(chunk_text)
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if len(chunk_text.strip()) > 50:
            chunks.append(chunk_text)
    if not chunks:
        st.error("No valid chunks extracted. Check PDF quality or OCR settings.")
    return chunks
