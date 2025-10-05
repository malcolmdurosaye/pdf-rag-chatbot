from PyPDF2 import PdfReader
from pdf2image import convert_from_path
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
        pdf_reader = PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text and len(page_text.strip()) > 10:
                text += page_text + "\n"
            else:
                pdf_file.seek(0)
                images = convert_from_path(pdf_file.name) if hasattr(pdf_file, 'name') else convert_from_path(pdf_file)
                for image in images:
                    # Preprocess image: grayscale, enhance contrast, reduce noise
                    image = image.convert('L')  # Grayscale
                    image = ImageEnhance.Contrast(image).enhance(2.0)  # Increase contrast
                    image = image.filter(ImageFilter.MedianFilter())  # Reduce noise
                    # Use multilingual OCR (eng+spa+fra for common languages; extend as needed)
                    ocr_text = pytesseract.image_to_string(image, lang='eng+spa+fra')
                    if len(ocr_text.strip()) > 10:  # Validate OCR output
                        text += ocr_text + "\n"
                    else:
                        st.warning(f"Low-quality OCR output on page {page_num + 1}. Consider improving PDF quality.")
    except Exception as e:
        st.error(f"Error extracting PDF: {str(e)}")
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