# app.py - This file contains the Python code for your backend server.

from flask import Flask, request, send_file, jsonify, render_template
from flask_cors import CORS # Used to allow your web page (frontend) to talk to this backend
import asyncio # Used to run asynchronous tasks like Playwright
from playwright.async_api import async_playwright # Powerful tool to generate PDF/PNG from HTML
import os
import io # Used to handle file data in memory
import requests # Used to make HTTP requests to the Gemini API
import json # Used to parse JSON responses from the API
from dotenv import load_dotenv # NEW: Import load_dotenv

# NEW: Load environment variables from .env file at the very beginning
load_dotenv()

# Initialize the Flask web application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS). This is crucial!
# It tells your browser that it's okay for your web page (which runs from a file or different address)
# to send requests to this Python server. Without it, your browser would block the communication.
CORS(app)

# Helper function to get font family CSS string
def get_font_family_css(font_name):
    if font_name == 'Arial':
        return 'Arial, sans-serif'
    elif font_name == 'Times New Roman':
        return '"Times New Roman", Times, serif'
    elif font_name == 'Georgia':
        return 'Georgia, serif'
    elif font_name == 'Open Sans':
        return '"Open Sans", sans-serif'
    else: # Default to Inter
        return '"Inter", sans-serif'

# Async function to generate PDF or PNG using Playwright
async def generate_file_with_playwright(html_content, file_type):
    # It's crucial to launch Playwright within an asyncio context.
    # The `async with` ensures the browser is properly closed.
    async with async_playwright() as p:
        # Launch a headless Chromium browser (runs in the background without a visible window)
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Set the HTML content of the page.
        # `wait_until='networkidle'` tries to wait until network activity calms down,
        # ensuring external resources like CDN CSS/fonts are loaded.
        await page.set_content(html_content, wait_until='networkidle')

        # Increase wait_for_timeout slightly to give more time for content rendering
        await page.wait_for_timeout(1000) # Increased to 1000 milliseconds (1 second)

        # --- FIX FOR PDF BACKGROUND AND CONTENT CLIPPING ---
        if file_type == 'pdf':
            # Explicitly set background to white using JavaScript before generating PDF
            await page.evaluate('''
                document.body.style.backgroundColor = '#ffffff';
                document.documentElement.style.backgroundColor = '#ffffff';
            ''')
            # For extra safety, ensure no margin/padding at the very root
            await page.evaluate('''
                document.body.style.margin = '0';
                document.body.style.padding = '0';
                document.documentElement.style.margin = '0';
                document.documentElement.style.padding = '0';
            ''')
            # Emulate print media. This can significantly affect how content flows and breaks across pages.
            await page.emulate_media(media='print')
        # --- END FIX ---

        if file_type == 'pdf':
            # Generate PDF.
            # `format='Letter'` sets standard paper size.
            # `print_background=True` ensures background colors/images are included.
            # Margins are set to give some breathing room.
            # `full_page=True` is not needed here as page breaks are handled by PDF format.
            pdf_buffer = await page.pdf(
                format='Letter', # Standard paper size (8.5 x 11 inches)
                print_background=True, # Include background colors/images
                margin={'top': '0.5in', 'right': '0.5in', 'bottom': '0.5in', 'left': '0.5in'}
            )
            await browser.close() # Close the browser instance
            return pdf_buffer
        elif file_type == 'png':
            # Generate PNG (screenshot).
            # Setting a viewport size helps standardize the screenshot dimensions.
            # `full_page=True` captures the entire scrollable height of the rendered content.
            await page.set_viewport_size({"width": 800, "height": 1100}) # Approximate letter size aspect ratio
            # For PNG, explicitly set screen media type
            await page.emulate_media(media='screen')

            png_buffer = await page.screenshot(full_page=True)
            await browser.close() # Close the browser instance
            return png_buffer
        else:
            await browser.close()
            raise ValueError("Unsupported file type for Playwright generation. Must be 'pdf' or 'png'.")

@app.route('/generate-pdf', methods=['POST'])
async def generate_pdf():
    # Get JSON data sent from the frontend
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    resume_data = data.get('resumeData')
    selected_font = data.get('selectedFont')
    selected_template = data.get('selectedTemplate')
    sections_order = data.get('sectionsOrder')
    base_font_size = data.get('baseFontSize') # NEW: Get baseFontSize
    bold_option = data.get('boldOption')     # NEW: Get boldOption

    if not resume_data or not sections_order:
        return jsonify({"error": "Missing resumeData or sections order in payload"}), 400

    # Render HTML using Jinja2 template.
    # The `resume_template.html` file must be in a 'templates' subfolder.
    rendered_html = render_template(
        'resume_template.html', # Path relative to the 'templates' folder
        resume_data=resume_data,
        selected_template=selected_template,
        font_family_css=get_font_family_css(selected_font),
        sections_order=sections_order,
        base_font_size=base_font_size, # NEW: Pass to template
        bold_option=bold_option      # NEW: Pass to template
    )

    try:
        # Call the async Playwright function to get PDF bytes
        pdf_bytes = await generate_file_with_playwright(rendered_html, 'pdf')
        # Send the generated PDF file back to the client
        return send_file(
            io.BytesIO(pdf_bytes), # Wrap bytes in a BytesIO object for send_file
            mimetype='application/pdf', # Set correct MIME type for PDF
            as_attachment=True, # Forces browser to download instead of displaying
            download_name=f"{resume_data['personalInfo']['name'].replace(' ', '-') or 'Resume'}.pdf" # Suggested filename
        )
    except Exception as e:
        app.logger.error(f"Error generating PDF: {e}", exc_info=True) # Log full traceback
        return jsonify({"error": "Failed to generate PDF", "details": str(e)}), 500

@app.route('/generate-png', methods=['POST'])
async def generate_png():
    # Get JSON data sent from the frontend
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    resume_data = data.get('resumeData')
    selected_font = data.get('selectedFont')
    selected_template = data.get('selectedTemplate')
    sections_order = data.get('sectionsOrder')
    base_font_size = data.get('baseFontSize') # NEW
    bold_option = data.get('boldOption')     # NEW

    if not resume_data or not sections_order:
        return jsonify({"error": "Missing resumeData or sections order in payload"}), 400

    # Render HTML using Jinja2 template
    rendered_html = render_template(
        'resume_template.html',
        resume_data=resume_data,
        selected_template=selected_template,
        font_family_css=get_font_family_css(selected_font),
        sections_order=sections_order,
        base_font_size=base_font_size, # NEW
        bold_option=bold_option      # NEW
    )

    try:
        # Call the async Playwright function to get PNG bytes
        png_bytes = await generate_file_with_playwright(rendered_html, 'png')
        # Send the generated PNG file back to the client
        return send_file(
            io.BytesIO(png_bytes), # Wrap bytes in a BytesIO object
            mimetype='image/png', # Set correct MIME type for PNG
            as_attachment=True,
            download_name=f"{resume_data['personalInfo']['name'].replace(' ', '-') or 'Resume'}.png"
        )
    except Exception as e:
        app.logger.error(f"Error generating PNG: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate PNG", "details": str(e)}), 500

@app.route('/generate-doc', methods=['POST'])
def generate_doc():
    # Get JSON data sent from the frontend
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    resume_data = data.get('resumeData')
    selected_font = data.get('selectedFont')
    selected_template = data.get('selectedTemplate')
    sections_order = data.get('sectionsOrder')
    base_font_size = data.get('baseFontSize') # NEW
    bold_option = data.get('boldOption')     # NEW

    if not resume_data or not sections_order:
        return jsonify({"error": "Missing resumeData or sections order in payload"}), 400

    # For .doc, we render the HTML directly and send it with application/msword mimetype.
    # This relies on Microsoft Word's ability to interpret HTML.
    # As discussed, for a TRUE .docx, you'd need a more advanced library or external tool
    # like python-docx or pandoc on the server. This is a pragmatic workaround.
    rendered_html_for_doc = render_template(
        'resume_template.html',
        resume_data=resume_data,
        selected_template=selected_template,
        font_family_css=get_font_family_css(selected_font),
        sections_order=sections_order,
        base_font_size=base_font_size, # NEW
        bold_option=bold_option      # NEW
    )

    doc_bytes = rendered_html_for_doc.encode('utf-8') # Encode HTML string to bytes

    # Send the HTML as a .doc file
    return send_file(
        io.BytesIO(doc_bytes),
        mimetype='application/msword', # Tells the browser/Word to treat it as a Word document
        as_attachment=True,
        download_name=f"{resume_data['personalInfo']['name'].replace(' ', '-') or 'Resume'}.doc"
    )

@app.route('/check-ats', methods=['POST'])
async def check_ats():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    resume_text = data.get('resumeText')
    job_description = data.get('jobDescription')

    if not resume_text or not job_description:
        return jsonify({"error": "Missing resume text or job description"}), 400

    # Retrieve the API key from an environment variable using python-dotenv.
    # It will first look for GEMINI_API_KEY in the actual environment,
    # then in a .env file if it exists.
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key: # Check if the key is empty or not found
        app.logger.error("Gemini API Key is missing. Please set the 'GEMINI_API_KEY' environment variable.")
        return jsonify({
            "error": "Gemini API Key is missing.",
            "details": "Please set the 'GEMINI_API_KEY' environment variable. If testing locally, ensure it's in your .env file or set in your shell before running app.py."
        }), 500


    # Construct the prompt for the Gemini LLM
    prompt = f"""
    You are an Applicant Tracking System (ATS) expert. Your task is to analyze a given resume against a job description.
    Provide a compatibility score (0-100%) and actionable suggestions for improvement to better match the job description.
    Also, list keywords from the job description that are present in the resume and those that are missing.

    Resume:
    ---
    {resume_text}
    ---

    Job Description:
    ---
    {job_description}
    ---

    Provide the output in a JSON format with the following structure:
    {{
        "score": <percentage_score_as_integer>,
        "keywords": {{
            "present": ["keyword1", "keyword2"],
            "missing": ["keyword3", "keyword4"]
        }},
        "suggestions": ["suggestion1", "suggestion2"]
    }}
    """

    try:
        # Call the Gemini API
        # Using gemini-2.0-flash for text generation
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "score": {"type": "INTEGER"},
                        "keywords": {
                            "type": "OBJECT",
                            "properties": {
                                "present": {"type": "ARRAY", "items": {"type": "STRING"}},
                                "missing": {"type": "ARRAY", "items": {"type": "STRING"}}
                            }
                        },
                        "suggestions": {"type": "ARRAY", "items": {"type": "STRING"}}
                    },
                    "required": ["score", "keywords", "suggestions"]
                }
            }
        }

        # Make the request to the Gemini API
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        # Parse the JSON response
        llm_result = response.json()
        
        # Access the generated content
        if llm_result.get('candidates') and len(llm_result['candidates']) > 0:
            content_part = llm_result['candidates'][0].get('content')
            if content_part and content_part.get('parts') and len(content_part['parts']) > 0:
                generated_text = content_part['parts'][0].get('text')
                if generated_text:
                    # Parse the string as JSON
                    ats_analysis = json.loads(generated_text)
                    return jsonify(ats_analysis)
                else:
                    return jsonify({"error": "LLM response content is empty"}), 500
            else:
                return jsonify({"error": "LLM response parts are missing"}), 500
        else:
            return jsonify({"error": "LLM candidates are missing"}), 500

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error calling Gemini API: {e}", exc_info=True)
        return jsonify({"error": "Failed to communicate with AI service", "details": str(e)}), 500
    except json.JSONDecodeError as e:
        app.logger.error(f"Error parsing LLM response JSON: {e}", exc_info=True)
        return jsonify({"error": "Failed to parse AI response", "details": str(e)}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error during ATS check: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500


if __name__ == '__main__':
    # When running 'python app.py' directly, this block executes.
    # Flask's development server can handle async routes, making it easy to test.
    # For production deployments, you'd use an ASGI server like Gunicorn + Uvicorn.
    app.run(debug=True, port=5000) # Run on port 5000, enable debug mode for development
