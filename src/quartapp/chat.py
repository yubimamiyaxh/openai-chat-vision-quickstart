# This is a new chat.py file that uses JSON Schema for patient data

import json
import os

# import additional packages
from quart import Blueprint, request, jsonify
import fitz  # PyMuPDF
from PIL import Image
from PIL import ImageOps
from io import BytesIO
import base64
import re
import asyncio


import azure.identity.aio
import openai
from quart import (
    Blueprint,
    Response,
    current_app,
    render_template,
    request,
    stream_with_context,
)

bp = Blueprint("chat", __name__, template_folder="templates", static_folder="static")


@bp.before_app_serving
async def configure_openai():
    openai_host = os.getenv("OPENAI_HOST", "github")
    # YUBI: this is gpt-4o
    bp.model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    if openai_host == "local":
        # Use a local endpoint like llamafile server
        current_app.logger.info("Using model %s from local OpenAI-compatible API with no key", bp.model_name)
        bp.openai_client = openai.AsyncOpenAI(api_key="no-key-required", base_url=os.getenv("LOCAL_OPENAI_ENDPOINT"))
    elif openai_host == "github":
        current_app.logger.info("Using model %s from GitHub models with GITHUB_TOKEN as key", bp.model_name)
        bp.openai_client = openai.AsyncOpenAI(
            api_key=os.environ["GITHUB_TOKEN"],
            base_url="https://models.inference.ai.azure.com",
        )
    else:
        client_args = {}
        # Use an Azure OpenAI endpoint instead,
        # either with a key or with keyless authentication
        # YUBI: remove key authentication step to make sure it doesn't go there
        '''
        if os.getenv("AZURE_OPENAI_KEY_FOR_CHATVISION"):
            # Authenticate using an Azure OpenAI API key
            # This is generally discouraged, but is provided for developers
            # that want to develop locally inside the Docker container.
            current_app.logger.info("Using model %s from Azure OpenAI with key", bp.model_name)
            client_args["api_key"] = os.getenv("AZURE_OPENAI_KEY_FOR_CHATVISION")
        '''
        # else:
        if os.getenv("RUNNING_IN_PRODUCTION"):
            client_id = os.getenv("AZURE_CLIENT_ID")
            current_app.logger.info(
                "Using model %s from Azure OpenAI with managed identity credential for client ID %s",
                bp.model_name,
                client_id,
            )
            azure_credential = azure.identity.aio.ManagedIdentityCredential(client_id=client_id)
        else:
            # should run this block
            tenant_id = os.environ["AZURE_TENANT_ID"]
            current_app.logger.info(
                "Using model %s from Azure OpenAI with Azure Developer CLI credential for tenant ID: %s",
                bp.model_name,
                tenant_id,
            )
            azure_credential = azure.identity.aio.AzureDeveloperCliCredential(tenant_id=tenant_id)
        client_args["azure_ad_token_provider"] = azure.identity.aio.get_bearer_token_provider(
            azure_credential, "https://cognitiveservices.azure.com/.default"
        )
        bp.openai_client = openai.AsyncAzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "2025-01-01-preview",
            **client_args,
        )

    # load the patient schema json template from data folder
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'patient_schema.json')
    try:
        with open(file_path, 'r') as f:
            bp.patient_schema = json.load(f)
        current_app.logger.info("Loaded patient schema from %s", file_path)
    except Exception as e:
        current_app.logger.error("Failed to load patient schema: %s", e)
        bp.patient_schema = {}  # Fallback or raise if critical


@bp.after_app_serving
async def shutdown_openai():
    await bp.openai_client.close()


@bp.get("/")
async def index():
    return await render_template("index.html")

async def convert_pdf_page_to_image(page):
    """Convert a PyMuPDF page to a PIL image."""
    # lower resolution of displayed PDF for performance
    pix = page.get_pixmap(dpi=100)
    img_bytes = pix.tobytes("png")
    return Image.open(BytesIO(img_bytes))

async def image_to_base64(img: Image.Image):
    """Convert a PIL image to a base64 string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

async def call_model_on_image(image_base64, user_message):
    # Example: send to model via HTTP or local function
    # YUBI: make sure that all ' characters are formatted correctly
    section_prompt = "The uploaded file is scanned medical documents of one or more medical patients. Identify the following information for each patient if it is in the documents: their full legal name, date of birth, sex, living address, email address, phone number, primary insurance name, primary insurance type, primary insurance Member ID number, primary insurance Group ID number, secondary insurance name, secondary insurance type, secondary insurance Member ID number, secondary insurance Group ID number, CPT code, and ICD code. The primary insurance may also be referred to as the main insurance or first insurance in these documents. There are two possible insurance types, Medicare and Commercial, where Commercial encompassses all insurances that are not Medicare. When a patient has both a commercial insurance and a Medicare only insurance, the Medicare insurance is the primary plan and the commercial insurance is the secondary plan. The Member ID number and the Group ID number consists of any combination of uppercase letters and numerical digits. In the returned information, the phone number should be returned as 10 digits with no dashes, parentheses, or spaces. In the returned information, the sex should be represented as either F for female or M for male. In the returned information, all of the commas should be removed from the living address. If there are multiple phone numbers listed for the patient, the returned information should provide their cell phone number. In the returned information, the date of birth should be written in MM/DD/YYYY format where the month, day, and year are represented numerically. A CPT code is a numerical five-digit code that represents medical services and procedures. If a code contains letters or symbols, it is not a CPT code. Return each CPT code as a string. If there is more than one CPT code, each code should be returned separately. An ICD code is an alphanumeric code that contains up to seven characters that represents a type of disease or health condition in a patient. If there is more than one ICD code, each code should be returned separately. For every piece of returned information, return it in a key-value pair separated by a colon where the key is the patient\'s full legal name and the value is the relevant returned information. All of the key-value pairs should then be returned as a comma separated list."
    
    # This sends all messages, so API request may exceed token limits
    all_messages = [{"role": "system", "content": "You are a helpful assistant."}]
    if image_base64:
        user_content = []
        user_content.append({"text": user_message, "type": "text"})
        user_content.append({"text": section_prompt, "type": "text"})
        user_content.append({"image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "auto"}, "type": "image_url"})
        all_messages.append({"role": "user", "content": user_content})

    # send to model
    chat_coroutine = await bp.openai_client.chat.completions.create(
        # Azure Open AI takes the deployment name as the model name
        model=bp.model_name,
        messages=all_messages,
        stream=True,
        temperature=0.5,
    )

    # save answers
    response_text = ""
    async for chunk in chat_coroutine:
        if chunk and chunk.choices:
            delta = chunk.choices[0].delta
            if delta and hasattr(delta, "content") and delta.content:
                response_text += delta.content

    return response_text

async def summarize_answers(partials):
    """Aggregate partial answers into a single string."""
    partials_connected = "\n".join(partials)
    # call model with final message prompt
    all_messages = [{"role": "system", "content": "You are a helpful assistant."}]
    patient_schema_file = bp.patient_schema

    final_prompt="This is a comma separated list of key-value pairs containing relevant information on one or more medical patients. Every key is a patient\'s full name and the associated value is one of the following: their full legal name, date of birth, sex, living address, email address, phone number, primary insurance name, primary insurance type, primary insurance Member ID number, primary insurance Group ID number, secondary insurance name, secondary insurance type, secondary insurance Member ID number, secondary insurance Group ID number, CPT code, or ICD code. There may be similar keys that can be reasonably assumed to belong to the same patient because the key is the patient\'s name. For example, some keys may include a middle initial, middle name, maiden name, switched order of first and last name, or spelled with different capitalization. Group the key-value pairs together in sets of similar keys and rename every key in each set with the same, longest full name that is known in each set. Then, use the aggregated data from these groupings to create an array of JSON data instances, where each data instance represents a unique patient. Return the full array of patients. There can be more than one CPT code for a patient. There can be more than one ICD code for a patient. If there are any missing values, they should be returned as \"null\" in the JSON data instance. Return only the raw JSON array. Do not wrap the response in markdown backticks. The JSON schema is attached to this message."

    # IDK if this check is necessary
    if partials_connected:
        user_content = []
        user_content.append({"text": partials_connected, "type": "text"})
        user_content.append({"text": final_prompt, "type": "text"})
        # add schema file to the user content
        user_content.append({"type": "text", "text": json.dumps(patient_schema_file)})
        all_messages.append({"role": "user", "content": user_content})
        

    # send to model
    chat_coroutine = await bp.openai_client.chat.completions.create(
        # Azure Open AI takes the deployment name as the model name
        model=bp.model_name,
        messages=all_messages,
        stream=True,
        temperature=0.5,
    )

    # save answers
    response_text = ""
    async for chunk in chat_coroutine:
        if chunk and chunk.choices:
            delta = chunk.choices[0].delta
            if delta and hasattr(delta, "content") and delta.content:
                response_text += delta.content

    return response_text

# helper function to validate patient fields returned from summarize_answers
def validate_patient_fields(patients):

    annotated = []
    for patient in patients:
        entry = {}
        for key, value in patient.items():
            if isinstance(value, str) and value in ["null", "None", ""]:
                valid = False
                reason = "Couldn't find the value in the document"
            elif key == "Date of Birth":
                valid = bool(re.match(r"\d{2}/\d{2}/\d{4}", str(value)))
                # YUBI: I can change these reasons to something more vague after I test
                reason = None if valid else "Invalid format, must be MM/DD/YYYY"
            elif key == "Sex":
                valid = value in {"M", "F"}
                reason = None if valid else "Must be 'M' or 'F'"
            elif key == "Phone Number":
                valid = bool(re.match(r"^\d{10}$", str(value)))
                reason = None if valid else "Must be 10 digits with no dashes, parentheses, or spaces"
            elif key in {"Primary Insurance Type", "Secondary Insurance Type"}:
                valid = value in {"Medicare", "Commercial"}
                reason = None if valid else "Must be 'Medicare' or 'Commercial'"
            elif key in {"Primary Insurance Member ID", "Primary Insurance Group ID",
                         "Secondary Insurance Member ID", "Secondary Insurance Group ID"}:
                valid = bool(re.match(r"^[A-Z0-9]+$", str(value)))
                reason = None if valid else "Must be alphanumeric with no spaces"
            elif key == "CPT Codes":
                # value must be a string containing 5 numbers only
                valid = bool(re.match(r"^\d{5}$", str(value)))
                reason = None if valid else "Must be numeric with 5 characters"
            elif key == "ICD Codes":
                valid = bool(re.match(r"^[A-Z0-9]{3,7}$", str(value)))
                reason = None if valid else "Must be alphanumeric with 3 to 7 characters"
            else:
                valid = True
                reason = None
            entry[key] = {"value": value, "valid": valid}
            if not valid:
                entry[key]["reason"] = reason
        annotated.append(entry)
    return annotated


@bp.route('/process_pdf', methods=['POST'])
async def process_pdf():
    uploaded_file = (await request.files)['file']
    # should I use get?
    # uploaded_file = (await request.files).get('file')
    if not uploaded_file:
        return jsonify({"error": "Missing file"}), 400

    user_message = (await request.form).get('message', '')

    try:
        # Don't need to wait for this function bc it isn't asynchronous 
        pdf_data = uploaded_file.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
    except Exception as e:
        return jsonify({"error": f"Failed to open PDF: {str(e)}"}), 500


    partial_answers = []

    # define helper function for batching and stacking images
    def stack_images_vertically(images):
        """Combine a list of PIL images vertically into one."""
        widths, heights = zip(*(img.size for img in images))
        total_height = sum(heights)
        max_width = max(widths)

        combined = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        y_offset = 0
        for img in images:
            combined.paste(ImageOps.expand(img, border=0, fill='white'), (0, y_offset))
            y_offset += img.height
        return combined

    # Process pages in batches of 2
    # YUBI: allow this variable to be set by the user
    # batch_size = int(os.getenv("BATCH_SIZE", 2))  # Default is 2
    batch_size = 2
    num_pages = len(doc)
    for i in range(0, num_pages, batch_size):
        try:
            images = []
            for j in range(i, min(i + batch_size, num_pages)):
                subdoc = fitz.open()
                subdoc.insert_pdf(doc, from_page=j, to_page=j)
                page = subdoc[0]
                pil_image = await convert_pdf_page_to_image(page)
                images.append(pil_image)

            if not images:
                continue

            if len(images) == 1:
                merged_image = images[0]
            else:
                merged_image = stack_images_vertically(images)

            img_base64 = await image_to_base64(merged_image)

            try:
                # time out the call to the model if it is taking too long
                # YUBI: do we want to limit to 60 seconds?
                result = await asyncio.wait_for(call_model_on_image(img_base64, user_message), timeout=90)
            except asyncio.TimeoutError:
                return jsonify({"error": f"Timeout on page {i}"}), 504
            
            partial_answers.append(result)

        except Exception as e:
            return jsonify({"error": f"Failed on pages {i}-{i+batch_size-1} using model {bp.model_name}: {str(e)}"}), 500

    '''
    for i in range(len(doc)):
        try:
            subdoc = fitz.open()
            subdoc.insert_pdf(doc, from_page=i, to_page=i)
            page = subdoc[0]
            pil_image = await convert_pdf_page_to_image(page)
            img_base64 = await image_to_base64(pil_image)
            result = await call_model_on_image(img_base64, user_message)

            try:
                # time out the call to the model if it is taking too long
                # YUBI: do we want to limit to 60 seconds?
                result = await asyncio.wait_for(call_model_on_image(img_base64, user_message), timeout=90)
            except asyncio.TimeoutError:
                return jsonify({"error": f"Timeout on page {i}"}), 504

            partial_answers.append(result)
        except Exception as e:
            # YUBI: added this error message but I'm not sure if it will cause issues
            return jsonify({"error": f"Failed on page {i} using a model name of {bp.model_name}: {str(e)}"}), 500
    '''
    
    # Final aggregation step
    try:
        final_answer = await summarize_answers(partial_answers)
    except Exception as e:
        return jsonify({"error": f"Failed during summarization: {str(e)}"}), 500
    
    try:
        patients_json = json.loads(final_answer)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Failed to parse model output as JSON: {str(e)}", "raw_output": final_answer}), 500

    try:
        annotated_patients = validate_patient_fields(patients_json)
    except Exception as e:
        print("Validation failed:", e)
        return {"error": "Validation error", "details": str(e)}, 500

    return jsonify({"patients": annotated_patients})

    
    # formatted_answer = await format_response(final_answer)
    # return jsonify({"answer": formatted_answer})