# This is a new chat.py file that uses asyncio.gather to parallelize the processing of PDF pages.
# This tried to complete the full payment processing flow
# switch to more concise prompts and less schema sending to reduce token usage

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
import json
import tiktoken  # For counting tokens (if available; otherwise approximate)


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
    
    # load EOB schema json template from data folder
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'EOB_schema.json')
    try:
        with open(file_path, 'r') as f:
            bp.EOB_schema = json.load(f)
        current_app.logger.info("Loaded EOB schema from %s", file_path)
    except Exception as e:
        current_app.logger.error("Failed to load EOB schema: %s", e)
        bp.EOB_schema = {}  # Fallback or raise if critical
    
    # load payment schema json template from data folder
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'payment_schema.json')
    try:
        with open(file_path, 'r') as f:
            bp.payment_schema = json.load(f)
        current_app.logger.info("Loaded payment schema from %s", file_path)
    except Exception as e:
        current_app.logger.error("Failed to load payment schema: %s", e)
        bp.payment_schema = {}  # Fallback or raise if critical
    
    # load match schema json template from data folder
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'match_schema.json')
    try:
        with open(file_path, 'r') as f:
            bp.match_schema = json.load(f)
        current_app.logger.info("Loaded match schema from %s", file_path)
    except Exception as e:
        current_app.logger.error("Failed to load match schema: %s", e)
        bp.match_schema = {}  # Fallback or raise if critical


@bp.after_app_serving
async def shutdown_openai():
    await bp.openai_client.close()


@bp.get("/")
async def index():
    return await render_template("index.html")

# Convert a PyMuPDF page to a PIL image.
async def convert_pdf_page_to_image(page):
    # YUBI: debugging statement, want to change back alter
    # updated dpi from 100 to 200 for payment to see if it improves accuracy
    pix = page.get_pixmap(dpi=100)
    img_bytes = pix.tobytes("png")
    return Image.open(BytesIO(img_bytes))

# Convert a PIL image to a base64 string.
async def image_to_base64(img: Image.Image):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Call the AI model with the image and user message
async def call_model_on_image(image_base64, user_message, processing_mode):
    section_prompt = ""
    user_content = []
    # YUBI: I am setting this right now, but we want this to be dynamic based on the number of pages in the PDF
    total_pg_count = 2000
    detail_level = "auto"  # Default detail level for images


    if processing_mode == "payment":
        section_prompt += f"This image is a section of a scanned document that may contain Explanation of Benefits (EOB) or payments for medical services. An EOB is titled \'Explanation of Benefits\' and includes a chart listing medical costs. The amount of money paid by the health insurance company (Amount Paid) is written in the most bottom-right entry of the EOB chart. Represent an EOB as a JSON object that contains the following fields: Patient Name (string) and Amount Paid (number). There are 2 types of Payment: Check and Virtual Card. A check appears as a wide, horizontally-oriented rectangular box enclosing a printed check number in the top right corner, a payer name in the upper left corner, a payment amount written in numeric form and spelled out in words, a signature line on the bottom right, and a long sequence of numbers printed in MICR format along the bottom. It is not a check if the rectangular box encloses a dense table. A virtual card typically includes a 16-digit card number, a CVV/CVV2 code, an expiration date written in MM/YY format, and a credit card company logo all grouped together inside an outlined rectangle with rounded corners. The card may appear alongside the text \'Mastercard Express ClaimsCard\' or \'Virtual Card\'. The card is displayed next to a payment Amount shown in dollar format. It is not a card if there are no numbers enclosed by the outlined rectangle or if the rectangle is near a \'U.S. Postage Paid\' stamp. Otherwise, if it looks similar to a payment, consider it a payment. For every payment in the document, extract the page number it is on. The page number is written as \'Page # of {total_pg_count}\' on every page, where # represents the page number. Represent a Payment as a JSON object that contains the following fields: Payer Name, Payee Name, Amount Paid, Payment Type, Payment Page Number, Card Number, CVV Code, Expiration Date, and Check Number. Missing fields should be 'null'. Return two arrays in a JSON object with the following keys: page_array and objects_array. page_array is an array of all the page numbers containing a Payment. objects_array is an array of all EOB and Payment JSON objects found. Format each array and your full response as raw JSON only. Empty arrays are allowed. Output raw JSON only; no extra text or formatting."
        # user_content.append({"type": "text", "text": json.dumps(bp.payment_schema)})
        # user_content.append({"type": "text", "text": json.dumps(bp.EOB_schema)})
    else:
        # default processing mode is billing
        # section_prompt += "The uploaded file is scanned medical documents of one or more medical patients. Identify the following information for each patient if it is in the documents: their full legal name, date of birth, sex, living address, email address, phone number, primary insurance name, primary insurance type, primary insurance Member ID number, primary insurance Group ID number, secondary insurance name, secondary insurance type, secondary insurance Member ID number, secondary insurance Group ID number, CPT code, and ICD code. The primary insurance may also be referred to as the main insurance or first insurance in these documents. There are two possible insurance types, Medicare and Commercial, where Commercial encompassses all insurances that are not Medicare. When a patient has both a commercial insurance and a Medicare only insurance, the Medicare insurance is the primary plan and the commercial insurance is the secondary plan. The Member ID number and the Group ID number consists of any combination of uppercase letters and numerical digits. In the returned information, the phone number should be returned as 10 digits with no dashes, parentheses, or spaces. In the returned information, the sex should be represented as either F for female or M for male. In the returned information, all of the commas should be removed from the living address. If there are multiple phone numbers listed for the patient, the returned information should provide their cell phone number. In the returned information, the date of birth should be written in MM/DD/YYYY format where the month, day, and year are represented numerically. A CPT code is a numerical five-digit code that represents medical services and procedures. If a code contains letters or symbols, it is not a CPT code. Return each CPT code as a string. If there is more than one CPT code, each code should be returned separately. An ICD code is an alphanumeric code that contains up to seven characters that represents a type of disease or health condition in a patient. If there is more than one ICD code, each code should be returned separately. For every piece of returned information, return it in a key-value pair separated by a colon where the key is the patient\'s full legal name and the value is the relevant returned information. All of the key-value pairs should then be returned as a comma separated list."
        # YUBI: testing simpler prompt
        section_prompt += (
        "The file contains scanned documents of medical patients. Extract the following for each patient if present: full name, date of birth, sex, address, email, phone, primary and secondary insurance name, type, member ID, group ID, CPT code, and ICD code. "
        "Insurance type is either Medicare or Commercial (which includes all others). If both are present, Medicare is the primary. "
        "Format each field as a separate key-value pair: the key is the patient’s full name, the value is 'Field Name: Field Value'. Repeat the patient name for each field. "
        "Format dates as MM/DD/YYYY. Do not include commas within values. Return all pairs as a single comma-separated list with no extra formatting or explanation. Omit any fields not found."
        )

    # This sends all messages, so API request may exceed token limits
    all_messages = [{"role": "system", "content": "You are a helpful assistant."}]
    if image_base64:
        user_content.append({"text": user_message, "type": "text"})
        user_content.append({"text": section_prompt, "type": "text"})
        user_content.append({"image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": detail_level}, "type": "image_url"})
        all_messages.append({"role": "user", "content": user_content})

    # YUBI: I'm going to use same AI model for both processing modes for now
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

    if processing_mode == "payment":
        try:
            data = json.loads(response_text)
            page_array = data.get('page_array', [])
            objects_array = data.get('objects_array', [])
        except json.JSONDecodeError:
            print("Failed to decode response as JSON.")
            page_array, objects_array = [], []

        # return an array of the pages with payments and an array of the JSON data instances
        return page_array, objects_array

    return response_text

# Call the AI model for follow-up questions
async def call_model_followup(prompt):
    
    # This sends all messages, so API request may exceed token limits
    all_messages = [{"role": "system", "content": "You are a helpful assistant."}]
    user_content = []
    user_content.append({"text": prompt, "type": "text"})
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


async def summarize_pages(partials):
    """Concatenate all arrays of pages into one array."""
    # assumes that input is a nested array of numbers
    merged = set()
    for pages in partials:
        merged.update(pages)
    # returns a single, flat array of numbers that are sorted in ascending order
    return sorted(merged)



# summarize answers function that batches the partial answers for batched calls to AI model
# returns a list of JSON data instances
# change token limit from 6000 to 12000
async def summarize_matches(partials, batch_token_limit=12000):
    """Aggregate partial answers into a single list of JSON objects by batching."""

    def count_tokens(text):
        try:
            enc = tiktoken.encoding_for_model(bp.model_name)
            return len(enc.encode(text))
        except Exception:
            return len(text.split())  # Fallback: approx 1 token per word

    # match_schema_file = bp.match_schema

    # YUBI: testing simpler prompt    
    summary_prompt = (
    "This is a JSON array containing two types of objects: Explanation of Benefits (EOB) objects and Payment objects."
    "EOB objects include the fields \'Patient Name\' and \'Amount Paid\'."
    "Payment objects include fields such as \'Payer Name\', \'Receiver Name\', \'Amount Paid\', \'Payment Type\', and other payment-specific fields."
    "Match each EOB object to a Payment object only if the \'Amount Paid\' values are exactly equal."
    "Return a new array of JSON objects, each representing one matched pair, with the following fields: Payer Name, Receiver Name, Patient Name, Amount Paid, Payment Type, Payment Page Number, Card Number, CVV Code, Expiration Date, Check Number. Fields can be \'null\' if they don’t exist."
    "If no matches are found, return an empty array. Output raw JSON only; no extra text or formatting. Do not include unmatched objects."
    )

    # Chunk partials to respect token limit per batch
    batches = []
    current_batch = []
    current_tokens = 0

    for part in partials:
        part_str = json.dumps(part)
        tokens = count_tokens(part_str)
        if current_tokens + tokens > batch_token_limit and current_batch:
            batches.append(current_batch)
            current_batch = [part_str]
            current_tokens = tokens
        else:
            current_batch.append(part_str)
            current_tokens += tokens


    if current_batch:
        batches.append(current_batch)

    all_json_objects = []

    for batch in batches:
        # partials_connected = "\n".join(batch)
        # YUBI: debugging statement, read in partials as a valid JSON array
        partials_connected = json.dumps([json.loads(p) for p in batch])
        all_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"text": partials_connected, "type": "text"},
                    {"text": summary_prompt, "type": "text"},
                    # {"type": "text", "text": json.dumps(match_schema_file)},
                ]
            }
        ]

        chat_coroutine = await bp.openai_client.chat.completions.create(
            model=bp.model_name,
            messages=all_messages,
            stream=True,
            temperature=0.5,
        )

        response_text = ""
        async for chunk in chat_coroutine:
            if chunk and chunk.choices:
                delta = chunk.choices[0].delta
                if delta and hasattr(delta, "content") and delta.content:
                    response_text += delta.content

        try:
            parsed_batch = json.loads(response_text)
            if isinstance(parsed_batch, list):
                all_json_objects.extend(parsed_batch)
            else:
                raise ValueError("Expected a list of JSON objects")
        except Exception as e:
            raise RuntimeError(f"Failed to parse model output as JSON: {str(e)}\nRaw response: {response_text}")

    return all_json_objects



# summarize answers function that batches the partial answers for batched calls to AI model
# returns a list of JSON data instances
async def summarize_answers(partials, processing_mode, batch_token_limit=6000):
    """Aggregate partial answers into a single list of JSON objects by batching."""

    def count_tokens(text):
        try:
            enc = tiktoken.encoding_for_model(bp.model_name)
            return len(enc.encode(text))
        except Exception:
            return len(text.split())  # Fallback: approx 1 token per word

    if processing_mode == "payment":
        schema_file = ""
        summary_prompt = "add prompt here"
    else:
        # default processing mode is billing
        schema_file = bp.patient_schema
        # YUBI: testing simpler prompt
        summary_prompt = (
            "This is a comma-separated list of key-value pairs about medical patients. "
            "Each key is a patient's full name; each value is a labeled field (e.g., 'Date of Birth: 01/01/1980'). "
            "Some names may refer to the same person despite differences (e.g., middle names, initials, or capitalization). "
            "Group similar names and use the longest full name in each group. "
            "Aggregate fields for each patient into a single JSON object. Each patient object must contain the following fields: Patient Name, Date of Birth, Sex, Address, Email, Phone, Primary Insurance Name, Primary Insurance Type, Primary Insurance Member ID, Primary Insurance Group ID, Secondary Insurance Name, Secondary Insurance Type, Secondary Insurance Member ID, Secondary Insurance Group ID, CPT Codes, and ICD Codes. All fields are strings, except CPT Codes and ICD Codes, which are arrays of strings that include all CPT and ICD codes found."
            "For other fields with conflicting values, choose the most likely one. "
            "Missing fields should be 'null'. Return an array of patient JSON objects. Output raw JSON only; no extra text or formatting."
        )

    # Chunk partials to respect token limit per batch
    batches = []
    current_batch = []
    current_tokens = 0

    for part in partials:
        tokens = count_tokens(part)
        if current_tokens + tokens > batch_token_limit and current_batch:
            batches.append(current_batch)
            current_batch = [part]
            current_tokens = tokens
        else:
            current_batch.append(part)
            current_tokens += tokens

    if current_batch:
        batches.append(current_batch)

    all_json_objects = []

    for batch in batches:
        partials_connected = "\n".join(batch)
        all_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"text": partials_connected, "type": "text"},
                    {"text": summary_prompt, "type": "text"},
                    #  YUBI: testing not having schema file
                    # {"type": "text", "text": json.dumps(schema_file)},
                ]
            }
        ]

        chat_coroutine = await bp.openai_client.chat.completions.create(
            model=bp.model_name,
            messages=all_messages,
            stream=True,
            temperature=0.5,
        )

        response_text = ""
        async for chunk in chat_coroutine:
            if chunk and chunk.choices:
                delta = chunk.choices[0].delta
                if delta and hasattr(delta, "content") and delta.content:
                    response_text += delta.content

        try:
            parsed_batch = json.loads(response_text)
            if isinstance(parsed_batch, list):
                all_json_objects.extend(parsed_batch)
            else:
                raise ValueError("Expected a list of JSON objects")
        except Exception as e:
            raise RuntimeError(f"Failed to parse model output as JSON: {str(e)}\nRaw response: {response_text}")

    return all_json_objects


# function to connect summaries of all JSON objects into a final answer
async def connect_summaries(all_json_objects, processing_mode):
    """Aggregate summarized chunks into a answer ."""

    json_input_str = json.dumps(all_json_objects)
    
    # call model with final message prompt
    all_messages = [{"role": "system", "content": "You are a helpful assistant."}]

    # YUBI EDIT: add schema file for the payment information
    if processing_mode == "payment":
        schema_file = ""
    else:
        # default processing mode is billing
        schema_file = bp.patient_schema

    final_prompt = ""
    if processing_mode == "payment":
        # YUBI: add payment prompt here
        final_prompt += "add prompt here"
    else:
        # default processing mode is billing
        # YUBI: testing simpler prompt
        final_prompt += ("This is a list of JSON data instances that each represent a patient. Review the list and combine any data instances that refer to the same patient. Data instances refer to the same patient if they have a similar Full Name (e.g., middle names, initials, or capitalization). "
            "Aggregate fields for each patient into a single JSON object. Each patient object must contain the following fields: Patient Name, Date of Birth, Sex, Address, Email, Phone, Primary Insurance Name, Primary Insurance Type, Primary Insurance Member ID, Primary Insurance Group ID, Secondary Insurance Name, Secondary Insurance Type, Secondary Insurance Member ID, Secondary Insurance Group ID, CPT Codes, and ICD Codes. All fields are strings, except CPT Codes and ICD Codes, which are arrays of strings that include all CPT and ICD codes found."
            "For other fields with conflicting values, choose the most likely one. "
            "Missing fields should be 'null'. Return an array of patient JSON objects. Output raw JSON only; no extra text or formatting.")

    # IDK if this check is necessary
    user_content = []
    user_content.append({"text": json_input_str, "type": "text"})
    user_content.append({"text": final_prompt, "type": "text"})
    # add schema file to the user content
    # user_content.append({"type": "text", "text": json.dumps(schema_file)})
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

    # I'm not doing any data cleaning right now and assuming that the model returns raw JSON exactly the way I want it

    return response_text


# helper function to validate patient fields returned from summarize_answers
# for billing processing mode
def validate_patient_fields(patients):

    annotated = []
    for patient in patients:
        entry = {}
        for key, value in patient.items():
            if isinstance(value, str) and value in ["null", "None", "", "N/A", "not provided", " "]:
                # do not highlight empty cells because they are already empty
                valid = True
            elif key == "Date of Birth":
                # value can be any arrangement of numbers and dashes or slashes, but can't have any alphabetic characters
                valid = bool(re.match(r"^\d{1,4}[-/]\d{1,4}[-/]\d{1,4}$", str(value)))
                reason = None if valid else "Date has invalid characters or format"
            elif key == "Sex":
                valid = value in {"M", "F", "Male", "Female"}
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
                # value must be an array of strings where each string is a 5-digit number and there are no more than 5 strings in the array
                valid = isinstance(value, list) and all(
                    isinstance(code, str) and re.match(r"^\d{5}$", code) for code in value
                ) and len(value) <= 5
                reason = None if valid else "Each CPT code must be a 5-digit number and there can be no more than 5 codes"
            elif key == "ICD Codes":
                # value must be an array of strings where each string is alphanumeric with 3 to 7 characters and there are no more than 5 strings in the array
                valid = isinstance(value, list) and all(
                    isinstance(code, str) and re.match(r"^[A-Z0-9]{3,7}$", code) for code in value
                ) and len(value) <= 5
                reason = None if valid else "Each ICD code must be alphanumeric with 3 to 7 characters and there can be no more than 5 codes"
            else:
                valid = True
                reason = None
            entry[key] = {"value": value, "valid": valid}
            
            if not valid:
                entry[key]["reason"] = reason
        annotated.append(entry)
    return annotated


# helper function to validate payment fields returned from summarize_answers
# for payment processing mode
# YUBI: write this function to validate payment fields later
# TODO
def validate_payment_fields(payments):
    # set everything to valid by default
    annotated = []
    for payment in payments:
        entry = {}
        for key, value in payment.items():
            valid = True
            entry[key] = {"value": value, "valid": valid}
        annotated.append(entry)
    return annotated

# Updated code to handle PDF processing in parallel
# YUBI: double check this
@bp.route('/process_pdf', methods=['POST'])
async def process_pdf():
    # Retrieve the uploaded PDF file from the request
    uploaded_file = (await request.files)['file']
    if not uploaded_file:
        return jsonify({"error": "Missing file"}), 400

    # Retrieve the optional user message sent along with the PDF
    user_message = (await request.form).get('message', '')
    processing_mode = (await request.form).get('processing_mode', 'billing')

    # Attempt to read and open the PDF using PyMuPDF
    try:
        pdf_data = uploaded_file.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
    except Exception as e:
        # Return 500 error if PDF cannot be opened
        return jsonify({"error": f"Failed to open PDF: {str(e)}"}), 500

    # Define the batch size (number of PDF pages processed together in one batch)
    # YUBI: debugging, I decrease the batch size to 1 from 2
    # COME BACK TO THIS AND CHANGE IT LATER
    batch_size = 2
    num_pages = len(doc)  

    # Set maximum number of concurrent batches allowed to avoid overloading downstream resources
    MAX_CONCURRENT_BATCHES = 4
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)  # Controls concurrency limit

    # Helper function to stack multiple images vertically into one tall image
    def stack_images_vertically(images):
        widths, heights = zip(*(img.size for img in images))
        total_height = sum(heights)
        max_width = max(widths)
        combined = Image.new('RGB', (max_width, total_height), (255, 255, 255))  # White background
        y_offset = 0
        for img in images:
            combined.paste(ImageOps.expand(img, border=0, fill='white'), (0, y_offset))
            y_offset += img.height
        return combined

    # Async function to process a batch of pages:
    # - converts pages to images,
    # - stacks images vertically if multiple,
    # - encodes image as base64,
    # - calls AI model with the image and user message.
    async def process_page_batch(start_idx: int):
        async with semaphore:  # Acquire semaphore before starting to limit concurrency
            images = []
            # Loop through pages in the batch
            for page_idx in range(start_idx, min(start_idx + batch_size, num_pages)):
                # Extract one page as a separate PDF document
                subdoc = fitz.open()
                subdoc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
                page = subdoc[0]
                # Convert PDF page to PIL image asynchronously
                pil_image = await convert_pdf_page_to_image(page)
                images.append(pil_image)

            if not images:
                return None  # No pages to process in this batch

            # Stack images vertically if multiple pages, else use single image
            merged_image = images[0] if len(images) == 1 else stack_images_vertically(images)
            # Convert merged image to base64 string for model input
            img_base64 = await image_to_base64(merged_image)

            try:
                # Call the AI model with a timeout to avoid hanging
                # EDIT HERE: enable parameters to be passed to this function
                # YUBI: the message to call_model_on_image should differ based on processing mode
                if processing_mode == "payment":
                    page_array, objects_array = await asyncio.wait_for(call_model_on_image(img_base64, user_message, processing_mode), timeout=90)
                    return page_array, objects_array
                else:
                    result = await asyncio.wait_for(call_model_on_image(img_base64, user_message, processing_mode), timeout=90)
                    return result
            except asyncio.TimeoutError:
                # Raise an error if processing times out for this batch
                raise RuntimeError(f"Timeout processing pages {start_idx}-{start_idx + batch_size - 1}")

    # Create async tasks for each batch of pages
    tasks = [asyncio.create_task(process_page_batch(i)) for i in range(0, num_pages, batch_size)]

    # YUBI: is this correct?

    partial_answers = []
    try:
        if processing_mode == "payment":
            # Run all batch tasks concurrently (limited by semaphore)
            batch_results = await asyncio.gather(*tasks)
            partial_pages = [r[0] for r in batch_results if r is not None]
            # partial_objects = [r[1] for r in batch_results if r is not None]
            # YUBI: recent debugging statement
            partial_objects = [obj for r in batch_results if r is not None for obj in r[1]]

            # YUBI: DEBUGGING by returning the partial objects and partial pages
            # return jsonify({"payments": partial_objects, "pages": partial_pages}), 200
        else:
            # Run all batch tasks concurrently (limited by semaphore)
            batch_results = await asyncio.gather(*tasks)
            # Filter out any None results (empty batches)
            partial_answers = [r for r in batch_results if r is not None]
    except RuntimeError as e:
        # Return 504 Gateway Timeout if any batch timed out
        return jsonify({"error": str(e)}), 504
    except Exception as e:
        # Return 500 for any other errors during batch processing
        return jsonify({"error": f"Batch processing failed: {str(e)}"}), 500

    # After all batches processed, aggregate partial answers into a final answer
    # EDIT HERE: enable parameters to be passed to this function
    if processing_mode == "payment":
        # have a different summarizing process for payment processing mode using partial_pages and partial_objects
        try:
            all_pages = await summarize_pages(partial_pages)
        except Exception as e:
            return jsonify({"error": f"Failed during summarization of pages: {str(e)}"}), 500
        
        try:
            all_matches = await summarize_matches(partial_objects)
        except Exception as e:
            return jsonify({"error": f"Failed during summarization of pages: {str(e)}"}), 500
        
        matches_json = all_matches
        
        '''
        # Parse the final aggregated model output as JSON
        try:
            # all_matches is already a list of json objects so you don't need to parse it
            matches_json = json.loads(all_matches)
        except json.JSONDecodeError as e:
            # Return 500 error with raw output for debugging if JSON parsing fails
            return jsonify({"error": f"Failed to parse model matches output as JSON: {str(e)}", "raw_output": all_matches}), 500
        '''

    else:  
        try:
            summarized_answer = await summarize_answers(partial_answers, processing_mode)
        except Exception as e:
            return jsonify({"error": f"Failed during summarization: {str(e)}"}), 500
    
        try:
            final_answer = await connect_summaries(summarized_answer, processing_mode)
        except Exception as e:
            return jsonify({"error": f"Failed during summarization: {str(e)}"}), 500

        # Parse the final aggregated model output as JSON
        try:
            answer_json = json.loads(final_answer)
        except json.JSONDecodeError as e:
            # Return 500 error with raw output for debugging if JSON parsing fails
            return jsonify({"error": f"Failed to parse model output as JSON: {str(e)}", "raw_output": final_answer}), 500

    # Validate the parsed data and annotate invalid fields

    if processing_mode == "payment":
        try:
            # YUBI: create a new function to validate payment fields
            annotated_payments = validate_payment_fields(matches_json)
        except Exception as e:
            current_app.logger.error("Validation failed: %s", e)
            return {"error": "Validation error", "details": str(e)}, 500

        # YUBI: EDIT json to ensure that the key is "payments" instead of "patients"
        # return two results to front end, one is annotated_payments and the other is all_pages formatted into a string
        # YUBI: does this work? what does 200 mean?
        return jsonify({"payments": annotated_payments, "pages": all_pages}), 200
    else:
        # default processing mode is billing
        try:
            annotated_patients = validate_patient_fields(answer_json)
        except Exception as e:
            current_app.logger.error("Validation failed: %s", e)
            return {"error": "Validation error", "details": str(e)}, 500

        # Return the validated and annotated patient data as JSON response
        return jsonify({"patients": annotated_patients})


# New route for follow-up
@bp.route("/followup", methods=["POST"])
async def followup():
    try:
        form = await request.form
        message = form["message"]
        # YUBI: Edit this function based on processing mode as well
        # change from previous_patients to previous_answer so that it is more general
        previous_answer_raw = form.get("previous_answer")


        # Parse the JSON string into an object
        try:
            previous_answer_json = json.loads(previous_answer_raw)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format for previous_patients"}), 400

        # Pretty-print the JSON for readability
        previous_answer_pretty = json.dumps(previous_answer_json, indent=2)

        # Build model message with context
        followup_prompt = (
            "The user has previously asked you to extract information from a scanned document. "
            "They now have a follow-up question. Below is the structured data from your previous response, "
            "and the user's follow-up question. Use this context to answer clearly and directly.\n\n"
            f"Previous extracted data:\n{previous_answer_pretty}\n\n"
            f"Follow-up question:\n{message}"
        )

        # Call model
        model_response = await call_model_followup(followup_prompt)

        return jsonify({"answer": model_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500