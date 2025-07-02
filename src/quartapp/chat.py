# This is a new chat.py file that will hopefully take in large PDFs
# chunk them into smaller PDFs then convert them to images, which are then iteratively called to the AI model
# and then the answers are all concatenated together and returned to front-end

import json
import os

# import additional packages
from quart import Blueprint, request, jsonify
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import base64


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


@bp.after_app_serving
async def shutdown_openai():
    await bp.openai_client.close()


@bp.get("/")
async def index():
    return await render_template("index.html")

async def convert_pdf_page_to_image(page):
    """Convert a PyMuPDF page to a PIL image."""
    pix = page.get_pixmap(dpi=200)
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
    section_prompt = "The uploaded PDF is scanned medical documents of one or more medical patients. Identify the following information for each patient if it is in the documents: their full legal name, date of birth, sex, living address, email address, phone number, primary insurance name, primary insurance type, primary insurance Member ID number, and primary insurance Group ID number. The primary insurance may also be referred to as the main insurance or first insurance in these documents. The Member ID number and the Group ID number consists of any combination of uppercase letters and numerical digits. There are two possible insurance types, Medicare and Commercial, where Commercial encompassses all insurances that are not Medicare. In the returned information, the phone number should be returned as 10 digits with no dashes, parentheses, or spaces. In the returned information, the sex should be represented as either F for female or M for male. In the returned information, all of the commas should be removed from the living address. If there are multiple phone numbers listed for the patient, the returned information should provide their cell phone number. In the returned information, the date of birth should be written in MM/DD/YYYY format where the month, day, and year are represented numerically. For every piece of returned information, return it in a key-value pair separated by a colon where the key is the patient\'s full legal name and the value is the relevant returned information. All of the key-value pairs should be returned as a comma separated list."
    
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

async def summarize_answers(partials, message):
    """Aggregate partial answers into a single string."""
    partials_connected = "\n".join(partials)
    # call model with final message prompt
    all_messages = [{"role": "system", "content": "You are a helpful assistant."}]

    # IDK if this check is necessary
    if partials_connected:
        user_content = []
        user_content.append({"text": partials_connected, "type": "text"})
        user_content.append({"text": message, "type": "text"})
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

    for i in range(len(doc)):
        try:
            subdoc = fitz.open()
            subdoc.insert_pdf(doc, from_page=i, to_page=i)
            page = subdoc[0]
            pil_image = await convert_pdf_page_to_image(page)
            img_base64 = await image_to_base64(pil_image)
            result = await call_model_on_image(img_base64, user_message)
            partial_answers.append(result)
        except Exception as e:
            # YUBI: added this error message but I'm not sure if it will cause issues
            return jsonify({"error": f"Failed on page {i} using a model name of {bp.model_name}: {str(e)}"}), 500


    # YUBI: this should ask model to group all information together
    # YUBI: make sure that all ' characters are formatted correctly
    final_prompt="Prompt: This is a comma separated list of key-value pairs containing relevant information on one or more medical patients. Every key is a patient\'s full name and the associated value is one of the following: their full legal name, date of birth, sex, living address, email address, phone number, primary insurance name, primary insurance type, primary insurance Member ID number, or primary insurance Group ID number. Create a table where there is one row per patient and the columns are each patient\'s full legal name, date of birth, sex, living address, email address, phone number, primary insurance name, primary insurance type, primary insurance Member ID number, or primary insurance Group ID number. If there is any missing information, write N/A in that table entry. Return the table as a comma separated list where each column is separated by a comma and each row is separated by a semicolon."
    # Final aggregation step
    try:
        final_answer = await summarize_answers(partial_answers, final_prompt)
    except Exception as e:
        return jsonify({"error": f"Failed during summarization: {str(e)}"}), 500


    # what is jsonify?
    return jsonify({"answer": final_answer})