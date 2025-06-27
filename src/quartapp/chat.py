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
        if os.getenv("AZURE_OPENAI_KEY_FOR_CHATVISION"):
            # Authenticate using an Azure OpenAI API key
            # This is generally discouraged, but is provided for developers
            # that want to develop locally inside the Docker container.
            current_app.logger.info("Using model %s from Azure OpenAI with key", bp.model_name)
            client_args["api_key"] = os.getenv("AZURE_OPENAI_KEY_FOR_CHATVISION")
        else:
            if os.getenv("RUNNING_IN_PRODUCTION"):
                client_id = os.getenv("AZURE_CLIENT_ID")
                current_app.logger.info(
                    "Using model %s from Azure OpenAI with managed identity credential for client ID %s",
                    bp.model_name,
                    client_id,
                )
                azure_credential = azure.identity.aio.ManagedIdentityCredential(client_id=client_id)
            else:
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
            api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "2024-05-01-preview",
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
    """Dummy placeholder — replace with actual model call logic."""
    # Example: send to model via HTTP or local function
    
    # This sends all messages, so API request may exceed token limits
    all_messages = [{"role": "system", "content": "You are a helpful assistant."}]
    if image_base64:
        user_content = []
        user_content.append({"text": user_message, "type": "text"})
        user_content.append({"image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "auto"}, "type": "image_url"})
        all_messages.append({"role": "user", "content": user_content})

    # send to model
    chat_coroutine = bp.openai_client.chat.completions.create(
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
        user_content.append({"text": message "type": "text"})
        all_messages.append({"role": "user", "content": user_content})

    # send to model
    chat_coroutine = bp.openai_client.chat.completions.create(
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

# app is not defined
@bp.route('/process_pdf', methods=['POST'])
async def process_pdf():
    uploaded_file = (await request.files)['file']
    user_message = (await request.form).get('message', '')

    pdf_data = await uploaded_file.read()
    doc = fitz.open(stream=pdf_data, filetype="pdf")

    partial_answers = []

    for i in range(len(doc)):  # One page per iteration
        subdoc = fitz.open()
        subdoc.insert_pdf(doc, from_page=i, to_page=i)

        # Convert that single-page doc to image
        page = subdoc[0]
        pil_image = await convert_pdf_page_to_image(page)
        img_base64 = await image_to_base64(pil_image)

        # Call AI model
        result = await call_model_on_image(img_base64, user_message)
        partial_answers.append(result)

    # YUBI; this should ask model to group all information together
    final_prompt = "Enter final prompt here"
    # Final aggregation step
    final_answer = await summarize_answers(partial_answers, final_prompt)

    # what is jsonify?
    return jsonify({"answer": final_answer})