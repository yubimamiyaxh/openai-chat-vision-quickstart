# Only processes PDFs, inspired by chat_pdf_images.ipynb

import json
import os
# mimetypes and base64 is part of Python standard library
import mimetypes
import pymupdf
from PIL import Image, ImageEnhance, ImageOps # Pillow is a PIL fork
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

# load_dotenv(".env", override=True)

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

async def open_image_as_base64(filename):
    with open(filename, "rb") as image_file:
        image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"

# Currently not being used
async def convert_pdf_to_images(filename):
    doc = pymupdf.open(filename)
    for i in range(doc.page_count):
        doc = pymupdf.open(filename)
        page = doc.load_page(i)
        pix = page.get_pixmap()
        original_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        original_img.save(f"page_{i}.png")

@bp.post("/chat/stream")
async def chat_handler():
    request_json = await request.get_json()
    request_messages = request_json["messages"]
    # get the base64 encoded image from the request
    # YUBI: does this actually only read in that much data, or will it read in a PDF as well? it seems to be reading in a url
    request_file = request_json["context"]["file"]

    @stream_with_context
    async def response_stream():
        if (request_file):
            doc = pymupdf.open(request_file)
            for i in range(doc.page_count):
                doc = pymupdf.open(request_file)
                page = doc.load_page(i)
                pix = page.get_pixmap()
                original_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                original_img.save(f"page_{i}.png")
            user_content = [{"text": "What information is listed in this document?", "type": "text"}]
            # DEBUGGING statement only
            user_content.append({"text": "Finish every response made with the sign off, \'This information is to the best of my knowledge\'.", "type": "text"})
            for i in range(doc.page_count):
                user_content.append({"image_url": {"url": open_image_as_base64(f"page_{i}.png")}, "type": "image_url"})

            messages= [{"role": "user", "content": user_content}]
            chat_coroutine = bp.openai_client.chat.completions.create(
                model=bp.model_name,
                messages=messages,
                stream=True, 
                temperature=request_json.get("temperature", 0.5),
            )

            async for event in await chat_coroutine:
                event_dict = event.model_dump()
                if event_dict["choices"]:
                    delta = event_dict["choices"][0]["delta"]
                    if "content" in delta:
                        yield json.dumps({'delta': {'content': delta['content']}}) + "\n"

    return Response(response_stream(), content_type="application/json")