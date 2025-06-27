# Python script sends multiple calls to the model, one for each image

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

# FUNCTION: open image converted from PDF as base64 image
async def open_image_as_base64(filename):
    with open(filename, "rb") as image_file:
        image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"

# FUNCTION: convert PDF to images
async def convert_pdf_to_images(filename):
    # This function converts a PDF file to images and saves them as PNG files
    doc = pymupdf.open(filename)
    img_names_list = []
    # Iterate through each page in the PDF and save the image
    for i in range(doc.page_count):
        doc = pymupdf.open(filename)
        page = doc.load_page(i)
        pix = page.get_pixmap()
        original_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_names_list.append(f"page_{i}.png")
        # I need to find a way to delete all of the saved images after the request is done
        # Or maybe it will just replace them when you run it again
        # Save the image as a PNG file in the current directory (src), switch to a different folder
        original_img.save(f"page_{i}.png")

    # return the list of image names
    return img_names_list

# Function: data pre-processing of all images to increase contras
async def preprocess_image(imagename):
    try:
        # Open the image
        image = Image.open(imagename)

        # Convert to grayscale
        gray_image = ImageOps.grayscale(image)

        # Increase contrast
        contrast_enhancer = ImageEnhance.Contrast(gray_image)
        enhanced_image = contrast_enhancer.enhance(2.0)  # You can adjust factor

        # Save the processed image (overwrite original)
        enhanced_image.save(imagename)

    except Exception as e:
        # debugging statement
        print(f"Error processing image '{imagename}': {e}")
    return

@bp.post("/chat/stream")
async def chat_handler():
    request_json = await request.get_json()
    request_messages = request_json["messages"]
    # get the base64 encoded image from the request
    # YUBI: does this actually only read in that much data, or will it read in a PDF as well? it seems to be reading in a url
    request_file = request_json["context"]["file"]

    @stream_with_context
    async def response_stream():
        all_messages = request_messages[0:-1]
        final_user_message = request_messages[-1]["content"]

        # If PDF
        if request_file and mimetypes.guess_type(request_file)[0] == "application/pdf":
            image_names = await convert_pdf_to_images(request_file)

            # TEST ON ONLY FIRST PAGE FOR DEBUGGING
            curr_image = await open_image_as_base64(image_names[0])

            try:
                await preprocess_image(curr_image)

                user_content = [
                    {"text": final_user_message, "type": "text"},
                    {"image_url": {"url": curr_image, "detail": "auto"}, "type": "image_url"}
                ]
                messages = all_messages + [{"role": "user", "content": user_content}]
                
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

            except Exception as e:
                yield json.dumps({'error': str(e)}) + "\n"
            
            # comment out for DEBUGGING
            '''
            for i, image_name in enumerate(image_names):
                try:
                    await preprocess_image(image_name)
                    image_base64 = await open_image_as_base64(image_name)

                    user_content = [
                        {"text": final_user_message, "type": "text"},
                        {"image_url": {"url": image_base64, "detail": "auto"}, "type": "image_url"}
                    ]

                    messages = all_messages + [{"role": "user", "content": user_content}]
                    
                    yield f"data: {json.dumps({'content': f'--- Page {i+1} ---'})}\n\n"

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

                except Exception as e:
                    yield json.dumps({'error': str(e)}) + "\n"
            '''

        # If single image
        elif request_file and mimetypes.guess_type(request_file)[0].startswith("image/"):
            try:
                await preprocess_image(request_file)

                user_content = [
                    {"text": final_user_message, "type": "text"},
                    {"image_url": {"url": request_file, "detail": "auto"}, "type": "image_url"}
                ]
                messages = all_messages + [{"role": "user", "content": user_content}]
                
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

            except Exception as e:
                yield json.dumps({'error': str(e)}) + "\n"

        # No file (text only)
        else:
            try:
                messages = all_messages + [request_messages[-1]]

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

            except Exception as e:
                yield json.dumps({'error': str(e)}) + "\n"

    return Response(response_stream(), content_type="application/json")