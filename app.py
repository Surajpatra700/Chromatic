from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline
import os
from dotenv import load_dotenv
import replicate

app = FastAPI()

# Loading environment variables
load_dotenv()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the frontend template
templates = Jinja2Templates(directory="templates")

# This Function generates lyrics using Hugging Face's GPT-NEO model
def generate_lyrics(prompt):
    # Initializing text generation pipeline with GPT-NEO model
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')  #EleutherAI/gpt-neo-1.3B #distilgpt2
    # Generate lyrics based on the prompt
    response = generator(prompt, max_length=50, temperature=0.7, do_sample=True)
    # Extracting generated text from response
    output = response[0]['generated_text']
    # Formating the generated lyrics
    cleaned_output = output.replace("\n", " ")
    formatted_lyrics = f"♪ {cleaned_output} ♪"
    return formatted_lyrics

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate-music")
async def generate_music(prompt: str = Form(...), duration: int = Form(...)):
    try:
        lyrics = generate_lyrics(prompt)
        prompt_with_lyrics = lyrics
        print(prompt_with_lyrics)
        output = replicate.run(
            "suno-ai/bark:b76242b40d67c76ab6742e987628a2a9ac019e11d56ab96c4e91ce03b79b2787",
            input={
                "prompt": prompt_with_lyrics,
                "text_temp": 0.7,
                "output_full": False,
                "waveform_temp": 0.7
            }
        )
        print(output)
        music_url = output['audio_out']
        music_path_or_url = music_url
        print(music_path_or_url)
        return JSONResponse(content={"url": music_path_or_url})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8900)


# # Define libraries
# from fastapi import FastAPI, Form, Request
# from fastapi.responses import JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# import os
# from dotenv import load_dotenv
# import replicate
# import openai

# app = FastAPI()

# # Load environment variables
# load_dotenv()

# # Ensure Replicate API key is set
# REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
# if not REPLICATE_API_TOKEN:
#     raise ValueError("Replicate API token not set in environment variables")

# # Ensure OpenAI API key is set
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# if not OPENAI_API_KEY:
#     raise ValueError("OpenAI API key not set in environment variables")

# openai.api_key = OPENAI_API_KEY

# # Mount static files directory
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Load HTML templates
# templates = Jinja2Templates(directory="templates")

# def generate_lyrics(prompt):
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a creative assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=50,
#             temperature=0.7
#         )
#         output = response.choices[0].message['content'].strip()
#         formatted_lyrics = f"♪ {output.replace('\n', ' ')} ♪"
#         return formatted_lyrics
#     except Exception as e:
#         print(f"OpenAI text generation error: {e}")
#         return str(e)

# @app.get("/")
# async def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/generate-music")
# async def generate_music(prompt: str = Form(...), duration: int = Form(...)):
#     try:
#         lyrics = generate_lyrics(prompt)
#         prompt_with_lyrics = lyrics
#         print(prompt_with_lyrics)

#         output = replicate.run(
#             "suno-ai/bark:b76242b40d67c76ab6742e987628a2a9ac019e11d56ab96c4e91ce03b79b2787",
#             input={
#                 "prompt": prompt_with_lyrics,
#                 "text_temp": 0.7,
#                 "output_full": False,
#                 "waveform_temp": 0.7,
#             }
#         )
#         print(output)
#         music_url = output["audio_out"]
#         music_path_or_url = music_url

#         print(music_path_or_url)
#         return JSONResponse(content={"url": music_path_or_url})
#     except Exception as e:
#         print(f"Error in /generate-music: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# # Start the server if this script is run directly
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app:app", host="0.0.0.0", port=9800, reload=True)
