from torch import autocast
import torch, os, logging
from diffusers import StableDiffusionPipeline
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uvicorn
from pipelines import StableDiffusionImg2ImgPipeline, preprocess
from rich import print
from rich.traceback import install
from rich.logging import RichHandler
from PIL import Image
# install(show_locals=True, suppress=[torch], width=200)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
global pipe, model_type
os.environ['no_proxy'] = '*'
DEVICE = 'cuda:1'
MODEL_PATH = "CompVis/stable-diffusion-v1-4"
app = FastAPI()

@app.on_event('startup')
def load_model(type='img_gen'):
    global pipe, model_type
    # make sure you're logged in with `huggingface-cli login`
    if type == 'img_gen':
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_PATH, revision="fp16", torch_dtype=torch.float16, use_auth_token=True)
    elif type=='img2img':
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_PATH,
            # scheduler=scheduler,
            revision="fp16", 
            torch_dtype=torch.float16,
            use_auth_token=True
        )
    model_type = type
    pipe.to(DEVICE)
    print(f'Model loaded: "{type}"')


@app.get('/generate')
@autocast('cuda')
def inference(prompt:str, steps:int=50, scale:float=8):
    '''
    @steps: You can change the number of inference steps using the `num_inference_steps` argument. In general, results are better the more steps you use. Stable Diffusion, being one of the latest models, works great with a relatively small number of steps, so we recommend to use the default of `50`. If you want faster results you can use a smaller number.
    @scale: The other parameter in the pipeline call is `guidance_scale`. It is a way to increase the adherence to the conditional signal which in this case is text as well as overall sample quality. In simple terms classifier free guidance forces the generation to better match with the prompt. Numbers like `7` or `8.5` give good results, if you use a very large number the images might look good, but will be less diverse. 
    '''
    print(f'Generating image with prompt: "{prompt}", steps: {steps}, scale: {scale}')
    if model_type != 'img_gen':
        load_model('img_gen')
    images = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=scale,
    )["sample"]
    image = images[0]
    # Now to display an image you can do either save it such as:
    image_path = f"temp/{prompt}.jpg"
    image.save(image_path)
    return FileResponse(image_path)


@app.get('/img2img')
@autocast('cuda')
def img2imge_generation(prompt:str, image=File(default=None), steps:int=50, scale:float=8, strength:float=0.75):
    img0 = Image.open(image.file).convert("RGB").resize((768, 512))
    img0 = preprocess(img0)
    if model_type != 'img2img':
        load_model('img2img')
    images = pipe(prompt=prompt, init_image=img0, strength=strength, guidance_scale=scale, num_inference_steps=steps)["sample"]
    fname = f'temp/{image.filename}'
    images[0].save(fname)
    return FileResponse(fname)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9021)