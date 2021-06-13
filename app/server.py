from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
import uvicorn, aiohttp, asyncio
from io import BytesIO, StringIO
from fastai.vision.all import *
from utils import *
import base64
import pdb

export_file_name = 'export.pkl'

path = Path(__file__).parent

templates = Jinja2Templates(directory='app/templates')
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)


async def setup_learner():
#     await download_file(export_file_url, path/'models'/export_file_name)
    defaults.device = torch.device('cpu')
    learn = load_learner(path/'saved'/export_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = await (data["file"].read())
    img = PILImage.create(BytesIO(img_bytes))
    orig_size = img.size   
    fi = float(data["fi"])
    Style = data["Style"]
    
    learn.dls.valid.after_item  = Pipeline([RatioResize(int(256*fi)), ToTensor()])
    learn.model_dir = '.'
    learn.load(Style, device='cpu', with_opt=False)
    
    pred_img = learn.predict(img)[0]
    pred_img = PILImage.create(pred_img).resize(orig_size)
    pred_img_bytes = pred_img.to_bytes_format()
    img_str = base64.b64encode(pred_img_bytes).decode()
    img_str = "data:image/png;base64," + img_str

    return templates.TemplateResponse('output.html', {'request' : request, 'b64val' : img_str})


@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app = app, host="0.0.0.0", port=8080)
