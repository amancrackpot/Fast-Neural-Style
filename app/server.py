from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
import uvicorn, aiohttp, asyncio
from io import BytesIO, StringIO
from fastai import *
from fastai.vision import *
import base64
import pdb
from utils import *

export_file_url = 'https://www.googleapis.com/drive/v3/files/1z5WRMshbw8Xz38UPU2-Fulk0f1ncU90B?alt=media&key=AIzaSyDvMHW-2yleU8G3OljOhzT49zTtf91xuYU'
export_file_name = 'export.pkl'
classes = ['a', 'b', 'c']

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
    learn = load_learner(path/'saved', export_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = await (data["file"].read())

    img = open_image(BytesIO(img_bytes))
    w, h = img.size   
    fi = float(data["fi"])
    Style = data["Style"]
    size_ = int(256*(w/h)*fi), int(256*fi)

    data_ = ( ImageImageList.from_folder(path=path, ignore_empty=True, recurse=False)
                    .split_none()
                    .label_empty()
                    .transform(size=size_)
                    .databunch(bs=1)
                    .normalize(imagenet_stats, do_y=True) )
    data_.c = 3

    learn.data = data_
    learn.load(Style)
    
    _,img_hr,losses = learn.predict(img)

    im = Image(img_hr.clamp(0,1))

    im_data = image2np(im.data*255).astype(np.uint8)

    img_io = BytesIO()

    PIL.Image.fromarray(im_data).save(img_io, 'PNG')

    img_io.seek(0)

    img_str = base64.b64encode(img_io.getvalue()).decode()
    img_str = "data:image/png;base64," + img_str

    return templates.TemplateResponse('output.html', {'request' : request, 'b64val' : img_str})


@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app = app, host="0.0.0.0", port=8080)
