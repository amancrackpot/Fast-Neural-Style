from io import BytesIO, StringIO
from fastai.vision.all import *
from utils import *
import pathlib
import platform
import streamlit as st
import time
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

path = Path(__file__).parent
learn = load_learner(path/'saved'/'export.pkl')
learn.model_dir = '.'

styles = sorted(['Anime','Depp','Inked','Deep','Paper','Ice','Fire Ice','Fire','Girl','Bird',
          'Butterfly','Picasso BW','Bold','Thunder','Vibrant','Sketch','Bad','Graffiti',
          'Colors','Cartoon','Rain','Krishna','Comic','Venetia','Buddha','Clocktower',
          'Lion','Neon','Watercolor','Little Village','Scream','Mosaic','Waterfall',
          'Candy','Udnie','Picasso','Strokes','Starry','Van Gogh','Magma','Cuphead',
          'Poly','Patterns','Crayon','Loose'])


st.set_page_config(layout='wide')
st.markdown(f""" <style>
.reportview-container .main .block-container{{
padding-top: 0rem;
padding-bottom: 0rem;
}}

div.stButton > button:first-child {{
width:100%;
}}
</style> """, unsafe_allow_html=True)



def style2pth(style):
    file = style.lower().replace(' ','_')+'.jpg'
    local_img_path = path/'static'/'images'/'style'/file
    return local_img_path

@st.cache(allow_output_mutation=True)
def get_images(styles):
    paths = np.array([style2pth(i) for i in styles])
    return np.array([PILImage.create(i) for i in paths])

style_imgs = get_images(styles)



def create_containers(img):
    col1, col2 = st.beta_columns(2)
    
    col1.header('Uploaded Image')
    col1.image(img, use_column_width=True)
    
    col2.header('Styled Image')
    pred_cont = col2.image('https://cdn.dribbble.com/users/563824/screenshots/4155980/untitled-11.gif')
    st.markdown('<hr>',unsafe_allow_html=True)    
    return pred_cont

def show_results(img, fi, style, pred_cont):        
    fi = 1.0+(fi/2)
    style_path = style.lower().replace(' ','')+'256'
    orig_size = img.size
    learn.dls.valid.after_item  = Pipeline([RatioResize(int(256*fi)), ToTensor()])
    learn.load(path/'saved'/style_path, device='cpu', with_opt=False)    
    pred_img = learn.predict(img)[0]
    pred_img = PILImage.create(pred_img).resize(orig_size)   
    time.sleep(3)
    pred_cont.image(pred_img) 





st.title('Neural Style Transfer')
st.markdown('<hr>',unsafe_allow_html=True)
with st.sidebar:
    st.title('Style Settings')
    menu = ['Demo','Upload', 'URL']
    choice = st.selectbox("Select Image Source", menu)
    cont = st.beta_container()
    
    col1, col2 = st.beta_columns(2)
    with col1:
        style = st.selectbox("Select your style", styles)
    with col2:
        fi = st.slider('Fineness',min_value=-1.,max_value=+1.,value=0.,step=0.1,format='%g')
        
    with st.beta_expander(f'Selected Style : {style}',True):
        st.image(style_imgs[styles.index(style)])
    btn = st.button('Render')



if choice == 'Upload':
    uploaded_file = cont.file_uploader("Upload an Image to style...", type=["jpg",'png','jpeg'])
    
    if btn and uploaded_file is not None:        
        try:
            img = PILImage.create(uploaded_file)
            pred_cont = create_containers(img)
            show_results(img, fi, style, pred_cont)
        except:
            st.error('Invalid File uploaded')

     
            
elif choice == 'URL':
    url = cont.text_input("Specify Image URL to style...")
        
    if btn and url is not '':
        try:
            content = requests.get(url).content
            img = BytesIO(content)
            img = PILImage.create(img)
            pred_cont = create_containers(img)
            show_results(img, fi, style, pred_cont)
        except:
            st.error('Invalid URL specified')


else:
    cont.write('Style an already available demo image')
    url = 'https://github.com/lengstrom/fast-style-transfer/raw/master/examples/content/chicago.jpg'
    if btn:
        content = requests.get(url).content
        img = BytesIO(content)
        img = PILImage.create(img)
        pred_cont = create_containers(img)
        show_results(img, fi, style, pred_cont)


 

st.header('Available Style Options')
st.write('')
cols = st.beta_columns(4)

cols[2].image(style_imgs[0])
for i in range(1,12):
    cols[0].image(style_imgs[i],caption=styles[i])
for i in range(12,23):
    cols[1].image(style_imgs[i],caption=styles[i])
for i in range(23,34):
    cols[2].image(style_imgs[i],caption=styles[i])
for i in range(34,45):
    cols[3].image(style_imgs[i],caption=styles[i])
       


    

