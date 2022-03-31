from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import sys
import traceback
import base64

mount_path = '/mnt/data'
module_path = os.path.join(mount_path,'neurips-2020-sevir/src')
library_path = os.path.join(mount_path, 'venv/lib/python3.7/site-packages')
sys.path.insert(0,library_path)
sys.path.insert(0,module_path)


from api.AnalyseNowCast import visualize_result
from mangum import Mangum

app = FastAPI(title="SevirApp")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/hello')
def hello():
  return {"message": "Hello World"}

@app.get('/event')
def event_query(request: Request, idx_id: str = ""):
  
  file_name = f"image_{int(datetime.now().timestamp())}.png"
  save_path = os.path.join(mount_path, 'export')
  file_path = os.path.join(save_path, file_name)
  try:
    fig,ax = plt.subplots(13,2,figsize=(5,20))
    fig.delaxes(ax[12][1])
    visualize_result([gan_model],x_test,y_test,int(idx_id),ax,labels=['cGAN+MAE'],save_path=file_path)
    with open(file_path, "rb") as file:
        image_bytes: bytes = base64.b64encode(file.read())
    return {"data": image_bytes}
  except Exception as e:
    message = traceback.format_exc()
    print(message)
    return "An internal error occurred"

handler = Mangum(app)
