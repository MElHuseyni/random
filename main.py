from fastapi import FastAPI, Request
import uvicorn
from planogram import check_panogram
import base64
import numpy as np
import cv2
import logging
import nest_asyncio


nest_asyncio.apply()

app = FastAPI()
my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)


@app.post("/get_planogram")
async def planogram_api(request: Request):
    req = await request.json()

    img_str = req['refrenceImg']
    jpg_original = base64.b64decode(img_str)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    reference_image = cv2.imdecode(jpg_as_np, flags=1)
    
    img_str2 = req['currentImg']
    jpg_original2 = base64.b64decode(img_str2)
    jpg_as_np2 = np.frombuffer(jpg_original2, dtype=np.uint8)
    current_image = cv2.imdecode(jpg_as_np2, flags=1)

    roi=req['rOIS']
    
    prodcut_inside_size=[]
    contours={}
    for i in roi:
        x=i["Polygon"].split(",")
        contours[i["rOIId"]] = [int(x[0]),int(x[1]),int(x[2]),int(x[3])]
        prodcut_inside_size.append(i["productSizeInsideArea"])        

    print("productSizeInsideArea",prodcut_inside_size)

    output={}


    contour_planogram = check_panogram(current_image, reference_image, contours,prodcut_inside_size)

    output["planogram"] = contour_planogram
    output["error"] = "no error"

    return output


if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        reload=True,
        port=3001,
    )
    