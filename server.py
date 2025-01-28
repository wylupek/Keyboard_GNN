from fastapi import FastAPI, Request
import uvicorn
from utils import database_utils
from utils.inference import inference as inference_fun
from utils.train import train as train_fun, LoadMode

from utils.current_setup import kwargs as current_setup_kwargs
from copy import deepcopy

app = FastAPI()

@app.get("/")
def hello():
    return {"message": "Hello, World!"}


@app.post("/upload_tsv")
async def upload_tsv(request: Request, username: str):
    tsv_data = await request.body()
    tsv_str = tsv_data.decode('utf-8')

    if not database_utils.load_str(tsv_str, username, skip_header=True):
        return {"message": "Error: Couldn't load the data"}
    database_utils.save_tsv(content=tsv_str, base_path="./datasets/training/", username=username)

    return {"message": "TSV data received successfully"}


@app.post("/train")
async def train(request: Request, username: str):
    tsv_data = await request.body()
    tsv_str = tsv_data.decode('utf-8')
    username_id = int(username)

    if not database_utils.load_str(tsv_str, username_id, skip_header=True):
        return {"message": "Error: Couldn't load the data"}
    database_utils.save_tsv(content=tsv_str, base_path="./datasets/training/", username=username_id)

    kwargs = deepcopy(current_setup_kwargs)
    kwargs.pop("threshold", None)  # remove the threshold
    train_fun('keystroke_data.sqlite', username_id, test_train_split=0, positive_negative_ratio=2, offset=10,
             **kwargs)

    return {"message": "TSV data received successfully. Training succeeded."}


@app.post("/inference")
async def inference(request: Request, username: str):
    tsv_data = await request.body()
    tsv_str = tsv_data.decode('utf-8')

    score, prediction = inference_fun(username, tsv_str)
    return {"message": "TSV data received successfully. Inference succeeded",
            "score": score,
            "prediction": prediction}


def main():
    import os
    from dotenv import load_dotenv
    load_dotenv()

    ssl_key_file = "ssl/key.pem"
    ssl_cert_file = "ssl/cert.pem"
    ssl_key_passphrase = os.getenv("SSL_PASSPHRASE")

    uvicorn.run(app,
                host="0.0.0.0",
                port=8000,
                ssl_keyfile=ssl_key_file,
                ssl_certfile=ssl_cert_file,
                ssl_keyfile_password=ssl_key_passphrase)


if __name__ == "__main__":
    database_utils.drop_table()
    database_utils.create_table()
    database_utils.load_dir("datasets/training")
    main()

