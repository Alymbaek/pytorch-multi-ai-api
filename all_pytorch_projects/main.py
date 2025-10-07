from fastapi import FastAPI
import uvicorn
from all_pytorch_projects.text_predict.api import ag_news
from all_pytorch_projects.audio_predict.api import speech_commands, gtzan, urban
from all_pytorch_projects.image_predict.api import mnist, fashion_mnist, cifar_100

app = FastAPI(title='ALL PyTorch Projects')

app.include_router(ag_news.ag_news)
app.include_router(speech_commands.speech_router)
app.include_router(gtzan.gtzan_router)
app.include_router(urban.urban_router)
app.include_router(mnist.mnist_router)
app.include_router(fashion_mnist.fashion_router)
app.include_router(cifar_100.cifar_router)






if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8002)






