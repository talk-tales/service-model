from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from .utils import PredictModel
import httpx
import asyncio
from google.cloud import storage
import os
import uuid
from datetime import datetime

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), 'talk-tales-project-storage.json')

def generate_unique_name(extension):
    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    # Generate a random UUID
    random_uuid = uuid.uuid4().hex
    # Combine the timestamp, UUID, and extension to create a unique name
    unique_name = f"{timestamp}_{random_uuid}.{extension}"
    return unique_name

def upload_cs_file(file_buffer, bucket_name="talktales-audio"): 
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    filename = generate_unique_name('wav')
    blob = bucket.blob(f"audio/{filename}")
    file_buffer.file.seek(0)
    blob.upload_from_file(file_buffer.file)
    return filename

async def save_dataset(file, target):
    filename = upload_cs_file(file)
    async with httpx.AsyncClient() as client:
                data = {'label': target, "filename":filename}
                response = await client.post('https://backend-service-bch64czzyq-et.a.run.app/dataset', data=data)



# Create your views here.
class TalkTalesModelApi(APIView):
    parser_classes = [MultiPartParser]
    predictModel = PredictModel()

    def post(self, request, format=None):
        try:
            audio_file = request.FILES['file']
            target = request.data['target']

            if (not audio_file or not target):
                raise Exception("Harus ada audio file dan target label!")

            # print(f"request body : {target}")
            print("[+] Predicting audio...")
            a = self.predictModel.predict(audio_file.read(), target)
            asyncio.run(save_dataset(audio_file, target))

            return JsonResponse({
                "success": True,
                "message": "Success predict audio",
                'data': a
            })
        except UserWarning as e:
            print(e)
            return JsonResponse({
                "success": False,
                "message": "Fail predict audio",
                'data': f'{e}'
            })
    
    def get(self, request):
        return JsonResponse({'result': "result get"})

