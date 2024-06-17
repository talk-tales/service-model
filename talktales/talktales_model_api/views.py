from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from .utils import PredictModel

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

            return JsonResponse({
                "success": True,
                "message": "Success predict audio",
                'data': a
            })
        except Exception as e:
            print(e)
            return JsonResponse({
                "success": False,
                "message": "Fail predict audio",
                'data': f'{e}'
            })
    
    def get(self, request):
        return JsonResponse({'result': "result get"})

