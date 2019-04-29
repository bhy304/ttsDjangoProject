from django.shortcuts import render
import json
from django.views.decorators.csrf import csrf_exempt
from django.core.serializers import serialize
from django.http import HttpResponse
import librosa as lr
from eval import eval_text

def index(request):
    #print("index.......")
    return render(request, 'tts/index.html')

# Ajax
@csrf_exempt # 403 error 제어
def synthesize(request):
    print("view 도착")
    text = request.POST['text'].strip()
    print(text)

    path_dir = '/static/audio/'
    audio_file = path_dir + eval_text(text)
    print(audio_file)
    audio = {"audio":audio_file}
    audio = json.dumps(audio)

    print(audio)
    return HttpResponse(audio, content_type="application/json")
