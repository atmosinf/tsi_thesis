from django.shortcuts import render

# Create your views here.

def videopred(request):
    return render(request, 'videopred.html')