from django.http import HttpResponseRedirect


##### redirects homepage to CV ####
def index(request):
      return HttpResponseRedirect('/CV/')
