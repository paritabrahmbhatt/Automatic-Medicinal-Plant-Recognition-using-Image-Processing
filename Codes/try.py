from classifynew import plant
output = plant

import webbrowser

if output == 'Nyctanthes Arbor - Tristis':
    webbrowser.open("file:///C:/Users/GEC DAHOD/Parita_finalyear_project/project-run/final/html/Tristis.html", new=1)

elif output == 'Lemon Leaf':
    webbrowser.open("file:///C:/Users/GEC DAHOD/Parita_finalyear_project/project-run/final/html/lemon.html", new=1)

elif output == 'Neem':
    webbrowser.open("file:///C:/Users/GEC DAHOD/Parita_finalyear_project/project-run/final/html/neem.html", new=1)

elif output == 'Basil':
        webbrowser.open("file:///C:/Users/GEC DAHOD/Parita_finalyear_project/project-run/final/html/basil.html", new=1)

elif output == 'Curry leaf':
        webbrowser.open("file:///C:/Users/GEC DAHOD/Parita_finalyear_project/project-run/final/html/curry.html", new=1)

elif output == 'Mango Leaf':
        webbrowser.open("file:///C:/Users/VYOMA/Desktop/project/final/html/mango.html", new=1)

else:
    print("Unrecognized")
