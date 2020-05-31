from textblob import TextBlob
from newspaper import Article
import nltk
art=Article("https://timesofindia.indiatimes.com/india/coronavirus-in-india-live-updates-pm-modi-address-nation-through-mann-ki-baat/liveblog/76114635.cms")
art.download()
art.parse()
nltk.download('punkt')
art.nlp()
text=art.summary
type(text)
textobj=TextBlob(b)
sen=textobj.sentiment.polarity
sen

if sen==0:
    print("neutal")
else if sen>0:
    print("positive")
else
    print("negative")