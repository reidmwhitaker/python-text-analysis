import nltk
import re

nltk.download('webtext')
document = nltk.corpus.webtext.open('grail.txt').read()

print(document[:1000])
print("\n\n")
snippet = document.split("\n")[8]
print(snippet)
re.search(r'coconuts', snippet)
#print(" 1 \n")
re.search(r'[a-z]', snippet)
#print(" 2 \n")
re.search(r'[a-z]!', snippet)
re.findall(r'(?:ARTHUR: )(.+)', document)[0:10]

p = re.compile(r'(?P<name>[A-Z ]+)(?:: +)(?P<line>.+)')
match = re.search(p, document)
print(match)
print(match.group('name'))
print(match.group('line'))

matches = re.findall(p, document)
chars = set([x[0] for x in matches])
print(chars)
print(len(chars))

chars_dict = {}
for c in chars:
    chars_dict[c] = re.findall(r'(?:' + c + ': )(.+)',document)
chars_dict_2 = {}
for c in chars:
    chars_dict_2[c] = [x[1] for x in matches if x[0]==c]

#chars_dict_3 = {}
#for n in chars:
#    chars_dict_3[n] = [x[1] for x in matches if x[0] == n]

print(chars_dict)
print(chars_dict["ARTHUR"])
print(chars_dict_2["ARTHUR"])
#print(chars_dict == chars_dict_2)
#print(chars_dict_3 == chars_dict_2)

arthur = ' '.join(chars_dict["ARTHUR"])
print(arthur[0:100])
nltk.download('punkt')
from nltk import word_tokenize, sent_tokenize
tokens = (word_tokenize(arthur))
print(len(tokens))
print(len(set(tokens)))

from nltk import collocations
fd = collocations.FreqDist(tokens)
print(fd.most_common()[:10])

#Challenge 2
nltk.download("stopwords")
def rem_punc_stop(text_string):
    from string import punctuation
    from nltk.corpus import stopwords

    for char in punctuation:
        text_string = text_string.replace(char, "")

    #return text_string
    toks = word_tokenize(text_string)
    toks_reduced = [x for x in toks if x.lower() not in stopwords.words('english')]
    return toks_reduced

def rem_punc_stop2(text_string):
    from string import punctuation
    from nltk.corpus import stopwords

    for i in punctuation:
        text_string = text_string.replace(i,"")
    #return text_string
    toks =word_tokenize(text_string)
    reduced_toks = [word for word in toks if word.lower() not in stopwords.words('english')]
    return reduced_toks

print(rem_punc_stop(arthur) == rem_punc_stop2(arthur))
print(rem_punc_stop(arthur))
#print(rem_punc_stop2(arthur))
#print(arthur)

tokens_reduced = rem_punc_stop(arthur)
fd2 = collocations.FreqDist(tokens_reduced)
print(fd2.most_common()[:10])

measures = collocations.BigramAssocMeasures()
c = collocations.BigramCollocationFinder.from_words(tokens_reduced)
c.nbest(measures.pmi, 10)
print(c.nbest(measures.likelihood_ratio, 10))

sents = sent_tokenize(arthur)
print(sents[0:10])

nltk.download("averaged_perceptron_tagger")
from nltk import pos_tag

toks_and_sents = [word_tokenize(s) for s in sent_tokenize(arthur)]
tagged_sents = [pos_tag(s) for s in toks_and_sents]

print()
print(tagged_sents[4])

#Challenge 3
def freq_pos(test_string):
    toks = [word_tokenize(s) for s in sent_tokenize(arthur)]
    tagged_sents = [pos_tag(s) for s in toks_and_sents]
    
