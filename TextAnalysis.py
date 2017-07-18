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
    toks = [word_tokenize(s) for s in sent_tokenize(test_string)]
    tagged_sents = [pos_tag(s) for s in toks]
    freqs_pos = {}
    for s in tagged_sents:
        for word in s:
            if word[1] in freqs_pos:
                freqs_pos[word[1]] = freqs_pos[word[1]] + 1
            else:
                freqs_pos[word[1]] = 1

    return freqs_pos

print(freq_pos(arthur))

tag_fd = nltk.FreqDist(tag for (word, tag) in [item for sublist in tagged_sents for item in sublist])
print(tag_fd.most_common())

snowball = nltk.SnowballStemmer('english')
print(snowball.stem('running'))
print(snowball.stem('eats'))
print(snowball.stem('embarassed'))
print(snowball.stem('cylinder') + ", " + snowball.stem('cylindrical'))
print(snowball.stem('vacation') + ", " + snowball.stem('vacate'))

nltk.download('wordnet')
wordnet = nltk.WordNetLemmatizer()
print(wordnet.lemmatize('vacation') + ", " + wordnet.lemmatize('vacate'))

tok_red_lem = [snowball.stem(w) for w in tokens_reduced]
fd3 = collocations.FreqDist(tok_red_lem)
print(fd3.most_common()[:15])

from textblob import TextBlob
blob = TextBlob(arthur)
net_pol = 0
for sentence in blob.sentences:
    pol = sentence.sentiment.polarity
    print(pol, sentence)
    net_pol += pol
print()
print("Net polarity of Arthur: ", net_pol)

from gensim import corpora, models, similarities

people = []
speeches = []
for k,v in chars_dict.items():
    people.append(k)
    new_string = ' '.join(v)  # join all dialogue pices
    toks = rem_punc_stop(new_string)  # remove puntuation and stop words, and tokenize
    stems = [snowball.stem(tok) for tok in toks]  # change words to stems
    speeches.append(stems)

#create a Gensim dictionary from the texts
dictionary = corpora.Dictionary(speeches)

#remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
#no_below is absolute # of docs, no_above is fraction of corpus
dictionary.filter_extremes(no_below=2, no_above=.70)

#convert the dictionary to a bag of words corpus for reference
corpus = [dictionary.doc2bow(i) for i in speeches]

#we run chunks of 15 books, and update after every 2 chunks, and make 10 passes
lda = models.LdaModel(corpus, num_topics=6,
                            update_every=2,
                            id2word=dictionary,
                            chunksize=15,
                            passes=10)

lda.show_topics()

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

corpus_lda = lda[corpus_tfidf]
for i, doc in enumerate(corpus_lda): # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    print(people[i],doc)
    print ()

print(lda.show_topics())

with open("./Intro_to_TextAnalysis/King_James_Bible.txt", "r") as f:
    bible = f.read()

from nltk.tokenize import sent_tokenize

bible = sent_tokenize(bible)
bible = [word_tokenize(s) for s in bible]

print(bible[10])

import gensim
model = gensim.models.word2vec.Word2Vec(bible, size=300, window=5, min_count=5, workers=4)
model.train(bible,  total_examples=model.corpus_count, epochs=model.iter)

print(model.most_similar('man'))
print(model.most_similar('woman'))
print(model.most_similar(positive=['king', 'woman'], negative=['man']))


