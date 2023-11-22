from datasets import load_dataset

#Yoda section of code, a lot taken from https://github.com/yevbar/Yoda-Script/blob/master/yoda.py
import spacy
punctuation = [',', '.', ';', '?']
nlp = spacy.load("en_core_web_sm")
import en_core_web_sm
nlp = en_core_web_sm.load()
comma = nlp('Hello, World')[1]

def sentify(text):
    output = []
    doc = nlp(text)
    for sent in doc.sents:
        sentence = []
        end_punctuation = sent[-1]
        for clause in clausify(sent[:-1]):
            sentence.append(yodafy(clause))
        sentence[-1].append(end_punctuation)
        output.append(sentence)
    return output

def clausify(sent):
    output = []
    cur = []

    prev_token = None
    for token in sent:
        if prev_token!= None and prev_token.dep_ == 'punct' and prev_token.text in punctuation and token.dep_ != "amod":
            #(token.dep_ == 'cc' or (token.dep_ == 'punct' and token.text in punctuation)):
            output.append(cur)
            #output.append([token])
            cur = [token]
        else:
            cur.append(token)
        prev_token = token
    if cur != []:
        output.append(cur)
    return output

def yodafy(clause):
    new_array = []
    state = False
    prev_token = None
    flag = False
    for token in clause:
        if state:
            new_array.append(token)
        elif not state and prev_token != None and (prev_token.dep_ == "ROOT" or prev_token.dep_ == "aux") and token.dep_ != "nsubj" or flag:
            state = True
            new_array.append(token)
        elif not state and prev_token != None and (prev_token.dep_ == "ROOT" or prev_token.dep_ == "aux"):
            flag = True
        prev_token = token
    if len(new_array) > 0 and new_array[len(new_array)-1].dep_ != 'punct':
        new_array.append(comma)
    prev_token = None
    for token in clause:
        new_array.append(token)
        if (token.dep_ == "ROOT" or token.dep_ == "aux") and flag == False:
            break
        elif prev_token != None and (prev_token.dep_ == "ROOT" or prev_token.dep_ == "aux") and flag:
            break
        prev_token = token
    return new_array

def yoda(string_):
    string = []
    #end_punctuation = string_[-1]
    yodafied = sentify(string_)
    for sentence in yodafied:
        sentence_ = ""
        for clause in sentence:
            for token in clause:
                if token.dep_ == 'NNP' or token.dep_ == 'NNPS' or token.text == 'I':
                    sentence_ += token.text + " "
                elif sentence_ == "" and token.dep_ == 'neg':
                    sentence_ += "Not" + " "
                elif sentence_ == "":
                    sentence_ += token.text[0].upper() + token.text[1:] + " "
                elif token.dep_ == 'punct':
                    sentence_ = sentence_[:len(sentence_)-1] + token.text + " "
                else:
                    sentence_+=token.text.lower() + " "
        string.append(sentence_)
    return "".join(string) #+ end_punctuation


#End of yoda functions
#Begin dataset functions

def get_dataset():
    x = load_dataset("generics_kb", name = 'generics_kb')['train']
    return x


def format_data_yoda(dataset, indices):
    output = []
    for i in indices:
        output.append("User: Tell me about " + dataset[i]['term'] + ".\n" +
                      "Yoda: " + yoda(dataset[i]['generic_sentence']))
    return output


def format_data_human(dataset, indices):
    output = []
    for i in indices:
        output.append("User: Tell me about " + dataset[i]['term'] + ".\n" +
                      "Human: " + dataset[i]['generic_sentence'])
    return output
        
def format_data_bert_model(dataset, indices_yoda, indices_human):
    output = []
    labels = []
    for i in indices_yoda:
        output.append(yoda(dataset[i]['generic_sentence']))
        labels.append("Yoda")
    for i in indices_human:
        output.append(dataset[i]['generic_sentence'])
        labels.append("Human")
    return output, labels
                      

        
        

