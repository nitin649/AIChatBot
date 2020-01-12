#creating input data from user input into bag of words
import tensorflow as tf
import json
import random
import numpy as np
#import tflearn
import nltk
import pickle
#from tensorflow.layers import Dense,Dropout,Flatten
from nltk.stem.lancaster import LancasterStemmer


stemmer=LancasterStemmer()

with open("E:/Python/Scripts/intents.json") as file:
    data=json.load(file)

model1=tf.keras.models.load_model('finalchatbot2')

try:
    with open("datachat2.pickle",'rb') as f:
        words,labels,training,ouput=pickle.load(f)
        print(words)
        print(labels)
except Exception:
        print("file cant be loaded")

def bag_of_words(s,words):
    bag=[0 for _ in range(len(words))]

    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w==se:
                bag[i]=1
    return np.array(bag)

#main method for chatting and prediction
def chat():
    print("Start talking with the bot(quit to stop)")
    while True:
        inp=input("you: ")
        if inp.lower()=="quit":
            break

        result=model1.predict([[bag_of_words(inp,words)]])
        print(result.shape)
        result_index=np.argmax(result)
        print(result_index)
        tag=labels[result_index]
        #print(result)
        #print(tag)

        if result[0,result_index] > 0.7:#if we get the prob less than 70 percent means computer do not get our question
            for tg in data['intents']:
                if tg['tag']==tag:
                    response=tg['responses']
            print(random.choices(response))
        else:
            print("dont understand your question")
    
        
#print(model1)
#print(len(words))
#print(labels)"""
