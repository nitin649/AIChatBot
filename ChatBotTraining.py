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


words=[]
labels=[]
docs_x=[]
docs_y=[]
for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds=nltk.word_tokenize(pattern)
        #print(wrds)
        words.extend(wrds)
   
        docs_x.append(wrds)
        docs_y.append(intent["tag"])#appendgin tag of the pattern as well
        
    if intent['tag'] not in labels:
        labels.append(intent["tag"])

print(words)
print(labels)
#stemming of word and remove duplicate words
words=[stemmer.stem(w.lower()) for w in words if w!="?"]
print(len(words))
words=sorted(list(set(words)))
print(len(words))
labels=sorted(labels)
training =[]
output=[]
#creating bag of words
out_empty =[0 for _ in range(len(labels))]
#print(out_empty)

for x,doc in enumerate(docs_x):
    bag=[]
    wrds=[stemmer.stem(w) for w in doc]

    for w in words:#if w is present in wrds we append 1 
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    #print(bag)
    output_row=out_empty[:]
    output_row[labels.index(docs_y[x])]=1#here we get the data from docs_y[x] x is the index
    #and then we using the value labels.index(values) place the one where the value is in the list
    #at corresponding location

    training.append(bag)
    output.append(output_row)
    with open("datachat2.pickle",'wb') as f:
        pickle.dump((words,labels,training,output),f)

#print(training)
#print(output)
#print(docs_x)
#print(docs_y)

training_data=np.array(training)
output_data=np.array(output)
#print(training_data.shape)
print(output_data)
print(output_data.shape)
"""print(docs_x)
print(docs_y)
print(labels)
print(words)"""
#print(data['intents creating and training of model

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(64,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(len(output_data[0]),activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


model.fit(training_data,output_data,epochs=100,batch_size=32,validation_split=0.2)
#creating input data from user input into bag of words
"""
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
    
        
print(model1)
print(len(words))
print(labels)"""

model.save("finalchatbot2")


