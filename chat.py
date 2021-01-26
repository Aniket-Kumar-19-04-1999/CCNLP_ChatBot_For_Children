import random
import json
import os
import torch
import tkinter

from tkinter import *
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


   
   
   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json',encoding="utf8") as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def clear():
    os.system( 'cls' )
	
def chatbot_response(sentence):
	
    if (sentence=="quit"):
        return ("Thank You")

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return(random.choice(intent['responses']))
    else:
        return ("I do not understand...")
        
 
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " +  msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END,"Shinchan: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
        if(res=="Thank You"):
            base.destroy()
    
clear() 
base = Tk()
base.title("Chat-Bot For Children")
base.geometry("1000x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="30", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=976,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=970)
EntryBox.place(x=128, y=401, height=90, width=850)
SendButton.place(x=6, y=401, height=90)

base.mainloop()