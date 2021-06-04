import pickle

def save_the_text(sentence):
    input_text = sentence
    pickle_out = open("sentences.pickle", "wb")
    pickle.dump(input_text, pickle_out)
    pickle_out.close()






