import pickle

def getText():
    pickle_in = open("sentences.pickle", "rb")
    output_text = pickle.load(pickle_in)

    return output_text

print(getText())