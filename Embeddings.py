#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info">
#     <h1>Natural Language Processing</h1>
#     <h3>General Information:</h3>
#     <p>Please do not add or delete any cells. Answers belong into the corresponding cells (below the question). If a function is given (either as a signature or a full function), you should not change the name, arguments or return value of the function.<br><br> If you encounter empty cells underneath the answer that can not be edited, please ignore them, they are for testing purposes.<br><br>When editing an assignment there can be the case that there are variables in the kernel. To make sure your assignment works, please restart the kernel and run all cells before submitting (e.g. via <i>Kernel -> Restart & Run All</i>).</p>
#     <p>Code cells where you are supposed to give your answer often include the line  ```raise NotImplementedError```. This makes it easier to automatically grade answers. If you edit the cell please outcomment or delete this line.</p>
#     <h3>Submission:</h3>
#     <p>Please submit your notebook via the web interface (in the main view -> Assignments -> Submit). The assignments are due on <b>Wednesday at 15:00</b>.</p>
#     <h3>Group Work:</h3>
#     <p>You are allowed to work in groups of up to two people. Please enter the UID (your username here) of each member of the group into the next cell. We apply plagiarism checking, so do not submit solutions from other people except your team members. If an assignment has a copied solution, the task will be graded with 0 points for all people with the same solution.</p>
#     <h3>Questions about the Assignment:</h3>
#     <p>If you have questions about the assignment please post them in the LEA forum before the deadline. Don't wait until the last day to post questions.</p>
#     
# </div>

# In[ ]:


'''
Group Work:
Enter the UID of each team member into the variables. 
If you work alone please leave the second variable empty.
'''
member1 = ''
member2 = ''


# 
# # Word2Vec and FastText Embeddings
# 
# In this assignment we will work on Word2Vec embeddings and FastText embeddings.
# 
# I prepared three dictionaries for you:
# 
# - ```word2vec_yelp_vectors.pkl```: A dictionary with 300 dimensional word2vec embeddings trained on the Google News Corpus, contains only words that are present in our Yelp reviews (key is the word, value is the embedding)
# - ```fasttext_yelp_vectors.pkl```: A dictionary with 300 dimensional FastText embeddings trained on the English version of Wikipedia, contains only words that are present in our Yelp reviews (key is the word, value is the embedding)
# - ```tfidf_yelp_vectors.pkl```: A dictionary with 400 dimensional TfIdf embeddings trained on the Yelp training dataset from last assignment (key is the word, value is the embedding)
# 
# In the next cell we load those into the dictionaries ```w2v_vectors```, ```ft_vectors``` and ```tfidf_vectors```.

# In[4]:


import pickle

with open('/srv/shares/NLP/word2vec_yelp_vectors.pkl', 'rb') as f:
    w2v_vectors = pickle.loads(f.read())
    
with open('/srv/shares/NLP/fasttext_yelp_vectors.pkl', 'rb') as f:
    ft_vectors = pickle.loads(f.read())
    
with open('/srv/shares/NLP/tfidf_yelp_vectors.pkl', 'rb') as f:
    tfidf_vectors = pickle.loads(f.read())
    
with open('/srv/shares/NLP/reviews_train.pkl', 'rb') as f:
    train = pickle.load(f)
    
with open('/srv/shares/NLP/reviews_test.pkl', 'rb') as f:
    test = pickle.load(f)
    
reviews = train + test


# ## Creating a vector model with helper functions [30 points]
# 
# In the next cell we have the class ```VectorModel``` with the methods:
# 
# - ```vector_size```: Returns the vector size of the model
# - ```embed```: Returns the embedding for a word. Returns None if there is no embedding present for the word
# - ```cosine_similarity```: Calculates the cosine similarity between two vectors
# - ```most_similar```: Given a word returns the ```top_n``` most similar words from the model, together with the similarity value, **sorted by similarity (descending)**.
# - ```most_similar_vec```: Given a vector returns the ```top_n``` most similar words from the model, together with the similarity value, **sorted by similarity (descending)**.
# 
# Your task is to complete these methods.
# 
# Example output:
# ```
# model = VectorModel(w2v_vectors)
# 
# vector_good = model.embed('good')
# vector_tomato = model.embed('tomato')
# 
# print(model.cosine_similarity(vector_good, vector_tomato)) # Prints: 0.05318105
# 
# print(model.most_similar('tomato')) 
# '''
# [('tomatoes', 0.8442263), 
#  ('lettuce', 0.70699364),
#  ('strawberry', 0.6888598), 
#  ('strawberries', 0.68325955), 
#  ('potato', 0.67841727)]
# '''
# 
# print(model.most_similar_vec(vector_good)) 
# '''
# [('good', 1.0), 
#  ('great', 0.72915095), 
#  ('bad', 0.7190051), 
#  ('decent', 0.6837349), 
#  ('nice', 0.68360925)]
# '''
# 
# ```

# In[ ]:


from typing import List, Tuple, Dict
import numpy as np
import math
   
class VectorModel:
    
    def __init__(self, vector_dict: Dict[str, np.ndarray]):
        self.vector_dict=vector_dict
        #raise NotImplementedError()
        
    def embed(self, word: str) -> np.ndarray:
        my_vector=np.zeros(len(word))
        counter=0
        dic={}                   
        for i in range(0,len(word)):
            if word[i] in dic.keys():
                my_vector[i]=dic[word[i]]
            else:    
                dic[word[i]]=counter 
                my_vector[i]=counter
                counter+=1  
                               
        return my_vector                 
        #raise NotImplementedError()
    
    def vector_size(self) -> int:
        return len(self.vector_dict)
        #raise NotImplementedError()
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(vec1)):
            x = vec1[i] 
            y = vec2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        return sumxy/math.sqrt(sumxx*sumyy)

        #raise NotImplementedError()

    def most_similar(self, word: str, top_n: int=5) -> List[Tuple[str, float]]:
        similar1={}
        similar=[]
        for word_vect,embeding in vector_dict.items():
            similar1[word_vect]=self.cosine_similarity(self.embed(word),embeding)
            
        {k: v for k, v in sorted(similar1.items(), key=lambda item: item[1])}
        keys=similar1.keys()
        values=similar1.values()
        for i in range(0,top_n):
            similar.append((keys[i],values[i]))
            
        return similar    
        #raise NotImplementedError()
        
    def most_similar_vec(self, vec: np.ndarray, top_n: int=5) -> List[Tuple[str, float]]:
        similar1={}
        similar=[]
        for word_vect,embeding in vector_dict.items():
            similar1[ word_vect]=self.cosine_similarity(vec,embeding)
            
        {k: v for k, v in sorted(similar1.items(), key=lambda item: item[1])}
        keys=similar1.keys()
        values=similar1.values()
        for i in range(0,top_n):
            similar.append((keys[i],values[i]))
            
        return similar    
        #raise NotImplementedError()
        


# In[ ]:


# This is a test cell, please ignore it


# ## Investigating similarity A) [10 points]
# 
# We now want to find the most similar words for a given input word for each model (Word2Vec, FastText and TfIdf).
# 
# Your input words are: ```['good', 'tomato', 'restaurant', 'beer', 'wonderful']```.
# 
# For each model and input word print the top three most similar words.

# In[ ]:


input_words = ['good', 'tomato', 'restaurant', 'beer', 'wonderful', 'dinner']

# YOUR CODE HERE
raise NotImplementedError()


# ## Investigating similarity B) [10 points]
# 
# Comment on the output from the previous task. Let us look at the output for the word ```wonderful```. How do the models differ for this word? Can you reason why the TfIdf model shows so different results?

# YOUR ANSWER HERE

# ## Investigating similarity C) [10 points]
# 
# Instead of just finding the most similar word to a single word, we can also find the most similar word given a list of positive and negative words.
# 
# For this we just sum up the positive and negative words into a single vector by calculating a weighted mean. For this we multiply each positive word with a factor of $+1$ and each negative word with a factor of $-1$. Then we get the most similar words to that vector.
# 
# You are given the following examples:
# 
# ```
# inputs = [
#     {
#         'positive': ['good', 'wonderful'],
#         'negative': ['bad']
#     },
#     {
#         'positive': ['tomato', 'lettuce'],
#         'negative': ['strawberry', 'salad']
#     }    
# ]
# ```

# In[ ]:


inputs = [
    {
        'positive': ['good', 'wonderful'],
        'negative': ['bad']
    },
    {
        'positive': ['tomato', 'lettuce'],
        'negative': ['strawberry', 'fruit']
    },
    {
        'positive': ['ceasar', 'chicken'],
        'negative': []
    }    
]
# YOUR CODE HERE
raise NotImplementedError()


# ## Investigating similarity D) [15 points]
# 
# We can use our model to find out which word does not match given a list of words.
# 
# For this we build the mean vector of all embeddings in the list.  
# Then we calculate the cosine similarity between the mean and all those vectors.
# 
# The word that does not match is then the word with the lowest cosine similarity to the mean.
# 
# Example:
# 
# ```
# model = VectorModel(w2v_vectors)
# doesnt_match(model, ['potato', 'tomato', 'beer']) # -> 'beer'
# ```

# In[ ]:


def doesnt_match(model, words):
    # YOUR CODE HERE
    raise NotImplementedError()
    
doesnt_match(model_tfidf, ['vegetable', 'strawberry', 'tomato', 'lettuce'])


# In[ ]:


# This is a test cell, please ignore it


# ## Document Embeddings A) [15 points]
# 
# Now we want to create document embeddings similar to the last assignment. For this you are given the function ```bagOfWords```. In the context of Word2Vec and FastText embeddings this is also called ```SOWE``` for sum of word embeddings.
# 
# Take the yelp reviews (```reviews```) and create a dictionary containing the document id as a key and the document embedding as a value.
# 
# Create the document embeddings from the Word2Vec, FastText and TfIdf embeddings.
# 
# Store these in the variables ```ft_doc_embeddings```, ```w2v_doc_embeddings``` and ```tfidf_doc_embeddings```

# In[ ]:


def bagOfWords(model: EmbeddingModel, doc: List[str]) -> np.ndarray:
    '''
    Create a document embedding using the bag of words approach
    
    Args:
        model     -- The embedding model to use
        doc       -- A document as a list of tokens
        
    Returns:
        embedding -- The embedding for the document as a single vector 
    '''
    embeddings = [np.zeros(model.vector_size())]
    n_tokens = 0
    for token in doc:
        embedding = model.embed(token)
        if embedding is not None:
            n_tokens += 1
            embeddings.append(embedding)
    if n_tokens > 0:
        return sum(embeddings)/n_tokens
    return sum(embeddings)


ft_doc_embeddings = dict()
w2v_doc_embeddings = dict()
tfidf_doc_embeddings = dict()

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


# This is a test cell, please ignore it


# ## Document Embeddings B) [10 points]
# 
# Create a vector model from each of the document embedding dictionaries. Call these ```model_w2v_doc```, ```model_ft_doc``` and ```model_tfidf_doc```.
# 
# Now find the most similar document (```top_n=1```) for document $438$ with each of these models. Print the text for each of the most similar reviews.

# In[ ]:


# First find the text for review 438
def find_doc(doc_id, reviews):
    for review in reviews:
        if review['id'] == doc_id:
            return review['text']
    
doc_id = 438

# Print it
print('Source document:')
print(find_doc(doc_id, reviews))

# YOUR CODE HERE
# Create the models
model_w2v_doc = None
model_ft_doc = None
model_tfidf_doc = None


raise NotImplementedError()

