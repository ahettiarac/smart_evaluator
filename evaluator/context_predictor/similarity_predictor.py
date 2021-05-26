from gensim.similarities import SoftCosineSimilarity,WordEmbeddingSimilarityIndex,SparseTermSimilarityMatrix
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess
from gensim.models import FastText
from collections.abc import Iterable

class SimilarityPredictor:

    fasttext_model = None
    dictionary = None

    def __init__(self,train=True,sentences=None):
        if train:
            if sentences != None and isinstance(sentences, Iterable):
                self.fasttext_model = FastText(sentences=sentences,epochs=50)
                self.fasttext_model.save("fasttext_model")
            else:
                ValueError("sentence should be valid list of iterable if train flag set to true")  
        else:          
            self.fasttext_model.load("fasttext_model")

    def prepare_for_prediction(self,user_abstract,teacher_abstract):
        documents_predicts = [user_abstract,teacher_abstract]
        self.dictionary = corpora.Dictionary([simple_preprocess(abstract) for abstract in documents_predicts])
        try:
            f = open('similarity_matrix.pkl')
            similarity_matrix = SparseTermSimilarityMatrix.load('similarity_matrix')
        except FileNotFoundError:
            similarity_index = WordEmbeddingSimilarityIndex(self.fasttext_model)
            similarity_matrix = SparseTermSimilarityMatrix(similarity_index, self.dictionary)
            similarity_matrix.save('similarity_matrix')
        finally:
            f.close()        
        student_abstract = self.dictionary.doc2bow(simple_preprocess(user_abstract))
        teacher_abstract = self.dictionary.doc2bow(simple_preprocess(teacher_abstract))
        softcossim = SoftCosineSimilarity([student_abstract,teacher_abstract],similarity_matrix)
        print("Soft cossine index size: {}".format(len(softcossim.index)))
        return softcossim[student_abstract]
        