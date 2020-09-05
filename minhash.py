from sys import getsizeof
import sys
import os
import string
import numpy as np
from sympy import symbols, solve
import time
import matplotlib.pyplot as plt

# path: path of a txt file
# k: the length of a shingle
# num_of_docs: the max documents the function process
# create a k-shingles set of a txt file 
def shingle_one_doc(path, k):
    shingle_for_one_doc = set()
    with open(path, encoding="latin-1") as file:
        # read a file
        text = file.read()
        text = str(text)
        trimed = text.replace(' ', '').replace('\n', '')
        for i in range(len(trimed) - k + 1):
            # loop through shingles in a file
            shingle_for_one_doc.add(trimed[i: i+k])
    return shingle_for_one_doc

# num_of_docs: the max documents the function process
# path: the path to a directory of files
# k: the length of a shingle
# return three things in a tuple:
# 1.a map from file id (start from 0) to its shringling set.
# 2.a numpy array with the union of all the shingles sets of files.
# 3.a map from file id to filename.
def shingle(num_of_docs, path, k=8):
    docid = 0
    count = num_of_docs
    docid_to_shingling_set = dict()
    docid_to_docname = dict()
    all_tokens = set()
    for filename in os.listdir(path):
        # if has already examed enough files
        if filename.endswith(".txt") and count > 0: 
            # only deal with txt files.
            file_path = os.path.join(path, filename)
            words = shingle_one_doc(file_path, k)
            docid_to_shingling_set[docid] = words
            all_tokens = all_tokens | words
            docid_to_docname[docid] = file_path
            docid += 1
        count = count - 1
    return docid_to_shingling_set, np.array(list(all_tokens)), docid_to_docname

# docid_to_shingling_set: a map from file id (start from 0) to its shringling set.
# all_tokens: a numpy array with the union of all the shingles sets of files.
# return a map from docid to a set of all the indexes of shingles it contains
def create_doc_signature(docid_to_shingling_set, all_tokens):
    docid_to_signature = dict()
    for docid in docid_to_shingling_set:
        #loop through each doc
        words = docid_to_shingling_set[docid]
        wordsSignature = set()
        for i in range(len(all_tokens)):
            # loop through all the tokens
            if all_tokens[i] in words:
                wordsSignature.add(i)
        docid_to_signature[docid] = wordsSignature
    return docid_to_signature

# docid_to_signature: a map from user_id to a set of all the 
# indexes of shingles it contains
# a permutation of all the shingles
# return an numpy array which stores signature for all files for a specific permutation
def min_hash(docid_to_signature, permutation):
    res = np.zeros(len(docid_to_signature))
    for i in range(len(docid_to_signature)):
        # for each doc
        doc_sig = docid_to_signature[i]
        for j in range(len(permutation)):
            # we store j
            if permutation[j] in doc_sig:
                res[i] = j
                break
    return res

# curr: current band of signature
# k: number of buckets
# dict: the bucket we want to modify
# hash each files in the band to k buckets
def create_buckets_for_one_band(curr,k, dict):
    # for each column hash
    # print(curr.shape[1])
    for i in range(curr.shape[1]):
        column = curr[:,i]
        # hash value is just the bucket number
        hash_value = hash(column.tostring()) % k
        if hash_value not in dict:
            dict[hash_value] = set()
        dict[hash_value].add(i)
            
# m: number of rows of the matrix
# n: number of columns of the matrix
# num_hashes: number of hash functions we use when doing min hash
# docid_to_signature: a map from user_id to a set of 
# all the indexes of shingles it contains
# return a signature matrix of documents
def generate_signature_matrix(m, n, num_hashes, docid_to_signature):
    sig_matrix = np.full((num_hashes, len(docid_to_signature)), np.inf)
    # generate hash function
    rand_matrix = np.random.randint(0,2^32,(num_hashes,2))

    for i in range(m):
        # for each column
        for j in range(n):
            if i in docid_to_signature[j]:
                for k in range(num_hashes):
                    # apply min hash algorithm
                    sig_matrix[k][j] = min(sig_matrix[k][j], (rand_matrix[k][0] * i + rand_matrix[k][1]) % m)
    return sig_matrix

# sig_matrix: signature matrix of the documents
# num_hashes: number of hash functions we use when doing min hash
# b: number of bands
# k: number of buckets
def lsh(sig_matrix, num_hashes, b, k=sys.maxsize):
    # LSH process
    r = int(num_hashes / b)
    # k = sys.maxsize
    # num of buckets
    dicts = [dict() for i in range(b)]
    for i in range(b):
        # for each band
        curr = sig_matrix[i*r:(i+1)*r,:]
        # a bucket is a dict from bucketid to set of docids
        create_buckets_for_one_band(curr, k, dicts[i])
    return show_result(dicts)

# return all the similar pairs in dicts
def show_result(dicts):
    similar_pairs = set()
    for word_dict in dicts:
        for key in word_dict:
            if len(word_dict[key]) > 1:
                similars = word_dict[key]
                for doc1 in similars:
                    for doc2 in similars:
                        if doc1 < doc2:
                            similar_pairs.add((doc1, doc2))
    return similar_pairs

# naively return all the similar pairs
def show_actual_result(docid_to_signature, threshold): 
    similar_pairs = set()
    for doc1 in docid_to_signature:
        for doc2 in docid_to_signature:
            overlap = docid_to_signature[doc1].intersection(docid_to_signature[doc2])
            union = docid_to_signature[doc1] | docid_to_signature[doc2]
            if (len(overlap) / len(union) >= threshold and doc1 < doc2):
                similar_pairs.add((doc1, doc2))
    return similar_pairs

# compute similarity data of given pairs of documents
def compute_threshold(docid_to_signature, similar_pairs, threshold):
    similarity_list = list()
    actual_similar = set()
    for doc1, doc2 in similar_pairs:
        overlap = docid_to_signature[doc1].intersection(docid_to_signature[doc2])
        union = docid_to_signature[doc1] | docid_to_signature[doc2]
        # print(overlap)
        # print(union)
        # print(matching_size)
        # print(all_tokens)
        similarity = len(overlap) / len(union)
        similarity_list.append(similarity)
        if similarity >= threshold:
            actual_similar.add((doc1,doc2))
    return np.mean(similarity_list), actual_similar

# take in a set of document id pairs and return the corresponding document name pairs
def id_to_name(similar_pairs, docid_to_docname):
    name_set = set()
    for doc1, doc2 in similar_pairs:
        name_set.add((docid_to_docname[doc1], docid_to_docname[doc2]))
    return name_set


if __name__ == "__main__":
    # path = "files/Figure Plagiarism corpus/Shape_Based_Figure Plagiarism/Plagiarised figures/Exact copy"
    # path = "files"
    # path = "corpus_test"
    # num_hashes = int(input("Enter number of hash functions: "))
    # path = input("Enter file directory: ")
    # threshold = float(input("Enter threshold: "))

    # section 6.1
    num_hashes = 100
    path = "corpus-20090418"
    num_of_docs_list = np.arange(10,101,10)
    # num_of_docs_list = [500, 600, 700, 800, 900, 1000]
    lsh_time_list = list()
    naive_time_list = list()
    for num_of_docs in num_of_docs_list:
        # print(num_of_docs)
        docid_to_shingling_set, all_tokens, docid_to_docname = shingle(num_of_docs, path, k=8)
        docid_to_signature = create_doc_signature(docid_to_shingling_set, all_tokens)


        threshold = 0.6
        x = symbols('x')
        expr = (1/x)**(x/num_hashes)-threshold
        b = int(solve(expr)[0])

        
        sig_matrix = generate_signature_matrix(len(all_tokens), len(docid_to_signature), num_hashes, docid_to_signature)
        start_time = time.time()
        similar_pairs = lsh(sig_matrix, b=20, num_hashes=num_hashes)
        lsh_finish = time.time()
        actual_similar_pairs = show_actual_result(docid_to_signature, threshold)
        naive_finish = time.time()

        lsh_time_list.append(lsh_finish - start_time)
        naive_time_list.append(naive_finish - lsh_finish)
    plt.plot(num_of_docs_list, lsh_time_list, label="lsh")
    plt.plot(num_of_docs_list, naive_time_list, label = "naive")
    plt.xlabel("input size")
    plt.ylabel("run time")
    plt.legend()
    plt.savefig("6.1.png")

    # section 6.3
    num_hashes = 100
    num_of_docs = 2
    thresholdlist = np.arange(0.1,1.0,0.1)
    false_postive = list()
    for i in range(len(thresholdlist)):
        threshold = thresholdlist[i]-0.1
        path = "files/" + str((i+1)/10)
        docid_to_shingling_set, all_tokens, docid_to_docname = shingle(num_of_docs, path, k=3)
        docid_to_signature = create_doc_signature(docid_to_shingling_set, all_tokens)
        
        lsh_res = 0
        for i in range (1000):
            sig_matrix = np.zeros((num_hashes, len(docid_to_signature)))
            for i in range(num_hashes):
                # create permutation
                permutation = np.arange(len(all_tokens))
                np.random.shuffle(permutation)
                sig_matrix[i] = min_hash(docid_to_signature, permutation)
            similar_pairs = lsh(sig_matrix, b=20, num_hashes=num_hashes)
            mean, lsh_similar = compute_threshold(docid_to_signature, similar_pairs, threshold)
            lsh_res += len(lsh_similar)
        false_postive.append(lsh_res / 1000)
    plt.plot(thresholdlist, false_postive)
    plt.xlabel("threshold")
    plt.ylabel("false postive")
    plt.savefig("6.3.png")