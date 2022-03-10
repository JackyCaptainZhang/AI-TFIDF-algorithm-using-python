import collections
import os
from nltk.stem.porter import *
import math
import numpy as np

global dirs
dirs = 'C:/Users/dyzhj/PycharmProjects/INT 104/Assignment 2/dataset'  # the path of the root folder(can be changed)
temlist = []
lis = []


def open_files(dirs):
    foldername = os.listdir(dirs)  # read the sub-folder name under the root folder
    global c
    c = -1  # to record how many files are there in the dataset,start with -1, and it is the index of the 'lis' below
    text_create('datasets')
    text_create('unique words')
    global fil
    fil = open('datasets', 'w')
    global FILE
    FILE = open('unique words', 'w')
    for i in range(len(foldername)):
        path = dirs + '/' + foldername[i]
        path_list = os.listdir(path)  # combine the path and read folders one by one
        fil.write('\n' + 'Class Labels : ' + foldername[i] + '\n' + '\n')  # write the class labels into the 'datasets'
        global file
        for file in path_list:  # read files in folders one by one
            f = open(path + '/' + file, 'r', encoding='Latin1')  # open files in folders one by one
            li = f.readlines()
            str1 = ''.join(li)  # turn the li(list) into str1(string)
            c += 1
            text_processing(str1)
            print("文件读取数量：" + str(c + 1))
    print('Unique Words生成中  等待大概40s')
    setA = set(temlist)  # remove the duplicated unique words in the lists
    global wordlists
    wordlists = list(setA)  # obtain the list of unique words from all files
    wordlists.sort(key=temlist.index)  # sort the order of the list
    sep = ' '
    FILE.write(sep.join(wordlists))  # write the unique word into the 'unique word' file


def text_create(name):  # create the text that store all the content of files
    path_ = dirs
    full_path = name
    file = open(full_path, 'w')


def text_processing(str1):
    regEx = re.compile(r'[^a-z]')  # define the regex expression of the non-alphabet letters
    stopwords = (open('stopwords.txt', 'r', encoding='Latin1')).readlines()  # read the stopwords text
    stopwords = map(lambda x: x.strip(), stopwords)  # remove the '\n' in the list
    stoplist = list(set(stopwords))  # store all the stopwords in the list
    stemmer = PorterStemmer()  # create a porter stemmer
    str2 = str1.lower()  # convert the words into lower case form
    str3 = regEx.split(str2)  # split the text as string type and the separator is non-alphabet letters
    lists = [item for item in str3 if item not in stoplist]  # remove the words which are in the stopwords list
    for i in range(lists.count('')):  # remove the '' in the text lists
        lists.remove('')
    plurals = lists
    lists = [stemmer.stem(plural) for plural in plurals]  # remove the word suffix by using stemmer potter
    sep = ' '
    fil.write('file name ' + file + ': (' + sep.join(lists) + ')' + '\n')  # write the file content into the text
    global lis
    lis.append(lists)  # store each file in the list individually to operate the TFIDF analysis below, the index is c
    global temlist
    temlist.extend(lists)  # get the lists of unique words from all files


# by now
# (c+1)(int) is the number of the file; lis[](list) contains each file content individually; woredlists(list) contains
# the list of unique  words from all files

# Main TFIDF analysis program!!!!!!!!!!!!!!
open_files(dirs)
tuplelist = tuple(wordlists)  # turn the list into tuples
dic1 = {}
dic2 = {}
tem = tuplelist
dic1 = {}.fromkeys(tuplelist, 0)  # set two dic according to the words in the unique words lists, keep the words order
# unchanged and the default val is 0
dic2 = {}.fromkeys(tuplelist, 0)
D1 = np.zeros((c + 1, len(dic1)))  # D1存储着每个单词的TF
for i in range(c + 1):
    g = -1
    dic2 = {}.fromkeys(tuplelist, 0)  # 每一次读取新的文章的时候初始化字典值
    dic = collections.Counter(lis[i])  # 计算该单词的TF
    dic2.update(dic)  # 写入词典
    for ke in dic2:
        g += 1
        D1[(i, g)] = dic2[ke]  # 将TF值一次写入D1中
        if ke in lis[i]:
            dic1[ke] += 1  # dic1储存着出现该单词的文章的数量
    print('TFIDF字典生成进度：' + str(i / c))
D2 = np.zeros((c + 1, len(dic1)))  # D2存储的是不考虑文章长度的TFIDF值
D = np.zeros((c + 1, len(dic1)))  # D存储的是最终考虑文章长度的TFIDF值！！！！！！！！
for s in range(c + 1):
    h = -1
    length = 0
    for k in dic1:
        h += 1
        a = math.log((c + 1) / dic1[k])  # get IDF Value
        b = D1[(s, h)] * a  # TF*IDF Value
        D2[(s, h)] = b  # Store the TF*IDF Value（without considering the length of the each article）
        length += b * b  # 考虑上每一篇文章的长度
    m = -1
    for j in dic1:
        m += 1
        o = (D2[(s, m)] / math.sqrt(length))  # 最终的TFIDF值
        D[(s, m)] = o  # 写入D
    print('TFIDF计算进度：' + str(s / c))

np.savez('train-20ng.npz', X=D)  # 保存D矩阵为npz文件
a = np.load('train-20ng.npz')
lst = a.files
print(tuplelist)
for item in lst:  # 遍历显示npz文件内容
    print(a[item])
