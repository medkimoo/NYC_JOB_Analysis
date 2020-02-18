import pandas as pd
import numpy as np
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import decomposition
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
# import matplotlib
from matplotlib import pyplot as plt
from data_explore import Nyc_job_read
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.mixture import GaussianMixture


# Lemmatizer object 
lemmatizer = WordNetLemmatizer()

# extract stop words
stop_words = set(stopwords.words('english'))

# create an object of stemming function
stemmer = SnowballStemmer("english")

# create a standar scaler object
scaler = StandardScaler()

#PCA component
n_components = 1001

def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)

def stopwords(text):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in stop_words]
    # joining the list of words with space separator
    return " ".join(text)


def stemming(text):    
    '''a function which stems each word in the given text'''
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text) 

def lemmatizing(text):
    '''a function which stems each word in the given text'''
    text = [lemmatizer.lemmatize(word) for word in text.split()]
    return " ".join(text) 

def plot_silhouette(data,cluster_labels,title=""):
    silhouette_avg    = silhouette_score(data, cluster_labels)
    silhouette_values = silhouette_samples(data, cluster_labels)
    fig,ax= plt.subplots(figsize=(10,8))
    n_clusters = np.unique(np.array(cluster_labels))
    y_lower = 10
    for i in n_clusters:
        cluster_silhouette_values = silhouette_values[cluster_labels == i]
        cluster_silhouette_values.sort()
        size_cluster = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster
        color = cm.nipy_spectral(float(i) /1)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                  0, cluster_silhouette_values,
                  facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster, str(i))
        y_lower = y_upper + 10 

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([]) 
    ax.set_xlim((-1,1))
    ax.set_ylim([0, len(data) + (len(n_clusters) + 1) * 10])
    ax.text(-0.12, (len(data) + (len(n_clusters) + 1) * 10)/2.,
            "Cluster label",rotation=90)
    ax.set_title(f"The silhouette plot {title}")
    ax.set_xlabel("The silhouette coefficient values")
    
    
def run_clusturing(data,method,parameter_name,parameter_values,cases_plot=[]):
    print(f'\n--------------------{method} --------------------------------------')
    print(f'- run for different {parameter_name}')
    Solvers = []
    for param in parameter_values:
        if method =="KMeans":
            solver = KMeans(n_clusters=param,init='k-means++')
        elif method == "MeanShift":
            solver = MeanShift(bandwidth=param)
        elif method == "AgglomerativeClustering":
            solver = AgglomerativeClustering(n_clusters=param,linkage='average')
        elif method == "GaussianMixture":
            solver = GaussianMixture(n_components=param, covariance_type='full')
        else:
            print('method not found !!!')
            return
        
        cluster_labels   = solver.fit_predict(data)
        silhouette_avg   = silhouette_score(data, cluster_labels)
        
        if method == "MeanShift":
            n_clusters = len(np.unique(cluster_labels))
            print(f"{parameter_name}={param:.2f}\tn_clusters found={n_clusters}")
        else:
            print(f"{parameter_name}={param}\tsilhouette_score={silhouette_avg:.2f}")
            
        if param in cases_plot:
            plot_silhouette(data,cluster_labels,title=f"{method} {parameter_name}={param}")
        
        Solvers.append(solver)
    return Solvers

plt.close("all")

data = Nyc_job_read('jobs.csv')
print("\n> Le nombre de lignes du dataset est: " , len(data.df), "lignes")
print("\n> Affichage de 5 premières lignes du dataset:")
print("\n", data.df.head(5))
print("\n> Affichage des features:")
print("\n", list(data.df.columns))


#data.df = data.df.set_index('Job ID')
df = data.df.copy() 
df['sep'] = ' '
df['Description'] = ''

for i in data.df.columns:
    df['Description'] = df['Description'] + df['sep'] + df[str(i)].astype(str)
    df = df.drop(i, axis = 1)
    
df = df.drop('sep', axis = 1)
    
df['Description'] = df['Description'].astype(str)
print("\n> Affichage des features concaténées dans une seule colonne Description:")
print("\n", df.head(5))

#print("\n> To test", df.iloc[i]['Description'])

for i in range(0,len(df)):

    # remove poncuation
    df.iloc[i] = df.iloc[i].apply(remove_punctuation)

    # remove stop words
    df.iloc[i] = df.iloc[i].apply(stopwords)

    # stemming
    df.iloc[i] = df.iloc[i].apply(lemmatizing)
    
    
print("\n> Affichage de la description après nettoyage:")    
print("\n", df.head(5))

print("\n ----- CountVectorizer Analyse------ ")
# create a count vectorizer object
count_vectorizer = CountVectorizer(analyzer = 'word',
                                   min_df = 10)

# fit the count vectorizer using the text data
train_data_count = count_vectorizer.fit_transform(df['Description']).toarray()

#Get the vocabulary
vocab = count_vectorizer.get_feature_names()

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_count, axis=0)

# Collect count in a dictionnary
counts = {}
for tag, count in zip(vocab, dist):
    counts = (tag, count)

#print(train_data.shape)
#print(count_vectorizer.get_feature_names())

# collect the vocabulary items used in the vectorizer
dictionary = count_vectorizer.vocabulary_.items()

# top 2000 most frequently repeated word dataframe
frequent_word = pd.DataFrame(dictionary, columns = ['Word', 'Index'])
frequent_word = frequent_word.set_index('Word')
print("\n>  les top 10 mots les plus fréquents sont:")   
print(frequent_word.head(10))


print("\n ----- TfidfVectorizer Analyse ------ ")
# create the object of tfid vectorizer
tfid_vectorizer = TfidfVectorizer("english")
# fit the vectorizer using the text data
train_data_tfid = tfid_vectorizer.fit_transform(df['Description']).toarray()

# feature names
vocab_tfid =  tfid_vectorizer.get_feature_names()

# Sum up the tfid score of each vocabulary word
dist_tfid = np.sum(train_data_tfid, axis=0)

# Collect count in a dictionnary
tfid = {}
for tag, score in zip(vocab_tfid, dist_tfid):
    counts = (tag, score)

# collect the vocabulary items used in the vectorizer
dictionary2 = tfid_vectorizer.vocabulary_.items()

# top word with tfid score 
frequent_word2 = pd.DataFrame(dictionary2, columns = ['Word', 'Index'])
frequent_word2 = frequent_word2.set_index('Word')
print("\n>  les top 10 mots les plus pertinents sont:")   
print(frequent_word2.head(10))
 
print("\n ----- N-Gram Analyse ------ ")
#n-gram (x, x)
#train_data2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))

print("\n ----- Scaling  ------ ")
train_data_count = scaler.fit_transform(train_data_count)
train_data_tfid = scaler.fit_transform(train_data_tfid)
print("\n data scaled ")

# PCA decomposition
print("\n ---- PCA decomposition on train_data_count -----")
pca = decomposition.PCA(n_components)
pca_train_data_count = pca.fit_transform(train_data_count)

variance_ratio       = pca.explained_variance_ratio_
cum_variance_ratio   = np.cumsum(variance_ratio)

ps = pd.DataFrame(pca_train_data_count)
print(ps.head()) 

print(f'Linear PCA with {pca.n_components_} compenents')
print('variance cumulative explained ratio:')
print(cum_variance_ratio[-1])

# PCA decomposition tfid
print("\n ---- PCA decomposition on train_data_tfid -----")
pca = decomposition.PCA(n_components)
pca_train_data_tfid = pca.fit_transform(train_data_tfid)

variance_ratio       = pca.explained_variance_ratio_
cum_variance_ratio   = np.cumsum(variance_ratio)

ps_tfid = pd.DataFrame(pca_train_data_tfid)
print(ps_tfid.head()) 

print(f'Linear PCA with {pca.n_components_} compenents')
print('variance cumulative explained ratio:')
print(cum_variance_ratio[-1])


# K-means
print('\n ------ k-mean clustring on train_data_count -----------')
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(ps)
    print('n_cluster is: ', n)
    inertia.append(algorithm.inertia_)

# Elbow method
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia (some of squared errors')
plt.show()

print('\n ------ k-mean clustring on train_data_tfid -----------')
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(ps_tfid)
    print('n_cluster is: ', n)
    inertia.append(algorithm.inertia_)

# Elbow method
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia (some of squared errors')
plt.show()

# Silouhette
K = range(2,20)
k_silhouette = [2,3,4,5,6,7,8,9,10]
solvers = run_clusturing(data= ps, method="KMeans",
                         parameter_name="n_clusters",
                         parameter_values=K,
                         cases_plot=k_silhouette)
# Silouhette tfid
K = range(2,20)
k_silhouette = [2,3,4,5,6,7,8,9,10]
solvers = run_clusturing(data= ps_tfid, method="KMeans",
                         parameter_name="n_clusters",
                         parameter_values=K,
                         cases_plot=k_silhouette)

R  = np.arange(6,10)
# Mean shift --------------------------------------------------------------
R_silhouette = [7]
solvers = run_clusturing(data=ps, method="MeanShift",
                             parameter_name="bandwidth",
                             parameter_values=R
                             )

solvers = run_clusturing(data=ps_tfid, method="MeanShift",
                             parameter_name="bandwidth",
                             parameter_values=R
                             )

    
# Agglomerative Clustering ------------------------------------------------
K = range(2,11)
k_silhouette = [10]
solvers = run_clusturing(data=ps, method="AgglomerativeClustering",
                             parameter_name="n_clusters",
                             parameter_values=K,
                             cases_plot=k_silhouette)
solvers = run_clusturing(data=ps_tfid, method="AgglomerativeClustering",
                             parameter_name="n_clusters",
                             parameter_values=K,
                             cases_plot=k_silhouette)

# Gaussian Mixture --------------------------------------------------------
K = range(2,11)
k_silhouette = K
solvers = run_clusturing(data=ps, method="GaussianMixture",
                             parameter_name="n_clusters",
                             parameter_values=K,
                             cases_plot=k_silhouette)
solvers = run_clusturing(data=ps_tfid, method="AgglomerativeClustering",
                             parameter_name="n_clusters",
                             parameter_values=K,
                             cases_plot=k_silhouette)

print(ps.shape)
print(ps_tfid.shape)
print(df['Description'].type)

