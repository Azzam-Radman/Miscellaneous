from sklearn.manifold import TSNE
embeddings = TSNE(n_components=2,init = 'pca').fit_transform(embeddings)
sns.scatterplot(x=embeddings[:,0],y=embeddings[:,1],hue= labels ,legend='full')
