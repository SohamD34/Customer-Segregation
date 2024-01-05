def person_df(dft):
    #We are not keeping a column names country as it is not of utility for clustering

    data_customer = pd.DataFrame({'No. of visits': dft.groupby('CustomerID')['InvoiceNo'].nunique(),
                                'Mean_purch': dft.groupby('CustomerID')['Total Price'].mean(),
                                'Sum_purch': dft.groupby('CustomerID')['Total Price'].sum(),
                                'Item_count': dft.groupby('CustomerID')['StockCode'].nunique(),
                                'Total_quantity':dft.groupby('CustomerID')['Quantity'].sum(),
                                'Avg_quant': dft.groupby('CustomerID')['Quantity'].mean()})
    
    data_customer['CustomerID'] = data_customer.index
    data_customer.reset_index(inplace=True,drop=True)
    return(data_customer)



def df_cluster_labels(dfi,k_clusters = 3):

    #Returns a dataframe with given clusters labels

    dpx = dfi.copy(deep=True)
    dpxx = dpx.copy(deep=True)
    dpxx.drop(['CustomerID'],axis=1)

    kmeans = KMeans(n_clusters=k_clusters)
    kmeans.fit(dpxx)
    
    labels_k = kmeans.labels_
    labels_k = list(labels_k)
    dpx['cluster_labels'] = labels_k
    centroids = kmeans.cluster_centers_
    
    return(dpx)



def descrip_dict(og_df,person_df):

    #Returns a dataframe which has CustomerIDs and corresponding
    #Descriptions of items
    #og_df is the dataframe from where the person_df is extracted
    #Dictionary will be {labells:{cust_ids:[desc]}}

    labells = list(person_df['cluster_labels'].unique())
    cust_id=[]
    desc=[]
    dtf=[]

    for i in labells:
        c_id = person_df[ person_df['cluster_labels']==i]['CustomerID']

        # Is a dataframe that will contain only customer IDs from cluster i

        d=[]
        for j in c_id:
            dpx = og_df[ og_df['CustomerID']==j]
            des = list(dpx['Description'])
            d.append([j,des])

        g = pd.DataFrame(d,columns=['CustomerID','Description'])
        dtf.append(g)

    return(dtf)    


def processlis(lis):
    out = []
    for i in lis:
    size = i.to_numpy().shape[0]
    if(size>=100):
        out.append(i)
    return out


def plot_dendrogram(model, **kwargs):

    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack( [model.children_, model.distances_, counts] ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



def Agglocls(df,cluster_count):
    model_new = AgglomerativeClustering(n_clusters=cluster_count,linkage='ward').fit(df)
    pred = model_new.fit_predict(df)

    plt.figure(figsize =(5, 5))
    plt.scatter(df['X1'], df['X0'], c = pred,cmap='rainbow')
    plt.show()
    clus = np.array(pred)
    clus = clus.reshape(-1,1)
    columns = list(df.columns)

    new_dvt['Clus'] = clus
    new_dvt.head()

    out = {}
    for i in range(cluster_count):

    clust = new_dvt.loc[new_dvt['Clus'] == i]
    print(clust.to_numpy().shape)
    out[i] = clust

    return out


def AggloDendo(df):
    model_ = AgglomerativeClustering(distance_threshold=0,n_clusters=None,linkage='ward').fit(df)
    plot_dendrogram(model_, truncate_mode="level", p=4)


def AggloPca(df,cluster_count,dvt,i,j):
    model_new = AgglomerativeClustering(n_clusters=cluster_count,linkage='ward').fit(df)
    #pred = model_new.fit_predict(df)

    pred = model_new.fit_predict(df)
    plt.figure(figsize =(5, 5))
    plt.scatter(df[i], df[j], c = pred,cmap='rainbow')
    plt.show()

    clus = np.array(pred)
    clus = clus.reshape(-1,1)
    #columns = list(final_dvt.columns)
    df['CustomerID']=dvt['CustomerID']

    df['cluster_labels'] = clus
    out = {}
    for i in range(cluster_count):

    clust = df.loc[df['cluster_labels'] == i]
    print(clust.to_numpy().shape)
    out[i] = clust

    return out


def apriori_rules(dfkk,support):
  ddp = dfkk.iloc[:100,:]
  transactions = ddp["Description"].to_numpy()

  te = TransactionEncoder()
  te_ary = te.fit(transactions).transform(transactions)
  dfp = pd.DataFrame(te_ary, columns=te.columns_)

  # Generating frequent itemsets
  frequent_itemsets = apriori(dfp, min_support=support, use_colnames=True)

  # Generating association rules
  rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

  rules = pd.DataFrame(rules)

  out = (pd.DataFrame(frequent_itemsets),rules)
  return out
