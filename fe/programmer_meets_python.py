from multiprocessing.dummy import Pool as ThreadPool
import concurrent.futures
import pandas as pd

pool = ThreadPool(4)
#results = pool.map(urllib2.urlopen, urls)
pool.close()
pool.join()


def getdiff(df):
    return df1["a"].diff(periods=1)





df1 = pd.DataFrame({
    "a": [1,2,3,3,4,4,5,5,4,3],
    "x": [1,2,3,4,5,6,7,8,9,10]
})


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    executor.map()


