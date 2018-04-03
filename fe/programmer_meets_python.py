from multiprocessing.dummy import Pool as ThreadPool
import concurrent.futures
import pandas as pd

pool = ThreadPool(4)
#results = pool.map(urllib2.urlopen, urls)
pool.close()
pool.join()


def getdiff(df):
    name="diff1"
    return name, df["a"].diff(periods=1)


def getdiff2(df, name, nam2):
    print(nam2)
    return name, df["a"].diff(periods=-1)


def directly_doit(df):
    df["direct"] = df["a"].diff(periods=-1)


df1 = pd.DataFrame({
    "a": [1,2,3,3,4,4,5,5,4,3],
    "x": [1,2,3,4,5,6,7,8,9,10]
})


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    future_list = []
    future_list.append(executor.submit(getdiff, df1))
    future_list.append(executor.submit(getdiff2, df1, "omg", "printer"))
    executor.submit(df1)

    for a_future in future_list:
        col_name, series = a_future.result()
        df1[col_name] = series

print(df1)

